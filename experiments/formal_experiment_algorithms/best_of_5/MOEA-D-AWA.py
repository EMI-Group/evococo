import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evomo.operators.selection import non_dominate_rank
from evox.operators.sampling import uniform_sampling

class MOEADAWA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_fe: int = 10000, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.max_fe = max_fe
        D = lb.numel()
        
        # Weight Initialization (Bug #13 Compliance)
        W_raw, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = int(actual_n)
        W_raw = W_raw.to(device)
        
        # Hyperparameters (Bug #2 Compliance)
        self.T = (self.pop_size + 9) // 10
        self.nr = (self.pop_size + 99) // 100
        self.nus = (self.pop_size + 19) // 20 # nus = 0.05 * N

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        
        # Weight Transformation (Bug #12 Compliance)
        W_inv = 1.0 / (W_raw + 1e-6)
        self.W = Mutable(W_inv / (torch.sum(W_inv, dim=1, keepdim=True) + 1e-6))
        
        # Neighborhood (Bug #19 Compliance)
        dist_w = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(dist_w, k=self.T, largest=False).indices.to(torch.int32))
        
        self.z = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.pi = Mutable(torch.ones((self.pop_size, 1), device=device))
        self.old_obj = Mutable(torch.full((self.pop_size, 1), torch.inf, device=device))
        
        # Archive (EP)
        self.archive_pop = Mutable(torch.zeros((int(1.5 * self.pop_size), D), device=device))
        self.archive_fit = Mutable(torch.full((int(1.5 * self.pop_size), n_objs), torch.inf, device=device))
        self.archive_size = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        
        self.fe = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0).values
        self.fe = self.fe + self.pop_size
        
        # Initial Tchebycheff
        diff = torch.abs(self.fit - self.z)
        self.old_obj = torch.max(diff * self.W, dim=1, keepdim=True).values
        
        # Initial Archive
        rank = non_dominate_rank(self.fit)
        mask = rank == 1
        num_nd = torch.sum(mask.int())
        self._update_archive(self.pop[mask], self.fit[mask], num_nd)

    def _update_archive(self, off_pop, off_fit, num_off):
        # Merge
        combined_pop = torch.cat([self.archive_pop[:self.archive_size], off_pop], dim=0)
        combined_fit = torch.cat([self.archive_fit[:self.archive_size], off_fit], dim=0)
        
        # ND Sort
        rank = non_dominate_rank(combined_fit)
        nd_mask = rank == 1
        
        # Deadlock Breaker: if no non-dominated solutions, take all (should not happen with rank 1)
        if torch.sum(nd_mask.int()) == 0:
            nd_mask = torch.ones(combined_fit.shape[0], dtype=torch.bool, device=self.lb.device)
            
        new_pop = combined_pop[nd_mask]
        new_fit = combined_fit[nd_mask]
        
        # Pruning if size > 1.5N
        limit = int(1.5 * self.pop_size)
        n_current = new_pop.shape[0]
        
        if n_current > limit:
            dist = torch.cdist(new_fit, new_fit) + torch.eye(n_current, device=self.lb.device) * 1e6
            sorted_dist, _ = torch.sort(dist, dim=1)
            score = torch.prod(sorted_dist[:, :min(self.n_objs, n_current)], dim=1)
            keep_idx = torch.topk(score, k=limit).indices
            new_pop = new_pop[keep_idx]
            new_fit = new_fit[keep_idx]
            n_current = limit
            
        self.archive_pop[:n_current] = new_pop
        self.archive_fit[:n_current] = new_fit
        self.archive_size = torch.tensor(n_current, dtype=torch.int32, device=self.lb.device)

    def _update_pi(self, old_obj, new_obj, pi):
        delta = (old_obj - new_obj) / (old_obj + 1e-6)
        new_pi = torch.where(delta > 0.001, torch.ones_like(pi), (0.95 + 0.05 * delta / 0.001) * pi)
        return new_pi

    def _awa_update_weights(self):
        # 1. Weight Pruning (Sparsity of self.fit)
        dist_pop = torch.cdist(self.fit, self.fit) + torch.eye(self.pop_size, device=self.lb.device) * 1e6
        min_dist = torch.min(dist_pop, dim=1).values
        
        actual_nus = min(self.nus, self.pop_size - 1)
        remove_idx = torch.topk(min_dist, k=actual_nus, largest=False).indices
        
        # 2. Weight Addition (EP relative to self.fit)
        ep_fit = self.archive_fit[:self.archive_size]
        ep_pop = self.archive_pop[:self.archive_size]
        
        if ep_fit.shape[0] > 0:
            dist_ep_pop = torch.cdist(ep_fit, self.fit)
            min_dist_ep = torch.min(dist_ep_pop, dim=1).values
            
            actual_add = min(actual_nus, ep_fit.shape[0])
            add_idx = torch.topk(min_dist_ep, k=actual_add).indices
            
            target_remove_idx = remove_idx[:actual_add]
            self.pop[target_remove_idx] = ep_pop[add_idx]
            self.fit[target_remove_idx] = ep_fit[add_idx]
            
            new_w_inv = 1.0 / (ep_fit[add_idx] + 1e-6)
            new_w = new_w_inv / (torch.sum(new_w_inv, dim=1, keepdim=True) + 1e-6)
            self.W[target_remove_idx] = new_w
            
            # Recompute Neighborhood
            dist_w = torch.cdist(self.W, self.W)
            self.B = torch.topk(dist_w, k=self.T, largest=False).indices.to(torch.int32)

    def step(self) -> None:
        device = self.lb.device
        # 1. Mating Selection (Utility-based)
        is_boundary = (self.W < 1e-3).sum(dim=1) == (self.n_objs - 1)
        boundary_idx = torch.where(is_boundary)[0]
        n_boundary = boundary_idx.numel()
        
        # Tournament selection for the rest (Fixing the crash by squeezing pi to 1D)
        n_needed = max(0, self.pop_size - n_boundary)
        # tournament_selection_multifit expects a list of 1D tensors for fitness
        tour_idx = tournament_selection_multifit(n_needed, [-self.pi.squeeze(-1)], tournament_size=2)
        
        # Generate Offspring
        rand_mask = torch.rand(self.pop_size, device=device) < 0.9
        
        neigh_rand = torch.randint(0, self.T, (self.pop_size, 2), device=device)
        pop_rand = torch.randint(0, self.pop_size, (self.pop_size, 2), device=device)
        
        p1_idx = torch.where(rand_mask, self.B[torch.arange(self.pop_size), neigh_rand[:, 0]].long(), pop_rand[:, 0])
        p2_idx = torch.where(rand_mask, self.B[torch.arange(self.pop_size), neigh_rand[:, 1]].long(), pop_rand[:, 1])
        
        offspring = simulated_binary(torch.cat([self.pop[p1_idx], self.pop[p2_idx]], dim=0), pro_c=1.0, dis_c=20.0)[:self.pop_size]
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        off_fit = self.evaluate(offspring)
        self.fe = self.fe + self.pop_size
        
        self.z = torch.min(self.z, torch.min(off_fit, dim=0).values)
        
        # Tchebycheff Update
        for i in range(self.pop_size):
            P = self.B[i].long()
            g_old = torch.max(torch.abs(self.fit[P] - self.z) * self.W[P], dim=1).values
            g_new = torch.max(torch.abs(off_fit[i] - self.z) * self.W[P], dim=1).values
            
            replace_mask = g_new <= g_old
            valid_idx = P[replace_mask]
            idx_to_replace = valid_idx[:self.nr]
            
            self.pop[idx_to_replace] = offspring[i]
            self.fit[idx_to_replace] = off_fit[i]
            
        new_obj_vals = torch.max(torch.abs(self.fit - self.z) * self.W, dim=1, keepdim=True).values
        self.pi = self._update_pi(self.old_obj, new_obj_vals, self.pi)
        self.old_obj = new_obj_vals
        
        self._update_archive(offspring, off_fit, self.pop_size)
        
        if self.fe > 0.8 * self.max_fe:
            self._awa_update_weights()

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADAWA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    jit_state_step()

    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")