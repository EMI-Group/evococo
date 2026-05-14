import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.operators.mutation import polynomial_mutation

class MOEAD_DRA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Bug #2: Ceil semantics for T and nr
        T = (pop_size + 9) // 10
        self.nr = (pop_size + 99) // 100
        
        # Weight Vectors & Neighborhood (Bug #13, #19)
        w, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = actual_n
        w = w.to(device)
        dist = torch.cdist(w, w)
        # Get indices of T nearest neighbors
        b = torch.topk(dist, T, largest=False, dim=1).indices.to(torch.int32)

        # Initialize State
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), 0.0, device=device))
        self.z = Mutable(torch.full((n_objs,), 1e10, device=device))
        self.pi = Mutable(torch.ones(self.pop_size, device=device))
        self.old_obj = Mutable(torch.zeros(self.pop_size, device=device))
        self.w = Mutable(w)
        self.b = Mutable(b)
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _tchebycheff_scalarization(self, fit: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Bug #10: MATLAB dim=2 -> PyTorch dim=1
        # fit: (N, M), z: (M,), w: (N, M)
        return torch.max(torch.abs(fit - z.unsqueeze(0)) * w, dim=1).values

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0).values
        self.old_obj = self._tchebycheff_scalarization(self.fit, self.z, self.w)

    def step(self) -> None:
        self.gen = self.gen + 1
        device = self.lb.device
        N = self.pop_size
        M = self.n_objs
        D = self.lb.numel()

        # MOEA/D-DRA performs 5 sub-generations per step
        for sub_gen in range(5):
            # 1. Subproblem Selection (DRA Logic)
            # Boundary subproblems are always selected
            is_boundary = (self.w <= 1e-6).sum(dim=1) == (M - 1)
            boundary_indices = torch.where(is_boundary)[0]
            
            num_to_select = (N // 5) - boundary_indices.numel()
            # Bug #27, #31: Tournament selection with [-pi]
            # tournament_selection_multifit expects (num_to_select, [fitness_tensors])
            tournament_indices = tournament_selection_multifit(num_to_select, [-self.pi], tournament_size=10)
            I = torch.cat([boundary_indices, tournament_indices])
            num_I = I.numel()

            # 2. Evolution
            rand_p = torch.rand(num_I, device=device)
            neigh_mask = rand_p < 0.9
            
            # Parent selection
            p_idx = torch.zeros((num_I, 2), dtype=torch.long, device=device)
            
            # Neighborhood selection indices
            neigh_rand = torch.stack([torch.randperm(self.b.shape[1], device=device)[:2] for _ in range(num_I)])
            neigh_parents = self.b[I.unsqueeze(1), neigh_rand].to(torch.long)
            
            # Global selection indices
            global_parents = torch.randint(0, N, (num_I, 2), device=device)
            
            p_idx = torch.where(neigh_mask.unsqueeze(1), neigh_parents, global_parents)
            
            # DE Variation: Offspring = pop[I] + 0.5 * (pop[p1] - pop[p2])
            # Then Polynomial Mutation
            x_i = self.pop[I]
            x_r1 = self.pop[p_idx[:, 0]]
            x_r2 = self.pop[p_idx[:, 1]]
            
            # DE/current-to-rand/1 style or similar to PlatEMO's OperatorDE
            off_pop = x_i + 0.5 * (x_r1 - x_r2)
            off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
            off_pop = clamp(off_pop, self.lb, self.ub)
            
            off_fit = self.evaluate(off_pop)

            # 3. Update Ideal Point
            self.z = torch.min(self.z, torch.min(off_fit, dim=0).values)

            # 4. Neighborhood Replacement
            idx_p_all = self.b[I].to(torch.long) # (num_I, T)
            
            # Vectorized Tchebycheff for all neighbors of all selected I
            # self.fit[idx_p_all]: (num_I, T, M)
            # self.w[idx_p_all]: (num_I, T, M)
            # self.z: (M,)
            g_old = torch.max(torch.abs(self.fit[idx_p_all] - self.z.unsqueeze(0).unsqueeze(0)) * self.w[idx_p_all], dim=2).values
            # off_fit: (num_I, M) -> (num_I, 1, M)
            g_new = torch.max(torch.abs(off_fit.unsqueeze(1) - self.z.unsqueeze(0).unsqueeze(0)) * self.w[idx_p_all], dim=2).values
            
            replace_mask = g_new <= g_old
            
            # Apply replacement limit nr
            cum_replace = torch.cumsum(replace_mask.to(torch.int32), dim=1)
            final_replace_mask = replace_mask & (cum_replace <= self.nr)
            
            # Sequential update for subproblems to handle overlapping neighborhoods correctly
            for k in range(num_I):
                mask_k = final_replace_mask[k]
                if mask_k.any():
                    target_indices = idx_p_all[k][mask_k]
                    self.pop[target_indices] = off_pop[k]
                    self.fit[target_indices] = off_fit[k]

        # 5. Utility Update (Every 10 Generations)
        if self.gen % 10 == 0:
            new_obj = self._tchebycheff_scalarization(self.fit, self.z, self.w)
            # Bug #12: Safe division
            delta = (self.old_obj - new_obj) / (self.old_obj + 1e-6)
            
            pi_mask = delta < 0.001
            # Update utility based on improvement
            pi_update = (0.95 + 0.05 * delta / 0.001) * self.pi
            self.pi = torch.where(pi_mask, pi_update, torch.ones_like(self.pi))
            self.old_obj = new_obj

if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_DRA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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