import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted

class MOEAURAW(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, delta: float = 0.9, nr: int = 2, nEP: int = 100, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.D = lb.numel()
        self.T = T
        self.delta = delta
        self.nr = nr
        self.nEP = nEP

        # 1. Greedy Weight Generation (Blueprint 3.A)
        pool_size = 5000
        pool = torch.rand(pool_size, n_objs, device=device)
        # Normalize pool to unit hyperplane
        pool = pool / (pool.sum(dim=1, keepdim=True) + 1e-6)
        
        selected_indices = torch.zeros(pop_size, dtype=torch.long, device=device)
        
        # Start with a random point
        start_idx = torch.randint(0, pool_size, (1,), device=device)[0]
        selected_indices[0] = start_idx
        
        # Greedy selection loop
        min_dists = torch.cdist(pool, pool[start_idx].unsqueeze(0)).squeeze(1)
        for i in range(1, pop_size):
            next_idx = torch.argmax(min_dists)
            selected_indices[i] = next_idx
            new_dist = torch.cdist(pool, pool[next_idx].unsqueeze(0)).squeeze(1)
            min_dists = torch.min(min_dists, new_dist)
            
        W = pool[selected_indices]
        # Transformation on W: W = 1./W / sum(1./W)
        W = 1.0 / (W + 1e-6)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-6)
        
        # Neighborhood
        dist_W = torch.cdist(W, W)
        B = torch.topk(dist_W, T, largest=False, sorted=True).indices

        # Initialize State
        self.pop = Mutable(torch.rand(pop_size, self.D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.zeros((pop_size, n_objs), device=device))
        self.W = Mutable(W)
        self.z = Mutable(torch.full((1, n_objs), 1e10, device=device))
        self.B = Mutable(B.to(torch.int32))
        self.archive_pop = Mutable(torch.zeros((nEP, self.D), device=device))
        self.archive_fit = Mutable(torch.zeros((nEP, n_objs), device=device))
        self.archive_size = Mutable(torch.zeros(1, dtype=torch.int32, device=device))
        self.iter = Mutable(torch.zeros(1, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        
        # Initial Archive is the non-dominated part of the population
        rank = non_dominate_rank(self.fit)
        mask = rank == 1
        initial_archive_pop = self.pop[mask]
        initial_archive_fit = self.fit[mask]
        
        # Handle size constraints
        num_found = initial_archive_pop.shape[0]
        actual_nEP = min(num_found, self.nEP)
        self.archive_pop[:actual_nEP] = initial_archive_pop[:actual_nEP]
        self.archive_fit[:actual_nEP] = initial_archive_fit[:actual_nEP]
        self.archive_size[0] = actual_nEP

    def _get_overcrowding_score(self, fit: torch.Tensor, k: int) -> torch.Tensor:
        n = fit.shape[0]
        dist = torch.cdist(fit, fit)
        # Avoid in-place fill_diagonal_ for JIT compatibility
        # Add a large value to the diagonal to ignore self-distances
        eye_inf = torch.eye(n, device=fit.device) * 1e10
        dist = dist + eye_inf
        
        actual_k = min(k, n - 1)
        # If only one individual, score is 0
        score = torch.where(
            torch.tensor(n > 1, device=fit.device),
            torch.prod(torch.topk(dist, actual_k, largest=False).values + 1e-6, dim=1),
            torch.zeros(n, device=fit.device)
        )
        return score

    def _greedy_furthest_selection(self, pool: torch.Tensor, current_set: torch.Tensor, n_to_add: int) -> torch.Tensor:
        min_dist_to_set = torch.cdist(pool, current_set).min(dim=1).values
        added_indices = torch.zeros(n_to_add, dtype=torch.long, device=pool.device)
        for i in range(n_to_add):
            idx = torch.argmax(min_dist_to_set)
            added_indices[i] = idx
            new_dist = torch.cdist(pool, pool[idx].unsqueeze(0)).squeeze(1)
            min_dist_to_set = torch.min(min_dist_to_set, new_dist)
        return added_indices

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        T = self.T

        # 1. Mating / Variation
        rand_mask = torch.rand(N, device=device) < self.delta
        local_indices = torch.gather(self.B, 1, torch.randint(0, T, (N, 2), device=device))
        global_indices = torch.randint(0, N, (N, 2), device=device)
        P_all = torch.where(rand_mask.unsqueeze(1), local_indices, global_indices).to(torch.long)
        
        # Generate offspring for each subproblem
        offspring = simulated_binary_half(torch.cat([self.pop[P_all[:, 0]], self.pop[P_all[:, 1]]], dim=0))
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        
        # 2. Environmental Selection (Tchebycheff)
        self.z = torch.min(self.z, off_fit.min(dim=0, keepdim=True).values)
        
        # Update the solutions in P by Tchebycheff approach
        # Note: MATLAB code updates Population(P) for each offspring i. 
        # In MOEA/D-URAW, P is either the neighborhood B(i,:) or a random permutation.
        for i in range(N):
            # Determine P for this offspring
            if torch.rand((), device=device) < self.delta:
                P = self.B[i].to(torch.long)
            else:
                P = torch.randperm(N, device=device)
            
            fit_i = off_fit[i]
            g_old = torch.max(torch.abs(self.fit[P] - self.z) * self.W[P], dim=1).values
            g_new = torch.max(torch.abs(fit_i - self.z) * self.W[P], dim=1).values
            
            improvement_mask = g_new <= g_old
            if improvement_mask.any():
                imp_indices = P[improvement_mask]
                # MATLAB: Population(P(find(g_old>=g_new,nr))) = Offsprings(i)
                # This means we take the first 'nr' indices where g_new <= g_old
                num_replace = min(self.nr, imp_indices.numel())
                sel_idx = imp_indices[:num_replace]
                
                self.pop[sel_idx] = offspring[i]
                self.fit[sel_idx] = fit_i

        # 3. Archive Update
        valid_archive_pop = self.archive_pop[:self.archive_size[0]]
        valid_archive_fit = self.archive_fit[:self.archive_size[0]]
        
        merged_pop = torch.cat([valid_archive_pop, offspring], dim=0)
        merged_fit = torch.cat([valid_archive_fit, off_fit], dim=0)
        merged_pop, u_idx = unique_rows_sorted(merged_pop)
        merged_fit = merged_fit[u_idx]
        
        rank = non_dominate_rank(merged_fit)
        mask_rank1 = rank == 1
        if mask_rank1.sum() == 0:
            mask_rank1 = torch.ones(len(merged_fit), dtype=torch.bool, device=device)
            
        new_ep_pop = merged_pop[mask_rank1]
        new_ep_fit = merged_fit[mask_rank1]
        
        if new_ep_pop.shape[0] > self.nEP:
            # Prune using overcrowding score (MATLAB uses product of distances)
            # We iteratively remove the worst until size is nEP
            curr_ep_fit = new_ep_fit
            curr_ep_pop = new_ep_pop
            # Vectorized pruning: although MATLAB does it one by one, 
            # we can use topk for efficiency if we don't need to re-calculate scores
            scores = self._get_overcrowding_score(curr_ep_fit, self.n_objs)
            keep_idx = torch.topk(scores, self.nEP, largest=True).indices
            new_ep_pop = curr_ep_pop[keep_idx]
            new_ep_fit = curr_ep_fit[keep_idx]
            
        curr_size = new_ep_pop.shape[0]
        self.archive_pop[:curr_size] = new_ep_pop
        self.archive_fit[:curr_size] = new_ep_fit
        self.archive_size[0] = curr_size

        # 4. Weight Adaptation
        self.iter += 1
        # adaptation_moment = round(ceil(Problem.maxFE/Problem.N)*0.05)
        # We use a fixed interval of 10 steps as a proxy for 5% of total iterations
        if self.iter % 10 == 0:
            nus = int(N * 0.05)
            if nus > 0 and self.archive_size[0] > 0:
                # Delete overcrowded subproblems
                scores_pop = self._get_overcrowding_score(self.fit, self.n_objs)
                # Smallest score means most crowded
                del_pop_idx = torch.topk(scores_pop, min(nus, self.archive_size[0]), largest=False).indices
                
                # Mask out deleted indices to find remaining population
                keep_mask = torch.ones(N, device=device, dtype=torch.bool)
                keep_mask[del_pop_idx] = False
                
                # Greedily select from archive to maximize diversity relative to current pop
                # MATLAB: find solutions in EP that are furthest from current Population
                add_idx = self._greedy_furthest_selection(self.archive_fit[:self.archive_size[0]], self.fit[keep_mask], del_pop_idx.numel())
                
                self.pop[del_pop_idx] = self.archive_pop[add_idx]
                self.fit[del_pop_idx] = self.archive_fit[add_idx]
                
                # Update weights for the new subproblems: W = 1./(fit-z) / sum(1./(fit-z))
                temp_W = 1.0 / (self.fit[del_pop_idx] - self.z + 1e-6)
                new_W = temp_W / (temp_W.sum(dim=1, keepdim=True) + 1e-6)
                self.W[del_pop_idx] = new_W
                
                # Re-calculate neighborhoods
                dist_W = torch.cdist(self.W, self.W)
                self.B = torch.topk(dist_W, T, largest=False, sorted=True).indices.to(torch.int32)

if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAURAW(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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