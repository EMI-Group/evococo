import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp, randint
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evomo.operators.selection import non_dominate_rank
from evox.operators.sampling import uniform_sampling

class hpaEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), 1e10, device=device))
        
        # Reference vectors (SLD)
        v, _ = uniform_sampling(pop_size, n_objs)
        self.v = Mutable(v.to(device))
        self.pop_size = self.v.shape[0] # Adjust to sampling
        
        self.max_obj = Mutable(torch.full((n_objs,), 1e10, device=device))
        self.psi = Mutable(torch.zeros((0,), device=device, dtype=torch.int32))
        self.iter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.max_obj = torch.max(self.fit, dim=0)[0]
        rank = non_dominate_rank(self.fit)
        self.psi = torch.where(rank == 1)[0].to(torch.int32)

    def _get_hyperplane_prominence(self, fit: torch.Tensor) -> torch.Tensor:
        N, M = fit.shape
        # Normalization within the front
        f_min = torch.min(fit, dim=0, keepdim=True)[0]
        f_max = torch.max(fit, dim=0, keepdim=True)[0]
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)
        
        dist = torch.cdist(norm_fit, norm_fit)
        # Find M nearest neighbors
        _, neighbors_idx = torch.topk(dist, k=M, largest=False)
        
        # Construct A and B for batch solve: A*w = 1
        # A shape: [N, M, M]
        A = norm_fit[neighbors_idx] 
        B = torch.ones((N, M, 1), device=fit.device)
        
        # Stability: Add epsilon to diagonal
        eye = torch.eye(M, device=fit.device).unsqueeze(0) * 1e-6
        # Batch solve for hyperplane weights
        weights = torch.linalg.solve(A + eye, B).squeeze(-1) # [N, M]
        
        # is_prominent = sum(fit * weights) < 1.0
        is_prominent = torch.sum(norm_fit * weights, dim=1) < 1.0
        return is_prominent

    def _angle_greedy_selection(self, F: torch.Tensor, n_select: int) -> torch.Tensor:
        N = F.shape[0]
        # Clamp n_select to available solutions
        n_select_clamped = torch.clamp(n_select, min=0, max=N)
        
        norm = torch.norm(F, dim=1, keepdim=True)
        cos_sim = (F @ F.t()) / (norm @ norm.t() + 1e-6)
        
        selected = torch.zeros(N, dtype=torch.bool, device=F.device)
        # Start with the most isolated point (min sum of cosine similarities)
        current_idx = torch.argmin(torch.sum(cos_sim, dim=1))
        selected[current_idx] = True
        
        min_cos = cos_sim[:, current_idx]
        selected_indices = torch.zeros(N, dtype=torch.long, device=F.device)
        selected_indices[0] = current_idx
        
        # Use a fixed range loop for JIT compliance
        for i in range(1, N):
            # We want to maximize the minimum angle -> minimize the maximum cosine
            # Mask already selected to avoid re-picking
            mask_val = torch.where(selected, 2.0, min_cos)
            next_idx = torch.argmin(mask_val)
            
            selected[next_idx] = True
            selected_indices[i] = next_idx
            # Update min_cos: for each point, the max cosine to the selected set
            min_cos = torch.maximum(min_cos, cos_sim[:, next_idx])
            
        return selected_indices[:n_select_clamped]

    def step(self) -> None:
        self.iter = self.iter + 1
        device = self.pop.device
        N = self.pop_size
        M = self.n_objs
        
        # 1. Mating Selection
        num_psi = self.psi.shape[0]
        num_psi_clamped = torch.clamp(torch.tensor(num_psi, device=device), max=N)
        psi_indices = self.psi[:num_psi_clamped].to(torch.long)
        
        rand_idx = randint(0, N, (N - num_psi_clamped,), device=device)
        mating_pool_idx = torch.cat([rand_idx, psi_indices])
        shuffled_idx = mating_pool_idx[torch.randperm(N, device=device)]
        
        # 2. Variation
        offspring = simulated_binary(self.pop[shuffled_idx], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        
        # 3. Pre-Selection Filtering
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        
        # Update Nadir Point (max_obj) - Blueprint 3.B.2
        self.max_obj = torch.minimum(self.max_obj, torch.max(merged_fit, dim=0)[0])
        
        is_feasible = (merged_fit <= self.max_obj).all(dim=1)
        num_feasible = torch.sum(is_feasible.to(torch.int32))
        
        # If too few feasible, keep all
        mask = torch.where(num_feasible >= 2, is_feasible, torch.ones(merged_pop.shape[0], dtype=torch.bool, device=device))
        filtered_pop = merged_pop[mask]
        filtered_fit = merged_fit[mask]
        
        # 4. Environmental Selection
        rank = non_dominate_rank(filtered_fit)
        
        # JIT-safe rank counting
        counts = torch.bincount(rank.to(torch.int64))
        # counts[0] is unused as rank starts at 1
        cum_counts = torch.cumsum(counts, dim=0)
        
        # Find max_rank such that cum_counts[max_rank] <= N
        rank_indices = torch.arange(cum_counts.shape[0], device=device)
        rank_fit_mask = (cum_counts <= N)
        max_rank = torch.max(torch.where(rank_fit_mask, rank_indices, torch.zeros_like(rank_indices)))
        
        count_at_max = cum_counts[max_rank]
        
        # Logic branching based on population filling
        if count_at_max == N:
            idx = torch.where(rank <= max_rank)[0]
            self.pop = filtered_pop[idx]
            self.fit = filtered_fit[idx]
            self.psi = torch.where(rank == 1)[0].to(torch.int32)
            
        elif counts[1] > N:
            # Case B: Front 1 is too large
            f1_idx = torch.where(rank == 1)[0]
            f1_fit = filtered_fit[f1_idx]
            
            is_prominent = self._get_hyperplane_prominence(f1_fit)
            prominent_idx = f1_idx[is_prominent]
            num_prom = prominent_idx.shape[0]
            
            # Use torch.where logic for JIT
            if num_prom > N:
                sub_idx = self._angle_greedy_selection(f1_fit[is_prominent], N)
                final_idx = prominent_idx[sub_idx]
                self.psi = torch.arange(N, device=device, dtype=torch.int32)
            else:
                n_rem = N - num_prom
                non_prominent_mask = ~is_prominent
                np_idx = f1_idx[non_prominent_mask]
                sub_idx = self._angle_greedy_selection(f1_fit[non_prominent_mask], n_rem)
                final_idx = torch.cat([prominent_idx, np_idx[sub_idx]])
                self.psi = torch.arange(num_prom, device=device, dtype=torch.int32)
            
            self.pop = filtered_pop[final_idx]
            self.fit = filtered_fit[final_idx]
            
        else:
            # Case C: Overfilled at Last Front
            selected_mask = rank <= max_rank
            last_front_mask = rank == (max_rank + 1)
            
            n_needed = N - count_at_max
            lf_idx = torch.where(last_front_mask)[0]
            
            # Angle greedy on last front
            sub_idx = self._angle_greedy_selection(filtered_fit[lf_idx], n_needed)
            final_idx = torch.cat([torch.where(selected_mask)[0], lf_idx[sub_idx]])
            
            self.pop = filtered_pop[final_idx]
            self.fit = filtered_fit[final_idx]
            
            # PSI are indices of Front 1 in the new population
            new_rank = non_dominate_rank(self.fit)
            self.psi = torch.where(new_rank == 1)[0].to(torch.int32)

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = hpaEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    # 1. Trigger JIT compilation (First step)
    jit_state_step()

    # 2. Pure execution (Remaining 49 steps)
    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            # Simple NaN filtering for metric calculation
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")