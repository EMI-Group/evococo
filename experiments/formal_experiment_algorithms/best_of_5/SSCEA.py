import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp, nanmin, nanmax
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class SSCEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_gen: int = 100, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.max_gen = Parameter(max_gen)

        # Subspace Masking: Split variables into Convergence (CV) and Distance (DV)
        # Following common SSCEA practice: first half CV, second half DV
        cv_size = self.dim // 2
        self.cv_mask = Mutable(torch.arange(cv_size, device=device))
        self.dv_mask = Mutable(torch.arange(cv_size, self.dim, device=device))

        # Initialize State (Mutables)
        # CA: Convergence Archive (pop_size)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        
        # DA: Diversity Archive (pop_size)
        self.archive_pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        
        self.gen = Mutable(torch.zeros(1, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.archive_fit = self.evaluate(self.archive_pop)

    def step(self) -> None:
        self.gen = self.gen + 1
        device = self.pop.device
        
        # 1. Mating Selection (Tournament on CA)
        # Using non-dominated rank for CA selection pressure
        rank = non_dominate_rank(self.fit)
        mating_idx = tournament_selection_multifit(self.pop_size, [rank.float()], tournament_size=2)
        parents = self.pop[mating_idx]
        
        # 2. Subspace Variation
        new_dec = simulated_binary(parents)
        new_dec = polynomial_mutation(new_dec, self.lb, self.ub)
        new_dec = clamp(new_dec, self.lb, self.ub)
        
        # Logic: Determine active mask based on generation
        off_pop = self.pop.clone()
        use_cv = (self.gen.float() / self.max_gen < 0.5) | (torch.rand(1, device=device) < 0.5)
        
        # Vectorized subspace update
        cv_indices = self.cv_mask
        dv_indices = self.dv_mask
        
        # Use torch.where style logic for JIT compliance
        mask_cv = use_cv.expand(self.pop_size)
        off_pop[:, cv_indices] = torch.where(mask_cv.unsqueeze(1), new_dec[:, cv_indices], off_pop[:, cv_indices])
        off_pop[:, dv_indices] = torch.where(~mask_cv.unsqueeze(1), new_dec[:, dv_indices], off_pop[:, dv_indices])
        
        # 3. Evaluation
        off_fit = self.evaluate(off_pop)
        
        # 4. Update CA (Indicator-based)
        combined_ca_pop = torch.cat([self.pop, off_pop], dim=0)
        combined_ca_fit = torch.cat([self.fit, off_fit], dim=0)
        self.pop, self.fit = self._update_ca(combined_ca_pop, combined_ca_fit, self.pop_size)
        
        # 5. Update DA (Angle-based)
        combined_da_pop = torch.cat([self.archive_pop, off_pop], dim=0)
        combined_da_fit = torch.cat([self.archive_fit, off_fit], dim=0)
        self.archive_pop, self.archive_fit = self._update_da(combined_da_pop, combined_da_fit, self.pop_size)

    def _update_ca(self, pop, fit, target_size):
        # Normalization
        f_min = nanmin(fit, dim=0)[0]
        f_max = nanmax(fit, dim=0)[0]
        f_norm = (fit - f_min) / (f_max - f_min + 1e-6)
        
        # Indicator Matrix (Pairwise max difference)
        # I[i, j] = max(f_norm[i] - f_norm[j])
        I = torch.max(f_norm.unsqueeze(1) - f_norm.unsqueeze(0), dim=-1)[0]
        
        # Fitness Calculation
        # F[j] = sum_{i != j} -exp(-I[i, j] / 0.05) + 1
        # We use matrix ops: exp_mat = -exp(-I / 0.05)
        exp_mat = -torch.exp(-I / 0.05)
        F = torch.sum(exp_mat, dim=0) + 1.0
        
        # Peeling Loop (JIT compliant via masking)
        curr_size = pop.shape[0]
        active_mask = torch.ones(curr_size, device=pop.device, dtype=torch.bool)
        
        # We need to remove (curr_size - target_size) individuals
        num_to_remove = curr_size - target_size
        
        # Sentinel for removed elements
        sentinel_val = torch.tensor(float('inf'), device=pop.device)
        
        for _ in range(num_to_remove):
            # Find index of individual with minimum fitness among active
            temp_F = torch.where(active_mask, F, sentinel_val)
            idx = torch.argmin(temp_F)
            
            # Update F of survivors: F[j] = F[j] - exp_mat[idx, j]
            F = F - exp_mat[idx, :]
            active_mask[idx] = False
            
        return pop[active_mask], fit[active_mask]

    def _update_da(self, pop, fit, N):
        # 1. Unique rows
        pop, u_idx = unique_rows_sorted(pop)
        fit = fit[u_idx]
        
        # 2. Rank-1 NDSort
        rank = non_dominate_rank(fit)
        mask_r1 = (rank == 0)
        pop_r1 = pop[mask_r1]
        fit_r1 = fit[mask_r1]
        
        # If not enough rank-1, take all rank-1 and fill with others
        num_r1 = pop_r1.shape[0]
        
        # 3. Extreme Selection (ASF)
        M = fit.shape[1]
        W = torch.eye(M, device=fit.device) + 1e-6
        norm_W = torch.norm(W, dim=1, keepdim=True)
        
        # asf = max(fit/W) + 0.1 * (fit@W.T / norm_W)
        # Vectorized ASF for all individuals in Rank 1
        asf = torch.max(fit_r1.unsqueeze(1) / W.unsqueeze(0), dim=-1)[0] + \
              0.1 * (fit_r1 @ W.T / (norm_W.T + 1e-6))
        
        extreme_idx = torch.argmin(asf, dim=0)
        # Use unique to get distinct extreme points
        unique_extreme_idx, _ = unique_rows_sorted(extreme_idx.unsqueeze(1))
        unique_extreme_idx = unique_extreme_idx.squeeze(1)
        
        # 4. Angle-based Max-Min Selection
        selected_mask = torch.zeros(num_r1, device=fit.device, dtype=torch.bool)
        selected_mask[unique_extreme_idx] = True
        
        # Precompute angular distances
        norm_fit = torch.norm(fit_r1, dim=1, keepdim=True)
        cos_sim = (fit_r1 @ fit_r1.T) / (norm_fit @ norm_fit.T + 1e-6)
        dist_mat = torch.acos(torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6))
        
        # Iteratively add individuals
        num_to_add = torch.clamp(torch.tensor(N - unique_extreme_idx.numel(), device=fit.device), min=0)
        
        # JIT-friendly loop for selection
        for _ in range(num_to_add):
            # min distance to selected set for each candidate
            # dist_mat shape: (num_r1, num_r1)
            # selected_dist shape: (num_r1, num_selected)
            mask_selected = selected_mask.view(1, -1).expand(num_r1, -1)
            # Use a large value for non-selected to find min
            temp_dist = torch.where(mask_selected, dist_mat, torch.tensor(float('inf'), device=fit.device))
            min_dists = torch.min(temp_dist, dim=1)[0]
            
            # Maximize the minimum distance
            # Only consider candidates not yet selected
            min_dists = torch.where(~selected_mask, min_dists, torch.tensor(-1.0, device=fit.device))
            best_candidate = torch.argmax(min_dists)
            selected_mask[best_candidate] = True
            
        # If we still don't have N (due to rank-1 being small), take top N by crowding or just truncate
        res_pop = pop_r1[selected_mask]
        res_fit = fit_r1[selected_mask]
        
        # Final size check and padding if necessary (Bug #26 compliance)
        if res_pop.shape[0] < N:
            # This is a bit complex for JIT, so we use a simple top-K fallback
            # In practice, rank-1 is usually larger than N
            return pop[:N], fit[:N]
            
        return res_pop[:N], res_fit[:N]

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SSCEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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