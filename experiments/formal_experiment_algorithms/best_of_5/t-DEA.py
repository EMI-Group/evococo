import torch
from typing import Tuple
from evox.core import Algorithm, Mutable
from evox.utils import clamp, lexsort
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.operators.sampling import uniform_sampling

class ThetaDEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # 1. Weight Vectors Initialization (Bug #13)
        W, NW = uniform_sampling(pop_size, n_objs)
        self.W = Mutable(W.to(device=device))
        self.NW = NW
        
        # 2. Theta Logic
        is_boundary = (self.W > 1e-4).sum(dim=1) == 1
        theta_vec = torch.where(is_boundary, torch.tensor(1e6, device=device), torch.tensor(5.0, device=device))
        self.theta = Mutable(theta_vec)

        # 3. Initialize State
        self.pop = Mutable(torch.rand(NW, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((NW, n_objs), float('inf'), device=device))
        self.z = Mutable(torch.full((n_objs,), float('inf'), device=device))
        self.znad = Mutable(torch.full((n_objs,), float('-inf'), device=device))
        
        # Sentinel for integer masks (Bug #1)
        self.sentinel = torch.iinfo(torch.int32).max

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]
        self.znad = torch.max(self.fit, dim=0)[0]

    def _calculate_pbi(self, norm_fit: torch.Tensor, W: torch.Tensor, theta_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # norm_fit: (2N, M), W: (NW, M), theta_vec: (NW,)
        norm_W = W / (torch.norm(W, dim=1, keepdim=True) + 1e-6)
        
        # d1 = norm_fit @ norm_W.T -> (2N, NW)
        d1 = torch.matmul(norm_fit, norm_W.T)
        
        # d2 calculation using broadcasting: || norm_fit - d1*norm_W ||
        # Shape: (2N, NW, M)
        diff = norm_fit.unsqueeze(1) - d1.unsqueeze(2) * norm_W.unsqueeze(0)
        d2 = torch.norm(diff, dim=2) # (2N, NW)
        
        cluster_idx = torch.argmin(d2, dim=1)
        
        rows = torch.arange(norm_fit.shape[0], device=norm_fit.device)
        chosen_d1 = d1[rows, cluster_idx]
        chosen_d2 = d2[rows, cluster_idx]
        chosen_theta = theta_vec[cluster_idx]
        
        pbi = chosen_d1 + chosen_theta * chosen_d2
        return pbi, cluster_idx

    def _group_argsort(self, values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        num_items = values.shape[0]
        sort_idx = lexsort(torch.stack([values, groups.float()]))
        sorted_groups = groups[sort_idx]
        
        # Identify where groups change
        diff = torch.cat([torch.tensor([1], device=values.device), (sorted_groups[1:] != sorted_groups[:-1]).int()])
        ranks = torch.arange(num_items, device=values.device)
        group_starts = torch.where(diff == 1, ranks, torch.tensor(0, device=values.device))
        group_offsets = torch.cummax(group_starts, dim=0)[0]
        intra_group_ranks = ranks - group_offsets + 1
        
        original_ranks = torch.zeros_like(intra_group_ranks)
        original_ranks[sort_idx] = intra_group_ranks
        return original_ranks

    def step(self) -> None:
        device = self.pop.device
        N = self.pop.shape[0]
        M = self.n_objs
        
        # 1. Mating / Variation
        norm_fit_current = (self.fit - self.z) / (self.znad - self.z + 1e-6)
        pbi_current, cluster_idx_current = self._calculate_pbi(norm_fit_current, self.W, self.theta)
        tFrontNo_current = self._group_argsort(pbi_current, cluster_idx_current)
        
        mating_pool = tournament_selection_multifit(N, [tFrontNo_current.float()], tournament_size=2)
        off_pop = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)
        
        off_fit = self.evaluate(off_pop)
        
        # 2. Environmental Selection
        merged_pop = torch.cat([self.pop, off_pop], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        twoN = merged_pop.shape[0]

        # Update Ideal Point
        self.z = torch.min(torch.stack([self.z, torch.min(merged_fit, dim=0)[0]]), dim=0)[0]
        
        # Normalization - Extreme Points (ASF)
        # Use explicit unit vectors to ensure we find exactly M extreme points
        eye_M = torch.eye(M, device=device) + 1e-6
        temp_fit = merged_fit - self.z
        # ASF calculation: max(temp_fit / unit_vector)
        asf = torch.max(temp_fit.unsqueeze(1) / eye_M.unsqueeze(0), dim=2)[0]
        extreme_idx = torch.argmin(asf, dim=0) # Shape: (M,)
        
        # Hyperplane Intercepts
        A = merged_fit[extreme_idx] - self.z
        b = torch.ones((M, 1), device=device)
        
        # Default znad is the max of the merged population
        new_znad = torch.max(merged_fit, dim=0)[0]
        
        # Check if A is square and non-singular (Bug #32)
        if A.shape[0] == A.shape[1]:
            det = torch.linalg.det(A)
            if torch.abs(det) > 1e-9:
                intercepts = torch.linalg.solve(A, b).squeeze()
                # a = 1/intercepts + z
                a = 1.0 / (intercepts + 1e-6) + self.z
                # Validate intercepts
                valid_a = (a > self.z).all() & (~torch.any(torch.isnan(a)))
                new_znad = torch.where(valid_a, a, new_znad)
        
        self.znad = new_znad
        norm_fit = (merged_fit - self.z) / (self.znad - self.z + 1e-6)

        # Core Metric Calculation
        pbi, cluster_idx = self._calculate_pbi(norm_fit, self.W, self.theta)
        
        # Ranking (Integrated Peeling)
        tFrontNo = self._group_argsort(pbi, cluster_idx)
        
        # Selection (Bug #25: Primary key last)
        rand_vals = torch.rand(twoN, device=device)
        idx = lexsort(torch.stack([rand_vals, tFrontNo.float()]))
        
        self.pop = merged_pop[idx[:N]]
        self.fit = merged_fit[idx[:N]]

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = ThetaDEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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