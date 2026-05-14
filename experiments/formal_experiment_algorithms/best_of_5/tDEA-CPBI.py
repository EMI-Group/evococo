import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp, lexsort
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.operators.sampling import uniform_sampling
from evomo.operators.selection import non_dominate_rank


class tDEACPBI(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Reference Weights
        w_samples, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = actual_n
        self.W = Mutable(w_samples.to(device))
        
        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        
        # Theta Logic (Boundary vs Others)
        boundary_mask = (self.W > 0).sum(dim=1) == 1
        theta = torch.where(boundary_mask, torch.tensor(1e6, device=device), torch.tensor(5.0, device=device))
        self.theta = Mutable(theta)
        
        # Ideal and Nadir
        self.z = Mutable(torch.zeros(n_objs, device=device))
        self.znad = Mutable(torch.ones(n_objs, device=device))

    def _calculate_intercepts(self, extreme_objs: torch.Tensor, nadir_fallback: torch.Tensor) -> torch.Tensor:
        device = extreme_objs.device
        M = self.n_objs
        # Shift extreme points by ideal point
        X = extreme_objs - self.z
        
        # Solve X * a = 1
        try:
            # Check determinant for singularity
            det = torch.linalg.det(X)
            # Use a small threshold for singularity check
            is_singular = torch.abs(det) < 1e-9
            
            a = torch.linalg.solve(X, torch.ones(M, device=device))
            intercepts = 1.0 / (a + 1e-6)
            
            # Fallback if intercepts are invalid or matrix is singular
            invalid_intercepts = (intercepts < 1e-6).any()
            final_znad = torch.where(is_singular | invalid_intercepts, nadir_fallback, self.z + intercepts)
        except RuntimeError:
            final_znad = nadir_fallback
            
        return final_znad

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]
        self.znad = torch.max(self.fit, dim=0)[0]

    def step(self) -> None:
        device = self.lb.device
        N = self.pop_size
        M = self.n_objs

        # 1. Mating / Variation
        mating_pool = tournament_selection_multifit(N, [self.fit.sum(dim=1)], tournament_size=2)
        parents = self.pop[mating_pool]
        offspring = simulated_binary(parents)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        off_fit = self.evaluate(offspring)
        
        # 2. Merge
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        total_size = merged_fit.shape[0]
        
        # 3. Environmental Selection
        self.z = torch.min(torch.stack([self.z, torch.min(merged_fit, dim=0)[0]]), dim=0)[0]
        
        # Extreme Points (ASF)
        E = torch.eye(M, device=device) + 1e-6
        f_norm = (merged_fit - self.z) / (torch.max(merged_fit, dim=0)[0] - self.z + 1e-6)
        asf_val = torch.max(f_norm.unsqueeze(1) / E.unsqueeze(0), dim=-1)[0] 
        extreme_idx = torch.argmin(asf_val, dim=0)
        extreme_objs = merged_fit[extreme_idx]
        
        # Update Nadir Point
        n_fallback = torch.max(merged_fit, dim=0)[0]
        self.znad = self._calculate_intercepts(extreme_objs, n_fallback)
        
        # Normalization
        norm_obj = (merged_fit - self.z) / (self.znad - self.z + 1e-6)
        
        # Core Metric Calculation (PBI & Clustering)
        w_norm = torch.norm(self.W, dim=1) + 1e-6
        d1 = torch.matmul(norm_obj, self.W.t()) / w_norm 
        
        w_unit = self.W / w_norm.unsqueeze(1)
        # Vectorized d2 calculation
        d2 = torch.norm(norm_obj.unsqueeze(1) - d1.unsqueeze(2) * w_unit.unsqueeze(0), dim=-1) 
        
        # Clustering
        min_d2, pi = torch.min(d2, dim=1) 
        
        # PBI: d1_assigned + theta * d2_assigned
        d1_assigned = torch.gather(d1, 1, pi.unsqueeze(1)).squeeze(1)
        pbi = d1_assigned + self.theta[pi] * min_d2
        
        # Selection Strategy (t-Dominance Sorting)
        front_no = non_dominate_rank(merged_fit)
        
        # Calculate tFrontNo (Intra-cluster ranking) - Vectorized
        # Sort by cluster index (pi) then by PBI.
        sort_idx = lexsort(torch.stack([pbi, pi.float()]))
        sorted_pi = pi[sort_idx]
        
        # Create a mask where a new cluster starts
        new_cluster_mask = torch.zeros(total_size, device=device, dtype=torch.int32)
        new_cluster_mask[0] = 1
        new_cluster_mask[1:] = (sorted_pi[1:] != sorted_pi[:-1]).int()
        
        global_rank = torch.arange(total_size, device=device)
        # Find the global index where each cluster starts
        cluster_start_indices = torch.where(new_cluster_mask == 1, global_rank, torch.zeros_like(global_rank))
        # Use inclusive scan to propagate the start index
        start_pos = torch.cummax(cluster_start_indices, dim=0)[0]
        intra_cluster_ranks = global_rank - start_pos
        
        # Map back to original order
        t_front_no = torch.zeros(total_size, device=device)
        t_front_no.scatter_(0, sort_idx, intra_cluster_ranks.float())
            
        # Final Selection via Lexsort (Primary key FrontNo last)
        idx = lexsort(torch.stack([t_front_no, front_no.float()]))
        
        self.pop = merged_pop[idx[:N]]
        self.fit = merged_fit[idx[:N]]

if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = tDEACPBI(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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