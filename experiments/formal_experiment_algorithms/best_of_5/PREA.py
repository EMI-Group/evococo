import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp, lexsort
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation

class PREA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.zmin = Mutable(torch.full((1, n_objs), torch.inf, device=device))
        self.i_matrix = Mutable(torch.full((pop_size, pop_size), torch.inf, device=device))

    def _calc_imatrix(self, F: torch.Tensor, zmin: torch.Tensor) -> torch.Tensor:
        F_shifted = F - zmin + 1e-6
        # Ratio Matrix: [L, L, M]
        R = F_shifted.unsqueeze(0) / F_shifted.unsqueeze(1)
        # Indicator Components
        Ir = torch.max(R - 1, dim=-1).values
        InvertIr = torch.max(1.0 / R - 1, dim=-1).values
        imatrix = torch.where(Ir <= 0, -InvertIr, Ir)
        imatrix.fill_diagonal_(torch.inf)
        return imatrix

    def _calc_parallel_dist(self, F: torch.Tensor) -> torch.Tensor:
        M = F.shape[1]
        diff = F.unsqueeze(1) - F.unsqueeze(0) # [K, K, M]
        sum_sq_diff = torch.sum(diff**2, dim=-1)
        sq_sum_diff = torch.sum(diff, dim=-1)**2
        dist = torch.sqrt(torch.relu(sum_sq_diff - (sq_sum_diff / (M + 1e-6))))
        return dist

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.zmin = torch.min(self.fit, dim=0, keepdim=True).values
        self.i_matrix = self._calc_imatrix(self.fit, self.zmin)

    def step(self) -> None:
        N = self.pop_size
        device = self.lb.device
        
        # 1. Mating / Variation
        neighbors = torch.argmin(self.i_matrix, dim=1)
        Ps = 0.8
        rand_mask = torch.rand(N, device=device) < Ps
        spouse_id = torch.where(rand_mask, neighbors, torch.randint(0, N, (N,), device=device))
        pool = torch.cat([torch.arange(N, device=device), spouse_id])
        
        crossovered = simulated_binary_half(self.pop[pool])
        off_pop = polynomial_mutation(crossovered, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)
        
        # 2. Evaluation
        off_fit = self.evaluate(off_pop)
        
        # 3. Merge
        combined_pop = torch.cat([self.pop, off_pop], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)
        
        # Update Global Ideal Point
        self.zmin = torch.min(torch.cat([self.zmin, combined_fit], dim=0), dim=0, keepdim=True).values
        
        # 4. Environmental Selection (PREA_Update)
        imatrix_full = self._calc_imatrix(combined_fit, self.zmin)
        fit_i = torch.min(imatrix_full, dim=1).values
        
        promising_mask = fit_i >= 0
        num_promising = torch.sum(promising_mask).item() # JIT-safe if used in simple logic
        
        if num_promising <= N:
            # Case A: Select by indicator fitness
            idx = lexsort(torch.stack([-fit_i]))[:N]
            self.pop = combined_pop[idx]
            self.fit = combined_fit[idx]
            self.i_matrix = self._calc_imatrix(self.fit, self.zmin)
        else:
            # Case B: Algorithm 1 - Promising Region Peeling to find Zmax
            active_mask = promising_mask.clone()
            current_count = num_promising
            temp_imatrix = imatrix_full.clone()
            # Mask non-promising
            temp_imatrix[~active_mask, :] = torch.inf
            temp_imatrix[:, ~active_mask] = torch.inf
            
            while current_count > N:
                # Recalculate indicator fitness for active set
                row_min = torch.min(temp_imatrix, dim=1).values
                worst_idx = torch.argmin(row_min)
                temp_imatrix[worst_idx, :] = torch.inf
                temp_imatrix[:, worst_idx] = torch.inf
                active_mask[worst_idx] = False
                current_count -= 1
            
            Zmax = torch.max(combined_fit[active_mask], dim=0).values
            
            # Boundary Filter
            keep_mask = torch.all(combined_fit <= Zmax, dim=1)
            K = torch.sum(keep_mask).item()
            
            # Algorithm 2: Parallel Distance Pruning
            F_filtered = combined_fit[keep_mask]
            P_filtered = combined_pop[keep_mask]
            fit_i_filtered = fit_i[keep_mask]
            
            dist_matrix = self._calc_parallel_dist(F_filtered)
            dist_matrix.fill_diagonal_(torch.inf)
            
            active_k = torch.ones(K, dtype=torch.bool, device=device)
            current_k = K
            
            while current_k > N:
                # Find closest pair
                flat_idx = torch.argmin(dist_matrix)
                i = flat_idx // K
                j = flat_idx % K
                
                # Remove the one with lower indicator fitness
                remove_idx = torch.where(fit_i_filtered[i] < fit_i_filtered[j], i, j)
                
                dist_matrix[remove_idx, :] = torch.inf
                dist_matrix[:, remove_idx] = torch.inf
                active_k[remove_idx] = False
                current_k -= 1
                
            self.pop = P_filtered[active_k]
            self.fit = F_filtered[active_k]
            self.i_matrix = self._calc_imatrix(self.fit, self.zmin)

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = PREA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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