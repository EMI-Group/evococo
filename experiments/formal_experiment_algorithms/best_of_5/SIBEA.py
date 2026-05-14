import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance
from evomo.operators.selection import non_dominate_rank


class SIBEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), float('inf'), device=device))  # [N,M]

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)  # [N,M]

    def step(self) -> None:
        N = self.pop_size
        device = self.pop.device
        M = self.n_objs

        # 1. Mating Selection (Bug #27 fix)
        # Vectorized Tournament Selection to avoid the gather error in tournament_selection_multifit
        # We use NDSort on current population to get selection pressure
        curr_fronts = non_dominate_rank(self.fit)
        idx1 = torch.randint(0, N, (N,), device=device)
        idx2 = torch.randint(0, N, (N,), device=device)
        
        # Lower front number is better
        sel_mask = curr_fronts[idx1] < curr_fronts[idx2]
        idx = torch.where(sel_mask, idx1, idx2)
        
        # Apply SBX and PM
        crossovered = simulated_binary(self.pop[idx], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(crossovered, self.lb, self.ub, pro_m=1.0/self.pop.shape[1], dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)
        
        off_fit = self.evaluate(offspring)

        # 2. Merge
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        # 3. Environmental Selection
        front_no = non_dominate_rank(merged_fit)
        
        # Identify the cutoff front
        # We sort front numbers and find the value at index N-1
        sorted_front_no, _ = torch.sort(front_no)
        cutoff_front_no = sorted_front_no[N-1]
        
        mask_better = front_no < cutoff_front_no
        mask_cutoff = front_no == cutoff_front_no
        
        num_better = torch.sum(mask_better.to(torch.int32))
        num_needed = N - num_better
        
        # Calculate HV loss for the cutoff front (Bug #34)
        fit_cutoff = merged_fit[mask_cutoff]
        
        # Reference Point: max(fit) + 0.1 (Bug #10: dim=0 for objective-wise max)
        ref_point = torch.max(merged_fit, dim=0)[0] + 0.1
        
        hv_loss = self._cal_hv_loss(fit_cutoff, ref_point)
        
        # Tie-breaking with Crowding Distance (Bug #34)
        cd = crowding_distance(merged_fit, mask_cutoff)
        cd_cutoff = cd[mask_cutoff]
        
        # Combine HV loss and CD into a single score for selection
        # Normalize CD to be a small tie-breaker
        # hv_loss is count of samples, so it's integer-based. CD is float.
        combined_score = hv_loss.to(torch.float32) + (cd_cutoff / (torch.max(cd_cutoff) + 1e-6)) * 0.1
        
        # Select top indices from the cutoff front
        _, top_rel_indices = torch.topk(combined_score, k=num_needed, largest=True)
        
        cutoff_indices = torch.where(mask_cutoff)[0]
        selected_cutoff_indices = cutoff_indices[top_rel_indices]
        
        # Final Selection
        better_indices = torch.where(mask_better)[0]
        final_indices = torch.cat([better_indices, selected_cutoff_indices])
        
        # Ensure exactly N (Deadlock Breaker / JIT Guard)
        self.pop = merged_pop[final_indices[:N]]
        self.fit = merged_fit[final_indices[:N]]

    def _cal_hv_loss(self, fit_front: torch.Tensor, ref_point: torch.Tensor) -> torch.Tensor:
        """
        Monte Carlo Hypervolume Contribution Estimation (Bug #34 & #29)
        """
        device = fit_front.device
        num_samples = 1000000
        M = fit_front.shape[1]
        N_L = fit_front.shape[0]
        
        # Handle empty front case
        if N_L == 0:
            return torch.zeros(0, device=device)
            
        zmin = torch.min(fit_front, dim=0)[0]
        
        # Generate Samples: samples = rand * (ref - zmin) + zmin (Bug #14)
        samples = torch.rand((num_samples, M), device=device) * (ref_point - zmin) + zmin
        
        # Vectorized Coverage Check (Bug #29)
        # fit_front: (N_L, M), samples: (num_samples, M)
        # is_covered: (num_samples, N_L)
        # We use broadcasting to check which individuals cover which samples
        is_covered = (fit_front.unsqueeze(0) <= samples.unsqueeze(1)).all(dim=-1)
        
        # Exclusive Contribution:
        # 1. Count how many individuals cover each sample
        covered_count = is_covered.sum(dim=1) # (num_samples,)
        
        # 2. A sample is exclusively covered if covered_count == 1
        # exclusive_mask: (num_samples, N_L)
        exclusive_mask = (covered_count == 1).unsqueeze(1) & is_covered
        
        # 3. Sum up exclusive samples for each individual
        hv_loss = exclusive_mask.sum(dim=0).to(torch.float32) # (N_L,)
        
        return hv_loss

if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SIBEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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