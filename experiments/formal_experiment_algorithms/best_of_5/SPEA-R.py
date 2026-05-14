import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evomo.operators.selection import non_dominate_rank
from evox.operators.sampling import uniform_sampling


class SPEAR(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Reference Directions
        w, _ = uniform_sampling(pop_size, n_objs)
        self.w = Mutable(w.to(device))
        
        # Niche Angle (theta) calculation
        norms = torch.linalg.norm(self.w, dim=1, keepdim=True)
        cos_sim = (self.w @ self.w.T) / (norms @ norms.T + 1e-6)
        dist_matrix = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
        # Set diagonal to inf to find nearest neighbor
        dist_matrix.fill_diagonal_(float('inf'))
        self.theta = Mutable(torch.max(torch.min(dist_matrix, dim=1).values))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Mating Selection (Nearest neighbor in objective space)
        idx = torch.randint(0, N, (N, 20), device=device)
        dist = torch.cdist(self.fit, self.fit)
        candidate_dists = torch.gather(dist, 1, idx)
        best_in_subset = torch.argmin(candidate_dists, dim=1)
        mating_pool_idx = torch.gather(idx, 1, best_in_subset.unsqueeze(1)).squeeze()
        
        # Variation
        crossovered = simulated_binary(self.pop[mating_pool_idx])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # 2. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # Normalization (Rank-1 Based)
        rank = non_dominate_rank(combined_fit)
        rank1_mask = (rank == 1)
        # Fallback if no rank 1 found (unlikely)
        rank1_fit = torch.where(rank1_mask.unsqueeze(1), combined_fit, combined_fit[0].unsqueeze(0))
        z_min = torch.min(rank1_fit, dim=0).values
        z_max = torch.max(rank1_fit, dim=0).values
        norm_fit = (combined_fit - z_min) / (z_max - z_min + 1e-6)

        # Association
        norm_fit_norm = torch.linalg.norm(norm_fit, dim=1, keepdim=True)
        w_norm = torch.linalg.norm(self.w, dim=1, keepdim=True).T
        cos_val = (norm_fit @ self.w.T) / (norm_fit_norm @ w_norm + 1e-6)
        dist_to_w = torch.acos(torch.clamp(cos_val, -1.0, 1.0))
        association = torch.argmin(dist_to_w, dim=1)
        min_angles = torch.min(dist_to_w, dim=1).values

        # Dominance Metrics
        # (X <= Y).all & (X < Y).any
        f_ext_1 = combined_fit.unsqueeze(1) # [2N, 1, M]
        f_ext_2 = combined_fit.unsqueeze(0) # [1, 2N, M]
        dom_mat = (f_ext_1 <= f_ext_2).all(-1) & (f_ext_1 < f_ext_2).any(-1)

        # Global Metrics
        S_g = dom_mat.sum(dim=1).float()
        R_g = dom_mat.T.float() @ S_g

        # Local Metrics
        niche_mask = (association.unsqueeze(1) == association.unsqueeze(0))
        local_dom = dom_mat & niche_mask
        S_l = local_dom.sum(dim=1).float()
        R_l = local_dom.T.float() @ S_l

        # Density and Final FV
        D_density = min_angles / (min_angles + self.theta + 1e-6)
        niche_counts = torch.bincount(association, minlength=N)[association]
        fv = torch.where(niche_counts > 1, R_l + D_density + R_g, R_l + D_density)

        # Niche-first Peeling
        choose = torch.zeros(2 * N, dtype=torch.bool, device=device)
        current_fv = fv.clone()
        sentinel = torch.iinfo(torch.int32).max
        
        # Loop for selection (Niche-first Peeling)
        while choose.sum() < N:
            remaining_mask = ~choose
            active_niches = association[remaining_mask]
            
            # Find best (min FV) in each unique niche
            niche_min_fv = torch.full((N,), float(sentinel), device=device)
            niche_min_fv.scatter_reduce_(0, active_niches, current_fv[remaining_mask], reduce='amin', include_self=False)
            
            # Identify individuals matching that min FV in their niche
            best_in_niche_mask = (current_fv == niche_min_fv[association]) & remaining_mask
            
            num_to_add = best_in_niche_mask.sum()
            current_total = choose.sum()
            
            # Deadlock Breaker
            if num_to_add == 0:
                remaining_indices = torch.where(~choose)[0]
                needed = N - current_total
                choose[remaining_indices[:needed]] = True
            # Overflow check
            elif (current_total + num_to_add) > N:
                batch_indices = torch.where(best_in_niche_mask)[0]
                # Sort the current batch by FV and take top
                sorted_batch_idx = torch.argsort(current_fv[batch_indices])
                sorted_batch = batch_indices[sorted_batch_idx]
                needed = N - current_total
                choose[sorted_batch[:needed]] = True
            else:
                choose[best_in_niche_mask] = True

        self.pop = combined_pop[choose]
        self.fit = combined_fit[choose]

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SPEAR(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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