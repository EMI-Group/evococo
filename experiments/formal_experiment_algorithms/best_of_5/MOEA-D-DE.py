import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling


class MOEADDE(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, 
                 T: int = 20, delta: float = 0.9, nr: int = 2, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.D = lb.numel()
        
        # Hyperparameters
        self.T = T
        self.delta = delta
        self.nr = nr

        # Initialize State (Mutables)
        # Weight Generation (Bug #13)
        w, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        self.w = Mutable(w.to(device))
        
        # Neighborhood Calculation (Section 3A)
        # Bug #19: Use broadcasting for distance matrix
        dist = torch.cdist(self.w, self.w, p=2)
        # Get T nearest neighbors for each weight vector
        self.neighbor_idx = Mutable(torch.topk(dist, k=self.T, largest=False).indices) # [N, T]

        # Population and Fitness
        self.pop = Mutable(torch.rand(self.pop_size, self.D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), 1e10, device=device))
        self.z = Mutable(torch.full((n_objs,), 1e10, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0).values

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        T = self.T
        
        # 1. Parent Selection (Section 3B)
        use_neighbor = torch.rand(N, device=device) < self.delta
        
        # Local parents: pick 2 from the T neighbors of each subproblem
        local_rand = torch.randint(0, T, (N, 2), device=device)
        idx_local = torch.gather(self.neighbor_idx, 1, local_rand)
        
        # Global parents: pick 2 from the whole population
        idx_global = torch.randint(0, N, (N, 2), device=device)
        
        # P contains the indices of the two parents for each of the N subproblems
        P = torch.where(use_neighbor.unsqueeze(1), idx_local, idx_global)
        
        # 2. Variation (DE/1/bin style)
        # offspring = pop_i + F * (pop_p1 - pop_p2)
        # In MOEA/D-DE, the base vector is usually the current individual i
        F = 0.5
        CR = 1.0 # Standard MOEA/D-DE often uses CR=1.0
        
        offspring = self.pop + F * (self.pop[P[:, 0]] - self.pop[P[:, 1]])
        
        # Binary crossover (simplified for CR=1.0, otherwise use mask)
        if CR < 1.0:
            cross_mask = torch.rand((N, self.D), device=device) < CR
            offspring = torch.where(cross_mask, offspring, self.pop)
            
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        # 3. Evaluation
        off_fit = self.evaluate(offspring)
        
        # 4. Update Ideal Point
        self.z = torch.min(self.z, torch.min(off_fit, dim=0).values)
        
        # 5. Environmental Selection (Tchebycheff Update)
        # Determine the competition pool for each offspring
        # If use_neighbor[i] is true, offspring i competes with its T neighbors
        # Otherwise, it competes with T random individuals from the population
        idx_global_pool = torch.randint(0, N, (N, T), device=device)
        idx_comp = torch.where(use_neighbor.unsqueeze(1), self.neighbor_idx, idx_global_pool)
        
        self._tchebycheff_update(offspring, off_fit, idx_comp)

    def _tchebycheff_update(self, off_pop, off_fit, idx_comp):
        device = self.pop.device
        N = self.pop_size
        T = self.T
        
        # Tchebycheff: g(f, w, z) = max_j (w_j * |f_j - z_j|)
        # Offspring score for its corresponding subproblem i: [N]
        # Note: In MOEA/D, offspring i is compared against subproblems in idx_comp[i]
        # using the weights of those subproblems.
        
        # Weights of the subproblems in the competition pool: [N, T, M]
        W_comp = self.w[idx_comp]
        # Fitness of the current individuals in those subproblems: [N, T, M]
        F_comp = self.fit[idx_comp]
        
        # g_old: [N, T] - Score of current solutions in the subproblems
        g_old = torch.max(W_comp * torch.abs(F_comp - self.z), dim=-1).values
        
        # g_new: [N, T] - Score of the NEW offspring if it were in those subproblems
        # off_fit is [N, M], we expand to [N, T, M] to compare with W_comp
        g_new = torch.max(W_comp * torch.abs(off_fit.unsqueeze(1) - self.z), dim=-1).values
        
        # Improvement Mask: [N, T]
        better_mask = g_new <= g_old
        
        # Limit Enforcement (nr): only replace up to nr solutions per offspring
        # We use a priority trick to pick the first nr 'True' values in each row
        # Add a small gradient to ensure stable selection of the first nr indices
        priority = better_mask.to(torch.float32) + torch.linspace(0.1, 0.0, T, device=device).unsqueeze(0)
        _, sorted_indices = torch.sort(priority, dim=1, descending=True)
        
        # Create a mask for the top nr elements
        rank = torch.zeros_like(sorted_indices)
        rank.scatter_(1, sorted_indices, torch.arange(T, device=device).expand(N, T))
        limit_mask = rank < self.nr
        
        final_mask = better_mask & limit_mask
        
        # Apply updates
        # rows: offspring index, cols: neighbor index
        rows, cols = torch.where(final_mask)
        update_sub_idx = idx_comp[rows, cols]
        
        # Vectorized update: if multiple offspring update the same subproblem, 
        # the last one in the 'rows' list wins. This is an acceptable vectorized approximation.
        self.pop[update_sub_idx] = off_pop[rows]
        self.fit[update_sub_idx] = off_fit[rows]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADDE(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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