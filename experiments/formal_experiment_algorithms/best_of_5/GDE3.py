import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp
from evox.operators.selection import crowding_distance
from evomo.operators.selection import non_dominate_rank


class GDE3(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, F: float = 0.5, CR: float = 0.5, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.F = Parameter(F)
        self.CR = Parameter(CR)
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))  # [N,M]

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)  # [N,M]

    def _de_operator(self, pop: torch.Tensor) -> torch.Tensor:
        N, D = pop.shape
        device = pop.device
        
        # Index Generation (Bug #14 Compliance)
        r1 = torch.randint(0, N, (N,), device=device)
        r2 = torch.randint(0, N, (N,), device=device)
        
        # Mutation
        mutant = pop + self.F * (pop[r1] - pop[r2])
        
        # Crossover
        cross_mask = torch.rand((N, D), device=device) < self.CR
        off_pop = torch.where(cross_mask, mutant, pop)
        
        # Boundary Handling
        return clamp(off_pop, self.lb, self.ub)

    def step(self) -> None:
        # 1. Mating / Variation
        off_pop = self._de_operator(self.pop)
        
        # 2. Evaluation
        off_fit = self.evaluate(off_pop)
        
        # 3. Environmental Selection (GDE3 Core)
        # One-to-One Comparison (Pareto Dominance - Bug #24 Compliance)
        off_dominates_p = (off_fit <= self.fit).all(dim=1) & (off_fit < self.fit).any(dim=1)
        p_dominates_off = (self.fit <= off_fit).all(dim=1) & (self.fit < off_fit).any(dim=1)
        non_dominated = ~off_dominates_p & ~p_dominates_off
        
        # Update Step (One-to-One Replacement)
        self.pop = torch.where(off_dominates_p.unsqueeze(1), off_pop, self.pop)
        self.fit = torch.where(off_dominates_p.unsqueeze(1), off_fit, self.fit)
        
        # Population Expansion
        combined_pop = torch.cat([self.pop, off_pop[non_dominated]], dim=0)
        combined_fit = torch.cat([self.fit, off_fit[non_dominated]], dim=0)
        
        # Global Truncation (Integrated Peeling - Bug #9 Compliance)
        rank = non_dominate_rank(combined_fit)
        N_total = combined_fit.shape[0]
        selected_mask = torch.zeros(N_total, dtype=torch.bool, device=combined_fit.device)
        selected_count = 0
        current_rank = 0
        
        # Peeling Loop (Bug #41: Iteration count is not data-dependent as rank is finite)
        while selected_count < self.pop_size:
            in_front = (rank == current_rank)
            num_in_front = in_front.sum()
            
            # Deadlock Breaker (Bug #9)
            is_empty = (num_in_front == 0)
            
            # Logic for selection
            can_fit_all = (selected_count + num_in_front <= self.pop_size) & (~is_empty)
            
            if can_fit_all:
                selected_mask = selected_mask | in_front
                selected_count = selected_count + num_in_front
            else:
                # Partial Front or Deadlock (Force select remaining)
                needed = self.pop_size - selected_count
                
                # Calculate Crowding Distance (Bug #6 Compliance)
                dist = crowding_distance(combined_fit, in_front)
                
                idx_in_front = torch.where(in_front)[0]
                # Sort by distance (descending)
                sub_sort = torch.argsort(dist[in_front], descending=True)
                selected_mask[idx_in_front[sub_sort[:needed]]] = True
                selected_count = self.pop_size
            
            current_rank = current_rank + 1

        # Final Update
        self.pop = combined_pop[selected_mask]
        self.fit = combined_fit[selected_mask]

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = GDE3(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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