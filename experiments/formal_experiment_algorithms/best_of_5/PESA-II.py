import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp, lexsort
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class PESA2(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, div: int = 10, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.div = Parameter(torch.tensor(div, dtype=torch.int32, device=device))
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))  # [N,M]

    def _get_grid_info(self, fit: torch.Tensor):
        N, M = fit.shape
        fmin = torch.min(fit, dim=0)[0]
        fmax = torch.max(fit, dim=0)[0]
        
        # Normalization (Bug #12)
        norm_fit = (fit - fmin) / (fmax - fmin + 1e-6)
        
        # Coordinates
        gloc = torch.floor(norm_fit * self.div).to(torch.int32)
        gloc = torch.clamp(gloc, 0, self.div - 1)
        
        # Flattening
        # Use int64 for weights to prevent overflow in high-dimensional objective space
        # weights = [div^0, div^1, ..., div^(M-1)]
        weights = (self.div.to(torch.int64)) ** torch.arange(M, device=fit.device, dtype=torch.int64)
        gid = (gloc.to(torch.int64) * weights).sum(dim=1)
        
        # Unique grids and crowding
        # We use torch.unique for JIT stability on 1D gid
        _, inverse_indices = torch.unique(gid, return_inverse=True, sorted=False)
        
        # crowd_g: count of individuals in each unique grid
        crowd_g = torch.bincount(inverse_indices)
        # crowd_per_ind: for each individual, how many individuals are in its grid
        crowd_per_ind = crowd_g[inverse_indices]
        
        return gid, crowd_per_ind

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        
        # 1. Mating Selection (Region-based Tournament)
        # We need grid info for the current population
        gid, crowd_per_ind = self._get_grid_info(self.fit)
        
        # Refined Mating: Tournament on individuals based on their grid crowding
        # Lower crowding is better for diversity
        mating_idx = tournament_selection_multifit(N, [crowd_per_ind.float()], tournament_size=2)
        
        # 2. Variation
        crossovered = simulated_binary(self.pop[mating_idx], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        
        # 3. Environmental Selection (Brutal Static Truncation)
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        
        # NDSort
        rank = non_dominate_rank(merged_fit)
        
        # Grid calculation for all merged individuals to get scores
        m_gid, m_crowd_per_ind = self._get_grid_info(merged_fit)
        
        # Score = Grid Crowding + Random Tie-break
        # We use a small random value to break ties within the same grid
        # Ensure the random tensor matches the size of m_crowd_per_ind (2*N)
        score = m_crowd_per_ind.float() + torch.rand(m_crowd_per_ind.shape[0], device=device)
        
        # Lexsort: Rank is primary (last), Score is secondary (first) (Bug #25)
        # We want lower rank first, then lower crowding score
        indices = lexsort(torch.stack([score, rank.float()]))
        
        # Truncate to N
        survivor_idx = indices[:N]
        self.pop = merged_pop[survivor_idx]
        self.fit = merged_fit[survivor_idx]

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = PESA2(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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