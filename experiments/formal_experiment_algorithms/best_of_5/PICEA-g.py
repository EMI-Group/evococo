import torch
from typing import Tuple
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class PICEAg(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, n_goal: int = 100, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.n_goal = n_goal
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        # Main Population
        initial_pop = torch.rand(pop_size, D, device=device) * (ub - lb) + lb
        self.pop = Mutable(initial_pop)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Archive
        self.archive_pop = Mutable(initial_pop.clone())
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Coevolving Goals
        self.goals = Mutable(torch.rand(n_goal, n_objs, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.archive_fit = self.fit.clone()
        
        # Initial Goal Range based on initial fitness
        g_min = self.fit.min(dim=0)[0]
        g_max = self.fit.max(dim=0)[0] * 1.2
        self.goals = g_min + torch.rand((self.n_goal, self.n_objs), device=self.fit.device) * (g_max - g_min + 1e-6)

    def _truncate_archive(self, pop: torch.Tensor, fit: torch.Tensor, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bug #30: Brutal Static Truncation
        dist = torch.cdist(fit, fit, p=2)
        # Fill diagonal with infinity to ignore self-distance
        dist.fill_diagonal_(float('inf'))
        # Find minimum distance for each individual (density measure)
        min_dist = dist.min(dim=1)[0]
        # Select individuals with largest minimum distances (most isolated)
        top_indices = torch.topk(min_dist, k=count, largest=True)[1]
        return pop[top_indices], fit[top_indices]

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        M = self.n_objs
        N_goal = self.n_goal

        # 1. Mating (Tournament Selection + SBX + PM)
        mating_idx = torch.randint(0, N, (N,), device=device)
        off_pop = simulated_binary(self.pop[mating_idx], pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)
        
        # 2. Evaluation
        off_fit = self.evaluate(off_pop)

        # 3. Archive Update
        combined_arc_pop = torch.cat([self.archive_pop, off_pop], dim=0)
        combined_arc_fit = torch.cat([self.archive_fit, off_fit], dim=0)
        
        # Filter unique to avoid duplicates in archive
        u_arc_pop, u_idx = unique_rows_sorted(combined_arc_pop)
        u_arc_fit = combined_arc_fit[u_idx]
        
        # Pareto Dominance for Archive (Bug #24)
        ranks = non_dominate_rank(u_arc_fit)
        is_rank1 = (ranks == 1)
        rank1_pop = u_arc_pop[is_rank1]
        rank1_fit = u_arc_fit[is_rank1]
        
        # Truncate if necessary
        num_rank1 = rank1_fit.shape[0]
        if num_rank1 > N:
            self.archive_pop, self.archive_fit = self._truncate_archive(rank1_pop, rank1_fit, N)
        else:
            indices = torch.arange(num_rank1, device=device)
            pad_indices = indices.repeat((N // num_rank1) + 1)[:N]
            self.archive_pop = rank1_pop[pad_indices]
            self.archive_fit = rank1_fit[pad_indices]

        # 4. Goal Generation (GeneGoal)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)
        g_min = combined_fit.min(dim=0)[0]
        g_max = combined_fit.max(dim=0)[0] * 1.2
        new_goals = g_min + torch.rand((N_goal, M), device=device) * (g_max - g_min + 1e-6)
        candidate_goals = torch.cat([self.goals, new_goals], dim=0) # [2*N_goal, M]

        # 5. Environmental Selection (Coevolutionary Logic)
        satisfy_mask = (combined_fit.unsqueeze(1) <= candidate_goals.unsqueeze(0)).all(dim=-1)
        
        # ng: Count solutions satisfying each goal
        ng = satisfy_mask.sum(dim=0).float() # [2*N_goal]
        
        # Solution Fitness (Fs) - Bug #23 & #29 Compliance
        inv_ng = 1.0 / (ng + 1e-6) # [2*N_goal]
        Fs = (satisfy_mask.float() @ inv_ng.unsqueeze(1)).squeeze() # [2N]
        
        # Goal Fitness (Fg) - Bug #12 & #41 Compliance
        Fg = 1.0 / (1.0 + (ng - 1.0) / (N - 1.0) + 1e-6)
        Fg = torch.where(ng == 0, torch.full_like(Fg, 0.5), Fg) # [2*N_goal]
        
        # Population Update
        combined_ranks = non_dominate_rank(combined_fit)
        is_rank1_comb = (combined_ranks == 1)
        # Prioritize Rank 1 solutions
        adjusted_Fs = torch.where(is_rank1_comb, Fs + Fs.max() + 1.0, Fs)
        pop_idx = torch.topk(adjusted_Fs, k=N, largest=True)[1]
        
        # Re-assigning correctly
        combined_all_pop = torch.cat([self.pop, off_pop], dim=0)
        self.pop = combined_all_pop[pop_idx]
        self.fit = combined_fit[pop_idx]
        
        # Goal Update
        goal_idx = torch.topk(Fg, k=N_goal, largest=True)[1]
        self.goals = candidate_goals[goal_idx]

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = PICEAg(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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