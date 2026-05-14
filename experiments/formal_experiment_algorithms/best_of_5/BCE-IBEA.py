import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evomo.operators.selection import nd_environmental_selection, non_dominate_rank


class BCE_IBEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, kappa: float = 0.05, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.kappa = Parameter(kappa)

        # Initialize State (Mutables)
        # PC: Pareto Population, NPC: Indicator Population
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.npc_pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.npc_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.nND = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.npc_fit = self.evaluate(self.npc_pop)
        # Initial selection to clean up random start
        self.pop, self.fit, self.nND = self._pc_selection(self.pop, self.fit, self.pop_size)
        self.npc_pop, self.npc_fit = self._environmental_selection_ibea(self.npc_pop, self.npc_fit, self.pop_size)

    def step(self) -> None:
        device = self.lb.device
        
        # 1. PC Exploration (Variation Source)
        dist_pc_npc = torch.cdist(self.pop, self.npc_pop)
        # Niche Radius calculation for mask
        dist_mat_pc = torch.cdist(self.fit, self.fit)
        sorted_dist_pc, _ = torch.sort(dist_mat_pc, dim=1)
        r = torch.mean(sorted_dist_pc[:, 2]) * (self.nND.float() / self.pop_size)
        
        neighbor_count = torch.sum(dist_pc_npc <= r, dim=1)
        mask = neighbor_count <= 1
        
        # If no unexplored, use all PC for variation to maintain pressure
        active_mask = torch.where(mask.any(), mask, torch.ones_like(mask))
        pc_subset = self.pop[active_mask]
        
        # 2. Mating (NPC uses Tournament, PC uses subset)
        # NPC Mating
        npc_fitness = self._cal_ibea_fitness(self.npc_fit)
        npc_mating_idx = tournament_selection_multifit(self.pop_size, [npc_fitness], tournament_size=2)
        npc_offspring = simulated_binary(self.npc_pop[npc_mating_idx])
        npc_offspring = polynomial_mutation(npc_offspring, self.lb, self.ub)
        npc_offspring = clamp(npc_offspring, self.lb, self.ub)
        
        # PC Mating (Exploration)
        num_pc_off = self.pop_size
        pc_mating_idx = torch.randint(0, pc_subset.shape[0], (num_pc_off,), device=device)
        pc_offspring = simulated_binary(pc_subset[pc_mating_idx])
        pc_offspring = polynomial_mutation(pc_offspring, self.lb, self.ub)
        pc_offspring = clamp(pc_offspring, self.lb, self.ub)
        
        # 3. Evaluation
        off_npc_fit = self.evaluate(npc_offspring)
        off_pc_fit = self.evaluate(pc_offspring)
        
        # 4. Environmental Selection
        # NPC Selection (IBEA)
        merged_npc_pop = torch.cat([self.npc_pop, npc_offspring], dim=0)
        merged_npc_fit = torch.cat([self.npc_fit, off_npc_fit], dim=0)
        self.npc_pop, self.npc_fit = self._environmental_selection_ibea(merged_npc_pop, merged_npc_fit, self.pop_size)
        
        # PC Selection (Niche Maintenance)
        merged_pc_pop = torch.cat([self.pop, pc_offspring], dim=0)
        merged_pc_fit = torch.cat([self.fit, off_pc_fit], dim=0)
        self.pop, self.fit, self.nND = self._pc_selection(merged_pc_pop, merged_pc_fit, self.pop_size)

    def _cal_ibea_fitness(self, fit: torch.Tensor) -> torch.Tensor:
        f_min = fit.min(dim=0)[0]
        f_max = fit.max(dim=0)[0]
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)
        
        # Indicator Matrix I_epsilon
        # I(x, y) = max_i (f_i(x) - f_i(y))
        I = torch.max(norm_fit.unsqueeze(1) - norm_fit.unsqueeze(0), dim=-1)[0]
        C = torch.max(torch.abs(I), dim=0)[0]
        
        # F(x_i) = sum_{j != i} -exp(-I(x_j, x_i) / (kappa * C_j))
        F = torch.sum(-torch.exp(-I / (C.unsqueeze(1) * self.kappa + 1e-6)), dim=0)
        return F

    def _environmental_selection_ibea(self, pop: torch.Tensor, fit: torch.Tensor, K: int) -> (torch.Tensor, torch.Tensor):
        N = pop.shape[0]
        f_min = fit.min(dim=0)[0]
        f_max = fit.max(dim=0)[0]
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)
        
        I = torch.max(norm_fit.unsqueeze(1) - norm_fit.unsqueeze(0), dim=-1)[0]
        C = torch.max(torch.abs(I), dim=0)[0]
        F = torch.sum(-torch.exp(-I / (C.unsqueeze(1) * self.kappa + 1e-6)), dim=0)
        
        active_mask = torch.ones(N, dtype=torch.bool, device=pop.device)
        sentinel = torch.finfo(torch.float32).max
        
        # Peeling loop
        for _ in range(N - K):
            # Masked argmin
            temp_F = torch.where(active_mask, F, sentinel)
            worst = torch.argmin(temp_F)
            
            # Update remaining fitness: F_j = F_j + exp(-I[worst, j] / (C[worst] * kappa))
            update = torch.exp(-I[worst, :] / (C[worst] * self.kappa + 1e-6))
            F = F + update
            active_mask[worst] = False
            
        return pop[active_mask], fit[active_mask]

    def _pc_selection(self, pop: torch.Tensor, fit: torch.Tensor, K: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # 1. Filter Rank 1
        rank = non_dominate_rank(fit)
        mask_r1 = (rank == 1)
        pop_r1 = pop[mask_r1]
        fit_r1 = fit[mask_r1]
        nND = torch.sum(mask_r1.int())
        
        N_r1 = pop_r1.shape[0]
        if N_r1 <= K:
            # If not enough Rank 1, fill with others using standard ND selection
            return nd_environmental_selection(pop, fit, K)[:2] + (nND,)

        # 2. Niche Maintenance Peeling
        dist_mat = torch.cdist(fit_r1, fit_r1)
        sorted_dist, _ = torch.sort(dist_mat, dim=1)
        # 3rd nearest neighbor (index 2)
        r = torch.mean(sorted_dist[:, 2]) * (nND.float() / K)
        
        # Diversity Metric Score = 1 - prod(clamp(dist/r, max=1))
        R = torch.clamp(dist_mat / (r + 1e-6), max=1.0)
        
        active_mask = torch.ones(N_r1, dtype=torch.bool, device=pop.device)
        
        for _ in range(N_r1 - K):
            # Recalculate score for active individuals
            R_masked = torch.where(active_mask.unsqueeze(0), R, torch.ones_like(R))
            # Also set diagonal to 1.0
            R_masked.fill_diagonal_(1.0)
            
            score = 1.0 - torch.prod(R_masked, dim=1)
            
            # Remove individual with maximum score
            temp_score = torch.where(active_mask, score, -1.0)
            worst = torch.argmax(temp_score)
            active_mask[worst] = False
            
        return pop_r1[active_mask], fit_r1[active_mask], nND


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = BCE_IBEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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