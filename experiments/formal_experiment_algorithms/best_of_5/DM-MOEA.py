import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit, crowding_distance
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted

class DMMOEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()
        self.dim = D

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.rand(pop_size, D, device=device) < 0.5)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        
        # Historical Buffers (Circular)
        self.dec_source = Mutable(torch.zeros((200, pop_size, D), device=device))
        self.mask_source = Mutable(torch.zeros((200, pop_size, D), device=device))
        self.var_fit = Mutable(torch.zeros(D, device=device))
        self.iter = Mutable(torch.zeros(1, dtype=torch.int32, device=device))
        
        self.sentinel = torch.iinfo(torch.int32).max

    def init_step(self) -> None:
        # A. Initialization (Variable Fitness Estimation)
        # Generate D individuals where each individual i has i-th bit set
        init_mask = torch.eye(self.dim, device=self.lb.device)
        # If D != pop_size, we adjust for the estimation phase
        eval_mask = init_mask.repeat((self.pop_size // self.dim) + 1, 1)[:self.pop_size]
        
        # Evaluate
        self.fit = self.evaluate(self.pop * eval_mask)
        
        # Rank-based score accumulation
        ranks = non_dominate_rank(self.fit)
        # Accumulate importance: lower rank (better) -> higher score
        scores = (ranks.max() - ranks).float()
        # Map back to variables based on the identity mask
        self.var_fit = torch.zeros(self.dim, device=self.lb.device)
        # Vectorized accumulation
        self.var_fit = self.var_fit.index_add(0, torch.arange(self.pop_size, device=self.lb.device) % self.dim, scores)

    def _batched_mlp_predict(self, history: torch.Tensor, steps: int) -> torch.Tensor:
        P, N, D = history.shape
        x = history.permute(1, 2, 0).reshape(N * D, P, 1)
        
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std
        
        t = torch.linspace(0, 1, P, device=history.device).view(1, P, 1).repeat(N * D, 1, 1)
        
        W1 = torch.ones((N * D, 1, 10), device=history.device) * 0.1
        b1 = torch.zeros((N * D, 1, 10), device=history.device)
        W2 = torch.ones((N * D, 10, 1), device=history.device) * 0.1
        b2 = torch.zeros((N * D, 1, 1), device=history.device)
        
        t_next = torch.full((N * D, 1, 1), 1.1, device=history.device)
        h_next = torch.sigmoid(t_next @ W1 + b1)
        pred_norm = h_next @ W2 + b2
        
        pred = pred_norm * x_std + x_mean
        return pred.view(N, D)

    def _grouped_mutation(self, pop: torch.Tensor) -> torch.Tensor:
        N, D = pop.shape
        idx = torch.argsort(pop, dim=1)
        group_size = D // 4
        g_idx = torch.randint(0, 4, (1,), device=pop.device)
        
        mask = torch.zeros((N, D), dtype=torch.bool, device=pop.device)
        start = g_idx * group_size
        end = (g_idx + 1) * group_size
        target_vars = idx[:, start:end]
        mask.scatter_(1, target_vars, True)
        
        mutated = polynomial_mutation(pop, self.lb, self.ub)
        return torch.where(mask, mutated, pop)

    def step(self) -> None:
        self.iter += 1
        N, D = self.pop_size, self.dim
        
        # 1. Mating / Variation
        # Manual Tournament for Variable Selection (Bug #36 fix)
        # We need to select variable indices for crossover/mutation logic
        # Sample random variable indices for tournament
        c1 = torch.randint(0, D, (N, D), device=self.lb.device)
        c2 = torch.randint(0, D, (N, D), device=self.lb.device)
        v_fit = self.var_fit
        winner_mask = v_fit[c1] > v_fit[c2]
        chosen_vars = torch.where(winner_mask, c1, c2)
        
        # Population Selection for Mating
        # We need FrontNo and CrowdDis from the current state
        ranks = non_dominate_rank(self.fit)
        cd = crowding_distance(self.fit, torch.ones(N, dtype=torch.bool, device=self.lb.device))
        # tournament_selection_multifit expects objectives to minimize
        mating_idx = tournament_selection_multifit(N, [ranks.float(), -cd])
        
        # Crossover Masks
        p1_mask = self.mask[mating_idx]
        p2_mask = self.mask[torch.roll(mating_idx, 1, dims=0)]
        off_mask = torch.where(torch.rand(N, D, device=self.lb.device) < 0.5, p1_mask, p2_mask)
        
        # Crossover Decs
        off_pop = simulated_binary(self.pop[mating_idx], pro_c=1.0, dis_c=20.0)
        off_pop = self._grouped_mutation(off_pop)
        off_pop = clamp(off_pop, self.lb, self.ub)
        
        # Apply Mask
        off_pop_masked = off_pop * off_mask.float()
        off_fit = self.evaluate(off_pop_masked)
        
        # 2. Environmental Selection (Integrated Peeling)
        merge_pop = torch.cat([self.pop, off_pop], dim=0)
        merge_mask = torch.cat([self.mask, off_mask], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)
        
        u_fit, u_idx = unique_rows_sorted(merge_fit)
        u_pop = merge_pop[u_idx]
        u_mask = merge_mask[u_idx]
        
        ranks = non_dominate_rank(u_fit)
        selected_indices = torch.full((N,), self.sentinel, dtype=torch.long, device=self.lb.device)
        count = 0
        max_rank = int(ranks.max())
        
        for r in range(max_rank + 1):
            mask_front = (ranks == r)
            num_in_front = mask_front.sum()
            
            cd = crowding_distance(u_fit, mask_front)
            
            if count + num_in_front <= N:
                indices = torch.where(mask_front, torch.arange(len(u_fit), device=self.lb.device), self.sentinel)
                valid_indices = indices[indices != self.sentinel]
                selected_indices[count:count+num_in_front] = valid_indices
                count += num_in_front
            else:
                needed = N - count
                cd_front = torch.where(mask_front, cd, torch.tensor(-1.0, device=self.lb.device))
                _, top_k = torch.topk(cd_front, needed)
                selected_indices[count:N] = top_k
                count = N
            
            if count >= N:
                break

        # Final Sync
        self.pop = u_pop[selected_indices]
        self.mask = u_mask[selected_indices]
        self.fit = u_fit[selected_indices]

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = DMMOEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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