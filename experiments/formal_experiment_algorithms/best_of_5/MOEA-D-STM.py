import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp, randint
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.operators.sampling import uniform_sampling

class MOEAD_STM(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        D = lb.numel()

        # 1. Weight Vectors Initialization (Bug #13)
        W, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        self.W = Mutable(W.to(device))

        # 2. Neighborhood Initialization (Bug #19)
        dist = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.argsort(dist, dim=1)[:, :self.T].to(torch.int32))

        # 3. State Initialization
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.z = Mutable(torch.full((1, n_objs), torch.inf, device=device))
        self.pi = Mutable(torch.ones(self.pop_size, device=device))
        self.old_obj = Mutable(torch.zeros(self.pop_size, device=device))
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True)[0]
        
        # Initial Tchebycheff for utility
        znad = torch.max(self.fit, dim=0)[0]
        norm_fit = (self.fit - self.z) / (znad - self.z + 1e-6)
        # g shape (N, N) -> g[i, j] is fitness of solution i for subproblem j
        g = torch.max(torch.abs(norm_fit.unsqueeze(1)) / (self.W.unsqueeze(0) + 1e-6), dim=2)[0]
        # oldObj is the Tchebycheff value of each solution on its corresponding subproblem
        self.old_obj = torch.diagonal(g, 0)

    def step(self) -> None:
        self.gen = self.gen + 1
        device = self.lb.device
        N = self.pop_size
        T = self.T

        # --- Mating Selection ---
        boundary = torch.any(self.W < 1e-3, dim=1)
        num_boundary = torch.sum(boundary).to(torch.int32)
        
        # Tournament on -pi (Bug #31)
        num_to_select = (N // 5) - num_boundary
        # Ensure num_to_select is at least 0
        num_to_select = torch.where(num_to_select > 0, num_to_select, torch.zeros_like(num_to_select))
        tour_idx = tournament_selection_multifit(int(num_to_select), [-self.pi], tournament_size=2)
        
        # Combine indices I (Subproblems chosen for reproduction)
        I = torch.cat([torch.where(boundary)[0], tour_idx])
        num_off = I.shape[0]
        
        # Parent Selection (Vectorized)
        mask = torch.rand(num_off, device=device) < 0.9
        P1 = torch.where(mask, self.B[I, randint(0, T, (num_off,), device=device)], randint(0, N, (num_off,), device=device))
        P2 = torch.where(mask, self.B[I, randint(0, T, (num_off,), device=device)], randint(0, N, (num_off,), device=device))
        P3 = torch.where(mask, self.B[I, randint(0, T, (num_off,), device=device)], randint(0, N, (num_off,), device=device))

        # Variation
        offspring = self.pop[P1] + 0.5 * (self.pop[P2] - self.pop[P3])
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # Update Ideal Point
        self.z = torch.min(torch.cat([self.z, off_fit], dim=0), dim=0, keepdim=True)[0]

        # --- Environmental Selection (STM) ---
        all_pop = torch.cat([self.pop, offspring], dim=0)
        all_fit = torch.cat([self.fit, off_fit], dim=0)
        twoN_pool = all_fit.shape[0]
        
        znad = torch.max(all_fit, dim=0)[0]
        norm_fit = (all_fit - self.z) / (znad - self.z + 1e-6) # (2N_pool, M)

        # Tchebycheff Matrix (2N_pool, N)
        g = torch.max(torch.abs(norm_fit.unsqueeze(1)) / (self.W.unsqueeze(0) + 1e-6), dim=2)[0]

        # Perpendicular Distance Matrix (2N_pool, N)
        W_norm = torch.norm(self.W, dim=1, keepdim=True) + 1e-6
        # Cosine similarity based distance
        d1 = torch.matmul(norm_fit, (self.W / W_norm).T) # (2N_pool, N)
        proj = d1.unsqueeze(2) * (self.W / W_norm).unsqueeze(0) # (2N_pool, N, M)
        dist_matrix = torch.norm(norm_fit.unsqueeze(1) - proj, dim=2) # (2N_pool, N)

        # Stable Matching
        sub_match = self._stable_matching(g, dist_matrix)
        
        self.pop = all_pop[sub_match]
        self.fit = all_fit[sub_match]

        # --- Utility Update ---
        if self.gen % 10 == 0:
            # Current Tchebycheff values for the matched pairs
            # sub_match contains solution indices for each subproblem 0..N-1
            new_obj = g[sub_match, torch.arange(N, device=device)]
            delta = (self.old_obj - new_obj) / (self.old_obj + 1e-6)
            self.pi = torch.where(delta > 0.001, torch.ones_like(self.pi), (0.95 + 0.05 * delta / 0.001) * self.pi)
            self.old_obj = new_obj

    def _stable_matching(self, g: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        device = g.device
        twoN, N = g.shape
        sentinel = torch.iinfo(torch.int32).max

        # Subproblem preferences: indices of solutions sorted by g (lower is better)
        # sub_pref[k, j] is the k-th best solution for subproblem j
        sub_pref = torch.argsort(g, dim=0) # (2N, N)
        
        # Solution preferences: rank of subproblems for each solution (lower rank is better)
        # sol_pref_rank[i, j] is the rank of subproblem j for solution i
        sol_pref_rank = torch.argsort(torch.argsort(dist, dim=1), dim=1) # (2N, N)

        sub_match = torch.full((N,), sentinel, dtype=torch.int32, device=device)
        sol_match = torch.full((twoN,), sentinel, dtype=torch.int32, device=device)
        next_proposal = torch.zeros(N, dtype=torch.int64, device=device)

        # Gale-Shapley Loop (Vectorized rounds)
        # Max iterations is 2N because each subproblem can propose to at most 2N solutions
        for _ in range(twoN):
            sub_free_mask = (sub_match == sentinel)
            if not torch.any(sub_free_mask):
                continue
                
            proposing_subs = torch.where(sub_free_mask)[0]
            # Each free subproblem proposes to its next best solution
            target_sols = sub_pref[next_proposal[proposing_subs], proposing_subs]
            
            # Ranks of the proposing subproblems according to the target solutions
            ranks = sol_pref_rank[target_sols, proposing_subs]
            
            # Handle multiple subproblems proposing to the same solution in one round
            # Sort by rank so scatter picks the one with the minimum rank (best preference)
            # Since scatter_ overwrites, we sort descending so the best (lowest rank) is written last
            sort_idx = torch.argsort(ranks, descending=True)
            sorted_targets = target_sols[sort_idx]
            sorted_proposers = proposing_subs[sort_idx]
            
            # best_sub_for_sol stores the best proposer for each solution in this round
            round_proposals = torch.full((twoN,), sentinel, dtype=torch.int32, device=device)
            round_proposals.scatter_(0, sorted_targets, sorted_proposers.to(torch.int32))
            
            # Solutions decide: accept if free or prefer new over current
            active_sols_mask = (round_proposals != sentinel)
            active_sols = torch.where(active_sols_mask)[0]
            
            current_subs = sol_match[active_sols]
            new_subs = round_proposals[active_sols]
            
            new_ranks = sol_pref_rank[active_sols, new_subs.to(torch.int64)]
            
            # Safe indexing for current ranks
            curr_ranks = torch.full_like(new_ranks, sentinel)
            matched_mask = (current_subs != sentinel)
            if torch.any(matched_mask):
                matched_sols = active_sols[matched_mask]
                matched_subs = current_subs[matched_mask].to(torch.int64)
                curr_ranks[matched_mask] = sol_pref_rank[matched_sols, matched_subs]
            
            accept = (new_ranks < curr_ranks)
            
            if torch.any(accept):
                sols_to_update = active_sols[accept]
                subs_to_accept = new_subs[accept]
                old_subs_to_free = current_subs[accept]
                
                # Free old subproblems
                valid_old_mask = (old_subs_to_free != sentinel)
                if torch.any(valid_old_mask):
                    sub_match.scatter_(0, old_subs_to_free[valid_old_mask].to(torch.int64), sentinel)
                
                # Set new matches
                sol_match.scatter_(0, sols_to_update.to(torch.int64), subs_to_accept)
                sub_match.scatter_(0, subs_to_accept.to(torch.int64), sols_to_update.to(torch.int32))
            
            # Increment proposal counter for all subproblems that were free at start of round
            next_proposal[proposing_subs] += 1

        return sub_match.to(torch.int64)

if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_STM(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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