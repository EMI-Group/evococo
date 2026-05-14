import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import nanmin, lexsort
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import crowding_distance
from evomo.operators.selection import non_dominate_rank

class MOEADM2M(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, K: int = 10, max_gen: int = 100):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.K = K
        self.S = pop_size // K
        self.max_gen = Parameter(torch.tensor(max_gen, device=device))
        self.dim = lb.numel()

        # Initialize Reference Vectors (Subproblem centers)
        W, _ = uniform_sampling(K, n_objs)
        self.W = Mutable(W.to(device))

        # Initialize State
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _associate(self, total_pop, total_fit, W, S):
        device = total_pop.device
        N_total = total_pop.shape[0]
        K = W.shape[0]
        
        # Cosine Similarity Partitioning
        f = total_fit - nanmin(total_fit, dim=0)[0]
        norm_f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-6)
        norm_W = W / (torch.norm(W, dim=1, keepdim=True) + 1e-6)
        # Use einsum for similarity: [2N, M] @ [K, M]^T -> [2N, K]
        cosine_sim = torch.einsum('nm,km->nk', norm_f, norm_W)
        sub_assignment = torch.argmax(cosine_sim, dim=1)

        next_pop_list = []
        next_fit_list = []

        # Subproblem Refinement Loop (K is small and constant)
        for k in range(K):
            mask = (sub_assignment == k)
            count = torch.sum(mask)
            
            # Case 1: Underfilled (including count == 0 deadlock breaker)
            if count < S:
                idx_in_sub = torch.where(mask)[0]
                fill_count = S - int(count)
                rand_idx = torch.randint(0, N_total, (fill_count,), device=device)
                
                next_pop_list.append(total_pop[idx_in_sub])
                next_pop_list.append(total_pop[rand_idx])
                next_fit_list.append(total_fit[idx_in_sub])
                next_fit_list.append(total_fit[rand_idx])
            
            # Case 2: Overfilled
            elif count > S:
                # Filter to only those in subproblem
                sub_indices = torch.where(mask)[0]
                sub_fit = total_fit[sub_indices]
                
                # ND Sort and Crowding Distance on the subset
                sub_rank = non_dominate_rank(sub_fit)
                # Bug #6: pass full fit and boolean mask
                sub_dist = crowding_distance(total_fit, mask)[sub_indices]
                
                # Lexsort: Primary key (rank) last. We want min rank and max dist.
                sel_internal = lexsort(torch.stack([-sub_dist, sub_rank.float()]))[:S]
                sel_idx = sub_indices[sel_internal]
                
                next_pop_list.append(total_pop[sel_idx])
                next_fit_list.append(total_fit[sel_idx])
                
            # Case 3: Exactly S
            else:
                idx_in_sub = torch.where(mask)[0]
                next_pop_list.append(total_pop[idx_in_sub])
                next_fit_list.append(total_fit[idx_in_sub])

        return torch.cat(next_pop_list, dim=0), torch.cat(next_fit_list, dim=0)

    def init_step(self) -> None:
        initial_fit = self.evaluate(self.pop)
        # Initial association to ensure subproblem structure
        new_pop, new_fit = self._associate(self.pop, initial_fit, self.W, self.S)
        self.pop = new_pop
        self.fit = new_fit

    def step(self) -> None:
        self.gen = self.gen + 1
        N = self.pop_size
        D = self.dim
        device = self.pop.device

        # 1. Mating Pool Selection
        sub_indices = torch.arange(N, device=device) // self.S
        rand_prob = torch.rand(N, device=device)
        
        # Local Selection
        offset = torch.randint(0, self.S, (N,), device=device)
        local_partner_idx = (sub_indices * self.S) + offset
        
        # Global Selection
        global_partner_idx = torch.randint(0, N, (N,), device=device)
        
        partner_idx = torch.where(rand_prob < 0.7, local_partner_idx, global_partner_idx)
        Parent1 = self.pop
        Parent2 = self.pop[partner_idx]

        # 2. Custom Variation Logic
        FE_ratio = self.gen.float() / self.max_gen
        rc = (1 - FE_ratio) ** 0.7
        rm = (1 - FE_ratio) ** 0.7
        
        # Crossover
        off_pop = Parent1 + (Parent2 - Parent1) * (torch.rand(N, D, device=device) * (1 + rc) - rc)
        
        # Mutation
        mut_mask = torch.rand(N, D, device=device) < (1.0 / D)
        off_pop = off_pop + (torch.rand(N, D, device=device) * (self.ub - self.lb) * rm * mut_mask)

        # Boundary Repair (Bug #5 Compliance)
        repair_low = self.lb + 0.5 * torch.rand(N, D, device=device) * (Parent1 - self.lb)
        repair_high = self.ub - 0.5 * torch.rand(N, D, device=device) * (self.ub - Parent1)
        
        off_pop = torch.where(off_pop < self.lb, repair_low, off_pop)
        off_pop = torch.where(off_pop > self.ub, repair_high, off_pop)

        # 3. Evaluation
        off_fit = self.evaluate(off_pop)

        # 4. Environmental Selection
        total_pop = torch.cat([self.pop, off_pop], dim=0)
        total_fit = torch.cat([self.fit, off_fit], dim=0)
        
        new_pop, new_fit = self._associate(total_pop, total_fit, self.W, self.S)
        self.pop = new_pop
        self.fit = new_fit

if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADM2M(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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