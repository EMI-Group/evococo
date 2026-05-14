import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling


class MOEADDU(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int = 20,
        K: int = 5,
        delta: float = 0.9,
        **kwargs
    ):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        self.K = K
        self.delta = delta
        D = lb.numel()

        # 1. Weight Generation (Bug #13)
        W, n_samples = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_samples
        W = W.to(device)

        # 2. Neighborhood (Bug #19)
        dist = torch.cdist(W, W)
        B = torch.topk(dist, T, largest=False).indices.to(torch.int32)

        # Initialize State
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.W = Mutable(W)
        self.B = Mutable(B)
        self.z = Mutable(torch.full((1, n_objs), torch.inf, device=device))
        self.znad = Mutable(torch.full((1, n_objs), -torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self.znad = torch.max(self.fit, dim=0, keepdim=True).values

    def _estimate_nadir(self, fit: torch.Tensor, z: torch.Tensor, znad: torch.Tensor) -> torch.Tensor:
        M = self.n_objs
        # Calculate ASF to find extreme points
        # Use broadcasting: (N, M) / (M, M) -> (N, M, M)
        # To find extreme point for each objective i, use weight vector e_i
        eye = torch.eye(M, device=fit.device) + 1e-6
        # asf shape: (N, M) where asf[i, j] is the asf of individual i for weight e_j
        asf = torch.max((fit.unsqueeze(1) - z) / eye, dim=2).values
        extreme_idx = torch.argmin(asf, dim=0)
        E = fit[extreme_idx]
        
        # Solve Hyperplane: A * intercepts = 1
        A = E - z
        try:
            b = torch.ones(M, 1, device=fit.device)
            intercepts = torch.linalg.solve(A, b).flatten()
            # znad = 1/intercepts + z
            new_znad = 1.0 / (intercepts + 1e-6) + z.flatten()
            
            # Fallback check (Bug #41: No .item())
            valid = torch.all(intercepts > 1e-6)
            res = torch.where(valid, new_znad, torch.max(fit, dim=0).values)
        except:
            res = torch.max(fit, dim=0).values
            
        return res.unsqueeze(0)

    def step(self) -> None:
        N = self.pop_size
        device = self.pop.device
        
        # 1. Parent Selection (Bug #14)
        rd = torch.rand(N, device=device)
        mask = rd < self.delta
        local_indices = torch.randint(0, self.T, (N, 2), device=device)
        global_indices = torch.randint(0, N, (N, 2), device=device)
        
        # Gather parents from B or global
        P_idx = torch.where(mask.unsqueeze(1), self.B[torch.arange(N, device=device).unsqueeze(1), local_indices], global_indices)
        
        # 2. Variation
        # simulated_binary expects (N, D) and pairs them
        off_pop = simulated_binary(torch.cat([self.pop, self.pop[P_idx[:, 0]]], dim=0))[:N]
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit = self.evaluate(off_pop)

        # 3. Update Ideal and Nadir Points
        self.z = torch.min(self.z, torch.min(off_fit, dim=0, keepdim=True).values)
        self.znad = self._estimate_nadir(torch.cat([self.fit, off_fit], dim=0), self.z, self.znad)

        # 4. Distance-based Update (Bug #29, Bug #41)
        # Cosine Similarity via Einsum (Requirement)
        norm_fit = off_fit - self.z
        norm_off = torch.sqrt(torch.einsum('ij,ij->i', norm_fit, norm_fit)) + 1e-6
        norm_W = torch.sqrt(torch.einsum('ij,ij->i', self.W, self.W)) + 1e-6
        
        # dot = norm_fit @ self.W.T
        dot = torch.einsum('ij,kj->ik', norm_fit, self.W)
        cosine_sim = dot / (norm_off.unsqueeze(1) * norm_W.unsqueeze(0))
        dist = 1.0 - cosine_sim
        
        # Identify K Nearest Weights
        idx_K = torch.topk(dist, self.K, dim=1, largest=False).indices # (N, K)
        
        # Modified Tchebycheff
        # g(x|w) = max_j [ (f_j - z_j) / ((znad_j - z_j + 1e-6) * w_j) ]
        scale = self.znad - self.z + 1e-6 # (1, M)
        
        # Calculate g_new for offspring at their K nearest weights
        # off_fit: (N, M) -> (N, 1, M)
        # W[idx_K]: (N, K, M)
        g_new = torch.max((off_fit.unsqueeze(1) - self.z) / (scale * self.W[idx_K] + 1e-6), dim=-1).values # (N, K)
        
        # Calculate g_old for current population at those same weights
        # self.fit[idx_K]: (N, K, M)
        g_old = torch.max((self.fit[idx_K] - self.z) / (scale * self.W[idx_K] + 1e-6), dim=-1).values # (N, K)
        
        # Update Logic (No Loops - Bug #29)
        improvement_mask = g_new <= g_old
        # Find first True in each row
        first_idx_in_K = torch.argmax(improvement_mask.to(torch.int32), dim=1)
        has_improvement = improvement_mask.any(dim=1)
        
        target_W_idx = idx_K[torch.arange(N, device=device), first_idx_in_K]
        
        # In-place Update via Masking
        update_indices = target_W_idx[has_improvement]
        source_indices = torch.arange(N, device=device)[has_improvement]
        
        # Note: If multiple offspring want to update the same weight, 
        # the last one in the index list wins in standard tensor assignment.
        self.pop[update_indices] = off_pop[source_indices]
        self.fit[update_indices] = off_fit[source_indices]


if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADDU(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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