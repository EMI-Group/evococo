import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling


class MOEAD_MRDL(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, 
                 T: int = 20, max_gen: int = 100, nmov: int = 10, gamma: float = 20.0, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        self.max_gen = max_gen
        self.nmov = nmov
        self.gamma = gamma
        D = lb.numel()

        # 1. Weights & Neighbors (Bug #13, #19)
        W, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        W = W.to(device)
        dist_matrix = torch.cdist(W, W)
        B = torch.topk(dist_matrix, T, largest=False, dim=1).indices

        # 2. State Initialization
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), float('inf'), device=device))
        self.W = Mutable(W)
        self.B = Mutable(B.to(torch.long))
        self.z = Mutable(torch.full((n_objs,), float('inf'), device=device))
        
        # MRDL History (Bug #1)
        self.all_egamma = Mutable(torch.full((max_gen,), -1.0, device=device))
        self.gen_counter = Mutable(torch.zeros(1, dtype=torch.int32, device=device))
        
        # Adaptive Parameters
        self.disC = Mutable(torch.full((1,), 20.0, device=device))
        self.pn = Mutable(torch.full((1,), 0.0, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0).values

    def _calculate_mrdl(self, off_fit: torch.Tensor, neighbor_fits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # off_fit: [N, M], neighbor_fits: [N, T, M], z: [M]
        N, M = off_fit.shape
        # Find nearest neighbor in objective space
        diff_to_neighbors = torch.norm(neighbor_fits - off_fit.unsqueeze(1), dim=2) # [N, T]
        nearest_idx = torch.argmin(diff_to_neighbors, dim=1) # [N]
        
        # conv_dir: direction from offspring to its nearest neighbor
        conv_dir = neighbor_fits[torch.arange(N, device=off_fit.device), nearest_idx] - off_fit
        
        # Cosine similarity between (off_fit - z) and conv_dir
        vec_a = off_fit - z.view(1, M)
        # cosine_similarity expects [N, M]
        cos_sim = torch.nn.functional.cosine_similarity(vec_a, conv_dir, dim=1)
        
        # MRDL = (||off_fit - z|| * cos_sim) / (||conv_dir|| + 1e-6)
        norm_a = torch.norm(vec_a, dim=1)
        norm_dir = torch.norm(conv_dir, dim=1)
        mrdl = (norm_a * cos_sim) / (norm_dir + 1e-6) # Bug #12
        return mrdl

    def step(self) -> None:
        N = self.pop_size
        M = self.n_objs
        device = self.pop.device
        gen = self.gen_counter[0]

        # 1. Mating Pool (Bug #14)
        indices = torch.randint(0, self.T, (N, 2), device=device)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, 2)
        p_idx = self.B[row_idx, indices] # [N, 2]
        
        # 2. Variation
        # Reshape to (N*2, D) to satisfy evox operator expectations in some versions
        mating_pop = self.pop[p_idx.reshape(-1)]
        offspring = simulated_binary(mating_pop, dis_c=self.disC[0])
        # SBX returns 2*N individuals, we only need N (one per subproblem)
        offspring = offspring[::2] 
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        
        # Gaussian Perturbation
        offspring = offspring + torch.randn(offspring.shape, device=device) * self.pn[0]
        offspring = clamp(offspring, self.lb, self.ub)
        
        off_fit = self.evaluate(offspring)
        
        # 3. Update Ideal Point
        self.z = torch.min(torch.stack([self.z, torch.min(off_fit, dim=0).values]), dim=0).values

        # 4. Batched Metric Calculation
        neighbor_fits = self.fit[self.B] # [N, T, M]
        neighbor_weights = self.W[self.B] # [N, T, M]
        
        diff_old = neighbor_fits - self.z.view(1, 1, M)
        g_old = torch.max(neighbor_weights * diff_old, dim=2).values
        
        diff_new = off_fit.unsqueeze(1) - self.z.view(1, 1, M)
        g_new = torch.max(neighbor_weights * diff_new, dim=2).values
        
        mrdl_vals = self._calculate_mrdl(off_fit, neighbor_fits, self.z)
        
        # 5. Sequential Update
        # Track MRDL of replaced individuals for adaptation
        replaced_mrdl_sum = torch.tensor(0.0, device=device)
        replaced_count = torch.tensor(0.0, device=device)

        for i in range(N):
            better_mask = (g_new[i] < g_old[i])
            if torch.any(better_mask):
                # In MRDL, we check if the MRDL of the offspring is acceptable
                # The MATLAB code calculates MRDL for each parent in PM.
                # Here we use the offspring's MRDL as a proxy for the update condition.
                if mrdl_vals[i] <= self.gamma:
                    update_indices = self.B[i][better_mask]
                    self.pop[update_indices] = offspring[i]
                    self.fit[update_indices] = off_fit[i]
                    replaced_mrdl_sum += mrdl_vals[i]
                    replaced_count += 1.0

        # 6. Adaptation (Log-Linear Regression)
        current_egamma = torch.where(replaced_count > 0, replaced_mrdl_sum / (replaced_count + 1e-6), torch.tensor(0.0, device=device))
        self.all_egamma[gen.to(torch.long)] = current_egamma
        
        curr_gen_int = gen.to(torch.long)
        n_samples = curr_gen_int + 1
        
        # Adaptation logic starts after some history is collected
        if n_samples > 1:
            # Moving Average
            start_idx = torch.clamp(n_samples - self.nmov, min=0)
            ma_egamma = torch.mean(self.all_egamma[start_idx:n_samples])
            
            # Regression on MA history (excluding current to predict current)
            # MATLAB: Y = log(MAEgamma(1:end-1))
            # We use all available history for stability
            Y = torch.log(self.all_egamma[:n_samples] + 1e-6).view(-1, 1)
            X = torch.stack([torch.ones(n_samples, device=device), 
                             torch.arange(n_samples, device=device, dtype=torch.float32)], dim=1)
            
            # Solve LSTSQ: Phi \ Y
            sol = torch.linalg.lstsq(X, Y).solution
            # Predict for current generation index
            predict = torch.exp(sol[0] + sol[1] * curr_gen_int.to(torch.float32))
            
            # Update Params based on current MRDL vs prediction
            # MATLAB: if allEgamma(end) > predictEgamma
            cond = current_egamma > predict
            
            new_disC = torch.where(cond, self.disC - 2.0, self.disC + 2.0)
            self.disC = torch.clamp(new_disC, 2.0, 30.0)
            
            new_pn = torch.where(cond, 0.5 * (current_egamma - predict), torch.tensor([0.0], device=device))
            self.pn = torch.clamp(new_pn, 0.0, 1.0)
        
        self.gen_counter = self.gen_counter + 1


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_MRDL(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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