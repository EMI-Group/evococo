import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp, randint
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted

class CLIA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        
        # Archive size constraint: 0.33 * M * N
        self.max_arc = int(0.33 * n_objs * pop_size)
        
        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        
        # Reference Vectors
        z_init, _ = uniform_sampling(pop_size, n_objs)
        self.Z = Mutable(z_init.to(device))
        self.nz = self.Z.shape[0]
        
        # Archive
        self.archive_pop = Mutable(torch.zeros((self.max_arc, self.dim), device=device))
        self.archive_fit = Mutable(torch.full((self.max_arc, n_objs), torch.inf, device=device))
        self.arc_len = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        
        # SVM State & Adaptation
        self.zmin = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.assoc_history = Mutable(torch.zeros(self.nz, dtype=torch.int32, device=device))
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        
        # Sentinel for Bug #1
        self.sentinel = torch.iinfo(torch.int32).max

    def _calculate_pdm(self, fit: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        # Bug #12: Safe Division
        norm = fit.norm(dim=1, keepdim=True)
        z_norm = Z.norm(dim=1, keepdim=True).t()
        # Cosine similarity
        cos = (fit @ Z.t()) / (norm @ z_norm + 1e-6)
        # Sine distance
        sine = torch.sqrt(torch.clamp(1.0 - cos**2, min=0.0))
        return norm * sine

    def _svm_predict(self, Z: torch.Tensor, SV_fit: torch.Tensor, SV_labels: torch.Tensor) -> torch.Tensor:
        # Gaussian Kernel K(x, y) = exp(-||x-y||^2 / (2*sigma^2))
        dist_sq = torch.cdist(Z, SV_fit, p=2)**2
        sigma = 0.1 # Smaller sigma for more localized influence
        K = torch.exp(-dist_sq / (2 * sigma**2 + 1e-6))
        # Decision function proxy: weighted sum of labels
        # SV_labels is (N_sv,), K is (N_z, N_sv)
        return (K @ SV_labels.float()) / (K.sum(dim=1) + 1e-6)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.zmin = torch.min(self.fit, dim=0)[0]
        # Initial Archive
        rank = non_dominate_rank(self.fit)
        mask = (rank == 1)
        num_arc = torch.sum(mask.int())
        actual_num = torch.minimum(num_arc, torch.tensor(self.max_arc, device=self.lb.device))
        indices = torch.where(mask)[0][:actual_num]
        self.archive_pop[:actual_num] = self.pop[indices]
        self.archive_fit[:actual_num] = self.fit[indices]
        self.arc_len = actual_num

    def step(self) -> None:
        self.gen += 1
        device = self.lb.device
        
        # 1. Mating
        mating_pool = randint(0, self.pop_size, (self.pop_size,), device=device)
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # 2. Archive Update
        valid_arc_fit = self.archive_fit[:self.arc_len]
        valid_arc_pop = self.archive_pop[:self.arc_len]
        combined_fit = torch.cat([valid_arc_fit, off_fit], dim=0)
        combined_pop = torch.cat([valid_arc_pop, offspring], dim=0)
        
        rank_arc = non_dominate_rank(combined_fit)
        mask_arc = (rank_arc == 1)
        arc_candidates_fit = combined_fit[mask_arc]
        arc_candidates_pop = combined_pop[mask_arc]
        
        num_cand = arc_candidates_fit.shape[0]
        if num_cand > self.max_arc:
            # Truncation based on diversity (PDM to current Z)
            norm_A = arc_candidates_fit - self.zmin
            pdm_arc = self._calculate_pdm(norm_A, self.Z)
            min_pdm = torch.min(pdm_arc, dim=1)[0]
            # Select those with largest PDM (most diverse/distant from reference points)
            _, idx_keep = torch.topk(min_pdm, self.max_arc)
            self.archive_fit[:] = arc_candidates_fit[idx_keep]
            self.archive_pop[:] = arc_candidates_pop[idx_keep]
            self.arc_len = torch.tensor(self.max_arc, device=device)
        else:
            self.archive_fit[:num_cand] = arc_candidates_fit
            self.archive_pop[:num_cand] = arc_candidates_pop
            self.arc_len = torch.tensor(num_cand, device=device)

        # 3. Environmental Selection (Cascade Clustering)
        all_pop = torch.cat([self.pop, offspring], dim=0)
        all_fit = torch.cat([self.fit, off_fit], dim=0)
        unique_pop, u_idx = unique_rows_sorted(all_pop)
        unique_fit = all_fit[u_idx]
        
        f_min = torch.min(unique_fit, dim=0)[0]
        self.zmin = torch.min(self.zmin, f_min)
        f_max = torch.max(unique_fit, dim=0)[0]
        norm_fit = (unique_fit - self.zmin) / (f_max - self.zmin + 1e-6)
        
        dist_matrix = self._calculate_pdm(norm_fit, self.Z)
        pi = torch.argmin(dist_matrix, dim=1) # Association
        
        # F-Metric: 5 * PDM + mean(norm_fit)
        pdm_vals = torch.min(dist_matrix, dim=1)[0]
        f_metric = 5.0 * pdm_vals + torch.mean(norm_fit, dim=1)
        
        # Vectorized Round-Robin Selection
        # For each individual, calculate its rank within its associated cluster
        # We use a large offset for sorting: cluster_rank * num_clusters + cluster_id
        # To get cluster_rank, we sort by f_metric within each cluster
        sort_val = pi.float() * 1e6 + f_metric 
        # This isn't quite round-robin. Let's use a more robust approach:
        # 1. Sort all by f_metric
        # 2. Assign a 'rank' within each cluster
        # 3. Sort by (rank, cluster_id)
        
        # Sort globally first to help calculate intra-cluster rank
        global_indices = torch.argsort(f_metric)
        sorted_pi = pi[global_indices]
        
        # Calculate intra-cluster rank using a cumulative sum trick
        # For each cluster, the i-th occurrence gets rank i
        one_hot_pi = torch.nn.functional.one_hot(pi, num_classes=self.nz)
        # Sort one_hot to match global_indices
        sorted_one_hot = one_hot_pi[global_indices]
        intra_cluster_rank = torch.cumsum(sorted_one_hot, dim=0) * sorted_one_hot
        intra_cluster_rank = torch.sum(intra_cluster_rank, dim=1)
        
        # Final selection key: Primary = intra_cluster_rank, Secondary = pi
        # Bug #25: Primary key last in lexsort
        selection_key = intra_cluster_rank.float() * 10000.0 + pi[global_indices].float()
        final_indices = global_indices[torch.argsort(selection_key)]
        
        # Select top N
        selected_idx = final_indices[:self.pop_size]
        self.pop = unique_pop[selected_idx]
        self.fit = unique_fit[selected_idx]

        # 4. Incremental SVM (Reference Vector Adaptation)
        if self.gen % 20 == 0:
            # Identify active reference vectors from current associations
            active_mask = torch.zeros(self.nz, dtype=torch.bool, device=device)
            active_mask[pi] = True
            
            # Training data: Archive (Positive) vs Inactive Z (Negative)
            inactive_indices = torch.where(~active_mask)[0]
            sv_fit = torch.cat([self.archive_fit[:self.arc_len], self.Z[inactive_indices]], dim=0)
            sv_labels = torch.cat([torch.ones(self.arc_len, device=device), 
                                   -torch.ones(inactive_indices.shape[0], device=device)], dim=0)
            
            if sv_fit.shape[0] > 0:
                scores = self._svm_predict(self.Z, sv_fit, sv_labels)
                
                # Adapt Z: Add noise to high scoring vectors
                # scores shape is (nz,)
                scale_noise = 0.05
                noise = torch.randn_like(self.Z) * scale_noise
                # Fix shape mismatch: ensure mask is (nz, 1)
                mask = (scores > 0).unsqueeze(1).expand(-1, self.n_objs)
                self.Z = torch.where(mask, self.Z + noise, self.Z)
                
                # Re-normalize Z to unit simplex/sphere
                self.Z = self.Z / (self.Z.norm(dim=1, keepdim=True) + 1e-6)

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = CLIA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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