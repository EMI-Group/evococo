import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp, lexsort
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance
from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class MOCell(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Grid Topology Precomputation
        G = int(pop_size**0.5)
        if G * G != pop_size:
            # Adjust pop_size to nearest square if necessary, though blueprint assumes N is square
            pass
        
        self.neighbor_idx = Parameter(self._get_moore_neighbors(pop_size).to(device))

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        
        # External Archive
        self.archive_pop = Mutable(torch.zeros((pop_size, self.dim), device=device))
        # Bug #1 Compliance: Use sentinel for integer/padded fitness
        sentinel = torch.iinfo(torch.int32).max
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), float(sentinel), device=device))
        self.archive_len = Mutable(torch.zeros(1, dtype=torch.int32, device=device))

    def _get_moore_neighbors(self, N: int) -> torch.Tensor:
        G = int(N**0.5)
        rows = torch.arange(G)
        cols = torch.arange(G)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        grid_r = grid_r.reshape(-1)
        grid_c = grid_c.reshape(-1)

        offsets = torch.tensor([-1, 0, 1])
        dr, dc = torch.meshgrid(offsets, offsets, indexing='ij')
        dr = dr.reshape(-1)
        dc = dc.reshape(-1)

        # (N, 9)
        neigh_r = (grid_r.view(-1, 1) + dr.view(1, -1)) % G
        neigh_c = (grid_c.view(-1, 1) + dc.view(1, -1)) % G
        
        return neigh_r * G + neigh_c

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial Archive Update
        self._update_archive(self.pop, self.fit)

    def _update_archive(self, off_pop, off_fit):
        # Merge current archive and offspring
        # Use archive_len to mask valid archive members
        mask = torch.arange(self.pop_size, device=self.pop.device) < self.archive_len
        valid_arc_pop = self.archive_pop[mask]
        valid_arc_fit = self.archive_fit[mask]
        
        merged_pop = torch.cat([valid_arc_pop, off_pop], dim=0)
        merged_fit = torch.cat([valid_arc_fit, off_fit], dim=0)
        
        # Unique rows to avoid duplicates (Bug #3)
        u_pop, u_idx = unique_rows_sorted(merged_pop)
        u_fit = merged_fit[u_idx]
        
        # Non-dominated sorting
        rank = non_dominate_rank(u_fit)
        is_rank0 = (rank == 0)
        
        sur_pop = u_pop[is_rank0]
        sur_fit = u_fit[is_rank0]
        num_sur = sur_pop.shape[0]
        
        # Pruning if archive exceeds pop_size
        if num_sur > self.pop_size:
            # Bug #21: Crowding distance on rank 0
            cd = crowding_distance(sur_fit, torch.ones(num_sur, dtype=torch.bool, device=sur_pop.device))
            # Bug #25: Lexsort primary key (CD) last? No, rank is same (0), so sort by CD descending
            # To get largest CD, we sort by -CD
            idx = lexsort(torch.stack([-cd]))
            sur_pop = sur_pop[idx[:self.pop_size]]
            sur_fit = sur_fit[idx[:self.pop_size]]
            num_sur = self.pop_size
            
        # Update Mutable Archive
        new_arc_pop = torch.zeros_like(self.archive_pop)
        new_arc_fit = torch.full_like(self.archive_fit, float(torch.iinfo(torch.int32).max))
        
        new_arc_pop[:num_sur] = sur_pop
        new_arc_fit[:num_sur] = sur_fit
        
        self.archive_pop = new_arc_pop
        self.archive_fit = new_arc_fit
        self.archive_len = torch.tensor([num_sur], dtype=torch.int32, device=self.pop.device)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        
        # 1. Mating / Variation
        rank = non_dominate_rank(self.fit).float()
        # Bug #21: Crowding distance per front is ideal, but MOCell uses it for selection pressure
        # We calculate CD on the whole population for the tournament as a diversity proxy
        cd = crowding_distance(self.fit, torch.ones(N, dtype=torch.bool, device=device))
        
        # Gather metrics for neighbors (N, 9)
        neighbor_ranks = rank[self.neighbor_idx]
        neighbor_cdists = cd[self.neighbor_idx]
        
        # Tournament Selection (Bug #27, #29)
        # Pick 2 random neighbors for each cell
        rand_idx = torch.randint(0, 9, (N, 2), device=device)
        # Map back to global indices
        idx_a = torch.gather(self.neighbor_idx, 1, rand_idx[:, 0:1]).squeeze()
        idx_b = torch.gather(self.neighbor_idx, 1, rand_idx[:, 1:2]).squeeze()
        
        # Compare rank (min) then cdist (max)
        rank_a = torch.gather(neighbor_ranks, 1, rand_idx[:, 0:1]).squeeze()
        rank_b = torch.gather(neighbor_ranks, 1, rand_idx[:, 1:2]).squeeze()
        cd_a = torch.gather(neighbor_cdists, 1, rand_idx[:, 0:1]).squeeze()
        cd_b = torch.gather(neighbor_cdists, 1, rand_idx[:, 1:2]).squeeze()
        
        # Winner selection logic
        mask_a = (rank_a < rank_b) | ((rank_a == rank_b) & (cd_a > cd_b))
        parent_idx = torch.where(mask_a, idx_a, idx_b)
        
        # Variation (SBX + PM)
        # OperatorGAhalf logic: use parent i and the selected neighbor
        mating_pool = torch.stack([torch.arange(N, device=device), parent_idx], dim=1).view(-1)
        off_pop = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        # simulated_binary returns 2N, we take N (one per cell)
        off_pop = off_pop[::2] 
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)
        
        # 2. Evaluation
        off_fit = self.evaluate(off_pop)
        
        # 3. Local Replacement (Bug #24)
        # off dominates current
        better = (off_fit <= self.fit).all(-1) & (off_fit < self.fit).any(-1)
        # current does not dominate off
        non_dominated = ~((self.fit <= off_fit).all(-1) & (self.fit < off_fit).any(-1))
        
        replace_mask = better | non_dominated
        self.pop = torch.where(replace_mask.unsqueeze(1), off_pop, self.pop)
        self.fit = torch.where(replace_mask.unsqueeze(1), off_fit, self.fit)
        
        # 4. Archive Update
        self._update_archive(off_pop, off_fit)
        
        # 5. Feedback from Archive
        # We'll use a fixed 20 and mask if archive_len < 20.
        rep_idx_arc = torch.randperm(self.archive_len, device=device)[:20]
        rep_idx_pop = torch.randperm(N, device=device)[:20]
        
        # Vectorized feedback
        self.pop[rep_idx_pop] = self.archive_pop[rep_idx_arc]
        self.fit[rep_idx_pop] = self.archive_fit[rep_idx_arc]

# === FIXED DEMO BLOCK ===
# This block MUST be appended at the end of the file.
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # MOCell must be replaced by your actual class name
    algo = MOCell(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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