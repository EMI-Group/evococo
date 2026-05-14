import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp, randint
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evomo.operators.selection import nd_environmental_selection
from evomo.utils import unique_rows_sorted


class MOEAD_VOV(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        
        # Parameters from MATLAB: G=100, C=9, theta=0.02
        self.G = 100
        self.theta = 0.02
        self.BP = pop_size * self.G * 9
        
        # Bug #2: Ceil semantics for neighborhood size T
        self.T = (pop_size + 9) // 10
        
        # 1. Weights Initialization (Bug #13)
        # Ensure uniform_sampling uses the correct device context
        with torch.device(device):
            w, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        self.w = Mutable(w.to(device))
        
        # 2. Neighborhood Initialization (Bug #19)
        dist_matrix = torch.sqrt(torch.sum((self.w.unsqueeze(1) - self.w.unsqueeze(0))**2, dim=-1))
        _, b = torch.topk(dist_matrix, k=self.T, largest=False)
        self.b = Mutable(b.to(torch.int32))
        
        # 3. State Initialization
        population = torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb
        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), 1e10, device=device))
        self.z = Mutable(torch.full((1, n_objs), 1e10, device=device))
        
        # Archive (N_ep = 5000 as per MATLAB)
        self.archive_pop = Mutable(torch.zeros((5000, self.dim), device=device))
        self.archive_fit = Mutable(torch.full((5000, n_objs), 1e10, device=device))
        self.archive_count = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        
        self.fe = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.fe = self.fe + self.pop_size
        self.z = torch.min(self.fit, dim=0, keepdim=True)[0]
        
        # Initial Archive
        num_to_add = min(self.pop_size, 5000)
        self.archive_pop[:num_to_add] = self.pop[:num_to_add]
        self.archive_fit[:num_to_add] = self.fit[:num_to_add]
        self.archive_count = torch.tensor(num_to_add, dtype=torch.int32, device=self.lb.device)

    def step(self) -> None:
        device = self.lb.device
        
        # 1. Mating: One offspring per subproblem from neighborhood
        idx_neighbors = randint(0, self.T, (self.pop_size, 2), device=device)
        # Use torch.gather to avoid loops for parent indexing
        row_idx = torch.arange(self.pop_size, device=device)
        p1_idx = self.b[row_idx, idx_neighbors[:, 0]].long()
        p2_idx = self.b[row_idx, idx_neighbors[:, 1]].long()
        
        parents1 = self.pop[p1_idx]
        parents2 = self.pop[p2_idx]
        offspring = simulated_binary(torch.cat([parents1, parents2], dim=0), pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring[:self.pop_size], self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        off_fit = self.evaluate(offspring)
        self.fe = self.fe + self.pop_size
        
        # Update Ideal Point
        self.z = torch.min(torch.min(off_fit, dim=0, keepdim=True)[0], self.z)
        
        # 2. Modified Tchebycheff Update (Vectorized)
        # neighbor_fit shape: (N, T, M)
        neighbor_fit = self.fit[self.b.long()]
        neighbor_w = self.w[self.b.long()]
        
        # g_old: (N, T)
        diff_old = torch.abs(neighbor_fit - self.z.unsqueeze(1))
        g_old = torch.max(diff_old / (neighbor_w + 1e-6), dim=-1)[0]
        
        # g_new: (N, T)
        diff_new = torch.abs(off_fit.unsqueeze(1) - self.z.unsqueeze(1))
        g_new = torch.max(diff_new / (neighbor_w + 1e-6), dim=-1)[0]
        
        # Update mask (N, T)
        mask = g_new <= g_old
        
        # Vectorized update across population
        # For each subproblem i, if any neighbor j in B(i) is improved by offspring(i)
        # This is slightly different from standard MOEA/D but matches the "one offspring per subproblem" logic
        for j in range(self.T):
            update_idx = self.b[:, j].long()
            m = mask[:, j]
            self.pop[update_idx] = torch.where(m.unsqueeze(1), offspring, self.pop[update_idx])
            self.fit[update_idx] = torch.where(m.unsqueeze(1), off_fit, self.fit[update_idx])

        # 3. Archive Maintenance (Only if FE <= BP)
        if self.fe <= self.BP:
            combined_fit = torch.cat([self.archive_fit[:self.archive_count], off_fit], dim=0)
            combined_pop = torch.cat([self.archive_pop[:self.archive_count], offspring], dim=0)
            
            # Bug #3: Unique rows
            u_pop, u_idx = unique_rows_sorted(combined_pop)
            u_fit = combined_fit[u_idx]
            
            # Bug #24: Non-dominated check
            is_nd = _get_non_dominated(u_fit)
            nd_pop = u_pop[is_nd]
            nd_fit = u_fit[is_nd]
            
            # Truncate if too large
            num_nd = nd_pop.shape[0]
            if num_nd > 5000:
                # MATLAB: EP(1:length(EP)-5000) = []; (Keep last 5000)
                self.archive_pop[:] = nd_pop[-5000:]
                self.archive_fit[:] = nd_fit[-5000:]
                self.archive_count = torch.tensor(5000, dtype=torch.int32, device=device)
            else:
                self.archive_pop[:num_nd] = nd_pop
                self.archive_fit[:num_nd] = nd_fit
                self.archive_count = torch.tensor(num_nd, dtype=torch.int32, device=device)

            # 4. Weight Adaptation (VOV Logic) - Every G generations
            # Use integer math for condition
            if (self.fe // self.pop_size) % self.G == 0:
                self._adapt_weights()

    def _adapt_weights(self):
        device = self.lb.device
        # 1. Generate W_large (Bug #13)
        with torch.device(device):
            w_large, _ = uniform_sampling(20000, self.n_objs)
        w_large = w_large.to(device)
        
        # 2. Normalize Archive
        arc_fit = self.archive_fit[:self.archive_count]
        f_min = torch.min(arc_fit, dim=0, keepdim=True)[0]
        f_max = torch.max(arc_fit, dim=0, keepdim=True)[0]
        f_norm = (arc_fit - f_min) / (f_max - f_min + 1e-6)
        
        # 3. VOV Selection via Cosine Distance
        # MATLAB: d2 = normObj.*sqrt(1-CosineVOV.^2);
        norm_obj = torch.norm(f_norm, dim=1, keepdim=True) # (N_arc, 1)
        norm_w = torch.norm(w_large, dim=1, keepdim=True) # (N_w, 1)
        
        # Cosine similarity: (N_w, N_arc)
        cosine_sim = (w_large @ f_norm.t()) / (norm_w @ norm_obj.t() + 1e-6)
        # d2 distance: (N_w, N_arc)
        d2 = norm_obj.t() * torch.sqrt(clamp(1.0 - cosine_sim**2, 0.0, 1.0))
        
        min_d2, min_idx = torch.min(d2, dim=1) # (N_w,)
        vov_mask = min_d2 < self.theta
        
        # Calculate VOV: W(i,:).*r where r = d1/norm(W(i,:))
        # d1 = normObj(I)*CosineVOV(I)
        best_cosine = torch.gather(cosine_sim, 1, min_idx.unsqueeze(1)).squeeze(1)
        best_norm_obj = torch.gather(norm_obj.squeeze(1).expand(w_large.size(0), -1), 1, min_idx.unsqueeze(1)).squeeze(1)
        d1 = best_norm_obj * best_cosine
        r = d1 / (norm_w.squeeze(1) + 1e-6)
        vovs = w_large * r.unsqueeze(1)
        vovs = vovs[vov_mask]
        
        # 4. Greedy Subset Selection (L0.5)
        candidate_w = torch.cat([f_norm, vovs], dim=0)
        u_cand, _ = unique_rows_sorted(candidate_w)
        candidate_w = u_cand
        
        num_cand = candidate_w.shape[0]
        if num_cand > self.pop_size:
            # Select extreme points first (Cosine distance to axes)
            eye_m = torch.eye(self.n_objs, device=device)
            cos_to_axes = (candidate_w @ eye_m.t()) / (torch.norm(candidate_w, dim=1, keepdim=True) + 1e-6)
            selected_indices = torch.argmax(cos_to_axes, dim=0)
            
            is_selected = torch.zeros(num_cand, dtype=torch.bool, device=device)
            is_selected[selected_indices] = True
            
            # Precompute distance matrix to save time in loop
            # LpNormD = pdist2(obj,obj,'minkowski',0.5);
            # We only need dists between candidates and selected
            
            for _ in range(self.pop_size - int(torch.sum(is_selected))):
                S = candidate_w[is_selected]
                P = candidate_w[~is_selected]
                
                dists = _minkowski_05_dist(P, S)
                min_dists = torch.min(dists, dim=1)[0]
                best_idx_in_P = torch.argmax(min_dists)
                
                p_indices = torch.where(~is_selected)[0]
                is_selected[p_indices[best_idx_in_P]] = True
            
            new_w = candidate_w[is_selected][:self.pop_size]
        else:
            new_w = torch.zeros((self.pop_size, self.n_objs), device=device)
            new_w[:num_cand] = candidate_w
            with torch.device(device):
                fill_w, _ = uniform_sampling(self.pop_size - num_cand, self.n_objs)
            new_w[num_cand:] = fill_w.to(device)

        # Normalize weights as per updateWeight.m: W = W./vecnorm(W,1,2);
        new_w = new_w / (torch.sum(torch.abs(new_w), dim=1, keepdim=True) + 1e-6)
        self.w = new_w
        
        # 5. Re-assignment: Population(i) = EP(I) where I minimizes Tchebycheff
        diff = torch.abs(arc_fit.unsqueeze(0) - self.z.unsqueeze(1)) # (N_pop, N_arc, M)
        tche_dists = torch.max(diff / (self.w.unsqueeze(1) + 1e-6), dim=-1)[0] # (N_pop, N_arc)
        best_arc_idx = torch.argmin(tche_dists, dim=1)
        
        self.pop = self.archive_pop[best_arc_idx]
        self.fit = self.archive_fit[best_arc_idx]
        
        # Update Neighborhood for new weights
        dist_matrix = torch.sqrt(torch.sum((self.w.unsqueeze(1) - self.w.unsqueeze(0))**2, dim=-1))
        _, b = torch.topk(dist_matrix, k=self.T, largest=False)
        self.b = b.to(torch.int32)

def _minkowski_05_dist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # (N_A, 1, M) - (1, N_B, M) -> (N_A, N_B, M)
    diff = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))
    return torch.sum((diff + 1e-6)**0.5, dim=-1)**2

def _get_non_dominated(fit: torch.Tensor) -> torch.Tensor:
    if fit.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=fit.device)
    dom = (fit.unsqueeze(1) <= fit.unsqueeze(0)).all(-1) & (fit.unsqueeze(1) < fit.unsqueeze(0)).any(-1)
    return ~dom.any(dim=0)

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_VOV(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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