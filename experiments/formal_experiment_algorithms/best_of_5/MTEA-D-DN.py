import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling

class MTEADDN(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, n_tasks: int = 2, dt: int = 20, beta: float = 0.9, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.n_tasks = n_tasks
        self.dt = dt
        self.beta = Parameter(beta)
        
        dim = lb.numel()
        # Weight Generation
        weights_single, actual_sub_pop_size = uniform_sampling(pop_size // n_tasks, n_objs)
        weights_single = weights_single.to(device=device)
        # Safe normalization
        weights_single = weights_single / (torch.norm(weights_single, dim=1, keepdim=True) + 1e-6)
        
        self.sub_pop_size = actual_sub_pop_size
        self.pop_size = actual_sub_pop_size * n_tasks
        
        # Initialize Population with Task ID in last column (1 to T)
        raw_pop = torch.rand(self.pop_size, dim, device=device) * (ub - lb) + lb
        task_ids = torch.repeat_interleave(torch.arange(1, n_tasks + 1, device=device), self.sub_pop_size).unsqueeze(1)
        self.pop = Mutable(torch.cat([raw_pop, task_ids.float()], dim=1))
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        # self.W: (T, sub_N, M)
        self.W = Mutable(weights_single.unsqueeze(0).repeat(n_tasks, 1, 1))
        
        # Calculate B: (T, sub_N, DT)
        dist = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(dist, k=dt, largest=False).indices.to(torch.int32))
        
        # Ideal points: (T, M)
        self.z = Mutable(torch.full((n_tasks, n_objs), torch.inf, device=device))
        
        # Inter-task Setup
        task_range = torch.arange(n_tasks, device=device)
        offsets = torch.randint(1, n_tasks, (n_tasks, self.sub_pop_size), device=device)
        self.B2k = Mutable((task_range.unsqueeze(1) + offsets) % n_tasks)
        self.B2 = Mutable(torch.randint(0, self.sub_pop_size, (n_tasks, self.sub_pop_size, dt), device=device))
        
        self.sentinel = torch.iinfo(torch.int32).max

    def _calc_tchebycheff(self, objs, weights, z):
        # objs: (N, M) or (N, DT, M)
        # weights: (N, DT, M)
        # z: (N, M)
        if objs.ndim == 2:
            diff = torch.abs(objs.unsqueeze(1) - z.unsqueeze(1)) # (N, 1, M)
        else:
            diff = torch.abs(objs - z.unsqueeze(1)) # (N, DT, M)
        return torch.max(weights * diff, dim=-1).values # (N, DT)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop[:, :-1])
        for t in range(self.n_tasks):
            task_mask = (self.pop[:, -1] == (t + 1))
            if task_mask.any():
                self.z[t] = torch.min(self.fit[task_mask], dim=0).values

    def step(self) -> None:
        device = self.lb.device
        N = self.pop_size
        T = self.n_tasks
        sub_N = self.sub_pop_size
        DT = self.dt
        
        flat_indices = torch.arange(N, device=device).view(T, sub_N)
        mask_dual = torch.rand(N, device=device) < self.beta
        mask_dual_reshaped = mask_dual.view(T, sub_N)
        
        # Parent Selection
        local_rand = torch.randint(0, DT, (T, sub_N, 2), device=device)
        p1_local = torch.gather(self.B, 2, local_rand[:, :, 0:1]).squeeze(2)
        p2_local = torch.gather(self.B, 2, local_rand[:, :, 1:2]).squeeze(2)
        
        dual_rand = torch.randint(0, DT, (T, sub_N), device=device)
        p2_dual = torch.gather(self.B2, 2, dual_rand.unsqueeze(2)).squeeze(2)
        
        idx_p1 = p1_local
        idx_p2 = torch.where(mask_dual_reshaped, p2_dual, p2_local)
        
        global_p1 = flat_indices.gather(1, idx_p1.long()).view(-1)
        target_tasks = torch.where(mask_dual_reshaped, self.B2k, torch.arange(T, device=device).unsqueeze(1))
        global_p2 = (target_tasks * sub_N + idx_p2.long()).view(-1)
        
        # DE Operator (Manual Vectorized Implementation to avoid module call error)
        # offspring = x1 + F * (x2 - x3). Here we use pop[i], pop[p1], pop[p2]
        F = 0.5
        CR = 0.9
        x_base = self.pop[:, :-1]
        x_p1 = self.pop[global_p1][:, :-1]
        x_p2 = self.pop[global_p2][:, :-1]
        
        # DE/rand/1 logic
        offspring_x = x_base + F * (x_p1 - x_p2)
        
        # Binomial Crossover
        rand_cross = torch.rand(offspring_x.shape, device=device)
        cross_mask = rand_cross < CR
        # Ensure at least one dimension is swapped
        j_rand = torch.randint(0, offspring_x.shape[1], (N,), device=device)
        cross_mask[torch.arange(N), j_rand] = True
        offspring_x = torch.where(cross_mask, offspring_x, x_base)
        
        offspring_x = polynomial_mutation(offspring_x, self.lb, self.ub)
        offspring_x = clamp(offspring_x, self.lb, self.ub)
        
        parent_tasks = self.pop[:, -1]
        transfer_eval_mask = (torch.rand(N, device=device) < 0.5) & mask_dual
        off_task_ids = torch.where(transfer_eval_mask, (self.B2k.view(-1) + 1).float(), parent_tasks)
        
        offspring_full = torch.cat([offspring_x, off_task_ids.unsqueeze(1)], dim=1)
        off_fit = self.evaluate(offspring_full[:, :-1])
        
        # Update Ideal Points
        for t in range(T):
            task_mask = (off_task_ids == (t + 1))
            if task_mask.any():
                self.z[t] = torch.min(self.z[t], torch.min(off_fit[task_mask], dim=0).values)
        
        eval_task_idx = (off_task_ids - 1).long()
        z_for_off = self.z[eval_task_idx]
        
        is_local_update = (off_task_ids == parent_tasks)
        # neighbors_rel: (N, DT)
        neighbors_rel = torch.where(is_local_update.unsqueeze(1), self.B.view(N, DT), self.B2.view(N, DT))
        neighbors_global = eval_task_idx.unsqueeze(1) * sub_N + neighbors_rel.long()
        
        # Gather weights for the neighbors
        W_task = self.W[eval_task_idx] # (N, sub_N, M)
        weights_neigh = torch.gather(W_task, 1, neighbors_rel.long().unsqueeze(2).expand(-1, -1, self.n_objs))
        
        fit_neighbors = self.fit[neighbors_global.view(-1)].view(N, DT, self.n_objs)
        g_old = self._calc_tchebycheff(fit_neighbors, weights_neigh, z_for_off)
        g_new = self._calc_tchebycheff(off_fit, weights_neigh, z_for_off)
        
        replace_mask = g_new < g_old
        
        # Update population and fitness using vectorized scatter
        # To handle multiple neighbors being updated by the same offspring, we iterate DT
        for d in range(DT):
            mask = replace_mask[:, d]
            if mask.any():
                idx = neighbors_global[mask, d]
                self.pop[idx] = offspring_full[mask]
                self.fit[idx] = off_fit[mask]
        
        # Shrinking / Reset Logic for B2
        # If an inter-task transfer failed to improve any neighbor in B2
        reset_mask = (~is_local_update) & (~torch.any(replace_mask, dim=1))
        if reset_mask.any():
            n_reset = torch.sum(reset_mask)
            p_tasks_idx = (parent_tasks[reset_mask].long() - 1)
            new_offsets = torch.randint(1, T, (n_reset,), device=device)
            new_B2k = (p_tasks_idx + new_offsets) % T
            self.B2k.view(-1)[reset_mask] = new_B2k
            self.B2.view(N, DT)[reset_mask] = torch.randint(0, sub_N, (n_reset, DT), device=device)

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MTEADDN(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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