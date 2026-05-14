import torch
from typing import Tuple
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp, lexsort
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit, crowding_distance

class rNSGA2(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, 
                 points: torch.Tensor, weights: torch.Tensor = None, delta: float = 0.1, max_fe: int = 10000):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.max_fe = Parameter(torch.tensor(max_fe, dtype=torch.float32, device=device))
        D = lb.numel()

        # Preferred Points and Weights
        self.points = Mutable(points.to(device))
        if weights is None:
            weights = torch.ones_like(points, device=device)
        self.weights = Mutable(weights.to(device))
        self.delta = Parameter(torch.tensor(delta, device=device))

        # Initialize State
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.fe = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        
        # Ranking and Distance
        self.front_no = Mutable(torch.full((pop_size,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.zeros(pop_size, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.fe = self.fe + self.pop_size
        
        delta_curr = 1.0 - (1.0 - self.delta) * (self.fe.float() / self.max_fe)
        self.front_no, self.crowd_dis = _r_nondominated_sort(
            self.fit, self.points, self.weights, delta_curr, self.pop_size
        )

    def step(self) -> None:
        # 1. Selection & Mating
        mating_pool = tournament_selection_multifit(
            self.pop_size, 
            fitnesses=[self.front_no.float(), -self.crowd_dis], 
            tournament_size=2
        )
        
        # 2. Variation
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        # 3. Evaluation
        off_fit = self.evaluate(offspring)
        self.fe = self.fe + self.pop_size
        
        # 4. Environmental Selection
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        
        delta_curr = 1.0 - (1.0 - self.delta) * (self.fe.float() / self.max_fe)
        new_front_no, new_crowd_dis = _r_nondominated_sort(
            merged_fit, self.points, self.weights, delta_curr, self.pop_size
        )
        
        # Select top N based on front_no and crowd_dis
        rank_indices = lexsort(torch.stack([-new_crowd_dis, new_front_no.float()]))
        survivor_idx = rank_indices[:self.pop_size]
        
        self.pop = merged_pop[survivor_idx]
        self.fit = merged_fit[survivor_idx]
        self.front_no = new_front_no[survivor_idx]
        self.crowd_dis = new_crowd_dis[survivor_idx]

def _r_nondominated_sort(fit, points, weights, delta_curr, N_target) -> Tuple[torch.Tensor, torch.Tensor]:
    N_total = fit.shape[0]
    K = points.shape[0]
    device = fit.device
    
    # Normalization
    f_min = fit.min(dim=0, keepdim=True).values
    f_max = fit.max(dim=0, keepdim=True).values
    
    # Pareto Dominance Matrix
    P = (fit.unsqueeze(1) <= fit.unsqueeze(0)).all(-1) & (fit.unsqueeze(1) < fit.unsqueeze(0)).any(-1)
    
    # Multi-point r-dominance
    all_front_nos = []
    
    for k in range(K):
        g = points[k:k+1]
        w = weights[k:k+1]
        
        # Weighted Distance
        dist = torch.sqrt(torch.sum(w * ((fit - g) / (f_max - f_min + 1e-6))**2, dim=1))
        dist_diff = (dist.unsqueeze(1) - dist.unsqueeze(0)) / (dist.max() - dist.min() + 1e-6)
        
        # r-Dominance Mask (R)
        R = P | ((~P.t()) & (dist_diff < -delta_curr))
        
        # Peeling process
        in_degree = R.sum(dim=0).to(torch.int32)
        front_no_k = torch.full((N_total,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device)
        
        curr_rank = 1
        mask_remaining = torch.ones(N_total, dtype=torch.bool, device=device)
        count = 0
        
        for _ in range(N_total):
            curr_front_mask = (in_degree == 0) & mask_remaining
            num_in_front = curr_front_mask.sum()
            
            is_deadlock = (num_in_front == 0) & mask_remaining.any()
            curr_front_mask = torch.where(is_deadlock, mask_remaining, curr_front_mask)
            num_in_front = curr_front_mask.sum()
            
            front_no_k = torch.where(curr_front_mask, torch.full_like(front_no_k, curr_rank), front_no_k)
            
            reduced_degrees = R[curr_front_mask, :].sum(dim=0)
            in_degree = in_degree - reduced_degrees
            
            mask_remaining = mask_remaining & (~curr_front_mask)
            count = count + num_in_front
            curr_rank = curr_rank + 1
            
        all_front_nos.append(front_no_k)

    # Final Rank: element-wise minimum across reference points
    final_front_no = torch.stack(all_front_nos, dim=0).min(dim=0).values
    
    # Crowding Distance
    final_crowd_dis = torch.zeros(N_total, device=device)
    for f in range(1, N_total + 1):
        mask = (final_front_no == f)
        cd = crowding_distance(fit, mask)
        final_crowd_dis = torch.where(mask, cd, final_crowd_dis)
        
    return final_front_no, final_crowd_dis

if __name__ == "__main__":
    import time
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = rNSGA2(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12), points=torch.tensor([[0.2, 0.2, 0.2]]))
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