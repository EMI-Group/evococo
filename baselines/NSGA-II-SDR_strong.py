import torch
from evox import Algorithm, utils

class NSGAIISDR(Algorithm):
    def __init__(self, problem, pop_size):
        self.problem = problem
        self.n = pop_size
        self.dim = problem.lower_bound.shape[0]
        self.m = None  # Number of objectives
        self.pop = None
        self.fit = None
        self.front_no = None
        self.crowd_dis = None
        self.zmin = None
        self.zmax = None

    def setup(self, key):
        # Initialization
        lb = self.problem.lower_bound
        ub = self.problem.upper_bound
        self.pop = lb + torch.rand((self.n, self.dim), device=lb.device) * (ub - lb)
        self.fit = self.problem.evaluate(self.pop)
        self.m = self.fit.shape[1]
        
        self.zmin = torch.min(self.fit, dim=0)[0]
        self.zmax = torch.max(self.fit, dim=0)[0]
        
        self.pop, self.fit, self.front_no, self.crowd_dis = self._environmental_selection(
            self.pop, self.fit, self.n, self.zmin, self.zmax
        )

    def init_step(self):
        pass

    def step(self):
        # Tournament Selection
        mating_pool = self._tournament_selection(2, self.n, self.front_no, -self.crowd_dis)
        
        # Genetic Operators (Simulated Binary Crossover and Polynomial Mutation)
        offspring = self._operator_ga(self.pop[mating_pool])
        off_fit = self.problem.evaluate(offspring)
        
        # Update Ideal and Nadir points
        self.zmin = torch.min(torch.cat([self.zmin.unsqueeze(0), off_fit], dim=0), dim=0)[0]
        mask_f1 = self.front_no == 1
        self.zmax = torch.max(self.fit[mask_f1], dim=0)[0]
        
        # Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)
        
        self.pop, self.fit, self.front_no, self.crowd_dis = self._environmental_selection(
            combined_pop, combined_fit, self.n, self.zmin, self.zmax
        )

    def _environmental_selection(self, pop, fit, n, zmin, zmax):
        # Normalization
        pop_obj = fit - zmin
        range_val = zmax - zmin
        range_val[range_val == 0] = 1e-6
        
        if 0.05 * torch.max(range_val) < torch.min(range_val):
            pop_obj = pop_obj / range_val
            
        # Unique solutions
        rounded_obj = torch.round(pop_obj * 1e6) / 1e6
        _, unique_indices = torch.unique(rounded_obj, dim=0, return_inverse=False, sorted=False)
        # Note: PyTorch unique doesn't return first occurrence indices like MATLAB. 
        # Using a manual approach for unique rows:
        unique_indices = self._get_unique_indices(rounded_obj)
        
        pop_obj = pop_obj[unique_indices]
        pop = pop[unique_indices]
        fit = fit[unique_indices]
        
        current_n = min(n, pop.shape[0])
        
        # SDR Non-dominated sorting
        front_no, max_fno = self._nd_sort_sdr(pop_obj, current_n)
        
        # Crowding Distance
        crowd_dis = self._crowding_distance(pop_obj, front_no)
        
        # Selection
        next_mask = front_no < max_fno
        last_front_indices = torch.where(front_no == max_fno)[0]
        
        num_needed = current_n - torch.sum(next_mask).item()
        if num_needed > 0:
            last_front_crowd = crowd_dis[last_front_indices]
            _, rank = torch.sort(last_front_crowd, descending=True)
            selected_in_last = last_front_indices[rank[:num_needed]]
            next_mask[selected_in_last] = True
            
        return pop[next_mask], fit[next_mask], front_no[next_mask], crowd_dis[next_mask]

    def _nd_sort_sdr(self, pop_obj, n_sort):
        n = pop_obj.shape[0]
        norm_p = torch.sum(pop_obj, dim=1)
        
        # Cosine similarity and Angle
        norm_obj = torch.nn.functional.normalize(pop_obj, p=2, dim=1)
        cosine = torch.mm(norm_obj, norm_obj.t())
        cosine = torch.clamp(cosine, -1.0, 1.0)
        cosine.fill_diagonal_(0)
        angle = torch.acos(cosine)
        
        min_angle, _ = torch.min(angle + torch.eye(n, device=angle.device) * 1e6, dim=1)
        temp = torch.sort(torch.unique(min_angle))[0]
        idx = min(int(torch.ceil(torch.tensor(n / 2.0))) - 1, temp.shape[0] - 1)
        min_a = temp[idx]
        
        theta = torch.pow(angle / min_a, 1.0)
        theta = torch.clamp(theta, min=1.0)
        
        # SDR Dominance
        # i dominates j if NormP(i)*Theta(i,j) < NormP(j)
        dominate = (norm_p.unsqueeze(1) * theta < norm_p.unsqueeze(0))
        
        front_no = torch.full((n,), float('inf'), device=pop_obj.device)
        max_fno = 0
        
        while torch.sum(front_no != float('inf')) < min(n_sort, n):
            max_fno += 1
            # current are those not dominated by any remaining individuals
            mask_inf = (front_no == float('inf'))
            # count how many remaining individuals dominate each individual
            dom_count = torch.sum(dominate & mask_inf.unsqueeze(0), dim=0)
            current = (dom_count == 0) & mask_inf
            
            if not torch.any(current):
                break
                
            front_no[current] = max_fno
            # Individuals in 'current' no longer dominate anyone for next fronts
            dominate[current, :] = False
            
        return front_no, max_fno

    def _crowding_distance(self, pop_obj, front_no):
        n, m = pop_obj.shape
        crowd_dis = torch.zeros(n, device=pop_obj.device)
        fronts = torch.unique(front_no)
        fronts = fronts[torch.isfinite(fronts)]
        
        for f in fronts:
            idx = torch.where(front_no == f)[0]
            f_obj = pop_obj[idx]
            f_n = idx.shape[0]
            if f_n <= 2:
                crowd_dis[idx] = float('inf')
                continue
            
            f_dis = torch.zeros(f_n, device=pop_obj.device)
            f_max = torch.max(f_obj, dim=0)[0]
            f_min = torch.min(f_obj, dim=0)[0]
            
            for i in range(m):
                _, rank = torch.sort(f_obj[:, i])
                f_dis[rank[0]] = float('inf')
                f_dis[rank[-1]] = float('inf')
                
                norm = f_max[i] - f_min[i]
                if norm > 0:
                    f_dis[rank[1:-1]] += (f_obj[rank[2:], i] - f_obj[rank[:-2], i]) / norm
            
            crowd_dis[idx] = f_dis
        return crowd_dis

    def _tournament_selection(self, k, n, front_no, crowd_dis):
        # front_no: smaller is better; crowd_dis: larger is better (input is -crowd_dis)
        # So we want smaller front_no and smaller -crowd_dis
        indices = torch.randint(0, front_no.shape[0], (n, k), device=front_no.device)
        selected_fronts = front_no[indices]
        selected_crowd = crowd_dis[indices]
        
        # Lexsort-like comparison
        winner_mask = (selected_fronts[:, 0] < selected_fronts[:, 1]) | \
                      ((selected_fronts[:, 0] == selected_fronts[:, 1]) & (selected_crowd[:, 0] < selected_crowd[:, 1]))
        
        winners = torch.where(winner_mask, indices[:, 0], indices[:, 1])
        return winners

    def _operator_ga(self, parent_pop, pc=1.0, pm=1.0, eta_c=20, eta_m=20):
        n, d = parent_pop.shape
        lb = self.problem.lower_bound
        ub = self.problem.upper_bound
        
        # SBX
        offspring = parent_pop.clone()
        for i in range(0, n - 1, 2):
            if torch.rand(1) < pc:
                u = torch.rand(d, device=parent_pop.device)
                beta = torch.where(u <= 0.5, (2*u)**(1/(eta_c+1)), (1/(2*(1-u)))**(1/(eta_c+1)))
                offspring[i] = 0.5 * ((1 + beta) * parent_pop[i] + (1 - beta) * parent_pop[i+1])
                offspring[i+1] = 0.5 * ((1 - beta) * parent_pop[i] + (1 + beta) * parent_pop[i+1])
        
        # Polynomial Mutation
        pm_val = pm / d
        mut_mask = torch.rand((n, d), device=parent_pop.device) < pm_val
        u = torch.rand((n, d), device=parent_pop.device)
        delta = torch.where(u <= 0.5, (2*u)**(1/(eta_m+1)) - 1, 1 - (2*(1-u))**(1/(eta_m+1)))
        
        offspring = torch.where(mut_mask, offspring + delta * (ub - lb), offspring)
        return torch.clamp(offspring, lb, ub)

    def _get_unique_indices(self, x):
        # Helper to get indices of unique rows (first occurrence)
        # Since torch.unique doesn't guarantee first occurrence index
        sorted_x, indices = torch.sort(torch.sum(x * torch.arange(1, x.shape[1]+1, device=x.device), dim=1))
        mask = torch.cat([torch.tensor([True], device=x.device), torch.any(x[1:] != x[:-1], dim=1)])
        # This is a simplified unique; for exact isomorphism we use a stable approach:
        # In high-performance tensor code, we often skip unique if it's just for diversity
        # but here we follow the MATLAB logic.
        unique_vals, inverse_indices = torch.unique(x, dim=0, return_inverse=True)
        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
        inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
        return inverse_indices.new_empty(unique_vals.size(0)).scatter_(0, inverse_indices, perm)