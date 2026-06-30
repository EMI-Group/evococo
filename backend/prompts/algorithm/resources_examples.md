# [Example 1: NSGA-II (Dominance-based)]
# Key features: Tournament selection, SBX/Poly mutation, Non-dominated sorting
from typing import Callable, Optional
import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp
from evomo.operators.selection import nd_environmental_selection

class NSGA2(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        
        # Dimensions and Bounds
        self.dim = lb.shape[0]
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        # Operators
        self.selection = selection_op if selection_op is not None else tournament_selection_multifit
        self.mutation = mutation_op if mutation_op is not None else polynomial_mutation
        self.crossover = crossover_op if crossover_op is not None else simulated_binary

        # Initialize State (Mutable)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.rank = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.dis = Mutable(torch.empty(self.pop_size, device=device).fill_(-torch.inf))

    def init_step(self):
        self.fit = self.evaluate(self.pop)
        _, _, self.rank, self.dis = nd_environmental_selection(self.pop, self.fit, self.pop_size)

    def step(self):
        # 1. Selection & Mating
        mating_pool = self.selection(self.pop_size, [-self.dis, self.rank])
        crossovered = self.crossover(self.pop[mating_pool])
        
        # 2. Mutation & Repair
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        # 3. Evaluation
        off_fit = self.evaluate(offspring)
        
        # 4. Merge & Environmental Selection
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.rank, self.dis = nd_environmental_selection(merge_pop, merge_fit, self.pop_size)


# [Example 2: RVEA (Decomposition/Reference-based)]
# Key features: Parameter, Reference Vector Adaptation, Uniform Sampling
from typing import Callable, Optional
import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, nanmax, nanmin, randint
from evomo.operators.selection import ref_vec_guided

class RVEA(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        alpha: float = 2.0,
        fr: float = 0.1,
        max_gen: int = 100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        device = torch.get_default_device() if device is None else device
        
        self.dim = lb.size(0)
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        # Algorithm Parameters
        self.alpha = Parameter(alpha)
        self.fr = Parameter(fr)
        self.max_gen = Parameter(max_gen)
        self.rv_adapt_every = Mutable(torch.max(torch.round(1 / self.fr), torch.tensor(1.0)))

        # Operators
        self.selection = selection_op if selection_op is not None else ref_vec_guided
        self.mutation = mutation_op if mutation_op is not None else polynomial_mutation
        self.crossover = crossover_op if crossover_op is not None else simulated_binary
        
        # Reference Vectors
        sampling, _ = uniform_sampling(self.pop_size, self.n_objs)
        v = sampling.to(device=device)
        v0 = v.clone()
        self.pop_size = v.size(0) # Adjust pop_size based on sampling

        # Initialize State
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self):
        self.rv_adapt_every = torch.max(torch.round(1 / self.fr), torch.tensor(1.0))
        self.fit = self.evaluate(self.pop)

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v.clone() * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector.clone()

    def _mating_pool(self):
        # Filter invalid solutions
        valid_mask = ~torch.isnan(self.pop).all(dim=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)
        
        # Random selection from valid solutions
        mating_pool = randint(0, num_valid, (self.pop_size,), device=self.pop.device)
        
        sentinel = torch.iinfo(torch.int32).max
        sorted_indices = torch.where(valid_mask, torch.arange(self.pop_size, device=self.device), sentinel)
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        
        pop = self.pop[sorted_indices[mating_pool]]
        return pop

    def _update_pop_and_rv(self, survivor, survivor_fit):
        self.pop = survivor
        self.fit = survivor_fit

        # Reference Vector Adaptation Logic
        # NOTE: Using standard if/else to comply with system constraints (Bug #5: No torch.cond)
        if (self.gen % self.rv_adapt_every) == 0:
            self.reference_vector = self._rv_adaptation(survivor_fit)
        else:
            self.reference_vector = self._no_rv_adaptation(survivor_fit)

    def step(self):
        self.gen = self.gen + 1
        
        # 1. Mating
        pop = self._mating_pool()
        crossovered = self.crossover(pop)
        
        # 2. Mutation
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        # 3. Evaluation
        off_fit = self.evaluate(offspring)
        
        # 4. Selection
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        survivor, survivor_fit = self.selection(
            merge_pop,
            merge_fit,
            self.reference_vector,
            (self.gen / self.max_gen) ** self.alpha,
        )

        self._update_pop_and_rv(survivor, survivor_fit)