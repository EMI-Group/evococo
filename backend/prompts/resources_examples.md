# Structural references, not replacement algorithms. Follow the supplied source
# for selection, state updates, variation parameters and helper-return retention.
# Import only symbols actually used; use the SDK reference for full signatures.

# Example 1: NSGA-II (dominance-based)
import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp
from evomo.operators.selection import nd_environmental_selection


class NSGA2(Algorithm):
    def __init__(self, pop_size, n_objs, lb, ub, **kwargs):
        super().__init__()
        self.pop_size, self.lb, self.ub = pop_size, lb, ub
        self.pop = Mutable(torch.rand(pop_size, lb.numel(), device=lb.device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=lb.device))
        self.rank = Mutable(torch.full((pop_size,), torch.inf, device=lb.device))
        self.dis = Mutable(torch.full((pop_size,), -torch.inf, device=lb.device))

    def init_step(self):
        self.fit = self.evaluate(self.pop)
        # This caller retains only ranks/distances, not selected pop/fit.
        _, _, self.rank, self.dis, _ = nd_environmental_selection(self.pop, self.fit, self.pop_size)

    def step(self):
        # Rank first, crowding second; SDK's last key is primary.
        indices = tournament_selection_multifit(self.pop_size, [-self.dis, self.rank])
        offspring = simulated_binary(self.pop[indices])
        offspring = clamp(polynomial_mutation(offspring, self.lb, self.ub), self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        self.pop, self.fit, self.rank, self.dis, _ = nd_environmental_selection(
            torch.cat([self.pop, offspring]), torch.cat([self.fit, off_fit]), self.pop_size
        )


# Example 2: RVEA (reference-vector-based)
from evox.core import Parameter
from evox.operators.sampling import uniform_sampling
from evox.utils import nanmax, nanmin, randint
from evomo.operators.selection import ref_vec_guided


class RVEA(Algorithm):
    def __init__(self, pop_size, n_objs, lb, ub, alpha=2.0, fr=0.1, max_gen=100, **kwargs):
        super().__init__()
        device = lb.device
        self.lb, self.ub = lb, ub
        self.alpha, self.fr, self.max_gen = Parameter(alpha), Parameter(fr), Parameter(max_gen)
        self.rv_adapt_every = Mutable(torch.round(1 / self.fr).clamp_min(1))
        vectors, _ = uniform_sampling(pop_size, n_objs)
        self.init_v = vectors.to(device=device)
        self.pop_size = self.init_v.shape[0]
        self.reference_vector = Mutable(self.init_v.clone())
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.pop = Mutable(torch.rand(self.pop_size, lb.numel(), device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

    def init_step(self):
        self.fit = self.evaluate(self.pop)

    def _mating_pool(self):
        valid = ~torch.isnan(self.pop).all(dim=1)
        count = valid.sum(dtype=torch.int32)
        choices = randint(0, count, (self.pop_size,), device=self.pop.device)
        sentinel = torch.iinfo(torch.int32).max
        indices = torch.where(valid, torch.arange(self.pop_size, device=self.pop.device), sentinel)
        indices = torch.argsort(indices, stable=True)
        return self.pop[indices[choices]]

    def step(self):
        self.gen = self.gen + 1
        offspring = simulated_binary(self._mating_pool())
        offspring = clamp(polynomial_mutation(offspring, self.lb, self.ub), self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        self.pop, self.fit = ref_vec_guided(
            torch.cat([self.pop, offspring]), torch.cat([self.fit, off_fit]),
            self.reference_vector, (self.gen / self.max_gen) ** self.alpha
        )
        if (self.gen % self.rv_adapt_every) == 0:
            ranges = nanmax(self.fit, dim=0)[0] - nanmin(self.fit, dim=0)[0]
            self.reference_vector = self.init_v * ranges
