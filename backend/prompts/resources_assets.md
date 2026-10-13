# ==========================================================
# ASSET LIBRARY (SDK)
# Preferentially use these APIs. Do NOT reinvent wheels.
# ==========================================================

# --- 1. Tensor Utilities (Primary Backend: EvoX) ---
from evox.utils import (
    clamp,           # Tensor clamping (works on float/int)
    clamp_float,     # Specific float clamping
    clamp_int,       # Specific int clamping
    clip,            # Alias for clamp
    maximum, minimum,
    maximum_float, minimum_float,
    maximum_int, minimum_int,
    lexsort,         # Lexicographical sort (keys: Tensor[k, N]) -> indices
    nanmin, nanmax,  # NaN-ignoring min/max
    randint,         # Random integers
)

# --- 2. Advanced Utilities (Extension: EvoMO) ---
# MUST use this for uniqueness to ensure deterministic behavior.
# [WARNING]:
# 1. Input 'x' MUST be 2D (N, D). If input is 1D, use .unsqueeze(1).
# 2. Returns a tuple (unique_rows, indices) by default.
# Signature: unique_rows_sorted(x, return_index=True).
# 'return_index' is a bool, not a second tensor to deduplicate.
# 'indices' has shape (K,) and indexes original rows; it is NOT an inverse map of shape (N,).
# With return_index=False, the result is only the unique-row tensor.
# Usage Pattern:
#   u_pop, u_idx = unique_rows_sorted(pop)
#   u_fit = fit[u_idx]
from evomo.utils import unique_rows_sorted

# --- 3. Selection Operators (Unified) ---
from evox.operators.selection import (
    tournament_selection,
    tournament_selection_multifit,
    crowding_distance  # Remember: crowding_distance(objs, mask)
)
from evomo.operators.selection import (
    nd_environmental_selection,
    non_dominate_rank,
)
# Usage:
#   # Single-objective tournament (min fitness wins)
#   parent_idx = tournament_selection(n_round=pop_size, fitness=fit, tournament_size=2)
#   parents = pop[parent_idx]
#
#   # Multi-criteria tournament: LAST key is primary; all keys are minimized.
#   # Rank first (minimize), crowding second (maximize):
#   parent_idx = tournament_selection_multifit(n_round=pop_size, fitnesses=[-crowding, rank], tournament_size=2)
#   # General ordering: f3 is primary, f2 secondary, f1 tertiary.
#   parent_idx = tournament_selection_multifit(n_round=pop_size, fitnesses=[f1, f2, f3], tournament_size=2)
#   parents = pop[parent_idx]
#   # Crowding distance calculation for a given front (mask is boolean)
#   # costs: (N, M), mask: (N,) bool
#   cd = crowding_distance(costs, mask)      # (N,), larger = more diverse
#   # The result covers ALL N rows: +inf is a valid front boundary;
#   # rows outside mask are -inf, NOT zero contributions.
#   # If the source stores each front's distances in a shared vector:
#   distances = torch.where(mask, cd, distances)
#   # Equivalent eager indexing: distances[mask] = cd[mask].
#   # Do not assign the full cd to distances[mask], or sum full masked
#   # results across fronts: +inf + -inf produces NaN and corrupts prior fronts.
#   # This update applies only to this SDK operator and per-front storage.
#   # Other density metrics, aggregation rules and source selection remain unchanged.
#
#   # Returns five values; selected_cv is None for unconstrained problems.
#   selected_pop, selected_fit, rank, cd, selected_cv = nd_environmental_selection(
#       pop, fit, topk
#   )
#   # Returned cd is computed on the cutoff front; earlier-front rows are -inf.
#   # With CV (EvoMO >= 0.3.1): nd_environmental_selection(pop, fit, topk, cv),
#   # or non_dominate_rank(fit, cv=cv). Use only if source comparisons/ties agree.
#   # SDK equal-total-CV rows compare objectives. PlatEMO NDSort(F,C,N) instead
#   # makes infeasible equal-CV rows tie via max(F,dim=0)+sum(max(C,0),dim=1).
#   # For that source, rank this local transformed copy without cv=; retain
#   # raw fit/CV state and use source-specified objectives for crowding.
#   # non_dominate_rank is zero-based: rank 0 is the first Pareto front.
#   rank = non_dominate_rank(fit)
#   # Input fit is an (N, M) objective-value tensor; this function recomputes Pareto dominance.
#   # NEVER pass an (N, N) custom dominance matrix. Peel that relation using the source logic.

# Optional CV state/evaluation only for source-supported algorithms:
from evomo.utils import parse_evaluate, register_lazy_buffer
#   register_lazy_buffer(self, "cv", device_like="pop")  # once, after self.pop exists
#   self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
#   # parse_evaluate only unpacks; it does not clamp or aggregate CV.
#   # cv is None for fitness-only results; slice supplied cv with pop/fit indices.
#   # CMOP.evaluate returns nonnegative (N,C) CV: max(G,0), max(abs(H)-eq_eps,0).
#   # This is not signed G/H; custom problem tuples are forwarded unchanged.
from evomo.workflows import UnifiedWorkflow
# UnifiedWorkflow supplies algorithm.evaluate, not algorithm.problem or FE/maxFE.

# Keep scalar arithmetic and indices on device (examples, not a new variation policy):
#   u = torch.rand(pop_size, 1, device=pop.device)  # one scalar per individual
#   shift = -b / torch.tan(torch.pi * u)           # no float/u.item() round trip
#   col = cv.index_select(1, con.reshape(1)).squeeze(1)
#   rf = (col <= tolerance).float().mean()        # keep rate as a scalar tensor
#   chosen = torch.argmax(score).reshape(1)
#   row = pop.index_select(0, chosen)              # no Python integer index
# Preserve source sampling dependencies; batch only independent draws/calculations.

# --- 4. Variation Operators (Standard) ---
from evox.operators.crossover import (
    simulated_binary,       # SBX (returns N offspring from N parents)
    simulated_binary_half,  # SBX-half (returns N/2 offspring from N parents)
)
# differential_evolution is a module, not a callable; use source DE equations.
# Usage:
#   # Expect x shape: (N, D), paired as x[:N/2] vs x[N/2:N]
#   off = simulated_binary(x, pro_c=1.0, dis_c=20.0)        # (N, D)
#   off_half = simulated_binary_half(x, pro_c=1.0, dis_c=20.0)  # (N/2, D)

from evox.operators.mutation import (
    polynomial_mutation,   # PM
)
# Signature: polynomial_mutation(x, lb, ub, pro_m=1.0, dis_m=20.0).
# Bounds lb and ub are required. x has shape (N, D).
# pro_m is the expected number of mutated variables, NOT the per-variable probability.
# The operator samples each variable with probability pro_m / D internally.
# For the default PlatEMO OperatorGA rate 1/D, pass pro_m=1.0 (or omit it).
# Do NOT pass pro_m=1.0/D: that would produce a per-variable rate of 1/D**2.
# For a source per-variable probability p, map it to pro_m=p*D.
# Usage:
#   off = polynomial_mutation(off, lb, ub, pro_m=1.0, dis_m=20.0)
from evox.operators.sampling import (
    uniform_sampling, # (points, n_samples) = uniform_sampling(n, m)
    latin_hypercube_sampling, 
    latin_hypercube_sampling_standard, 
    grid_sampling
)
