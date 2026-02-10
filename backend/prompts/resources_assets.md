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
# Usage:
#   # Single-objective tournament (min fitness wins)
#   parent_idx = tournament_selection(n_round=pop_size, fitness=fit, tournament_size=2)
#   parents = pop[parent_idx]
#
#   # Multi-objective / multi-criteria tournament (lexicographic on stacked fitnesses)
#   parent_idx = tournament_selection_multifit(n_round=pop_size, fitnesses=[f1, f2, f3], tournament_size=2)
#   parents = pop[parent_idx]
#   # Crowding distance calculation for a given front (mask is boolean)
#   # costs: (N, M), mask: (N,) bool
#   cd = crowding_distance(costs, mask)      # (N,), larger = more diverse

# --- 4. Variation Operators (Standard) ---
from evox.operators.crossover import (
    simulated_binary,       # SBX (returns N offspring from N parents)
    simulated_binary_half,  # SBX-half (returns N/2 offspring from N parents)
    differential_evolution, # DE
)
# Usage:
#   # Expect x shape: (N, D), paired as x[:N/2] vs x[N/2:N]
#   off = simulated_binary(x, pro_c=1.0, dis_c=20.0)        # (N, D)
#   off_half = simulated_binary_half(x, pro_c=1.0, dis_c=20.0)  # (N/2, D)

from evox.operators.mutation import (
    polynomial_mutation,   # PM
)
from evox.operators.sampling import (
    uniform_sampling, # (points, n_samples) = uniform_sampling(n, m)
    latin_hypercube_sampling, 
    latin_hypercube_sampling_standard, 
    grid_sampling
)
