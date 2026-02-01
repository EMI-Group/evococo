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
# Standard EvoX Selection
from evox.operators.selection import (
    tournament, 
    tournament_selection_multifit,
    crowding_distance  # Remember: crowding_distance(objs, mask)
)

# Advanced / Multi-Objective Selection (EvoMO)
from evomo.operators.selection import (
    non_dominate_rank,           # Fast Non-dominated Sorting
    nd_environmental_selection,  # NSGA-II style environmental selection
    ref_vec_guided,              # RVEA/MOEA-D style selection
)

# --- 4. Variation Operators (Standard) ---
from evox.operators.crossover import (
    simulated_binary,      # SBX
    differential_evolution # DE
)
from evox.operators.mutation import (
    polynomial_mutation,   # PM
    gaussian_mutation
)
from evox.operators.sampling import uniform_sampling