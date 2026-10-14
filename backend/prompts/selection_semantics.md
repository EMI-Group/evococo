# Selection Semantics Contract

The Analyst report and Architect blueprint must contain a section named
`Selection Semantics Contract`. Derive each parent/survivor selection and sorting
site from the source; do not assume Pareto rank or automatically prepend CV.

Use one compact row per site: stable name, exact source expression, raw criteria
and signs, minimize/maximize directions, semantic order (highest priority first),
target tensors and source tie rule. Lower-priority criteria apply only when all
higher-priority criteria tie. Record source CV aggregation and tolerances where used.

The Architect adds the exact target API arguments, including signs and SDK order.
Coder and Fixers preserve them with a short priority comment at each call.

## SDK mapping

EvoX `lexsort` and `tournament_selection_multifit` use the LAST key as primary;
all keys are minimized. Negate maximize criteria, then reverse semantic order
exactly once. Tournament `fitnesses` is a list/tuple of 1D tensors, not a stack.

For source `TournamentSelection(2, N, FrontNo, -CrowdDis)`, semantic order is rank
ascending, then crowding descending. Use
`tournament_selection_multifit(n_round=N, fitnesses=[-crowding, rank], tournament_size=2)`;
lexsort uses the same key order. Never replace lexicographic ordering with a
weighted score: finite weights cannot preserve priority for unbounded crowding.

Give two concrete winner examples per distinct comparison/tie policy: conflicting
criteria and tied primary criteria. Rank/crowding must also cover infinity:
A=(0,0) beats B=(1,100); B=(0,2) beats A=(0,1); A=(0,0) beats B=(1,+inf).
Sites with identical policies share examples; do not repeat the general SDK rules.

Examples are reasoning checks, not executed tests. Repairs must not reverse
priorities to fix errors or improve metrics. Mark unknown semantics unresolved;
if no selection/sorting sites exist, state `not applicable`.
