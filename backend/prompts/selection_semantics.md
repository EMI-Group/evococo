# Selection Semantics Contract

For EVERY parent-selection, survivor-selection, or multi-key sorting site, declare
the source semantics before choosing an SDK call. Derive them from the supplied
MATLAB source; do not assume that all algorithms prioritize Pareto rank.

The Analyst report and Architect blueprint MUST contain a section named
`Selection Semantics Contract`. For each site, give:

1. The exact source call/expression and a stable site name (e.g. `parent_mating`).
2. A compact criteria listing with `priority (1 = highest)`, `raw criterion`, `minimize/maximize`,
   `source expression/sign`, and `target tensor`. A negative source expression
   must not accidentally be negated twice.
3. `semantic_order`: the comparison criteria from HIGHEST to LOWEST priority.
4. The tie rule. Lower-priority criteria may affect the winner ONLY when every
   higher-priority criterion is equal. Retain source tie behavior; do not invent
   random selection or a new scalar objective.

The Architect MUST additionally give `target_api` and the EXACT argument list in
SDK order, including signs. Copy the Analyst's priorities and directions without
changing them. The Coder and both Fixers MUST preserve this declaration at each
call site (a short comment is sufficient in Python output).

## EvoX mapping: last key has highest priority

Both `evox.utils.lexsort` and
`evox.operators.selection.tournament_selection_multifit` use the LAST key as the
PRIMARY key. SDK key order is therefore LOWEST to HIGHEST priority. Convert
maximize criteria to minimizing keys by negation, then reverse the semantic
priority order exactly once. `fitnesses` must be a list/tuple of 1D tensors,
not an already stacked tensor.

For the source `TournamentSelection(2, N, FrontNo, -CrowdDis)`:

| priority | raw criterion | direction | source expression | target tensor |
| --- | --- | --- | --- | --- |
| 1 | Pareto/SDR front number | minimize | FrontNo | rank |
| 2 | crowding distance | maximize | -CrowdDis | crowding |

- `semantic_order`: rank ascending, then crowding descending.
- `target_api`: `tournament_selection_multifit`.
- Exact SDK call: `tournament_selection_multifit(n_round=N, fitnesses=[-crowding, rank], tournament_size=2)`.
- Corresponding lexsort keys: `[-crowding, rank]`.
- `fitnesses=[rank, -crowding]` is WRONG for this source: it prioritizes crowding.
- Never replace this lexicographic comparison with `rank - epsilon * crowding`.
  No fixed epsilon preserves the priority for unbounded/infinite crowding.

For each distinct comparison ordering, write at least two concrete winner examples: one where
criteria disagree and one where the primary criterion ties. For rank/crowding,
also cover infinite crowding:

- A=(rank 0, crowding 0), B=(rank 1, crowding 100): A wins.
- A=(rank 0, crowding 1), B=(rank 0, crowding 2): B wins.
- A=(rank 0, crowding 0), B=(rank 1, crowding +inf): A wins.

Sites with identical criteria, directions and tie behavior may reference one
shared set of examples instead of repeating it. Each site must still declare its
own source expression, priorities, tie behavior and exact target API arguments.
Do not share examples across sites whose comparison or tie behavior differs.
Use a compact per-site row for the criteria instead of repeating the template
table and general SDK rules; preserve all specified fields.

These examples describe the semantic requirement, not proof that generated code
has been tested. An optimization/repair must not reverse the order to fix a type
or shape error. If a helper's comparison semantics cannot be established from
the supplied source/reference, explicitly mark them unresolved instead of
guessing. If there are no selection/sorting sites, explicitly state `not applicable`.
