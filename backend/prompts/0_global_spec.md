# System Architecture Standard (EvoCoCo)

## 0. Target Environment (Source-Defined Constraint Support)
* Preserve constraints only when the source uses them in optimization decisions;
  names or algorithm labels alone are not evidence. Otherwise remain unconstrained.
* Retain source CV aggregation, tolerances, priorities and ties; do not replace
  them with a generic constraint policy. Report missing semantics as unresolved.
* Supported algorithms accept both fitness-only and `(fitness, cv)` evaluations.
  Keep evaluated fitness/CV aligned with the population; SDK details are in the reference.

## 1. Input Interface
The algorithm is a class inheriting from `evox.core.Algorithm`.
Use the Coder's constructor interface; the workflow supplies `self.evaluate(pop)`, not `self.problem`.

## 2. Tensor Standards (PyTorch)
All calculations must use `torch.Tensor` on GPU (cuda) if available.
* **Variable Naming (STRICT)**:
    * **Population (Decision Vars)**: MUST be named `self.pop`. Shape `(N, D)`.
    * **Fitness (Objective Vals)**: MUST be named `self.fit`. Shape `(N, M)`.
* **Bounds**: `lower_bound` and `upper_bound` are 1D tensors of shape `(D,)`.

## 3. Key Methods
* `init_step(self)`: 
    * Initialize internal states.
    * **Must set**: `self.pop` and `self.fit`.
* `step(self)`: 
    * Perform one iteration (mating -> mutation -> selection).
    * Update `self.pop` and `self.fit` **in-place**.

## 4. Forbidden Anti-Patterns (STRICTLY PROHIBITED)

1. **Batch Independent Calculations**: Vectorize independent individual, dimension and pairwise calculations rather than processing them with Python loops.
2. **Compiled Kernels**: Keep hot-path values and indices as tensors; minimize host reads using the JIT rule in RAG. Preserve necessary control boundaries in the smallest helper and keep independent calculations compilable. Logs may extract scalars. Disabling the entire step is a performance downgrade.
3. **NO Dynamic Lists Extraction**: Do NOT use `selected_indices = []` and `extend()`. Operations must stay in Tensor space.
4. **Preserve Sequential Semantics**: Retain source-required peeling, sequential updates, stopping conditions and update order. Do not add deadlock exits or fallback termination, or replace iterative selection with a different sort or fixed iteration budget.
    * When a source-equivalent global sort is valid, use `evox.utils.lexsort` with keys ordered from lowest to highest priority. For rank ascending followed by crowding descending, use keys `[-CD, Rank]`. Do NOT substitute a weighted score such as `Rank - CD * 1e-4`: it changes lexicographic priority, especially for infinite crowding. Slice `self.pop = pop[sorted_idx[:N]]` only when this preserves the source selection semantics.
5. **NO Numpy**: Use `torch` exclusively unless absolutely necessary.

## 5. Required Tensorization Paradigms
* **Random Sampling**: Instead of looping to find targets, use `torch.where`, boolean masking (`x[mask] = ...`), or `torch.multinomial` for batched probabilistic sampling without `for` loops.
* **In-Place Updates**: Instead of `X = X + Y`, use `X.add_(Y)` or `X += Y` if memory profiling is tight.
* **Component Re-use**: Do not manually rewrite standard functions if EvoX/EvoMo (`non_dominate_rank`, `crowding_distance`) can process the variables directly in a batched manner.

## 6. Algorithmic Fidelity (ABSOLUTE RULE)
* **NO "SIMPLIFIED FOR VECTORIZATION"**: You MUST 100% faithfully translate the mathematical objectives, fitness calculations, and archive pruning from the original MATLAB code. 
* **EXACT MATHEMATICAL ISOMORPHISM**: When translating scientific and mathematical elements (e.g., Jacobian matrices, multi-term fitness weighting, penalty formulas), you must perform a strict operator-by-operator mapping. NEVER alter the original code's constants (e.g., if MATLAB has `theta=0.2`, do not invent `theta=0.5` or hardcode your own assumptions). NEVER reduce multi-term formulas (e.g., if MATLAB uses `dis(:,1) + 0.1*dis(:,2)`, use `topk(k=2)` or `sort` to perfectly replicate the first and second neighbor weights; do NOT substitute it with a naive `min()` operation).
* **Do NOT** simplify complex metrics (e.g., higher-order distances, multi-stage selections, dual-fitness contributions) into single-dimension proximities just because it is easier to write in PyTorch without a loop. 
* **Loop Transformations**: Vectorize independent work without flattening source-dependent sequential updates. Mathematical fidelity takes priority over an implementation preference.
