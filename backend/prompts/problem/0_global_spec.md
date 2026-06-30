# System Architecture Standard (EvoCoder - Problem Translation)

## 0. Target Environment
* **Optimization Type**: Unconstrained Evolutionary Optimization Problems.
* **Focus**: Translating problem definitions, objective evaluations, and pareto front calculations.

## 1. Input Interface
The problem is a class inheriting from `evox.core.Problem`.
It must accept initialization parameters and provide methods for evaluation.

## 2. Tensor Standards (PyTorch)
All calculations must use `torch.Tensor` on GPU (cuda) if available.
* **Variable Naming (STRICT)**:
    * **Input (Decision Vars)**: `X`. Shape `(N, D)`.
    * **Output (Objective Vals)**: `f`. Shape `(N, M)`.
* **Bounds**: `lower_bound` and `upper_bound` should be defined if applicable.

## 3. Key Methods
* `__init__(self, ...)`: 
    * Initialize internal states, `d` (decision dim), `m` (objectives).
* `evaluate(self, X: torch.Tensor) -> torch.Tensor`: 
    * Takes decision variables `X` of shape `(N, D)`.
    * Returns objective values `f` of shape `(N, M)`.
* `pf(self)`: 
    * Optional. Returns the true pareto front for the problem if mathematically defined.

## 4. Forbidden Anti-Patterns (STRICTLY PROHIBITED)
1. **NO Individual-level Loops**: Do NOT write `for i in range(N):` to process solutions.
2. **NO CPU-GPU Sync**: Do NOT use `.item()` or `.tolist()` in the main execution paths. This forces the GPU to halt and wait for the CPU.
3. **NO Dynamic Lists Extraction**: Do NOT use `selected_indices = []` and `extend()`. Operations must stay in Tensor space.
4. **NO Numpy**: Use `torch` exclusively unless absolutely necessary.

## 5. Required Tensorization Paradigms
* **Vectorized Math**: The `evaluate` function must handle a batch size `N`. Matrix multiplications, broadcasting, and element-wise operations should be leveraged.
* **Component Re-use**: Do not manually rewrite standard functions if EvoX has them.

## 6. Algorithmic Fidelity (ABSOLUTE RULE)
* **EXACT MATHEMATICAL ISOMORPHISM**: You must perfectly replicate the objective function's mathematical equations from the original MATLAB code. NEVER alter constants or simplify mathematical forms.
* **NO PYTHON FOR-LOOP FALLBACKS**: You MUST force vectorization using advanced boolean indexing, broadcasting (`A.unsqueeze(1) - B.unsqueeze(0)`), or `torch.where`.