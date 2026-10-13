### 📚 Reference Context: PlatEMO to EvoX Migration Guide

**1. Source Framework Identity (PlatEMO)**
The input MATLAB code is built upon **PlatEMO**.
* **`Population` Structure**: Bundles `decs` (Variables), `objs` (Objectives), `cons` (Constraints), and `adds` (Masks).
* **Implicit Slicing**: `Population(Index)` slices all internal arrays.

**2. The "Rosetta Stone": Variable Mapping Table**
You must distinguish between **Persistent State** (maintained across generations) and **Local/Temporary Variables**.

| PlatEMO Context | Source Var (MATLAB) | **EvoX (Target) Name** | **Shape** | **Rule** |
| :--- | :--- | :--- | :--- | :--- |
| **Main Population** (State) | `Population.decs`, `Dec` | **`self.pop`** | $(N, D)$ | **PERSISTENT**. Must be `self.pop`. |
| **Main Objectives** (State) | `Population.objs`, `Obj` | **`self.fit`** | $(N, M)$ | **PERSISTENT**. Must be `self.fit`. |
| **Offspring/Temporary** | `OffDec`, `Offspring.decs` | `off_pop` / `vals` | $(N, D)$ | **LOCAL**. Do NOT use `self.` prefix. |
| **Archive** (If exists) | `Archive`, `ExternalPop` | **`self.archive`** | $(N', D)$ | **PERSISTENT**. Treat as separate state. |
| **Sparse Mask** | `Mask` | **`self.mask`** | $(N, D)$ | Pair with `self.pop`. |
| **Constraint Violation** (if used) | `Population.cons`, `CV` | `self.cv` / local aggregate | $(N,C)$ / $(N,)$ | Source-defined constraint policy. |
| **Bounds** | `Problem.lower/upper` | `self.lb`, `self.ub` | $(D,)$ | Stored in `__init__`. |

**3. Logic Translation Protocols**

* **A. State Persistence Protocol (CRITICAL)**
    * **Rule**: Only use `self.` for variables that **must survive to the next `step()`**.
    * **Scenario 1 (Main Update)**: If `Dec` is updated in place (e.g., `Dec(Next,:)`), map it to `self.pop = self.pop[indices]`.
    * **Scenario 2 (Calculation)**: If `Dec` is just used for intermediate calculation (e.g., `Dist = pdist2(Dec, Dec)`), use `self.pop` or the local variable name appropriately.

* **B. The Decoupling Protocol**
    * *Source*: `Population = Population(Next)`
    * *Target*: You must explicitly instruct the Coder to slice **all** active tensors synchronously.
    * *Instruction*: "Slice `self.pop`, `self.fit`, and active `self.cv`/`self.mask` using the same `Next` indices."

* **C. The Evaluation Protocol**
    * *Source*: `Problem.Evaluation(Dec)` or `Problem.Evaluation(Dec .* Mask)`
    * *Target*: `self.evaluate(...)`.
    * Evaluation may also return CV when the source supports constraints.
    * *Instruction*: If SparseEA, ensure the input to evaluate is `self.pop * self.mask`.

* **D. The Duplicate Removal Protocol**
    * *Source*: `unique(Population.objs, 'rows')`
    * *Target*: `evomo.utils.unique_rows_sorted`.
    * *Instruction*: "Calculate unique indices on the source's specified values, then apply those indices to `self.pop`, `self.fit`, and active `self.cv`/`self.mask`."
