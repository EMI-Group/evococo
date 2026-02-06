# System Architecture Standard (EvoCoder)

## 0. Target Environment (Unconstrained Only)
* **Optimization Type**: Unconstrained Evolutionary Optimization.
* **Constraint Handling**: **STRICTLY FORBIDDEN**.
    * **Remove Logic**: Any logic related to constraint violation (`CV`), feasibility matrices, or penalty functions in the source MATLAB code **MUST BE REMOVED**.
    * **Assume Feasible**: The converted Python code must treat the problem as having **no constraints** (i.e., every solution is feasible).
    * **Objective Sorting**: Sorting and selection must rely **solely on Objective Values** (Fitness).

## 1. Input Interface
The algorithm is a class inheriting from `evox.Algorithm`.
It must accept a `problem` object and configuration parameters.

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

## 4. Forbidden Patterns
* **NO** `numpy` (unless absolutely necessary). Use `torch`.
* **NO** Python `for` loops for calculating distances or checking dominance. Use `torch` broadcasting.
* **NO** Iterative Selection: Do not write logic like "put front 1, then front 2, then...".
  * **Correct Pattern**:
    1. Calculate metrics (Rank, Density) for the *entire* merged population.
    2. Sort the entire population using `torch.argsort` or `evox.utils.lexsort` based on primary key (Rank) and secondary key (Density).
    3. Slice the top N: `self.pop = sorted_pop[:N]`.