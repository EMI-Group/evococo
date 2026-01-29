# System Architecture Standard (EvoCoder)

## 1. Input Interface
The algorithm is a class inheriting from `evox.Algorithm`.
It must accept a `problem` object and configuration parameters.

## 2. Tensor Standards (PyTorch)
All calculations must use `torch.Tensor` on GPU (cuda) if available.
* **Population (Dec)**: Shape `(N, D)` where N=pop_size, D=dim.
* **Population (Obj)**: Shape `(N, M)` where M=num_objectives.
* **Bounds**: `lower_bound` and `upper_bound` are 1D tensors of shape `(D,)`.

## 3. Key Methods
* `init_step(self)`: Initialize internal states (e.g., population and fitness).
* `step(self)`: Perform one iteration (mating -> mutation -> selection), updating internal states in-place.

## 4. Forbidden Patterns
* **NO** `numpy` (unless absolutely necessary). Use `torch`.
* **NO** Python `for` loops for calculating distances or checking dominance. Use `torch` broadcasting.
* **NO** Iterative Selection: Do not write logic like "put front 1, then front 2, then...".
  * **Correct Pattern**:
    1. Calculate metrics (Rank, Density) for the *entire* merged population.
    2. Sort the entire population using `torch.argsort` or `evox.utils.lexsort` based on primary key (Rank) and secondary key (Density).
    3. Slice the top N: `pop = sorted_pop[:N]`.