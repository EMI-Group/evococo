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
* `setup(self, key)`: Initialize state (population).
* `step(self, state)`: Perform one iteration (mating -> mutation -> selection).
    * **Input**: Current `state`.
    * **Output**: New `state` containing the updated population.

## 4. Forbidden Patterns
* **NO** `numpy` (unless absolutely necessary). Use `torch`.
* **NO** Python `for` loops for calculating distances or checking dominance. Use `torch` broadcasting.