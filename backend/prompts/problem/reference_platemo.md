### 📚 Reference Context: PlatEMO to EvoX Migration Guide (Problem Mode)

**1. Source Framework Identity (PlatEMO Problem)**
The input MATLAB code is built upon **PlatEMO** and inherits from `PROBLEM` (e.g. `classdef XXX < PROBLEM`).

**2. Variable Mapping Table**
You must map the PlatEMO problem properties to EvoX `evox.core.Problem`.

| PlatEMO Context | Source Var (MATLAB) | **EvoX (Target) Name** | **Description** |
| :--- | :--- | :--- | :--- |
| **Objectives** | `obj.M` | `self.m` | Number of objectives |
| **Dimensions** | `obj.D` | `self.d` | Number of decision variables |
| **Lower Bound** | `obj.lower` | `self.lower_bound` | Should be a Tensor `(D,)` |
| **Upper Bound** | `obj.upper` | `self.upper_bound` | Should be a Tensor `(D,)` |
| **Evaluation** | `CalObj(obj, X)` | `evaluate(self, X)` | Must handle batched input `X` of shape `(N, D)` |
| **Pareto Front**| `GetOptimum(obj, N)`| `pf(self)` | Calculate or sample true pareto front |

**3. Logic Translation Protocols**
* **A. Initialization (`__init__`)**: Translate the setting of `obj.M`, `obj.D`, `obj.lower`, `obj.upper` into PyTorch tensor properties.
* **B. Evaluation (`evaluate`)**: Map the MATLAB `CalObj` logic directly. `X` will be a PyTorch tensor of shape `(N, D)`. Do NOT use for loops to process the `N` individuals.
* **C. Pareto Front (`pf`)**: Translate `GetOptimum` logic exactly using PyTorch operations (like `torch.linspace`, `torch.meshgrid`).
