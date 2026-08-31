### 📚 Reference Context: Custom/Standalone MATLAB to EvoX Migration Guide

**1. Target Environment & EvoX Standard**
Regardless of the input MATLAB code structure, the final target is always the **EvoX** framework:
* **`self.pop`**: Decision variables of shape `(N, D)` (where `N` is population size, `D` is dimension).
* **`self.fit`**: Objective/fitness values of shape `(N, M)` (where `M` is number of objectives).
* **`self.lb`, `self.ub`**: Bounds of shape `(D,)`.

**2. Custom MATLAB Analysis Requirements (Dynamic Structure & Shape Forensics)**
The input code is custom or standalone MATLAB/Octave code (not built on PlatEMO). Individual coding styles vary. You must inspect the code structure to extract the following mappings dynamically:
* **Shape Layout Analysis (CRITICAL)**: Determine if the MATLAB code represents individuals/population as column vectors or row vectors.
  - **Column-Major Layout (Common in standalone code)**: If the population matrix (e.g. `X`, `P.x`) has shape `(D, N)` (where individuals are columns) and objective matrix (e.g. `F`, `P.f`) has shape `(M, N)`:
    - You must explicitly instruct the Coder to transpose these matrices to row-major format in EvoX, i.e., `self.pop` of shape `(N, D)` and `self.fit` of shape `(N, M)`.
    - You must translate column indexing `X(:, i)` to row indexing `pop[i]` in Python.
    - You must translate column concatenation `[P.x, O.x]` to row concatenation `torch.cat([self.pop, off_pop], dim=0)`.
  - **Row-Major Layout**: If the population already uses shape `(N, D)` and objectives use shape `(N, M)`, map them directly.
* **Variable Role Forensics**: Identify which variables correspond to:
  - The population (decision variables) ➡️ `self.pop`
  - The objectives/fitness values ➡️ `self.fit`
  - The lower and upper bounds (e.g., `xrange`, `MinValue`, `MaxValue`) ➡️ `self.lb`, `self.ub`
  - The population size (e.g., `options.mu`, `mu`) ➡️ `self.pop_size`
* **Evaluation Interface Analysis**: Identify how the objective functions are evaluated. For example, is there a function handle `f` evaluated via `feval(f, x)` or `fobjeval(f, x)`?
  - Map the custom evaluation logic to EvoX's `self.evaluate(pop)`.
  - Ensure the population passed to `self.evaluate` has shape `(N, D)`.

**3. Logic Translation Protocols**
* **State Persistence**: Only use `self.` for variables that must survive to the next step.
* **Slicing & Filtering**: If the MATLAB code filters the population by slicing indices, translate this to slicing both `self.pop` and `self.fit` synchronously.
