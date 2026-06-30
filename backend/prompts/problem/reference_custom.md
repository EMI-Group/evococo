### 📚 Reference Context: Custom MATLAB to EvoX Migration Guide (Problem Mode)

**1. Target Environment & EvoX Standard**
The input is a custom mathematical optimization problem. The target is the **EvoX** `evox.core.Problem` framework:
* **`self.d`**: Integer. Dimension of decision variables.
* **`self.m`**: Integer. Number of objectives.
* **`evaluate(self, X)`**: Returns objective values of shape `(N, M)` given `X` of shape `(N, D)`.

**2. Custom MATLAB Analysis Requirements**
The input code is custom or standalone MATLAB/Octave code. You must inspect the code structure to extract the mappings:
* **Shape Layout Analysis (CRITICAL)**: Determine if the MATLAB code represents inputs as column vectors `(D, N)` or row vectors `(N, D)`.
  - In EvoX, `X` is always `(N, D)`. If the MATLAB formula assumes column vectors (e.g. `sum(X.^2, 1)` to sum over dimensions), you must adjust the PyTorch implementation to sum over `dim=1` instead of `dim=0`.
* **Variable Role Forensics**: Identify constants, bounds, scaling factors, and objective equations.

**3. Logic Translation Protocols**
* **Strict Math Fidelity**: Keep mathematical operations isomorphic. Do not alter coefficients or structural math.
* **Batched Processing**: Ensure that any custom equations support batch processing without Python loops.

**4. How to Handle Multiple Tiered Subnets (e.g. PB1 to PB15)**
If the MATLAB code computes identical spatial integrals across multiple subnets (e.g. 10 Tier-1 nodes and 5 Tier-2 nodes), DO NOT hardcode the output integrals!
Instead:
- **Avoid 4D+ Tensor Broadcasting**: Do NOT try to broadcast the subnets along with all spatial dimensions (`r_i`, `R`, `Theta`) simultaneously. This will lead to Out of Memory (OOM) errors and timeouts!
- **Use Sequential Loops**: You ARE EXPLICITLY ALLOWED to use a simple python `for s in range(num_subnets):` loop to process each subnet sequentially. Inside the loop, perform the vectorized integration using `torch.sum` over the spatial grids (`R`, `Theta`, `H`), and accumulate the result into a total tensor.
- NEVER hardcode integration values (e.g. `[1.852, ...]`). It must be mathematically computed by PyTorch at runtime inside the loop!
