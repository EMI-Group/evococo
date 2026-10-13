```python
# === FIXED DEMO BLOCK ===
# This block MUST be appended at the end of the file.
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # <YourAlgoName> must be replaced by your actual class name
    algo = <YourAlgoName>(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    # 1. Trigger JIT compilation (First step)
    jit_state_step()

    # 2. Pure execution (Remaining 99 steps)
    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 100):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            # Validate the full population; never hide invalid individuals.
            if fit.numel() == 0 or not torch.isfinite(fit).all():
                raise ValueError("Runtime validation failed: empty or non-finite fitness")
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-100 (99 steps): {exec_time:.4f}s (Avg: {exec_time / 99:.4f}s/gen)")
```
