```python
# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import json
    import time
    import torch
    from evox.problems.numerical import Sphere
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")
    torch.manual_seed(42)
    direction = "<ObjectiveDirection>"

    class FitnessColumn(torch.nn.Module):
        def forward(self, fitness):
            return (fitness if direction == "min" else -fitness).unsqueeze(-1)

    algo = <YourAlgoName>(pop_size=100, n_objs=1, lb=torch.full((12,), -5.0), ub=torch.full((12,), 5.0))
    workflow = StdWorkflow(algo, Sphere(), fitness_transform=FitnessColumn())
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)
    jit_state_step()
    torch.cuda.synchronize()
    exec_start = time.perf_counter()
    for i in range(1, 100):
        jit_state_step()
        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            if fit.ndim != 2 or fit.shape[1] != 1 or fit.numel() == 0 or not torch.isfinite(fit).all():
                raise ValueError("Invalid single-objective fitness")
            best = fit.min() if direction == "min" else fit.max()
            print("SINGLE_OBJECTIVE_METRICS: " + json.dumps({
                "generation": i + 1, "best_fitness": best.item(), "direction": direction,
            }))
    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-100 (99 steps): {exec_time:.4f}s (Avg: {exec_time / 99:.4f}s/gen)")
```
