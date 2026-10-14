```python
# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import json
    import time
    import torch
    from evox.core import Problem
    from evox.problems.numerical import Sphere
    from evomo.workflows import UnifiedWorkflow

    torch.set_default_device("cuda")
    direction = "<ObjectiveDirection>"

    class FitnessColumn(torch.nn.Module):
        def forward(self, fitness):
            return (fitness if direction == "min" else -fitness).unsqueeze(-1)

    class ConstrainedSphere(Problem):
        def evaluate(self, pop):
            fitness = pop.square().sum(dim=1)
            cv = torch.cat((1 - pop[:, :1], pop[:, :1] - 2), dim=1).clamp_min(0)
            return fitness, cv

    cases = [("Sphere", Sphere(), False), ("ConstrainedSphere", ConstrainedSphere(), True)]
    for problem_name, problem, constrained in cases:
        torch.manual_seed(42)
        algo = <YourAlgoName>(pop_size=100, n_objs=1, lb=torch.full((12,), -5.0), ub=torch.full((12,), 5.0))
        workflow = UnifiedWorkflow(algo, problem, fitness_transform=FitnessColumn(), device="cuda")
        workflow.init_step()
        step = torch.compile(workflow.step)
        step()
        torch.cuda.synchronize()
        exec_start = time.perf_counter()
        for generation in range(2, 101):
            step()
            if generation % 5 == 0:
                fit = workflow.algorithm.fit
                if fit.ndim != 2 or fit.shape[1] != 1 or fit.numel() == 0 or not torch.isfinite(fit).all():
                    raise ValueError("Invalid single-objective fitness")
                best = fit.min() if direction == "min" else fit.max()
                row = dict(problem=problem_name, generation=generation, direction=direction, best_fitness=best.item())
                if constrained:
                    cv = workflow.algorithm.cv
                    if cv is None or cv.ndim not in (1, 2) or cv.shape[0] != fit.shape[0] or not torch.isfinite(cv).all() or (cv < 0).any():
                        raise ValueError("Invalid single-objective CV")
                    total_cv = cv.sum(dim=1) if cv.ndim == 2 else cv
                    feasible = total_cv <= 0
                    feasible_best = (fit[feasible].min() if direction == "min" else fit[feasible].max()).item() if feasible.any() else None
                    row.update(population=fit.shape[0], feasible_count=feasible.sum().item(), mean_cv=total_cv.mean().item(), feasible_best_fitness=feasible_best)
                print("SINGLE_OBJECTIVE_METRICS: " + json.dumps(row))
        torch.cuda.synchronize()
        exec_time = time.perf_counter() - exec_start
        if constrained:
            print(f"ConstrainedSphere execution time for Gen 2-100: {exec_time:.4f}s")
        else:
            print(f"Execution time for Gen 2-100 (99 steps): {exec_time:.4f}s (Avg: {exec_time / 99:.4f}s/gen)")
```
