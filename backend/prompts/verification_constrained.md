```python
# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import json
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evomo.problems.constrained import MW4
    from evomo.workflows import UnifiedWorkflow

    torch.set_default_device("cuda")
    constrained_problem = MW4(d=12, m=3)
    cases = [
        ("DTLZ2", DTLZ2(m=3), torch.zeros(12), torch.ones(12), False),
        ("MW4", constrained_problem, constrained_problem.lb, constrained_problem.ub, True),
    ]
    for problem_name, prob, lb, ub, has_constraints in cases:
        algo = <YourAlgoName>(pop_size=100, n_objs=3, lb=lb, ub=ub)
        pf = prob.pf()
        workflow = UnifiedWorkflow(algo, prob, device="cuda")
        workflow.init_step()
        jit_state_step = torch.compile(workflow.step)
        jit_state_step()
        torch.cuda.synchronize()
        exec_start = time.perf_counter()
        for i in range(1, 100):
            jit_state_step()
            if (i + 1) % 5 == 0:
                fit = workflow.algorithm.fit
                if fit.numel() == 0 or not torch.isfinite(fit).all():
                    raise ValueError("Runtime validation failed: empty or non-finite fitness")
                if has_constraints:
                    cv = workflow.algorithm.cv
                    if cv is None or cv.shape[0] != fit.shape[0] or not torch.isfinite(cv).all():
                        raise ValueError("Runtime validation failed: missing, misaligned or non-finite CV")
                    total_cv = cv.sum(dim=1) if cv.ndim == 2 else cv
                    feasible = total_cv <= 0
                    feasible_igd = igd(fit[feasible], pf).item() if feasible.any() else None
                    print("CONSTRAINT_METRICS: " + json.dumps({
                        "problem": problem_name, "generation": i + 1,
                        "population": fit.shape[0], "feasible_count": feasible.sum().item(),
                        "mean_cv": total_cv.mean().item(), "feasible_igd": feasible_igd,
                    }))
                else:
                    print(f"Gen {i + 1} IGD: {igd(fit, pf)}")
        torch.cuda.synchronize()
        exec_time = time.perf_counter() - exec_start
        if has_constraints:
            print(f"MW4 execution time for Gen 2-100: {exec_time:.4f}s")
        else:
            print(f"Execution time for Gen 2-100 (99 steps): {exec_time:.4f}s (Avg: {exec_time / 99:.4f}s/gen)")
```
