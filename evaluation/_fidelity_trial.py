#!/usr/bin/env python3
"""Run one isolated optimization-fidelity trial.

This internal worker is invoked by ``run_optimization_fidelity_benchmark.py``. Each trial
uses a fresh process so a failed algorithm cannot corrupt the remaining runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _benchmark_problems import ALL_PROBLEMS, create_problem
from _common import (
    RESULT_PREFIX_FIDELITY,
    finite_fitness_rows,
    instantiate_algorithm,
    load_algorithm_class,
    setup_torch_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal optimization-fidelity trial")
    parser.add_argument("--algorithm-file", type=Path, required=True)
    parser.add_argument("--problem", choices=ALL_PROBLEMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--objectives", type=int, default=3)
    parser.add_argument("--execution", choices=("eager", "compile"), default="eager")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import evox
    import torch
    from evox.metrics import igd
    from evox.workflows import StdWorkflow

    setup_torch_device(args.device, args.seed)

    problem, dimension, lower_bound, upper_bound = create_problem(
        args.problem,
        args.objectives,
        args.device,
    )
    algorithm_file = args.algorithm_file.expanduser().resolve()
    algorithm_class = load_algorithm_class(
        algorithm_file, evox, module_prefix="evococo_fidelity_"
    )
    kwargs = {
        "pop_size": args.pop_size,
        "n_objs": args.objectives,
        "lb": lower_bound,
        "ub": upper_bound,
    }
    algorithm = instantiate_algorithm(algorithm_class, problem, args.pop_size, **kwargs)

    workflow = StdWorkflow(algorithm, problem)
    workflow.init_step()
    step = workflow.step
    if args.execution == "compile":
        step = torch.compile(step)

    if args.device == "cuda":
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    for _ in range(args.generations):
        step()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    fit = workflow.algorithm.fit
    if fit is None:
        raise RuntimeError("The algorithm did not expose a final fitness tensor")
    if fit.dim() == 3:
        fit = fit[0]
    fit = finite_fitness_rows(fit)

    result = {
        "igd": igd(fit, problem.pf()).item(),
        "runtime_s": elapsed,
        "final_population_size": int(fit.shape[0]),
        "dimension": dimension,
        "device": args.device,
    }
    print(f"{RESULT_PREFIX_FIDELITY}{json.dumps(result, allow_nan=False)}", flush=True)


if __name__ == "__main__":
    main()
