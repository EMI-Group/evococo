#!/usr/bin/env python3
"""Run one isolated optimization-fidelity trial.

This internal worker is invoked by ``run_optimization_fidelity_benchmark.py``. Each trial
uses a fresh process so a failed algorithm cannot corrupt the remaining runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from _benchmark_problems import ALL_PROBLEMS, create_problem


RESULT_PREFIX = "EVOCOCO_FIDELITY_RESULT="


def load_algorithm_class(path: Path, evox_module):
    module_name = f"evococo_fidelity_{path.stem.replace('-', '_')}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    expected_name = path.stem.replace("-", "").replace("_", "")
    candidate = getattr(module, expected_name, None)
    if isinstance(candidate, type):
        try:
            if issubclass(candidate, evox_module.core.Algorithm):
                return candidate
        except TypeError:
            pass

    for attr_name in dir(module):
        candidate = getattr(module, attr_name)
        if not isinstance(candidate, type):
            continue
        try:
            if (
                issubclass(candidate, evox_module.core.Algorithm)
                and candidate is not evox_module.core.Algorithm
            ):
                return candidate
        except TypeError:
            continue

    raise RuntimeError(f"No EvoX Algorithm subclass found in {path}")


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

    import torch
    import evox
    from evox.metrics import igd
    from evox.workflows import StdWorkflow

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in this process"
        )

    torch.set_default_device(args.device)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    problem, dimension, lower_bound, upper_bound = create_problem(
        args.problem,
        args.objectives,
        args.device,
    )
    algorithm_file = args.algorithm_file.expanduser().resolve()
    algorithm_class = load_algorithm_class(algorithm_file, evox)
    kwargs = {
        "pop_size": args.pop_size,
        "n_objs": args.objectives,
        "lb": lower_bound,
        "ub": upper_bound,
    }

    try:
        algorithm = algorithm_class(**kwargs)
    except TypeError as standard_error:
        try:
            algorithm = algorithm_class(problem=problem, pop_size=args.pop_size)
        except TypeError:
            raise standard_error

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
    fit = fit[torch.all(torch.isfinite(fit), dim=1)]
    if fit.shape[0] == 0:
        raise RuntimeError("The final fitness tensor contains no finite rows")

    result = {
        "igd": igd(fit, problem.pf()).item(),
        "runtime_s": elapsed,
        "final_population_size": int(fit.shape[0]),
        "dimension": dimension,
        "device": args.device,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, allow_nan=False)}", flush=True)


if __name__ == "__main__":
    main()
