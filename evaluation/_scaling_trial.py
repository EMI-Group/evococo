#!/usr/bin/env python3
"""Run one isolated scaling trial.

This is an internal worker used by ``run_computational_scalability_benchmark.py``. Keeping one
trial per process prevents CUDA allocations and torch.compile state from
leaking between algorithms, scales, or repetitions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _common import (
    RESULT_PREFIX_SCALING,
    finite_fitness_rows,
    instantiate_algorithm,
    load_algorithm_class,
    setup_torch_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal isolated benchmark trial")
    parser.add_argument("--algorithm-file", type=Path, required=True)
    parser.add_argument("--pop-size", type=int, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--objectives", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--execution", choices=("eager", "compile"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--recompile-limit", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import evox
    import torch
    import torch._dynamo
    from evomo.problems.numerical import DTLZ3
    from evox.metrics import igd
    from evox.workflows import StdWorkflow

    torch._dynamo.config.recompile_limit = args.recompile_limit
    setup_torch_device(args.device, args.seed)

    algorithm_file = args.algorithm_file.expanduser().resolve()
    algorithm_class = load_algorithm_class(
        algorithm_file, evox, module_prefix="evococo_eval_"
    )
    problem = DTLZ3(m=args.objectives, d=args.dimension)
    kwargs = {
        "pop_size": args.pop_size,
        "n_objs": args.objectives,
        "lb": torch.zeros(args.dimension),
        "ub": torch.ones(args.dimension),
    }
    algorithm = instantiate_algorithm(algorithm_class, problem, args.pop_size, **kwargs)

    workflow = StdWorkflow(algorithm, problem)
    workflow.init_step()
    step = workflow.step
    if args.execution == "compile":
        step = torch.compile(step)

    for _ in range(args.warmup):
        step()

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
        "total_time_s": elapsed,
        "time_per_generation_s": elapsed / args.generations,
        "igd": igd(fit, problem.pf()).item(),
        "final_population_size": int(fit.shape[0]),
        "device": args.device,
    }
    print(f"{RESULT_PREFIX_SCALING}{json.dumps(result, allow_nan=False)}", flush=True)


if __name__ == "__main__":
    main()
