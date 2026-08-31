#!/usr/bin/env python3
"""Run repeated single-GPU optimization-fidelity benchmark suites."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from _benchmark_problems import ALL_PROBLEMS, PROBLEM_SUITES, suite_problems


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("_fidelity_trial.py")
DEFAULT_ALGORITHM_DIR = ROOT / "experiments" / "generated_algorithms"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation_results" / "fidelity"
RESULT_PREFIX = "EVOCOCO_FIDELITY_RESULT="
RAW_FIELDS = (
    "algorithm",
    "algorithm_file",
    "problem",
    "run",
    "seed",
    "pop_size",
    "generations",
    "objectives",
    "execution",
    "status",
    "igd",
    "runtime_s",
    "final_population_size",
    "dimension",
    "device",
    "error",
)
SUMMARY_FIELDS = (
    "algorithm",
    "problem",
    "pop_size",
    "generations",
    "objectives",
    "execution",
    "expected_runs",
    "successful_runs",
    "coverage",
    "mean_igd",
    "median_igd",
    "std_igd",
    "mean_runtime_s",
    "reference_mean_igd",
    "igd_difference",
    "relative_igd_increase",
    "classification",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated IGD evaluations for generated algorithms on one GPU. "
            "The default protocol uses 21 independent seeds."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--algorithm-dir", type=Path)
    source.add_argument("--algorithm-file", type=Path, nargs="+")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument(
        "--suite",
        choices=(*PROBLEM_SUITES, "all"),
        default="DTLZ",
        help="Problem suite to run (default: DTLZ)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=ALL_PROBLEMS,
        help="Optional subset from the selected suite",
    )
    parser.add_argument("--runs", type=positive_int, default=21)
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--pop-size", type=positive_int, default=100)
    parser.add_argument("--generations", type=positive_int, default=100)
    parser.add_argument("--objectives", type=positive_int, default=3)
    parser.add_argument("--execution", choices=("eager", "compile"), default="eager")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=positive_int, default=3600)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help="Optional PlatEMO summary with Algorithm, Problem, and MeanIGD columns",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=0.10)
    parser.add_argument("--relative-tolerance", type=float, default=2.0)
    parser.add_argument("--minimum-coverage", type=probability, default=0.80)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        parser.error("tolerances must be non-negative")
    selected_suite = suite_problems(args.suite)
    if args.problems is None:
        args.problems = list(selected_suite)
    else:
        outside_suite = [
            problem for problem in args.problems if problem not in selected_suite
        ]
        if outside_suite:
            parser.error(
                f"problems do not belong to suite {args.suite}: "
                f"{', '.join(outside_suite)}"
            )
    return args


def discover_algorithms(args: argparse.Namespace) -> list[Path]:
    if args.algorithm_file:
        files = list(
            dict.fromkeys(path.expanduser().resolve() for path in args.algorithm_file)
        )
    else:
        directory = (args.algorithm_dir or DEFAULT_ALGORITHM_DIR).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Algorithm directory does not exist: {directory}")
        files = sorted(directory.glob("*.py"))

    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"Algorithm file does not exist: {', '.join(missing)}")

    if args.algorithms:
        requested = set(args.algorithms)
        files = [
            path for path in files if path.stem in requested or path.name in requested
        ]
        found = {path.stem for path in files} | {path.name for path in files}
        unmatched = sorted(name for name in requested if name not in found)
        if unmatched:
            raise ValueError(
                f"Requested algorithms were not found: {', '.join(unmatched)}"
            )

    if not files:
        raise ValueError("No Python algorithm files were selected")
    return sorted(files, key=lambda path: path.name.casefold())


def check_cuda(gpu: int) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip().startswith("1 "):
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"GPU {gpu} is not available to this Python environment. {detail}"
        )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RAW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def trial_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        row["algorithm"],
        row["problem"],
        int(row["seed"]),
        int(row["pop_size"]),
        int(row["generations"]),
        int(row["objectives"]),
        row["execution"],
    )


def load_references(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Reference CSV has no header: {path}")
        normalized = {name.casefold(): name for name in reader.fieldnames}
        required = ("algorithm", "problem", "meanigd")
        missing = [name for name in required if name not in normalized]
        if missing:
            raise ValueError(
                "Reference CSV requires Algorithm, Problem, and MeanIGD columns"
            )
        references: dict[tuple[str, str], float] = {}
        for row in reader:
            if "validruns" in normalized and "expectedruns" in normalized:
                valid_runs = int(row[normalized["validruns"]])
                expected_runs = int(row[normalized["expectedruns"]])
                if valid_runs != expected_runs:
                    continue
            value = float(row[normalized["meanigd"]])
            if math.isfinite(value):
                references[
                    (row[normalized["algorithm"]], row[normalized["problem"]])
                ] = value
    return references


def classify(
    generated: float,
    reference: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> str:
    if generated <= reference:
        return "Improved"
    absolute_difference = generated - reference
    relative_increase = absolute_difference / reference if reference > 0 else math.inf
    if (
        absolute_difference <= absolute_tolerance
        or relative_increase <= relative_tolerance
    ):
        return "Preserved"
    return "IGD Degradation"


def run_trial(
    algorithm_file: Path,
    problem: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(WORKER),
        "--algorithm-file",
        str(algorithm_file),
        "--problem",
        problem,
        "--seed",
        str(seed),
        "--pop-size",
        str(args.pop_size),
        "--generations",
        str(args.generations),
        "--objectives",
        str(args.objectives),
        "--execution",
        args.execution,
        "--device",
        args.device,
    ]
    env = dict(os.environ)
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {"status": "timeout", "error": f"Exceeded {args.timeout}s: {error}"}

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
            result.update({"status": "success", "error": ""})
            return result

    detail = (completed.stderr or completed.stdout).strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    return {
        "status": "failed",
        "error": f"Worker exited with code {completed.returncode}: {detail}",
    }


def write_summary(
    raw_path: Path,
    summary_path: Path,
    *,
    expected_seeds: set[int],
    references: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> None:
    grouped: dict[tuple[object, ...], dict[int, dict[str, str]]] = {}
    for row in read_rows(raw_path):
        seed = int(row["seed"])
        if seed not in expected_seeds:
            continue
        group = (
            row["algorithm"],
            row["problem"],
            int(row["pop_size"]),
            int(row["generations"]),
            int(row["objectives"]),
            row["execution"],
        )
        if row["status"] == "success":
            grouped.setdefault(group, {})[seed] = row
        else:
            grouped.setdefault(group, {})

    summary_rows: list[dict[str, object]] = []
    for group in sorted(
        grouped, key=lambda item: (str(item[0]).casefold(), str(item[1]))
    ):
        algorithm, problem, pop_size, generations, objectives, execution = group
        successes = grouped[group]
        igds = [float(row["igd"]) for row in successes.values()]
        runtimes = [float(row["runtime_s"]) for row in successes.values()]
        coverage = len(successes) / len(expected_seeds)
        mean_igd = statistics.fmean(igds) if igds else ""
        reference = references.get((str(algorithm), str(problem)), "")
        difference: float | str = ""
        relative_increase: float | str = ""
        if isinstance(mean_igd, float) and isinstance(reference, float):
            difference = mean_igd - reference
            relative_increase = difference / reference if reference > 0 else math.inf
        classification = ""
        if coverage < args.minimum_coverage:
            classification = "Insufficient Coverage"
        elif isinstance(mean_igd, float) and isinstance(reference, float):
            classification = classify(
                mean_igd,
                reference,
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
            )

        summary_rows.append(
            {
                "algorithm": algorithm,
                "problem": problem,
                "pop_size": pop_size,
                "generations": generations,
                "objectives": objectives,
                "execution": execution,
                "expected_runs": len(expected_seeds),
                "successful_runs": len(successes),
                "coverage": coverage,
                "mean_igd": mean_igd,
                "median_igd": statistics.median(igds) if igds else "",
                "std_igd": statistics.stdev(igds) if len(igds) > 1 else "",
                "mean_runtime_s": statistics.fmean(runtimes) if runtimes else "",
                "reference_mean_igd": reference,
                "igd_difference": difference,
                "relative_igd_increase": relative_increase,
                "classification": classification,
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    temporary.replace(summary_path)


def main() -> int:
    args = parse_args()
    try:
        algorithms = discover_algorithms(args)
        references = load_references(args.reference_csv)
        if args.device == "cuda":
            check_cuda(args.gpu)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    raw_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.csv"
    seeds = [args.seed_base + offset for offset in range(args.runs)]
    expected_seeds = set(seeds)
    completed = {
        trial_key(row) for row in read_rows(raw_path) if row.get("status") == "success"
    }
    total = len(algorithms) * len(args.problems) * len(seeds)
    attempted = 0
    skipped = 0

    print(
        f"Algorithms: {len(algorithms)}; suite: {args.suite}; "
        f"problems: {', '.join(args.problems)}"
    )
    print(f"Runs per pair: {args.runs}; execution: {args.execution}")
    print(
        f"Device: {args.device}{f' (physical GPU {args.gpu})' if args.device == 'cuda' else ''}"
    )
    print(f"Output: {output_dir}")

    try:
        for algorithm_file in algorithms:
            for problem in args.problems:
                for run, seed in enumerate(seeds, start=1):
                    key = (
                        algorithm_file.stem,
                        problem,
                        seed,
                        args.pop_size,
                        args.generations,
                        args.objectives,
                        args.execution,
                    )
                    if key in completed:
                        skipped += 1
                        continue
                    attempted += 1
                    print(
                        f"[{attempted + skipped}/{total}] {algorithm_file.stem} "
                        f"{problem} run={run} seed={seed} ... ",
                        end="",
                        flush=True,
                    )
                    result = run_trial(algorithm_file, problem, seed, args)
                    row: dict[str, object] = {
                        "algorithm": algorithm_file.stem,
                        "algorithm_file": str(algorithm_file),
                        "problem": problem,
                        "run": run,
                        "seed": seed,
                        "pop_size": args.pop_size,
                        "generations": args.generations,
                        "objectives": args.objectives,
                        "execution": args.execution,
                        "igd": "",
                        "runtime_s": "",
                        "final_population_size": "",
                        "dimension": "",
                        "device": args.device,
                        "error": "",
                        **result,
                    }
                    append_row(raw_path, row)
                    if row["status"] == "success":
                        print(
                            f"IGD={float(row['igd']):.6g}, "
                            f"time={float(row['runtime_s']):.3f}s"
                        )
                    else:
                        print(
                            f"{str(row['status']).upper()}: {str(row['error'])[:180]}"
                        )
                write_summary(
                    raw_path,
                    summary_path,
                    expected_seeds=expected_seeds,
                    references=references,
                    args=args,
                )
    except KeyboardInterrupt:
        print("\nInterrupted; completed trials are preserved.", file=sys.stderr)
        return 130

    print(f"Done. Attempted {attempted}, resumed/skipped {skipped} successful trials.")
    print(f"Raw trials: {raw_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
