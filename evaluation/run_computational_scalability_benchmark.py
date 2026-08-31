#!/usr/bin/env python3
"""Single-GPU computational-scalability benchmark with PlatEMO speedups.

The benchmark uses DTLZ3 and reproduces the population- and dimension-scaling
protocol from the EvoCoCo experiments while isolating every EvoX trial in a
fresh subprocess.  When a PlatEMO scaling CSV is supplied, speedup is computed
as PlatEMO time per generation divided by EvoX time per generation.
"""

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
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("_scaling_trial.py")
DEFAULT_ALGORITHM_DIR = ROOT / "experiments" / "generated_algorithms"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation_results" / "scaling"
POPULATION_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384)
DIMENSION_SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
RAW_FIELDS = (
    "algorithm",
    "algorithm_file",
    "scaling",
    "size",
    "pop_size",
    "dimension",
    "objectives",
    "execution",
    "repeat",
    "seed",
    "generations",
    "warmup",
    "recompile_limit",
    "status",
    "total_time_s",
    "time_per_generation_s",
    "igd",
    "final_population_size",
    "device",
    "error",
)
SUMMARY_FIELDS = (
    "algorithm",
    "scaling",
    "size",
    "pop_size",
    "dimension",
    "objectives",
    "generations",
    "warmup",
    "recompile_limit",
    "eager_successes",
    "compile_successes",
    "eager_mean_time_per_generation_s",
    "compile_mean_time_per_generation_s",
    "eager_mean_igd",
    "compile_mean_igd",
    "eager_compile_paired_runs",
    "paired_eager_mean_time_per_generation_s",
    "paired_compile_mean_time_per_generation_s",
    "compile_vs_eager_speedup_x",
    "evox_execution_for_speedup",
    "platemo_successes",
    "platemo_timeouts",
    "speedup_paired_runs",
    "paired_platemo_mean_time_per_generation_s",
    "paired_evox_mean_time_per_generation_s",
    "speedup_x",
    "timeout_lower_bound_pairs",
    "timeout_lower_bound_speedup_x",
)
RESULT_PREFIX = "EVOCOCO_TRIAL_RESULT="


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark generated EvoX algorithms on one GPU with torch.compile, "
            "an eager fallback, and optional PlatEMO/EvoX speedup measurement."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--algorithm-dir",
        type=Path,
        help=f"Directory of generated .py files (default: {DEFAULT_ALGORITHM_DIR})",
    )
    source.add_argument(
        "--algorithm-file",
        type=Path,
        nargs="+",
        help="One or more generated algorithm files",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Only run these filename stems, for example AGE-MOEA PREA",
    )
    parser.add_argument(
        "--scaling",
        choices=("population", "dimension", "pop", "dim"),
        default="population",
    )
    parser.add_argument(
        "--sizes",
        type=positive_int,
        nargs="+",
        help="Scale values; defaults to the paper's seven values for the selected mode",
    )
    parser.add_argument("--fixed-dimension", type=positive_int, default=100)
    parser.add_argument("--fixed-population", type=positive_int, default=1000)
    parser.add_argument("--objectives", type=positive_int, default=3)
    parser.add_argument(
        "--executions",
        choices=("compile", "eager", "both"),
        default="compile",
        help=(
            "Execution mode (default: compile). Use eager as a fallback when "
            "an algorithm cannot be compiled; use both for an additional "
            "compile-versus-eager diagnostic."
        ),
    )
    parser.add_argument("--repeats", type=positive_int, default=11)
    parser.add_argument("--generations", type=positive_int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0, help="One physical GPU index")
    parser.add_argument(
        "--timeout", type=positive_int, default=3600, help="Seconds per trial"
    )
    parser.add_argument("--recompile-limit", type=positive_int, default=512)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--platemo-csv",
        type=Path,
        help=(
            "Matching PlatEMO population- or dimension-scaling CSV. When set, "
            "summary.csv reports the paper's PlatEMO/EvoX speedup."
        ),
    )
    parser.add_argument(
        "--platemo-timeout-seconds",
        type=positive_int,
        default=3 * 60 * 60,
        help="PlatEMO timeout used for separately reported lower-bound speedups",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
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


def execution_modes(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("eager", "compile")
    return (value,)


def trial_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        row["algorithm"],
        row["scaling"],
        int(row["size"]),
        int(row["pop_size"]),
        int(row["dimension"]),
        int(row["objectives"]),
        row["execution"],
        int(row["repeat"]),
        int(row["seed"]),
        int(row["generations"]),
        int(row["warmup"]),
        int(row["recompile_limit"]),
    )


def run_trial(
    *,
    algorithm_file: Path,
    pop_size: int,
    dimension: int,
    execution: str,
    repeat: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(WORKER),
        "--algorithm-file",
        str(algorithm_file),
        "--pop-size",
        str(pop_size),
        "--dimension",
        str(dimension),
        "--objectives",
        str(args.objectives),
        "--seed",
        str(seed),
        "--generations",
        str(args.generations),
        "--warmup",
        str(args.warmup),
        "--execution",
        execution,
        "--device",
        args.device,
        "--recompile-limit",
        str(args.recompile_limit),
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


def optional_mean(values: Iterable[float]) -> float | str:
    values = list(values)
    return statistics.fmean(values) if values else ""


def parse_platemo_raw_times(value: str) -> dict[int, float | str]:
    """Parse PlatEMO's one-based RawTimes list without mixing failure states."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return {}

    parsed: dict[int, float | str] = {}
    for repeat, token in enumerate(value.split(","), start=1):
        token = token.strip()
        try:
            number = float(token)
        except ValueError:
            parsed[repeat] = token.upper()
        else:
            parsed[repeat] = number if math.isfinite(number) else "FAILED"
    return parsed


def load_platemo_times(
    path: Path | None,
    *,
    scaling: str,
    fixed_dimension: int,
    generations: int,
) -> dict[tuple[str, str, int, int, int], dict[int, float | str]]:
    """Load total runtime per repeat keyed by the matching EvoX configuration."""
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"PlatEMO CSV does not exist: {path}")

    baselines: dict[tuple[str, str, int, int, int], dict[int, float | str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"PlatEMO CSV has no header: {path}")
        normalized = {name.casefold(): name for name in reader.fieldnames}
        required = {"algorithm", "popsize", "rawtimes"}
        if scaling == "dimension":
            required.add("dimension")
        missing = sorted(required - normalized.keys())
        if missing:
            raise ValueError(f"PlatEMO CSV is missing columns: {', '.join(missing)}")

        for row in reader:
            algorithm = row[normalized["algorithm"]].strip()
            pop_size = int(row[normalized["popsize"]])
            if scaling == "population":
                size = pop_size
                dimension = fixed_dimension
            else:
                dimension = int(row[normalized["dimension"]])
                size = dimension

            row_generations = generations
            if "generations" in normalized and row[normalized["generations"]].strip():
                row_generations = int(row[normalized["generations"]])
            key = (algorithm, scaling, size, pop_size, dimension)
            if key in baselines:
                raise ValueError(f"Duplicate PlatEMO configuration in {path}: {key}")
            # Store total times. They are divided by this row's generation count
            # here so later pairing cannot accidentally use the EvoX count.
            times = parse_platemo_raw_times(row[normalized["rawtimes"]])
            baselines[key] = {
                repeat: value / row_generations if isinstance(value, float) else value
                for repeat, value in times.items()
            }
    return baselines


def write_summary(
    raw_path: Path,
    summary_path: Path,
    *,
    platemo_times: dict[tuple[str, str, int, int, int], dict[int, float | str]],
    platemo_timeout_seconds: int,
) -> None:
    rows = read_rows(raw_path)
    successful: dict[tuple[object, ...], dict[str, str]] = {}
    groups: set[tuple[object, ...]] = set()
    for row in rows:
        group = (
            row["algorithm"],
            row["scaling"],
            int(row["size"]),
            int(row["pop_size"]),
            int(row["dimension"]),
            int(row["objectives"]),
            int(row["generations"]),
            int(row["warmup"]),
            int(row["recompile_limit"]),
        )
        groups.add(group)
        if row["status"] == "success":
            successful[trial_key(row)] = row

    output_rows: list[dict[str, object]] = []
    for group in sorted(
        groups, key=lambda item: (item[0].casefold(), item[1], item[2])
    ):
        (
            algorithm,
            scaling,
            size,
            pop_size,
            dimension,
            objectives,
            generations,
            warmup,
            recompile_limit,
        ) = group
        eager = {
            (int(row["repeat"]), int(row["seed"])): row
            for row in successful.values()
            if (
                row["algorithm"] == algorithm
                and row["scaling"] == scaling
                and int(row["size"]) == size
                and int(row["pop_size"]) == pop_size
                and int(row["dimension"]) == dimension
                and int(row["objectives"]) == objectives
                and int(row["generations"]) == generations
                and int(row["warmup"]) == warmup
                and int(row["recompile_limit"]) == recompile_limit
                and row["execution"] == "eager"
            )
        }
        compiled = {
            (int(row["repeat"]), int(row["seed"])): row
            for row in successful.values()
            if (
                row["algorithm"] == algorithm
                and row["scaling"] == scaling
                and int(row["size"]) == size
                and int(row["pop_size"]) == pop_size
                and int(row["dimension"]) == dimension
                and int(row["objectives"]) == objectives
                and int(row["generations"]) == generations
                and int(row["warmup"]) == warmup
                and int(row["recompile_limit"]) == recompile_limit
                and row["execution"] == "compile"
            )
        }
        eager_compile_pairs = sorted(set(eager) & set(compiled))
        all_eager_times = [
            float(row["time_per_generation_s"]) for row in eager.values()
        ]
        all_compile_times = [
            float(row["time_per_generation_s"]) for row in compiled.values()
        ]
        eager_times = [
            float(eager[repeat]["time_per_generation_s"])
            for repeat in eager_compile_pairs
        ]
        compile_times = [
            float(compiled[repeat]["time_per_generation_s"])
            for repeat in eager_compile_pairs
        ]
        eager_mean = optional_mean(eager_times)
        compile_mean = optional_mean(compile_times)
        compile_vs_eager_speedup: float | str = ""
        if (
            isinstance(eager_mean, float)
            and isinstance(compile_mean, float)
            and compile_mean > 0
        ):
            compile_vs_eager_speedup = eager_mean / compile_mean

        # The paper compares PlatEMO against compiled EvoX. An eager result is
        # used only when no compiled trial succeeded for this configuration.
        evox_execution = "compile" if compiled else "eager" if eager else ""
        evox = compiled if compiled else eager
        evox_by_repeat = {int(row["repeat"]): row for row in evox.values()}
        baseline = platemo_times.get(
            (str(algorithm), str(scaling), size, pop_size, dimension), {}
        )
        completed_platemo = {
            repeat: value
            for repeat, value in baseline.items()
            if isinstance(value, float)
        }
        platemo_timeouts = {
            repeat for repeat, value in baseline.items() if value == "TIMEOUT"
        }
        speedup_pairs = sorted(set(evox_by_repeat) & set(completed_platemo))
        paired_platemo_times = [completed_platemo[repeat] for repeat in speedup_pairs]
        paired_evox_times = [
            float(evox_by_repeat[repeat]["time_per_generation_s"])
            for repeat in speedup_pairs
        ]
        paired_platemo_mean = optional_mean(paired_platemo_times)
        paired_evox_mean = optional_mean(paired_evox_times)
        paper_speedup: float | str = ""
        if (
            isinstance(paired_platemo_mean, float)
            and isinstance(paired_evox_mean, float)
            and paired_evox_mean > 0
        ):
            paper_speedup = paired_platemo_mean / paired_evox_mean

        timeout_pairs = sorted(set(evox_by_repeat) & platemo_timeouts)
        timeout_lower_bounds = [
            (platemo_timeout_seconds / generations)
            / float(evox_by_repeat[repeat]["time_per_generation_s"])
            for repeat in timeout_pairs
            if float(evox_by_repeat[repeat]["time_per_generation_s"]) > 0
        ]

        output_rows.append(
            {
                "algorithm": algorithm,
                "scaling": scaling,
                "size": size,
                "pop_size": pop_size,
                "dimension": dimension,
                "objectives": objectives,
                "generations": generations,
                "warmup": warmup,
                "recompile_limit": recompile_limit,
                "eager_successes": len(eager),
                "compile_successes": len(compiled),
                "eager_mean_time_per_generation_s": optional_mean(all_eager_times),
                "compile_mean_time_per_generation_s": optional_mean(all_compile_times),
                "eager_mean_igd": optional_mean(
                    float(row["igd"]) for row in eager.values()
                ),
                "compile_mean_igd": optional_mean(
                    float(row["igd"]) for row in compiled.values()
                ),
                "eager_compile_paired_runs": len(eager_compile_pairs),
                "paired_eager_mean_time_per_generation_s": eager_mean,
                "paired_compile_mean_time_per_generation_s": compile_mean,
                "compile_vs_eager_speedup_x": compile_vs_eager_speedup,
                "evox_execution_for_speedup": evox_execution if baseline else "",
                "platemo_successes": len(completed_platemo) if baseline else "",
                "platemo_timeouts": len(platemo_timeouts) if baseline else "",
                "speedup_paired_runs": len(speedup_pairs) if baseline else "",
                "paired_platemo_mean_time_per_generation_s": paired_platemo_mean,
                "paired_evox_mean_time_per_generation_s": paired_evox_mean,
                "speedup_x": paper_speedup,
                "timeout_lower_bound_pairs": len(timeout_pairs) if baseline else "",
                "timeout_lower_bound_speedup_x": optional_mean(timeout_lower_bounds),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_path.replace(summary_path)


def format_result(row: dict[str, object]) -> str:
    if row["status"] != "success":
        return f"{str(row['status']).upper()}: {str(row['error'])[:180]}"
    return (
        f"{float(row['time_per_generation_s']):.6f}s/gen, IGD={float(row['igd']):.6g}"
    )


def main() -> int:
    args = parse_args()
    args.scaling = {"pop": "population", "dim": "dimension"}.get(
        args.scaling, args.scaling
    )
    try:
        algorithms = discover_algorithms(args)
        platemo_times = load_platemo_times(
            args.platemo_csv,
            scaling=args.scaling,
            fixed_dimension=args.fixed_dimension,
            generations=args.generations,
        )
        if args.device == "cuda":
            check_cuda(args.gpu)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    sizes = args.sizes or (
        list(POPULATION_SIZES)
        if args.scaling == "population"
        else list(DIMENSION_SIZES)
    )
    modes = execution_modes(args.executions)
    output_dir = args.output_dir.expanduser().resolve()
    raw_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.csv"
    completed_keys = {
        trial_key(row) for row in read_rows(raw_path) if row.get("status") == "success"
    }
    total = len(algorithms) * len(sizes) * args.repeats * len(modes)
    skipped = 0
    attempted = 0

    print(f"Algorithms: {len(algorithms)}")
    print(f"Scaling: {args.scaling}; sizes: {sizes}")
    print(f"Executions: {', '.join(modes)}; repeats: {args.repeats}")
    print(
        f"Device: {args.device}{f' (physical GPU {args.gpu})' if args.device == 'cuda' else ''}"
    )
    print(f"Output: {output_dir}")
    if args.platemo_csv:
        print(f"PlatEMO baseline: {args.platemo_csv.expanduser().resolve()}")

    try:
        for algorithm_file in algorithms:
            for size in sizes:
                if args.scaling == "population":
                    pop_size, dimension = size, args.fixed_dimension
                else:
                    pop_size, dimension = args.fixed_population, size
                for repeat in range(1, args.repeats + 1):
                    seed = args.seed_base + repeat - 1
                    for execution in modes:
                        key = (
                            algorithm_file.stem,
                            args.scaling,
                            size,
                            pop_size,
                            dimension,
                            args.objectives,
                            execution,
                            repeat,
                            seed,
                            args.generations,
                            args.warmup,
                            args.recompile_limit,
                        )
                        if key in completed_keys:
                            skipped += 1
                            continue

                        attempted += 1
                        print(
                            f"[{attempted + skipped}/{total}] {algorithm_file.stem} "
                            f"{args.scaling}={size} rep={repeat} {execution} ... ",
                            end="",
                            flush=True,
                        )
                        result = run_trial(
                            algorithm_file=algorithm_file,
                            pop_size=pop_size,
                            dimension=dimension,
                            execution=execution,
                            repeat=repeat,
                            seed=seed,
                            args=args,
                        )
                        row: dict[str, object] = {
                            "algorithm": algorithm_file.stem,
                            "algorithm_file": str(algorithm_file),
                            "scaling": args.scaling,
                            "size": size,
                            "pop_size": pop_size,
                            "dimension": dimension,
                            "objectives": args.objectives,
                            "execution": execution,
                            "repeat": repeat,
                            "seed": seed,
                            "generations": args.generations,
                            "warmup": args.warmup,
                            "recompile_limit": args.recompile_limit,
                            "total_time_s": "",
                            "time_per_generation_s": "",
                            "igd": "",
                            "final_population_size": "",
                            "device": args.device,
                            "error": "",
                            **result,
                        }
                        append_row(raw_path, row)
                        print(format_result(row), flush=True)
                    write_summary(
                        raw_path,
                        summary_path,
                        platemo_times=platemo_times,
                        platemo_timeout_seconds=args.platemo_timeout_seconds,
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
