"""Shared helpers for the EvoCoCo evaluation benchmarks.

Centralizes logic used by more than one benchmark entry point (``run_*.py``)
or isolated trial worker (``_*_trial.py``).  Organization:

- Shared constants: default paths and the worker result-marker prefixes that
  entry points and workers agree on.
- Entry-point helpers: argparse validators, algorithm discovery, GPU checks,
  CSV reading/writing, and worker-subprocess invocation.
- Worker helpers: loading generated algorithm modules and common trial setup.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGORITHM_DIR = ROOT / "experiments" / "generated_algorithms"
DEFAULT_FIDELITY_OUTPUT_DIR = ROOT / "evaluation_results" / "fidelity"
DEFAULT_SCALING_OUTPUT_DIR = ROOT / "evaluation_results" / "scaling"
RESULT_PREFIX_FIDELITY = "EVOCOCO_FIDELITY_RESULT="
RESULT_PREFIX_SCALING = "EVOCOCO_TRIAL_RESULT="

# --- Entry-point helpers -------------------------------------------------
# Shared by run_optimization_fidelity_benchmark.py and
# run_computational_scalability_benchmark.py.


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


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


def append_row(path: Path, row: dict[str, object], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def run_worker_process(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    result_prefix: str,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {"status": "timeout", "error": f"Exceeded {timeout}s: {error}"}

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(result_prefix):
            result = json.loads(line[len(result_prefix) :])
            result.update({"status": "success", "error": ""})
            return result

    detail = (completed.stderr or completed.stdout).strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    return {
        "status": "failed",
        "error": f"Worker exited with code {completed.returncode}: {detail}",
    }


# --- Worker helpers ------------------------------------------------------
# Shared by _fidelity_trial.py and _scaling_trial.py.


def load_algorithm_class(path: Path, evox_module, *, module_prefix: str) -> type:
    """Load a generated algorithm module and return its EvoX ``Algorithm`` subclass.

    The module is registered in ``sys.modules`` under a name that is unique
    per process and per caller (via ``module_prefix``), so modules loaded by
    concurrent processes (each trial runs in its own process) never collide.
    A class named after the file stem is preferred; otherwise the module is
    scanned for any ``Algorithm`` subclass.
    """
    module_name = f"{module_prefix}{path.stem.replace('-', '_')}_{os.getpid()}"
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
            # Intentional duck-typing guard: module attributes that are not
            # classes (functions, builtins, lambdas) are not valid
            # ``issubclass`` arguments.
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
            # Same duck-typing guard as above: skip non-class attributes.
            continue

    raise RuntimeError(f"No EvoX Algorithm subclass found in {path}")


def setup_torch_device(device: str, seed: int) -> None:
    """Validate ``device``, make it the default, and seed every RNG."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in this process"
        )
    torch.set_default_device(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def instantiate_algorithm(
    algorithm_class: type, problem: object, pop_size: int, **kwargs: object
) -> object:
    """Instantiate an algorithm, falling back to ``(problem=..., pop_size=...)``.

    Generated algorithms may reject the generic ``lb``/``ub``/``n_objs``
    keyword set, so fall back to the problem-based signature.  If the
    fallback also fails, the original ``TypeError`` is re-raised.
    """
    try:
        return algorithm_class(**kwargs)
    except TypeError as standard_error:
        try:
            return algorithm_class(problem=problem, pop_size=pop_size)
        except TypeError:
            raise standard_error


def finite_fitness_rows(fit: torch.Tensor) -> torch.Tensor:
    """Drop rows containing non-finite values; raise if no finite rows remain."""
    fit = fit[fit.isfinite().all(dim=1)]
    if fit.shape[0] == 0:
        raise RuntimeError("The final fitness tensor contains no finite rows")
    return fit
