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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGORITHM_DIR = ROOT / "experiments" / "generated_algorithms"
DEFAULT_FIDELITY_OUTPUT_DIR = ROOT / "evaluation_results" / "fidelity"
DEFAULT_SCALING_OUTPUT_DIR = ROOT / "evaluation_results" / "scaling"
RESULT_PREFIX_FIDELITY = "EVOCOCO_FIDELITY_RESULT="
RESULT_PREFIX_SCALING = "EVOCOCO_TRIAL_RESULT="

# --- Entry-point helpers -------------------------------------------------
# Owned by the entry-point refactor (run_optimization_fidelity_benchmark.py,
# run_computational_scalability_benchmark.py).

# --- Worker helpers ------------------------------------------------------
# Owned by the worker refactor (_fidelity_trial.py, _scaling_trial.py).
