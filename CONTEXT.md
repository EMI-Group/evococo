# EvoCoCo — Repo Root

## Intent
EvoCoCo translates MATLAB/PlatEMO multi-objective evolutionary algorithms (MOEAs) into high-performance PyTorch/EvoX code via an LLM multi-agent pipeline (backend), then benchmarks the generated algorithms (evaluation) and runs batch/one-shot translation experiments (experiments). Python 3.x monorepo with a FastAPI backend, CLI benchmark/experiment runners, and a web UI (frontend).

## API Surface (repo-level)
- `requirements.txt` — fixed dependency set; NO new dependencies allowed.
- `.ruff.toml` — `extend-exclude = ["experiments/generated_algorithms"]`; plain `python -m ruff check .` must print "All checks passed!" (generated output correctly excluded).
- Repo-wide validation gate (currently GREEN at HEAD `59ce4cc`): `compileall` on backend/evaluation/experiments, `ruff check` clean, backend import smoke, `--help` on all evaluation/experiments entry points, clean `git status`.

## Constraints
- `experiments/generated_algorithms/` is GENERATED OUTPUT — never edit; always excluded from lint (root `.ruff.toml` and explicit `--exclude 'experiments/generated_algorithms/**'`).
- `experiments/matlab_code/` is input data — read-only.
- Keep each file under ~1000 lines (current: backend/engine.py 681, backend/storage.py 97, backend/stats.py 128, evaluation/_common.py 243, experiments/_common.py 106).
- Cross-directory imports are load-bearing: experiments/ and evaluation/ import backend modules (`backend.engine.run_pipeline`, `backend.executor.*`, `backend.config` constants) — never rename/change their public API.
- `torch`/`evox` are NOT installed in dev worktrees and must NOT be installed for validation; `--help`, compileall, ruff, and backend imports work without them.

## Routing Table
- `./backend` → FastAPI service: `config.py`, `engine.py`, `executor.py`, `generator.py`, `storage.py`, `stats.py`, `main.py`; `prompts/` (LLM templates), `database/` (rag_db.json). Cross-imported by evaluation/ and experiments/.
- `./evaluation` → benchmark CLIs: `run_migration_reliability_benchmark.py`, `run_optimization_fidelity_benchmark.py`, `run_computational_scalability_benchmark.py`; shared `_common.py`, `_benchmark_problems.py`, internal `_*_trial.py` workers.
- `./experiments` → `batch_translate.py`, `batch_one_shot.py`, `one_shot.py`, `_common.py`; `generated_algorithms/` (read-only artifacts), `matlab_code/` (input data).
- `./frontend` → web UI (seed editor; `initialScript` preserved verbatim).
- `./docs` → documentation.

## Known Issues
- `.env` is auto-created from `.env.example` on first `backend.config` import (prints a warning; has a GBK-safe fallback). `.env` is gitignored — `git status` stays clean.
- First `backend.config` import on a GBK (Chinese-Windows) console can crash with `UnicodeEncodeError` if `.env` doesn't exist yet — one-time; workaround `PYTHONIOENCODING=utf-8`.
- ruff does not support `-g '!...'` exclusion patterns on this version; use `--exclude 'experiments/generated_algorithms/**'`.

## Notes for Agents (worktree tooling gotchas)
- `run_powershell` output capture quirk in this environment: **python stdout and PowerShell cmdlet output are NOT captured** by the tool wrapper (only native `git` output and error/progress records appear). Workaround: have python write results to a file inside the repo and read it with `read_file`, or observe via `git status`/`git log`.
- PowerShell 5.1 `>` / `*>` redirects produce UTF-16 files — decode via python (`raw.decode('utf-16')`) before reading.
- Validation one-liners that must stay green: `python -m compileall -q backend evaluation experiments`; `python -m ruff check backend evaluation experiments --exclude 'experiments/generated_algorithms/**'`; `python -c "from backend import engine, executor, generator, config, main; from backend.storage import load_rag_db; from backend.stats import aggregate_stage_metrics"`; `--help` on all evaluation/experiments entry points.
