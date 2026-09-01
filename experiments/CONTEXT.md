# Experiments

## Intent
Batch runners and a one-shot baseline for translating MATLAB multi-objective evolutionary algorithms (MOEAs) into high-performance PyTorch/EvoX code via LLM. `one_shot.py` is the shared translation core (single LLM call with a fixed prompt template); `batch_translate.py` runs the full EvoCoCo pipeline (`backend.engine.run_pipeline`); `batch_one_shot.py` fans out one-shot translations across algorithms × repeats. `_common.py` centralizes shared path/env/file I/O helpers.

## API Surface
- `batch_translate.py`: `--input_dir` (required), `--output_dir` (default `benchmark_results`), `--repeats` (default 5), `--repeat-concurrency` (default 1). Writes a `batch_manifest.json` plus per-run `<algo>_runN.py` and `<algo>_runN_stats.json` files into the output dir.
- `batch_one_shot.py`: `--input_dir` (default `experiments/all_matlab_algorithms`), `--output_dir` (default `experiments/baseline-48`), `--repeats` (default 5), `--concurrency` (default 5). Writes `<algo>_runN.py` per run; skips outputs that already exist.
- `one_shot.py`: `--input`/`-i` (required), `--output`/`-o` (default `experiments/baselines/one_shot_output.py`). Single-file translation; prints `[Success]`/`[Failed]`.
- `_common.py` (no CLI): constants `MATLAB_EXTENSIONS` and `EXCLUDED_INPUT_FILES`; helpers `project_root()`, `ensure_repo_root_on_path()`, `load_dotenv_from_root()`, `setup_litellm_env()`, `set_win32_event_loop_policy()`, `is_matlab_source()`, `read_matlab_source()`, `algorithm_name()`, `write_json()`, `read_json()`.

## Constraints
- `generated_algorithms/` is GENERATED OUTPUT — never edit; excluded from linting via the root `.ruff.toml` (`extend-exclude`).
- `matlab_code/` is input data — treat as read-only.
- Keep CLI flags, defaults, help text, and user-facing prints stable (README documents `batch_translate`).
- No new dependencies (stdlib + already-imported `dotenv`/`openai` only).
- File I/O in async functions must go through `asyncio.to_thread(...)` (ruff ASYNC230).
- Batch scripts import `backend.*` after `ensure_repo_root_on_path()`; `batch_one_shot.py` must call `setup_litellm_env()` before importing `one_shot` (it constructs an `AsyncOpenAI` client at import time).

## Routing Table
- `generated_algorithms/` → 48 generated EvoX algorithm artifacts (read-only reference data).
- `matlab_code/` → MATLAB source input data (may not exist yet).

## Known Issues
- Three intentional broad `except Exception` fallbacks (each documented with a rationale comment; ruff does not flag them because they call `logger.exception`):
  - `batch_one_shot.py` worker retry loop: transient network/API errors are retried up to 5 attempts.
  - `batch_translate.py` `process_file`: a failed pipeline run is recorded as failed stats and the batch continues.
  - `one_shot.py` `one_shot_translate_with_metrics`: documented contract to never raise; returns error metrics instead.
- `python -m ruff check experiments -g '!...'` is not supported by the installed ruff; use `--exclude 'experiments/generated_algorithms/**'` or rely on the root config.
- On Chinese-Windows GBK consoles, the FIRST import of `backend.config` (via any runner) crashes with `UnicodeEncodeError` if `.env` doesn't exist yet: `backend/config.py` auto-creates `.env` from the template while printing a `⚠️` emoji that GBK can't encode. One-time issue — once `.env` exists the print is skipped. Workaround if needed: run with `PYTHONIOENCODING=utf-8`. (`backend/config.py` is outside this directory's scope.)

## Test Strategy
- `python -m compileall -q experiments` (whole dir, includes generated artifacts).
- `python -m ruff check experiments` (root config excludes `generated_algorithms`).
- `python experiments/<script> --help` smoke tests.
- NEVER run actual translations (costly LLM calls) unless explicitly asked.
