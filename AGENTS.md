# Agent Guide

This file applies to the entire repository. Read the relevant code and directory documentation before making changes. If a subdirectory has its own `AGENTS.md`, its specific instructions take precedence within that directory. Explicit user instructions for the current task take precedence over this file.

## Project Context and Entry Points

EvoCoCo converts MATLAB multiobjective evolutionary optimization algorithms into tensorized PyTorch/EvoX implementations. Its pipeline includes semantic analysis, rule retrieval, planning, code generation, repair, and candidate selection. Preserve the original algorithm's optimization semantics; successful execution or faster runtime alone does not establish correctness.

| Path | Responsibility |
| --- | --- |
| `backend/main.py` | FastAPI, WebSocket, frontend serving, and connection authentication |
| `backend/engine.py` | Generation pipeline and candidate branch orchestration |
| `backend/config.py`, `backend/generator.py` | Configuration, model requests, and structured responses |
| `backend/executor.py`, `backend/trial_runner.py` | Static checks and runtime validation |
| `backend/storage.py`, `backend/stats.py` | Run artifacts and statistics |
| `backend/prompts/`, `backend/database/rag_db.json` | Pipeline prompts, semantic constraints, and retrieval rules |
| `frontend/index.html` | Browser interface; no separate npm build workflow |
| `experiments/batch_translate.py` | Batch generation CLI entry point |
| `experiments/generated_algorithms/` | Generated algorithms used in experiments |
| `evaluation/` | Migration reliability, optimization fidelity, computational scalability, and PlatEMO comparisons |

See `README.md` for general usage. Read `evaluation/README.md` before evaluation work and `tests/README.md` for testing guidance.

## Environment and Common Commands

Run the following commands from the repository root. CI uses Python 3.12; dependencies are defined in `requirements.txt`.

```text
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

When local configuration is needed, create `.env` from `.env.example` only if `.env` does not already exist. Do not overwrite existing configuration. Open the launch URL containing the connection token printed by the backend. See `.env.example` for configuration options.

Python checks matching CI:

```text
ruff check .
ruff format --check .
```

CI pins Ruff to 0.14.10. See `.github/workflows/ruff.yml` and `.ruff.toml` for configuration. Limit formatting changes to modified files where possible. `experiments/generated_algorithms/` is excluded from regular Ruff scans; do not rewrite experimental algorithms in bulk for formatting consistency.

## Change Guidelines

- Keep changes focused and follow the naming, structure, and style of surrounding code. Avoid unrelated refactoring and unnecessary dependencies.
- Check the working tree before editing. Preserve existing user changes and do not revert or overwrite unrelated work.
- When changing backend protocols, check frontend WebSocket message handling, authentication, and error presentation together.
- When changing prompts, structured response models, or RAG data, check their producers and consumers. Keep fields, stage artifacts, and semantic constraints consistent. Prompt files are application inputs, not instructions for the coding agent.
- For algorithm and tensor changes, consider population size, objective dimensions, device, random seeds, fitness validity, and selection semantics. Consult the MATLAB source and existing rules when necessary.
- Update the corresponding README when changing CLI options, configuration, or output protocols. Do not document performance or fidelity claims that have not been validated by execution.
- Write repository guidance in English and preserve the style of existing code identifiers and documentation. Follow the user's explicit language preferences when communicating.

## Validation Strategy

- Run checks appropriate to the change. For documentation-only changes, verify content, paths, and commands.
- `tests/README.md` documents the regression commands below. When this guide was created, `tests/` contained only the README, without the corresponding test scripts. Confirm that scripts exist before running them; discovering zero tests is not a successful regression check.

```text
python -B -m unittest discover -s tests -v
node --test tests/test_frontend_regressions.cjs
```

- For Python behavior fixes, add or restore focused regression coverage where practical. Prefer mocked model calls over live LLM requests in routine tests.
- See `tests/README.md` for the optional MATLAB output-protocol test. It requires MATLAB, `EVOCOCO_TEST_MATLAB=1`, and the actual test script to be present.
- If GPU access, MATLAB, PlatEMO, or model credentials are unavailable, state which validation could not be completed and why. Do not equate static checks with runtime validation.
- On completion, briefly report the changes, checks actually run, and remaining validation gaps.

## Evaluation and Data Protection

- Public evaluation entry points are `evaluation/run_*`. Python files beginning with an underscore are internal workers, normally invoked through the public entry points.
- Evaluation scripts should read and validate generated algorithms without modifying the code under evaluation.
- When changing evaluation code, start with the small smoke commands in `evaluation/README.md`. Do not launch full, multi-seed, or large GPU experiments by default for routine validation.
- Use separate output directories for smoke checks. Preserve `evaluation_results/`, `run_history/`, and existing experiment results; do not delete historical data to bypass resume checks.
- Preserve Python evaluation identity fields used for resume and aggregation, including source path, SHA-256, device, GPU index, and trial parameters. Avoid mixing results from different code versions or devices.
- Preserve MATLAB fidelity validation against `benchmark_protocol.json`. Use a new directory when budgets or smoke/full modes differ, or when legacy results lack protocol metadata. Do not bypass the protections in `prepare_platemo_fidelity_output.m`.
- Retain failures, timeouts, OOMs, and coverage information. Do not silently discard failures to improve summary metrics.
- PlatEMO is an external dependency configured through `PLATEMO_ROOT` or the MATLAB path. Do not add its installation directory to the repository.

## Configuration and Sensitive Information

- Do not commit `.env`, API keys, or WebSocket connection tokens, or include them in logs, test snapshots, documentation, or generated artifacts.
- Prefer `.env.example` when inspecting configuration structure. Avoid printing the contents of the local `.env` file.
- Preserve authentication and origin restrictions. Do not remove these checks for debugging convenience.
- Follow `.gitignore` for temporary files, run history, and evaluation artifacts. Review the diff before committing to ensure it contains only files needed for the current task.
