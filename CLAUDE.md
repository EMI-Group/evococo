# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the backend server** (from project root):
```bash
python -m uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

**Open the frontend**: Open `frontend/index.html` directly in a browser. The backend must be running first at `http://localhost:8000`.

**Run the batch translator** to convert MATLAB files:
```bash
python experiments/batch_translate.py --input_dir ./experiments/matlab_code
```

**Run the benchmark evaluator** against a directory of generated Python files:
```bash
python evaluation/benchmark.py --dir ./experiments/benchmark_results
```

**Lint code with Ruff** (same codes ignored as the pipeline):
```bash
ruff check --ignore E501,E402,E722,E731,E741,E701,E702,E703,I <file>
```

## Architecture

EvoCoCo translates MATLAB evolutionary algorithm code into tensorized PyTorch/EvoX implementations using a multi-agent LLM pipeline. The frontend sends MATLAB code over WebSocket → backend orchestrates a 7-stage tournament → best Python implementation returned.

### Backend modules (`backend/`)

- **`main.py`** — FastAPI app with a single WebSocket endpoint (`/ws`) that receives MATLAB code and streams progress updates back to the client.
- **`engine.py`** — Orchestrates the full 7-stage pipeline (`run_pipeline`). Stages 1–3 run sequentially; Stage 4 (Coder) fans out into `NUM_BRANCHES=6` parallel branches with different tensorization strategies; Stages 5–6 fix errors per branch; Stage 7 selects the winner by IGD metric.
- **`generator.py`** — Async OpenAI-compatible LLM client wrapper. Loads prompt templates from `backend/prompts/`, performs variable substitution, and calls the configured model.
- **`executor.py`** — Code execution sandbox: AST syntax check → Ruff static check → isolated `exec()` trial with PyTorch/EvoX. Manages temporary workspaces under `temp_workspace/`.
- **`config.py`** — Central configuration: `NUM_BRANCHES`, the 6 `STRATEGIES_SHORT/FULL`, Ruff ignore codes, workspace/history paths.

### 7-Stage Pipeline

Each stage corresponds to a prompt template in `backend/prompts/`:

| Stage | File | Role |
|-------|------|------|
| 1 | `1_analyst.md` | Deconstructs MATLAB into logical blocks |
| 2 | `2_rag_selector.md` | Selects applicable bug-fix rules from `database/rag_db.json` |
| 3 | `3_architect.md` | Designs the tensorization blueprint |
| 4 | `4_coder.md` | Generates Python code (×6 branches, each with a different strategy) |
| 5 | `5_static_fixer.md` | Fixes Ruff/AST errors |
| 6 | `6_runtime_fixer.md` | Fixes execution errors |
| 7 | `7_selector.md` | Picks the best branch by IGD convergence |

### RAG Knowledge Base (`backend/database/rag_db.json`)

Contains ~20 named bug patterns (e.g., "Bug #1 (Integer Sentinel)", "Bug #7 (SDR Dominance Helper)"). The RAG Selector stage matches relevant rules to the input algorithm; matched rules are injected into the Coder prompt. Some rules are marked `always_apply`.

### WebSocket message protocol

The backend sends JSON messages via the `send_update()` callback:
- `type`: `"init"`, `"log"`, `"step_start"`, `"result_ir"`, `"step_done"`, `"fatal"`
- Log messages include tags: `SYS`, `INFO`, `PERF`, `ERR`, `FAIL`

### LLM configuration (`.env`)

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
OPENAI_TEMPERATURE=0.2
```

The client in `generator.py` is OpenAI-SDK-compatible, pointed at a LiteLLM proxy by default.

### Generated code conventions

All generated algorithms must:
- Subclass `evox.Algorithm`
- Accept `(pop_size, n_objs, lb, ub)` constructor signature
- Use PyTorch tensors throughout — no Python loops over populations
- Avoid `.item()` / `.tolist()` (JIT-compliance requirement)

The global specification enforcing these rules lives in `backend/prompts/0_global_spec.md`.

### Run artifacts

- `run_history/` — Timestamped directories (e.g., `20260413_022603_NSGAIISDR/`) with per-branch code and logs. Session IDs include the algorithm name.
- `temp_workspace/` — Isolated execution dirs per branch, auto-cleaned after `MAX_RETAINED_WORKSPACES=150` total.
