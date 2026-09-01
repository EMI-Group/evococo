# backend — EvoCoCo FastAPI Service

## Intent
FastAPI backend for the EvoCoCo pipeline: takes a MATLAB algorithm (PlatEMO classdef or custom/standalone) via WebSocket and runs a multi-agent LLM orchestration that translates/optimizes it into PyTorch. Pipeline stages: Analyst → RAG rule selection → Architect → Tournament (NUM_BRANCHES parallel Coder → Static Fixer → Runtime Fixer branches) → Judge (best branch wins) → final code with a performance-stats header. Python 3.x, async (asyncio + FastAPI + WebSocket).

## API Surface

### `config.py` — module-level constants (LOAD-BEARING: imported by other modules, incl. outside backend)
`NUM_BRANCHES`, `ACTIVE_LLM_PROVIDER`, `REASONING_EFFORT`, `LLM_PROVIDERS`, `STRATEGIES_SHORT`, `STRATEGIES_FULL`, `IGNORE_RUFF_CODES`, `MAX_RETAINED_WORKSPACES`, `BASE_WORKSPACE_DIR`, `HISTORY_DIR`, `PROMPTS_DIR`, `DATABASE_DIR`, `RULES_DB_PATH`, `GLOBAL_SPEC_PATH`, `ENV_PATH`, `ENV_EXAMPLE_PATH`, `BACKEND_DIR`, `PROJECT_ROOT`.
External importers: `experiments/batch_translate.py` (NUM_BRANCHES, REASONING_EFFORT), `experiments/one_shot.py` (LLM_PROVIDERS). At import time config auto-copies `.env.example` → `.env` if missing (with an encoding-safe print fallback) and loads dotenv.

### `generator.py` — LLM I/O
- `generate_llm_response(prompt_filename: str, response_model: type[BaseModel] | None = None, metrics_out: dict | None = None, **kwargs)` — loads the prompt template from `prompts/`, calls the OpenAI-compatible API (provider from config), returns either a pydantic model (structured mode), cleaned text (default mode), or an error string `"Error generating response: ..."` on failure (it never raises).
- Module constants: `ACTIVE_PROVIDER`, `API_KEY`, `BASE_URL`, `MODEL_NAME`, `TEMPERATURE`, `LLM_CONNECT_TIMEOUT`, `LLM_READ_TIMEOUT`, `LLM_NETWORK_RETRY_INTERVAL`, `PROMPT_DIR` (batch_translate.py imports ACTIVE_PROVIDER, MODEL_NAME).
- `_wait_for_provider_network()` pauses DeepSeek calls while the endpoint is unreachable (retry loop).

### `executor.py` — code execution / static checks
- `setup_workspace(session_id: str) -> str`
- `cleanup_workspace(session_id: str)` — keeps the workspace, triggers global cleanup
- `cleanup_old_workspaces(base_dir: str, max_retained: int = MAX_RETAINED_WORKSPACES)`
- `check_syntax_with_ruff(code: str, session_id: str | None = None) -> tuple[bool, str]` — needs `ruff` on PATH or next to `sys.executable`
- `execute_code(code_str: str, session_id: str | None = None, filename="algo_script.py") -> (success, stdout, stderr)`
- `execute_code_trial(code_str: str, session_id: str, filename="algo_script.py") -> dict` — dict keys: success, stdout, stderr, last_igd, igd_history, has_nan, is_converging, duration, exec_time
- `parse_igd_from_stdout(stdout) -> list[float]`, `parse_exec_time_from_stdout(stdout) -> float`
External importer: `evaluation/run_migration_reliability_benchmark.py` (check_syntax_with_ruff, execute_code_trial).

### `engine.py` — pipeline orchestration
- `run_pipeline(matlab_code: str, status_callback)` — async, runs the full pipeline; returns `perf_stats` dict. `status_callback(type_, title, message, step_id=None, extra_data=None, is_success=None, icon=None)` is invoked as an awaitable with positional call sites like `status_callback("log", title, msg, step_id=...)` — the callback must accept this exact signature (main.py's send_update does).
- `run_single_branch_lifecycle(branch_idx, base_session_id, matlab_code, blueprint_md, constraints_str, asset_lib_content, few_shot_content, run_dir, status_callback)` — one tournament branch.
- Models: `JudgeResult` (reasoning, winning_branch_id, code), `RagResult` (selected_bug_numbers).
- Re-exports config names GLOBAL_SPEC_PATH, HISTORY_DIR, MAX_RETAINED_WORKSPACES, PROMPTS_DIR, RULES_DB_PATH (`# noqa: F401`) for backward-compat; consumers now live in storage.py.
External importer: `experiments/batch_translate.py` (run_pipeline).

### `storage.py` — data loading + artifact I/O (extracted from engine)
- `load_rag_db()` → list ([] if file missing/unparseable — intentional fallback)
- `load_global_spec()` / `load_resource(filename)` → str ("" if missing/unreadable — intentional fallback)
- `ensure_history_dir(algo_name="UnknownAlgo") -> str` — creates `run_history/<timestamp>_<algo_name>`, triggers cleanup
- `save_artifact(run_dir, filename, content)` — best-effort, logs + skips on failure

### `stats.py` — perf-stats helpers (extracted from engine)
- `aggregate_stage_metrics(perf_stats)` — RESETS totals to 0 then recomputes token/LLM-time totals from all stage + branch metrics
- `format_stats_header(perf_stats, judge_result=None) -> str` — builds the `# ====...` header comment prepended to final code; judge lines only when judge_result is not None

### `main.py` — FastAPI entrypoint
- `GET /` → `{"status": "EvoCoCo Backend is Running"}`
- `WS /ws` — receives JSON `{"code": "<matlab>"}`, runs `run_pipeline` with a hoisted `send_update` callback; structured logging via `logging.getLogger("evococo.backend.main")`.
- Windows: sets Proactor event-loop policy when `sys.platform == "win32"`.

## Constraints
- Public API stability is load-bearing: config names above, `run_pipeline(matlab_code, status_callback)`, `check_syntax_with_ruff`/`execute_code_trial` from executor, `ACTIVE_PROVIDER`/`MODEL_NAME` from generator, and the `execute_code_trial` dict keys are consumed by code OUTSIDE ./backend (experiments/, evaluation/) — never rename or change semantics.
- No new dependencies (requirements.txt is fixed). For async file I/O use `asyncio.to_thread`.
- Keep each file under ~1000 lines.
- Intentional fallback patterns that MUST be preserved: load_rag_db → [], load_global_spec/load_resource → "", save_artifact best-effort skip, generate_llm_response → error string, judge failure → fallback to min-IGD successful branch (default results[0]), stats save never crashes the pipeline. Broad `except Exception` blocks at these boundaries carry `# noqa: BLE001` + an explanatory comment — keep them commented when touched.
- `status_callback` is invoked positionally by engine (e.g. `status_callback("log", "SYS", msg)` and with keywords `step_id=`, `extra_data=`, `is_success=`) — a callback whose first parameter is `websocket` would break the engine contract.
- Workspaces live under `BASE_WORKSPACE_DIR` (temp_workspace/); run artifacts under `HISTORY_DIR` (run_history/); MAX_RETAINED_WORKSPACES caps retained dirs.

## Routing Table
- `./backend/prompts/` → LLM prompt templates (0_global_spec.md … 7_selector.md, reference_*.md, resources_*.md). Content assets — do not rewrite prompt wording; `load_resource`/`generate_llm_response` reference them by filename.
- `./backend/database/` → `rag_db.json`, the RAG rules DB loaded by `load_rag_db()` (data asset; id format `bug #<n>`).

## Known Issues
- `check_syntax_with_ruff` replaces the absolute temp file path with "script.py" in ruff output via string `.replace()`. On Windows, if ruff reports the path with different slash conventions than the constructed `file_path`, the replace misses and the absolute path leaks into the error string sent to the LLM (cosmetic; pre-existing).
- `generate_llm_response` returns an error string instead of raising — callers that `await` it and treat the result as code must tolerate a non-code error string (pre-existing design; engine's branch lifecycle handles it by attempting static-fix).
- `config.py` prints a startup notice (emoji) when auto-copying `.env.example` — has a UnicodeEncodeError fallback so imports never crash on non-UTF-8 (GBK) consoles.

## Notes for Agents
- The static-check step shells out to `ruff`; if ruff is missing, `check_syntax_with_ruff` reports "not found" and engine skips static checks (by design).
- Ruff lint gate for this repo: run `python -m ruff check backend` — must stay clean. `python -m compileall -q backend` and `python -c "from backend import engine, executor, generator, config, main"` are the standard import sanity checks (the import check needs a console that can encode the config notice, or PYTHONIOENCODING=utf-8 on Windows).
- engine.py was split from 842 → ~680 lines by extracting storage.py (I/O) and stats.py (perf aggregation/formatting); further splitting is possible (e.g. branch lifecycle) if it grows again.

## Design Decisions
- engine.py split into `storage.py` + `stats.py` (2025 repo-wide quality pass) to separate I/O fallbacks and the large stats-header formatter from orchestration; behaviors verified equivalent (A/B checks against the original inline logic).
- Stats aggregation intentionally resets totals to 0 then recomputes from stage metrics (original behavior kept — do not "optimize" into incremental accumulation without checking all consumers).
- Broad exception catches at pipeline boundaries are deliberate resilience (an LLM/network failure in one stage must not lose the whole run) — they log and fall back, they do not silently pass.
