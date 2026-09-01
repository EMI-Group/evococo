# EvoCoCo Repository Optimization — Technical Report

**Date:** 2026-02
**Scope:** Full-repository code-quality pass on the `dev-genesis` codebase (backed by `main` lineage).
**Verdict:** All validation gates GREEN. 92 lint findings eliminated (51 backend + 16 evaluation + 25 experiments), 4 shared modules extracted, dead code and CDN dependencies removed, and a frontend security/lifecycle hardening completed — all without changing any public API, CLI flag, or benchmark semantics.

---

## 1. Repository Overview

EvoCoCo is a multi-agent framework that automatically tensorizes evolutionary multiobjective optimization (EMO) software: it takes concrete MATLAB implementations (e.g. PlatEMO MOEAs), treats the underlying optimization mechanism as the *semantic constraint*, and generates GPU-oriented PyTorch/EvoX implementations through a coordinated multi-agent pipeline.

The repository is a Python monorepo with four functional areas:

| Area | Role | Entry points |
|---|---|---|
| `backend/` | FastAPI service; 7-stage tournament pipeline | `main.py` (`/`, `/ws`), `engine.run_pipeline` |
| `evaluation/` | Reliability / optimization-fidelity / computational-scalability benchmarks | 3 `run_*.py` CLIs |
| `experiments/` | Batch & one-shot MATLAB→Python translation runners | 3 CLI scripts |
| `frontend/` | Dependency-free browser UI over the WebSocket API | `index.html` |

`experiments/generated_algorithms/` contains 48 generated algorithm artifacts (read-only, excluded from lint).

---

## 2. Baseline Assessment (Pre-Optimization)

Established by each sub-team before any change:

| Check | backend | evaluation | experiments |
|---|---|---|---|
| `compileall` | PASS | PASS | PASS |
| `ruff check` | **51 findings** | **16 findings** | **25 findings** |
| Largest file | `engine.py` — 842 lines | `run_computational_scalability_benchmark.py` — 733 lines | `one_shot.py` — 369 lines |

**Headline pre-existing defects found:**

1. **`backend/engine.py` (842 lines)** — a single module mixing data loading, artifact I/O, per-branch pipeline orchestration, performance-stats accumulation, a ~80-line stats-header formatter, and judge-result handling.
2. **Bare/blank exception swallowing** — 15 `BLE001` blind `except` sites in backend, 3 in experiments; several silently dropped real failures (e.g. missing files, malformed JSON, failed artifact writes).
3. **`UnicodeEncodeError` crash on GBK (Chinese-Windows) consoles** — `backend/config.py` printed a `⚠️` emoji when auto-creating `.env` on first import; this crashed `python -m compileall` and `run_migration_reliability_benchmark.py --help` on such consoles.
4. **Frontend constraint violations** — 6 CDNs (Tailwind, marked, MathJax, Font Awesome, highlight.js, Google Fonts) made the UI fail offline/`file://`; backend messages were interpolated into `innerHTML` unescaped (**XSS**); WebSocket reconnect had no backoff and no dedup (double-connect risk); `stopPipeline()` was contradicted by the auto-reconnect `onclose` handler.
5. **Triplicated logic across sibling scripts** — `sys.path.append`/env-setup in all 3 experiments scripts; algorithm discovery / GPU selection / CSV result-writing duplicated across the 3 evaluation CLIs; 4× code-fence-stripping pattern inside `engine.py`.
6. **Style debt** — `os.path` instead of `pathlib.Path`, blocking `open()` inside `async` functions (`ASYNC230`), `str(e)` inside f-strings (`RUF010`), `Optional`-style defaults (`RUF013`), import-order violations (`I001`), dead code (`extract_json`, unused `# noqa` comments, an unreachable `type==='fatal'` branch in the frontend).

---

## 3. Optimization Work by Area

### 3.1 Backend (`backend/`) — 51 → 0 ruff findings

**Structural refactor — `engine.py` 842 → 681 lines, two new cohesive modules:**

- **`backend/storage.py` (97 lines)** — all data/artifact I/O helpers (`load_rag_db`, `load_global_spec`, `load_resource`, `ensure_history_dir`, `save_artifact`), modernized to `pathlib` + `os.makedirs(exist_ok=True)` with intentional fallbacks preserved (`[]` / `""` on missing input) and bare `except: pass` replaced by logged `except Exception`.
- **`backend/stats.py` (128 lines)** — performance-metrics aggregation (`aggregate_stage_metrics`, preserving the original reset-then-recompute semantics) and the stats-header formatter (`format_stats_header`, byte-identical output).
- **`engine.py`** — deduplicated the 4× code-fence-stripping pattern into `_strip_code_fences`; replaced fragile `"judge_result" in locals()` checks with an explicit `judge_result = None` variable; removed dead `extract_json` (verified unreferenced repo-wide); `re.I` → `re.IGNORECASE`; `str(e)` → `{e!s}`.

**Concurrency & correctness fixes:**

- `executor.py`: blocking `open()` in async code → `asyncio.to_thread` (`ASYNC230` ×2); `stdout/stderr=PIPE` → `capture_output=True` with explicit `check=False` (the expected-nonzero-returncode flow is intentional); `session_id: str = None` → `str | None` (`RUF013`).
- `generator.py`: env-var defaults kept as strings and converted after `os.getenv` (`PLW1508` ×4) — behavior-identical; `response_model`/`metrics_out` → `| None`; `PROMPT_DIR` → `pathlib`.

**Startup crash fix:** `config.py`'s emoji print now has a scoped `UnicodeEncodeError` fallback (`[WARNING]` ASCII on GBK consoles, emoji on UTF-8), eliminating the one-time import crash. All 18 load-bearing config names kept as `str` (cross-module `os.path.join` callers).

**Observability:** `main.py` raw `print()` → structured `logging` (`logger "evococo.backend.main"`); the `send_update` closure hoisted out of the WS loop (signature unchanged); unnecessary `pass` removed. Public API — `run_pipeline(code, status_callback)`, routes `/` and `/ws`, all config names — unchanged.

### 3.2 Evaluation (`evaluation/`) — 16 → 0 ruff findings

**Dedup via new shared module `evaluation/_common.py` (243 lines):** constants (`ROOT`, algorithm dirs, result prefixes), `discover_algorithms()`, `check_cuda()`, CSV helpers (`read_rows`/`append_row` with field-order-preserving round-trip), `run_worker_process()` (subprocess + result-prefix parsing), `load_algorithm_class()` (module-prefix aware), `setup_torch_device()`, `instantiate_algorithm()`, `finite_fitness_rows()`. Uses `TYPE_CHECKING` for torch annotations (import-free lint).

| File | Before | After |
|---|---|---|
| `run_optimization_fidelity_benchmark.py` | 515 | 417 |
| `run_computational_scalability_benchmark.py` | 733 | ~640 |
| `run_migration_reliability_benchmark.py` | 221 | 246 (UTF-8 guard + `to_thread`) |
| `_fidelity_trial.py` | 147 | 99 |
| `_scaling_trial.py` | 151 | ~110 |
| `_benchmark_problems.py` | 101 | 101 (annotations only) |

**Every argparse flag, default, `RAW_FIELDS`/`SUMMARY_FIELDS` and output format preserved** — verified via `--help` diffs against `git show`. Legitimately differing worker logic (fidelity vs. scaling timed loops, result keys, warmup, recompile limits) intentionally kept separate. The pre-existing `--help` crash was fixed in-scope with a best-effort UTF-8 stdout/stderr reconfigure guard (root cause fixed at the source in `backend/config.py`).

### 3.3 Experiments (`experiments/`) — 25 → 0 ruff findings

**Dedup via new shared module `experiments/_common.py` (106 lines):** `project_root()` / `ensure_repo_root_on_path()` (replaces 3× duplicated `sys.path.append`), `load_dotenv_from_root()`, `setup_litellm_env()` (exact env mapping preserved), `set_win32_event_loop_policy()`, `is_matlab_source()` / `read_matlab_source()` (dir-walk concatenation preserved byte-for-byte), `algorithm_name()`, `write_json()` / `read_json()`.

| File | Before | After |
|---|---|---|
| `batch_translate.py` | 300 | 297 |
| `batch_one_shot.py` | 161 | 170 |
| `one_shot.py` | 369 | 362 |

Critical execution-order semantics preserved: `ensure_repo_root_on_path` → `load_dotenv_from_root` → `setup_litellm_env` **before** `from one_shot import ...`. Worker retry loop (5 attempts, `(attempt+1)*15`s backoff), skip-if-exists, the never-raise error-metrics contract (all 15 metric keys), manifest keys, cost block, and prompt templates are byte-identical. `ASYNC230` ×13 fixed via `asyncio.to_thread`; dead `if stats_data:` removed (exactly one stats write per run).

### 3.4 Frontend (`frontend/index.html`) — full rework

886 → 2,441 lines (self-contained). Nothing outside the file was modified; `initialScript` seed preserved **byte-identical** (verified programmatically).

- **Zero external dependencies:** all 6 CDNs + Google Fonts removed; system font stack, inline SVG icon sprite, tiny regex tokenizers (`highlightMatlab`/`highlightPython`) replacing highlight.js, and a minimal safe `mdToHtml()` renderer (escapes first, then fenced code/headings/…) replacing marked. Now works offline and from `file://`.
- **WebSocket lifecycle hardened:** origin-derived `wsUrl()` (`file://` → `ws://localhost:8000/ws`); exponential reconnect backoff (1s→2s→… capped 30s, reset on open); `clearPendingReconnect()` + socket null-out prevent double-connects; a `manualStop` flag suppresses false "link severed" alarms; unexpected mid-run disconnect resets the UI and aborts the run; run button disabled while connecting/processing.
- **Security:** all backend strings rendered via `textContent`/`escapeHtml()` — the previous `innerHTML` interpolation (stored XSS) is gone; markdown links href-whitelisted.
- **Maintainability:** no inline handlers (all `addEventListener` in `bindEvents()`); dead code removed (unreachable `type==='fatal'` branch, unused params, MathJax typeset); CSS de-duplicated via custom properties under `:root` / `[data-theme="dark"]`, drop-in theming persisted to `localStorage` with a head bootstrap to avoid theme flash.
- **Accessibility & UX:** `role="tablist"/"tab"/"tabpanel"` + `aria-selected` + tabindex management, `aria-live="polite"` status region, `role="log"` terminal, keyboard-resizable panels (arrows, Shift=40px), `aria-busy` on the run button, `noscript` fallback; responsive at 960px/560px breakpoints; copy with `execCommand('copy')` fallback; log capped at `MAX_LOG_LINES = 300`.

**Protocol contract preserved verbatim:** server→client `{type, title, message, step_id, extra_data, is_success, icon}`; client→server `{code}`.

---

## 4. Consolidated Metrics

| Metric | Before | After |
|---|---|---|
| Ruff findings (backend) | 51 | **0** |
| Ruff findings (evaluation) | 16 | **0** |
| Ruff findings (experiments) | 25 | **0** |
| `backend/engine.py` | 842 lines | 681 lines |
| Largest evaluation file | 733 lines | ~640 lines |
| New shared modules | — | `storage.py`, `stats.py`, `_common.py` ×2 |
| Frontend external CDNs | 6 + Google Fonts | **0** |
| Frontend XSS sink (`innerHTML` interpolation) | present | **removed** |
| Files over 1000 lines | 0 | 0 |
| Directory CONTEXT.md docs | 0 | **5** (root + 4 subtrees) |
| Diff | — | 23 files, +3,379 / −1,417 |

---

## 5. Validation (Final Gate — Post-Merge)

Executed in a clean worktree against the merged tree; every check passed:

| # | Check | Result |
|---|---|---|
| 1 | `python -m compileall -q backend evaluation experiments` (incl. generated_algorithms) | PASS (exit 0) |
| 2 | `python -m ruff check backend evaluation experiments --exclude 'experiments/generated_algorithms/**'` | All checks passed (exit 0) |
| 2b | `python -m ruff check .` (plain, proving `.ruff.toml` exclusion) | All checks passed (exit 0) |
| 3 | Import smoke (`backend` engine/executor/generator/config/main/storage/stats) | PASS (exit 0) |
| 4 | `--help` on all 6 evaluation/experiments entry points | PASS (all exit 0) |
| 5 | `git status` clean; no conflict markers anywhere | PASS |
| 6 | All key modules tracked and < 1000 lines | PASS |

> Note: `torch`/`evox` are not installed in the dev worktree and are intentionally **not** installed for validation (no check requires them). End-to-end GPU/LLM benchmark execution should be smoke-verified once on a machine with the full dependency stack — semantics were preserved by construction (identical flags, CSV columns, JSON schemas, worker result lines) plus torch-free functional tests of the shared helpers.

---

## 6. Known Issues / Follow-up Recommendations

**Left unfixed (documented in subtree CONTEXT.md files):**

1. `check_syntax_with_ruff` path sanitization (`file_path.replace(..., "script.py")`) can miss on Windows slash-convention mismatches, leaking the absolute temp path into the ruff error string sent to the LLM — cosmetic; fix requires output parsing beyond a refactor's scope.
2. `generate_llm_response` returns an error string instead of raising — pre-existing design contract (engine handles it via the static-fix loop); kept intentionally.
3. Three intentional broad `except Exception` fallbacks in experiments (LLM retry loop, batch-failure recording, one-shot never-raise contract) — each now logs via `logger.exception` with a rationale comment; converting to narrow exceptions is possible future work.
4. Frontend: MathJax removal means `$…$` math renders literally; regex tokenizers are best-effort (no full syntax highlighting). Acceptable trade-offs for the zero-dependency constraint.
5. Fidelity vs. scaling worker internals still differ (timed loops, result keys, warmup) — extracting further would over-engineer; kept separate deliberately.

**Recommended next steps (beyond this pass):**

- Add an automated CI job running the exact validation gate (compileall + ruff + `--help` smoke) on every PR — `.github/` currently only lints via ruff (see `.github/workflows`).
- Add lightweight unit tests for the new shared modules (`evaluation/_common.py` CSV round-trip, `experiments/_common.py` source discovery, `backend/stats.py` header formatting) — these are the highest-value, lowest-cost test targets.
- Smoke-run one real migration + one fidelity benchmark on a GPU box to lock in end-to-end behavior after the refactor.
- Consider pinning `requirements.txt` versions (currently unpinned except `>=` on torch/evox/evomo/ruff) for reproducible environments.

---

## 7. How to Reproduce

```bash
# Lint & syntax gate (mirrors the validation gate)
python -m compileall -q backend evaluation experiments
python -m ruff check .          # requires ruff; .ruff.toml excludes generated output

# CLI smoke
python evaluation/run_migration_reliability_benchmark.py --help
python evaluation/run_optimization_fidelity_benchmark.py --help
python evaluation/run_computational_scalability_benchmark.py --help
python experiments/batch_translate.py --help
python experiments/one_shot.py --help
```

See subtree `CONTEXT.md` files (`backend/`, `evaluation/`, `experiments/`, `frontend/`) for area-specific API surfaces, constraints, and known issues.
