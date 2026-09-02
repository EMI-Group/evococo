import asyncio
import datetime
import json
import logging
import os
import re
import traceback

from pydantic import BaseModel, Field

from .config import (
    GLOBAL_SPEC_PATH,  # noqa: F401 - kept as module-level re-export
    HISTORY_DIR,  # noqa: F401 - kept as module-level re-export
    MAX_RETAINED_WORKSPACES,  # noqa: F401 - kept as module-level re-export
    NUM_BRANCHES,
    PROMPTS_DIR,  # noqa: F401 - kept as module-level re-export
    RULES_DB_PATH,  # noqa: F401 - kept as module-level re-export
    STRATEGIES_FULL,
    STRATEGIES_SHORT,
)
from .executor import (
    check_syntax_with_ruff,
    cleanup_workspace,
    execute_code_trial,
)
from .generator import generate_llm_response
from .stats import aggregate_stage_metrics, format_stats_header
from .storage import (
    ensure_history_dir,
    load_global_spec,
    load_rag_db,
    load_resource,
    save_artifact,
)

logger = logging.getLogger(__name__)


# --- Pydantic models ---


class JudgeResult(BaseModel):
    """Pydantic model describing the structured extraction for the final Judge output."""

    reasoning: str = Field(description="Detailed analysis of all branches.")
    winning_branch_id: int = Field(
        description="The integer ID of the branch that won the tournament (e.g. 0, 1, 2, 3, or 4)."
    )
    code: str = Field(
        description="The FULL, UNMODIFIED Python code of the winning branch, no markdown ticks."
    )


class RagResult(BaseModel):
    """Pydantic model describing the structured extraction for the RAG step."""

    selected_bug_numbers: list[int] = Field(
        description="A list of bug integers extracted from the rules."
    )


# --- Core logic: single branch lifecycle ---


def _strip_code_fences(code: str) -> str:
    """Strip markdown code fences (```python / ```) from generated code."""
    return code.replace("```python", "").replace("```", "").strip()


async def run_single_branch_lifecycle(
    branch_idx: int,
    base_session_id: str,
    matlab_code: str,
    blueprint_md: str,
    constraints_str: str,
    asset_lib_content: str,
    few_shot_content: str,
    run_dir: str,
    status_callback,
):
    """
    Independently run a code generation branch: Coder -> Static Fix -> Runtime Fix
    """
    branch_id = f"br{branch_idx}"
    session_id = os.path.join(base_session_id, branch_id)

    # Log label
    L_TAG = f"Branch {branch_idx}"

    # Strategy injection: 5 different tensorization schools
    current_strategy_short = STRATEGIES_SHORT[branch_idx % len(STRATEGIES_SHORT)]
    current_strategy_full = STRATEGIES_FULL[branch_idx % len(STRATEGIES_FULL)]

    temp_offset = 0.08 * branch_idx
    current_temp = 0.6 + temp_offset

    # [LOG] Branch start
    await status_callback(
        "log", L_TAG, f"Start | Strat: {current_strategy_short} | T={current_temp:.2f}"
    )

    enhanced_constraints = (
        f"{constraints_str}\n\n"
        f"!!! CRITICAL REQUIREMENT FOR THIS BRANCH !!!\n"
        f"{current_strategy_full}\n"
        f"!!! DO NOT USE PYTHON LOOPS FOR CALCULATION !!!"
    )

    current_code = ""
    branch_metrics = {"coder": {}, "static_fixes": [], "runtime_fixes": []}

    try:
        # =================================================
        # STEP 4: CODER
        # =================================================
        coder_metrics = {}
        current_code = await generate_llm_response(
            "4_coder.md",
            constraints=enhanced_constraints,
            blueprint_plan=blueprint_md,
            asset_library=asset_lib_content,
            few_shot_examples=few_shot_content,
            temperature=current_temp,
            metrics_out=coder_metrics,
        )
        branch_metrics["coder"] = coder_metrics
        current_code = _strip_code_fences(current_code)

        if run_dir:
            save_artifact(run_dir, f"4_code_draft_{branch_id}.py", current_code)

        await status_callback("log", L_TAG, "Coder finished draft.")

        # =================================================
        # STEP 5: STATIC FIXER
        # =================================================
        MAX_STATIC_RETRIES = 2
        for i in range(MAX_STATIC_RETRIES + 1):
            is_valid, error_msg = await check_syntax_with_ruff(current_code, session_id)

            if is_valid:
                await status_callback("log", L_TAG, "Static Check: PASSED.")
                break

            if "not found" in error_msg and "ruff" in error_msg:
                await status_callback("log", L_TAG, "Ruff missing, skipping check.")
                break

            await status_callback(
                "log", L_TAG, f"Static Check Failed (Try {i + 1}). Fixing..."
            )

            if i < MAX_STATIC_RETRIES:
                static_metrics = {}
                current_code = await generate_llm_response(
                    "5_static_fixer.md",
                    error_log=error_msg,
                    previous_code=current_code,
                    metrics_out=static_metrics,
                )
                branch_metrics["static_fixes"].append(static_metrics)
                current_code = _strip_code_fences(current_code)
            else:
                await status_callback(
                    "log", L_TAG, "Static Check failed, proceeding anyway."
                )

        if run_dir:
            save_artifact(run_dir, f"5_code_static_final_{branch_id}.py", current_code)

        # =================================================
        # STEP 6: RUNTIME FIXER
        # =================================================
        MAX_RUNTIME_RETRIES = 2
        best_igd_in_branch = float("inf")
        best_history = []

        for attempt in range(MAX_RUNTIME_RETRIES + 1):
            await status_callback(
                "log", L_TAG, f"Runtime Attempt {attempt + 1} executing..."
            )

            report = await execute_code_trial(current_code, session_id)

            if run_dir:
                save_artifact(
                    run_dir, f"6_code_exec_{branch_id}_try{attempt}.py", current_code
                )
                save_artifact(
                    run_dir,
                    f"6_log_{branch_id}_try{attempt}.txt",
                    f"IGD: {report['last_igd']}\nErr: {report['stderr']}",
                )

            if report["success"]:
                if not report["has_nan"]:
                    best_igd_in_branch = report["last_igd"]
                    best_history = report["igd_history"]

                    # [LOG] Success log (PERF color)
                    await status_callback(
                        "log",
                        "PERF",
                        f"[{L_TAG}] Success! IGD: {best_igd_in_branch:.5f}",
                    )

                    cleanup_workspace(session_id)
                    return {
                        "success": True,
                        "code": current_code,
                        "igd": best_igd_in_branch,
                        "igd_history": best_history,
                        "exec_time": report.get("exec_time", -1.0),
                        "branch_idx": branch_idx,
                        "metrics": branch_metrics,
                    }
                else:
                    await status_callback("log", "ERR", f"[{L_TAG}] NaN Detected.")

            if attempt < MAX_RUNTIME_RETRIES:
                err_summary = report["stderr"]
                if report["has_nan"]:
                    err_summary = "Runtime Warning: NaN values detected."
                elif not report["success"] and not err_summary:
                    err_summary = f"Runtime Error: Execution failed. output: {report['stdout'][-200:]}"

                # [LOG] Failure brief
                short_err = err_summary.split("\n")[-1][:60]
                await status_callback("log", "ERR", f"[{L_TAG}] Fix... ({short_err})")

                runtime_metrics = {}
                current_code = await generate_llm_response(
                    "6_runtime_fixer.md",
                    constraints=constraints_str,
                    blueprint_plan=blueprint_md,
                    error_summary=err_summary,
                    previous_code=current_code,
                    matlab_code=matlab_code,
                    metrics_out=runtime_metrics,
                )
                branch_metrics["runtime_fixes"].append(runtime_metrics)
                current_code = _strip_code_fences(current_code)
            else:
                await status_callback(
                    "log", "FAIL", f"[{L_TAG}] Failed after max retries."
                )

        cleanup_workspace(session_id)
        return {
            "success": False,
            "code": current_code,
            "igd": float("inf"),
            "igd_history": [],
            "exec_time": -1.0,
            "branch_idx": branch_idx,
            "metrics": branch_metrics,
        }

    # Intentional: a branch crash is a fallback boundary -> return a failure result
    except Exception as e:  # noqa: BLE001
        await status_callback("log", "FATAL", f"[{L_TAG}] Crashed: {e!s}")
        traceback.print_exc()
        cleanup_workspace(session_id)
        return {
            "success": False,
            "code": current_code,
            "igd": float("inf"),
            "igd_history": [],
            "exec_time": -1.0,
            "branch_idx": branch_idx,
            "metrics": branch_metrics,
        }


# --- Main process Pipeline ---


async def run_pipeline(matlab_code: str, status_callback):
    # 0. Load resources
    rag_db = load_rag_db()
    global_spec_content = load_global_spec()
    asset_lib_content = load_resource("resources_assets.md")
    few_shot_content = load_resource("resources_examples.md")

    # Detect if the input MATLAB code is a PlatEMO classdef algorithm
    is_platemo = bool(
        re.search(r"classdef\s+[A-Za-z0-9_]+\s*<\s*ALGORITHM", matlab_code)
    )

    if is_platemo:
        print(">>> [INFO] Detected PlatEMO algorithm standard.")
        reference_content = load_resource("reference_platemo.md")
    else:
        print(">>> [INFO] Detected custom/standalone MATLAB algorithm standard.")
        reference_content = load_resource("reference_custom.md")

    # Extract algorithm name from matlab_code
    algo_name = "UnknownAlgo"
    m_class = re.search(r"classdef\s+([A-Za-z0-9_]+)", matlab_code)
    if m_class:
        algo_name = m_class.group(1)
    else:
        m_func = re.search(
            r"function\s+.*?(?:=\s*|\s+)([A-Za-z0-9_]+)\s*\(", matlab_code
        )
        if m_func:
            algo_name = m_func.group(1)

    perf_stats = {
        "algorithm": algo_name,
        "session_id": "UnknownSession",
        "total_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "total_llm_time": 0.0,
        "stages": {},
    }

    # 1. Initialization
    # Intentional: if the history dir cannot be created, fall back to run_dir=None
    try:
        run_dir = ensure_history_dir(algo_name)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED TO CREATE HISTORY DIR: {e}")
        traceback.print_exc()
        run_dir = None

    session_id = (
        os.path.basename(run_dir)
        if run_dir
        else f"{datetime.datetime.now(datetime.timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}_{algo_name}"
    )
    perf_stats["session_id"] = session_id

    def save_perf_stats():
        if not run_dir:
            return
        aggregate_stage_metrics(perf_stats)
        save_artifact(run_dir, "performance_stats.json", perf_stats)

    if run_dir:
        save_artifact(run_dir, "0_input.m", matlab_code)

    await status_callback("init", "Pipeline Initialized", f"Session: {session_id}")
    await status_callback("log", "SYS", f"Session created at {session_id}")
    await status_callback(
        "log",
        "SYS",
        f"Detected algorithm style: {'PlatEMO' if is_platemo else 'Custom/Standalone'}",
    )

    # Track the judge outcome (None until the Judge step produces a result).
    # Used by the stats header to decide judge status / winning branch lines.
    judge_result = None

    try:
        # Step 1: Analyst
        await status_callback(
            "step_start", "Step 1: Analyst", "Analyzing Logic...", step_id="analyst"
        )

        analyst_metrics = {}
        # [Modify] Inject reference_context into LLM request
        ir_md = await generate_llm_response(
            "1_analyst.md",
            matlab_code=matlab_code,
            global_spec=global_spec_content,
            reference_context=reference_content,  # <--- Here is the injection point
            metrics_out=analyst_metrics,
        )
        perf_stats["stages"]["analyst"] = analyst_metrics

        if run_dir:
            save_artifact(run_dir, "1_analyst_ir.md", ir_md)
        await status_callback("result_ir", "", ir_md)
        await status_callback(
            "step_done",
            "Analyst Done",
            "Report Generated",
            step_id="analyst",
            is_success=True,
        )
        await status_callback("log", "INFO", "Analyst report generated.")

        # Step 2: RAG
        await status_callback(
            "step_start", "Step 2: RAG", "Scanning Knowledge...", step_id="rag"
        )

        rules_context_list = []
        for r in rag_db:
            tag = "[UNIVERSAL]" if r.get("always_apply") else "[CONDITIONAL]"
            desc = r.get("description", "No description")
            keywords = ", ".join(r.get("keywords", []))
            rules_context_list.append(
                f"ID: {r['id']}\nType: {tag}\nDesc: {desc}\nKeywords: {keywords}"
            )
        all_rules_desc = "\n---\n".join(rules_context_list)

        rag_metrics = {}
        rag_result = await generate_llm_response(
            "2_rag_selector.md",
            response_model=RagResult,
            rules_context=all_rules_desc,
            analyst_report=ir_md,
            metrics_out=rag_metrics,
        )
        perf_stats["stages"]["rag"] = rag_metrics
        selected_bug_numbers = rag_result.selected_bug_numbers

        # ... (Rule matching logic) ...
        def _bug_no_from_id(s: str):
            m = re.search(r"\bbug\b\s*#\s*(\d+)", str(s), flags=re.IGNORECASE)
            return int(m.group(1)) if m else None

        no_to_rule = {}
        for r in rag_db:
            no = _bug_no_from_id(r.get("id", ""))
            if no is not None:
                no_to_rule[no] = r

        matched_rules_list = []
        seen_ids = set()

        for no in selected_bug_numbers:
            r = no_to_rule.get(no)
            if r and r.get("id") not in seen_ids:
                matched_rules_list.append(
                    f"[{r['id']}] {r.get('instruction', r.get('description', ''))}"
                )
                seen_ids.add(r["id"])
        for r in rag_db:
            if r.get("always_apply") and r.get("id") not in seen_ids:
                matched_rules_list.append(
                    f"[{r['id']}] {r.get('instruction', r.get('description', ''))}"
                )
                seen_ids.add(r["id"])

        matched_rules_text = "\n".join(matched_rules_list)
        if run_dir:
            save_artifact(run_dir, "2_rag_selection.json", rag_result.model_dump())

        await status_callback(
            "log", "RAG", f"Selected {len(selected_bug_numbers)} specific rules."
        )
        await status_callback(
            "step_done",
            "RAG Done",
            f"Found {len(selected_bug_numbers)} Rules",
            step_id="rag",
            extra_data=selected_bug_numbers,
            is_success=True,
        )

        # Step 3: Architect
        await status_callback(
            "step_start",
            "Step 3: Architect",
            "Designing Blueprint...",
            step_id="architect",
        )
        architect_metrics = {}
        blueprint_md = await generate_llm_response(
            "3_architect.md",
            rag_rules=matched_rules_text,
            analyst_report=ir_md,
            metrics_out=architect_metrics,
        )
        perf_stats["stages"]["architect"] = architect_metrics
        if run_dir:
            save_artifact(run_dir, "3_blueprint.md", blueprint_md)
        await status_callback("result_blueprint", "", blueprint_md)
        await status_callback(
            "step_done",
            "Architect Done",
            "Blueprint Ready",
            step_id="architect",
            is_success=True,
        )
        await status_callback("log", "ARCH", "Blueprint ready.")

        # Step 4-6: Tournament
        await status_callback(
            "step_start",
            "Step 4-6: Tournament",
            f"Racing {NUM_BRANCHES} candidates...",
            step_id="tournament",
        )
        await status_callback(
            "log", "RACE", f"Starting {NUM_BRANCHES} parallel branches..."
        )

        tasks = []
        for i in range(NUM_BRANCHES):
            tasks.append(
                run_single_branch_lifecycle(
                    i,
                    session_id,
                    matlab_code,
                    blueprint_md,
                    matched_rules_text,
                    asset_lib_content,
                    few_shot_content,
                    run_dir,
                    status_callback,
                )
            )

        results = await asyncio.gather(*tasks)

        perf_stats["stages"]["branches"] = []
        for res in results:
            b_metrics = res.get("metrics", {})
            perf_stats["stages"]["branches"].append(
                {
                    "branch_idx": res["branch_idx"],
                    "strategy": STRATEGIES_SHORT[
                        res["branch_idx"] % len(STRATEGIES_SHORT)
                    ],
                    "success": res["success"],
                    "igd": res["igd"],
                    "metrics": b_metrics,
                }
            )

        # Step 7: Selector
        await status_callback(
            "step_start",
            "Step 7: Judge",
            "Selecting Best Implementation...",
            step_id="selector",
        )
        await status_callback("log", "JUDGE", "Evaluating candidates...")

        candidates_data = []
        valid_candidates_exist = False
        for res in results:
            if res["success"]:
                valid_candidates_exist = True
            candidates_data.append(
                {
                    "branch_id": res["branch_idx"],
                    "success": res["success"],
                    "final_igd": res["igd"],
                    "igd_history": res["igd_history"],
                    "exec_time": res.get("exec_time", -1.0),
                    "code_snippet": res["code"],
                }
            )

        if run_dir:
            save_artifact(run_dir, "7_candidates_raw.json", candidates_data)

        candidates_json = json.dumps(candidates_data, indent=2)

        final_code = ""

        if not valid_candidates_exist:
            await status_callback(
                "log", "FAIL", "All branches failed. Fallback to Br0."
            )
            final_code = results[0]["code"]
            await status_callback(
                "step_done",
                "Tournament Failed",
                "No valid candidates.",
                step_id="selector",
                is_success=False,
            )
        else:
            try:
                judge_metrics = {}
                # 3. Send request strictly expecting JudgeResult JSON
                judge_result = await generate_llm_response(
                    "7_selector.md",
                    response_model=JudgeResult,
                    candidates_list=candidates_json,
                    metrics_out=judge_metrics,
                )
                perf_stats["stages"]["judge"] = judge_metrics

                if run_dir:
                    save_artifact(
                        run_dir, "7_judge_raw.json", judge_result.model_dump()
                    )

                # =========================================================
                # [Core Logic] Clean Extractions
                # =========================================================
                reasoning = judge_result.reasoning
                final_code = judge_result.code
                # =========================================================

                # [LOG] Live judge verdict
                await status_callback("log", "JUDGE", "Verdict Reached:")
                for line in reasoning.split("\n"):
                    if line.strip():
                        # Truncate overly long lines to prevent UI overflow
                        disp_line = (
                            line.strip()[:120] + "..."
                            if len(line.strip()) > 120
                            else line.strip()
                        )
                        await status_callback("log", "REASON", disp_line)

                if run_dir:
                    save_artifact(run_dir, "7_judge_reasoning.txt", reasoning)

                await status_callback(
                    "step_done",
                    "Selector Done",
                    f"Winner: Branch {judge_result.winning_branch_id}",
                    step_id="selector",
                    is_success=True,
                    extra_data={"report": reasoning},
                )

            # Intentional: judge failure is a fallback boundary -> pick best successful branch
            except Exception as e:  # noqa: BLE001
                await status_callback("log", "ERR", f"Judge Failed: {e}. Fallback.")
                best_res = min(
                    [r for r in results if r["success"]],
                    key=lambda x: x["igd"],
                    default=results[0],
                )
                final_code = best_res["code"]

                await status_callback(
                    "step_done",
                    "Selector Fallback",
                    f"Winner: Branch {best_res['branch_idx']}",
                    step_id="selector",
                    is_success=True,
                )

        # Save performance statistics and log summary
        try:
            save_perf_stats()
            total_p = perf_stats["total_tokens"]["prompt"]
            total_c = perf_stats["total_tokens"]["completion"]
            total_t = perf_stats["total_tokens"]["total"]
            total_time = perf_stats["total_llm_time"]
            await status_callback(
                "log",
                "STATS",
                f"Total LLM Time: {total_time:.2f}s | Total Tokens: {total_t:,} (Prompt: {total_p:,}, Completion: {total_c:,})",
            )
        # Intentional: stats logging is best-effort and must not break the pipeline
        except Exception as stats_err:  # noqa: BLE001
            print(f">>> [WARNING] Failed to log/save performance stats: {stats_err}")

        # Keep performance metadata separate from the generated algorithm so
        # FINAL_OUTPUT.py remains a clean, directly reusable source file.
        stats_report = format_stats_header(perf_stats, judge_result)
        if run_dir:
            save_artifact(run_dir, "PERFORMANCE_STATS.txt", stats_report)

        await status_callback("result_code", "Final Optimized Code", final_code)
        if run_dir:
            save_artifact(run_dir, "FINAL_OUTPUT.py", final_code)

        await status_callback(
            "step_done",
            "Success",
            "Pipeline Complete!",
            step_id="finish",
            is_success=True,
        )
        return perf_stats

    # Intentional: outer pipeline crash boundary -> save diagnostics and report fatal
    except Exception as e:  # noqa: BLE001
        print(">>> [ERROR] Pipeline Crashed:")
        traceback.print_exc()
        try:
            save_perf_stats()
        except Exception as stats_err:  # noqa: BLE001 - intentional: best-effort stats save on crash path
            logger.warning(
                "Failed to save performance stats after pipeline crash: %s", stats_err
            )
        await status_callback("log", "FATAL", str(e))
        if run_dir:
            save_artifact(run_dir, "FATAL_ERROR.txt", traceback.format_exc())
        await status_callback("fatal", "System Error", str(e))
