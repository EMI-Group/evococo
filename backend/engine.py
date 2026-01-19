import asyncio
import json
import traceback
import os
import datetime
import uuid
import time  # 用于计时
from .generator import generate_llm_response
from .executor import execute_code, check_syntax_with_ruff

# --- 配置部分 ---
RAG_DB = [
    {
        "keywords": ["crowding", "distance", "cd"],
        "id": "Bug #6",
        "rule": "CrowdingDistance requires full population + boolean mask.",
    },
    {
        "keywords": ["dominance", "sort", "nds"],
        "id": "Bug #7",
        "rule": "Implement helper _calculate_sdr_dominance_matrix.",
    },
    {
        "keywords": ["rank", "front", "index"],
        "id": "Bug #1",
        "rule": "Never init int tensors with torch.inf.",
    },
    {
        "keywords": ["unique"],
        "id": "Bug #3",
        "rule": "Use evomo.utils.unique_rows_sorted instead of torch.unique.",
    },
    {"keywords": ["ceil"], "id": "Bug #2", "rule": "Do not use torch.ceil on scalars."},
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_DIR = os.path.join(PROJECT_ROOT, "run_history")


# --- 辅助函数 ---


def ensure_history_dir():
    """确保 history 目录存在，并返回当前运行的独立时间戳目录"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(HISTORY_DIR, timestamp)
    os.makedirs(run_dir)
    return run_dir


def save_artifact(run_dir, filename, content):
    """保存中间产物"""
    path = os.path.join(run_dir, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(content, (dict, list)):
                f.write(json.dumps(content, indent=2, ensure_ascii=False))
            else:
                f.write(str(content))
        print(f">>> [SAVED] {filename}")
    except Exception as e:
        print(f">>> [SAVE ERROR] Failed to save {filename}: {e}")


def extract_json(text):
    """仅用于 RAG Step 解析 JSON"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[0]
        return json.loads(text.strip())
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {}


# --- 核心 Pipeline ---


async def run_pipeline(matlab_code: str, status_callback):
    print(">>> [DEBUG] run_pipeline started!")

    # 1. 初始化运行目录和 Session
    try:
        run_dir = ensure_history_dir()
    except Exception:
        run_dir = None

    session_id = os.path.basename(run_dir) if run_dir else str(uuid.uuid4())[:8]

    if run_dir:
        save_artifact(run_dir, "0_input.m", matlab_code)

    await status_callback(
        "init",
        "Pipeline Initialized",
        f"Session: {session_id}",
    )

    try:
        # =================================================
        # STEP 1: ANALYST
        # =================================================
        print(">>> [DEBUG] Step 1: Analyst")
        await status_callback(
            "step_start",
            "Step 1: Analyst",
            "Analyzing Logic...",
            step_id="analyst",
            icon="fa-magnifying-glass",
        )

        ir_md = await generate_llm_response("1_analyst.md", matlab_code=matlab_code)

        if run_dir:
            save_artifact(run_dir, "1_analyst_ir.md", ir_md)

        await status_callback("result_ir", "", ir_md)
        await status_callback(
            "step_done", "Analyst Done", "Report Generated", step_id="analyst"
        )

        # =================================================
        # STEP 2: RAG
        # =================================================
        print(">>> [DEBUG] Step 2: RAG")
        await status_callback(
            "step_start",
            "Step 2: RAG",
            "Scanning Knowledge...",
            step_id="rag",
            icon="fa-book-open",
        )

        all_rules_desc = "\n".join([f"[ID: {r['id']}] {r['rule']}" for r in RAG_DB])
        rag_str = await generate_llm_response(
            "2_rag_selector.md", rules_context=all_rules_desc, analyst_report=ir_md
        )

        rag_json = extract_json(rag_str)
        selected_ids = rag_json.get("selected_rule_ids", [])
        matched_rules_text = "\n".join(
            [f"[{r['id']}] {r['rule']}" for r in RAG_DB if r["id"] in selected_ids]
        )

        if run_dir:
            save_artifact(run_dir, "2_rag_selection.json", rag_json)

        await status_callback(
            "step_done",
            "RAG Done",
            f"Found {len(selected_ids)} Rules",
            step_id="rag",
            extra_data=selected_ids,
        )

        # =================================================
        # STEP 3: ARCHITECT
        # =================================================
        print(">>> [DEBUG] Step 3: Architect")
        await status_callback(
            "step_start",
            "Step 3: Architect",
            "Designing Blueprint...",
            step_id="architect",
            icon="fa-compass-drafting",
        )

        blueprint_md = await generate_llm_response(
            "3_architect.md", rag_rules=matched_rules_text, analyst_report=ir_md
        )

        if run_dir:
            save_artifact(run_dir, "3_blueprint.md", blueprint_md)

        await status_callback("result_blueprint", "", blueprint_md)
        await status_callback(
            "step_done", "Architect Done", "Blueprint Ready", step_id="architect"
        )

        # =================================================
        # STEP 4: CODER (Initial Generation)
        # =================================================
        print(">>> [DEBUG] Step 4: Coder (Draft)")
        await status_callback(
            "step_start",
            "Step 4: Coder",
            "Drafting Initial Code...",
            step_id="coder_draft",
            icon="fa-pen-nib",
        )

        constraints_str = matched_rules_text if matched_rules_text else "None"

        # 使用纯生成模式的 4_coder.md
        current_code = await generate_llm_response(
            "4_coder.md", constraints=constraints_str, blueprint_plan=blueprint_md
        )
        current_code = current_code.replace("```python", "").replace("```", "").strip()

        if run_dir:
            save_artifact(run_dir, "4_code_draft.py", current_code)

        await status_callback("result_code", "", current_code)
        await status_callback(
            "step_done", "Coder Done", "Draft Generated", step_id="coder_draft"
        )

        # =================================================
        # STEP 5: STATIC FIXER (Ruff Analysis)
        # =================================================
        print(">>> [DEBUG] Step 5: Static Analysis")
        MAX_STATIC_RETRIES = 3
        static_pass = False

        for i in range(MAX_STATIC_RETRIES + 1):
            check_id = f"static_{i}"
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            await status_callback(
                "step_start",
                f"Step 5: Static Check ({i + 1})",
                f"Running Ruff [{current_time}]...",
                step_id=check_id,
                icon="fa-microscope",
            )

            # --- [计时开始] Ruff ---
            t_start = time.time()
            is_valid, error_msg = check_syntax_with_ruff(current_code, session_id)
            t_end = time.time()

            duration_msg = f"Ruff Check ({i + 1}) took: {t_end - t_start:.4f}s"
            print(f">>> [PERF] {duration_msg}")
            await status_callback("log", "PERF", duration_msg)
            # --- [计时结束] ---

            # 【防御性跳过】
            if "not found" in error_msg and "ruff" in error_msg:
                print(">>> [WARN] Ruff tool not found. Skipping static check.")
                await status_callback(
                    "log", "WARN", "⚠️ Ruff tool not found. Skipping static check."
                )
                await status_callback(
                    "step_done",
                    "Skipped",
                    "Ruff Missing",
                    step_id=check_id,
                    is_success=True,
                )
                static_pass = True
                break

            if is_valid:
                static_pass = True
                await status_callback("log", "SYSTEM", "Ruff Check Passed ✅")
                await status_callback(
                    "step_done",
                    "Static Pass",
                    "Code Valid",
                    step_id=check_id,
                    is_success=True,
                )
                break

            if i < MAX_STATIC_RETRIES:
                await status_callback(
                    "step_done",
                    "Static Issues",
                    "Fixing...",
                    step_id=check_id,
                    is_success=False,
                )

                fix_id = f"static_fix_{i}"
                await status_callback(
                    "step_start",
                    f"Step 5: Static Fix ({i + 1})",
                    "Applying Fixes...",
                    step_id=fix_id,
                    icon="fa-wrench",
                )

                # --- [计时开始] LLM Fix ---
                t_llm_start = time.time()
                current_code = await generate_llm_response(
                    "5_static_fixer.md", error_log=error_msg, previous_code=current_code
                )
                t_llm_end = time.time()

                llm_msg = (
                    f"LLM Static Fix ({i + 1}) took: {t_llm_end - t_llm_start:.4f}s"
                )
                print(f">>> [PERF] {llm_msg}")
                await status_callback("log", "PERF", llm_msg)
                # --- [计时结束] ---

                current_code = (
                    current_code.replace("```python", "").replace("```", "").strip()
                )

                if run_dir:
                    save_artifact(
                        run_dir, f"5_code_static_fix_{i + 1}.py", current_code
                    )
                await status_callback("result_code", "", current_code)
                await status_callback(
                    "step_done", "Fixed", "New Version", step_id=fix_id
                )
            else:
                # 即使静态检查失败多次，也尝试进入运行时
                await status_callback(
                    "log",
                    "WARN",
                    "Static check failed multiple times. Proceeding to Runtime anyway.",
                )

                # 手动结束上一个 Check Step，防止前端转圈
                await status_callback(
                    "step_done",
                    "Check Failed",
                    "Force Run",
                    step_id=check_id,
                    is_success=False,
                )

                static_pass = True
                break

        # =================================================
        # STEP 6: RUNTIME VERIFIER (Execution & Repair)
        # =================================================
        if static_pass:
            print(">>> [DEBUG] Step 6: Runtime Verification")
            # 【修改点 1/2】改为 3，提供 3 次修复机会 (Fix 1, 2, 3)
            MAX_RUNTIME_RETRIES = 3

            for attempt in range(1, MAX_RUNTIME_RETRIES + 2):
                exec_id = f"runtime_{attempt}"

                await status_callback(
                    "step_start",
                    f"Step 6: Execute ({attempt})",
                    "Running in Sandbox...",
                    step_id=exec_id,
                    icon="fa-play",
                )

                print(f">>> [DEBUG] Executing code (Length: {len(current_code)})")

                # --- 运行时计时 ---
                t_exec_start = time.time()
                success, output, error = execute_code(current_code, session_id)
                t_exec_end = time.time()

                exec_msg = (
                    f"Runtime Exec ({attempt}) took: {t_exec_end - t_exec_start:.4f}s"
                )
                print(f">>> [PERF] {exec_msg}")
                await status_callback("log", "PERF", exec_msg)
                # ----------------

                if run_dir:
                    save_artifact(
                        run_dir,
                        f"6_exec_log_{attempt}.txt",
                        f"STDOUT:\n{output}\nSTDERR:\n{error}",
                    )

                if success:
                    await status_callback(
                        "step_done",
                        "Success",
                        "Pipeline Complete!",
                        step_id=exec_id,
                        is_success=True,
                    )
                    await status_callback("log", "STDOUT", output)
                    return

                # 如果失败且有重试次数，则修复
                if attempt <= MAX_RUNTIME_RETRIES:
                    await status_callback(
                        "step_done",
                        "Runtime Error",
                        "Repairing...",
                        step_id=exec_id,
                        is_success=False,
                    )
                    await status_callback("log", "STDERR", error)

                    repair_id = f"runtime_fix_{attempt}"
                    await status_callback(
                        "step_start",
                        f"Step 6: Repair ({attempt})",
                        "Logic Repair...",
                        step_id=repair_id,
                        icon="fa-heart-pulse",
                    )

                    # --- 【修改点 2/2】LLM 运行时修复计时 + 前端推送 ---
                    t_fix_start = time.time()
                    # 使用 6_runtime_fixer.md 进行逻辑修复 (增量修复)
                    current_code = await generate_llm_response(
                        "6_runtime_fixer.md",
                        constraints=constraints_str,
                        blueprint_plan=blueprint_md,
                        error_summary=f"Runtime Error:\n{error}",
                        previous_code=current_code,
                    )
                    t_fix_end = time.time()

                    rt_fix_msg = f"LLM Runtime Fix ({attempt}) took: {t_fix_end - t_fix_start:.4f}s"
                    print(f">>> [PERF] {rt_fix_msg}")
                    await status_callback("log", "PERF", rt_fix_msg)
                    # ---------------------------------

                    current_code = (
                        current_code.replace("```python", "").replace("```", "").strip()
                    )

                    if run_dir:
                        save_artifact(
                            run_dir, f"6_code_runtime_fix_{attempt}.py", current_code
                        )
                    await status_callback("result_code", "", current_code)
                    await status_callback(
                        "step_done", "Repaired", "Ready to Retry", step_id=repair_id
                    )
                else:
                    await status_callback(
                        "step_done",
                        "Failed",
                        "Max Retries Reached",
                        step_id=exec_id,
                        is_success=False,
                    )
                    return

    except Exception as e:
        print(">>> [ERROR] Pipeline Crashed:")
        traceback.print_exc()
        if run_dir:
            save_artifact(run_dir, "FATAL_ERROR.txt", traceback.format_exc())
        await status_callback("fatal", "System Error", str(e))