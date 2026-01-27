import asyncio
import json
import traceback
import os
import datetime
import uuid
import time

# 注意：这里导入了 _load_prompt，请确保 generator.py 里没有把 _load_prompt 设为私有
from .generator import generate_llm_response
from .executor import execute_code, check_syntax_with_ruff

# --- 路径配置 ---

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
HISTORY_DIR = os.path.join(PROJECT_ROOT, "run_history")
PROMPTS_DIR = os.path.join(BACKEND_DIR, "prompts")
DATABASE_DIR = os.path.join(BACKEND_DIR, "database")

# 规则库路径
RULES_DB_PATH = os.path.join(DATABASE_DIR, "rag_db.json")
# 全局规范路径
GLOBAL_SPEC_PATH = os.path.join(PROMPTS_DIR, "0_global_spec.md")


# --- 数据加载辅助函数 ---


def load_rag_db():
    """从 JSON 文件加载 RAG 规则库"""
    if os.path.exists(RULES_DB_PATH):
        try:
            with open(RULES_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "rules" in data:
                    return data["rules"]
                return data
        except Exception as e:
            print(f">>> [ERROR] Failed to load rag_db.json: {e}")
            return []
    else:
        print(f">>> [WARN] Rules DB not found at {RULES_DB_PATH}")
        return []


def load_global_spec():
    """从 Markdown 文件加载全局规范"""
    if os.path.exists(GLOBAL_SPEC_PATH):
        try:
            with open(GLOBAL_SPEC_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f">>> [ERROR] Failed to load 0_global_spec.md: {e}")
            return "Error: Global Spec not found."
    return ""


def load_resource(filename):
    """加载资源文件 (SDK, Examples)"""
    path = os.path.join(PROMPTS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f">>> [ERROR] Failed to load {filename}: {e}")
    return ""


# --- 通用辅助函数 ---


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

    # 0. 加载外部数据
    rag_db = load_rag_db()
    global_spec_content = load_global_spec()
    # 加载 Prompt 资源
    asset_lib_content = load_resource("resources_assets.md")
    few_shot_content = load_resource("resources_examples.md")

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

        ir_md = await generate_llm_response(
            "1_analyst.md", matlab_code=matlab_code, global_spec=global_spec_content
        )

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

        # 构建 RAG Context
        rules_context_list = []
        for r in rag_db:
            tag = "[UNIVERSAL]" if r.get("always_apply") else "[CONDITIONAL]"
            desc = r.get("description", "No description")
            keywords = ", ".join(r.get("keywords", []))
            rule_entry = f"ID: {r['id']}\nType: {tag}\nDesc: {desc}\nKeywords: {keywords}"
            rules_context_list.append(rule_entry)

        all_rules_desc = "\n---\n".join(rules_context_list)

        rag_str = await generate_llm_response(
            "2_rag_selector.md", rules_context=all_rules_desc, analyst_report=ir_md
        )

        rag_json = extract_json(rag_str)
        selected_ids = rag_json.get("selected_rule_ids", [])

        # 构建 matched_rules_text
        matched_rules_list = []
        for r in rag_db:
            if r["id"] in selected_ids:
                instruction = r.get("instruction", r.get("description", ""))
                matched_rules_list.append(f"[{r['id']}] {instruction}")

        matched_rules_text = "\n".join(matched_rules_list)

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

        # 【核心修改】
        # 1. 移除了 execution_mode 和 error_summary (因为 prompt v3.1 删除了这两个占位符)
        # 2. 注入了 asset_library and few_shot_examples
        current_code = await generate_llm_response(
            "4_coder.md",
            constraints=constraints_str,
            blueprint_plan=blueprint_md,
            asset_library=asset_lib_content,    # <--- SDK
            few_shot_examples=few_shot_content, # <--- 范例
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

            # 如果检测失败
            if i < MAX_STATIC_RETRIES:
                if run_dir:
                    save_artifact(run_dir, f"5_ruff_error_{i + 1}.txt", error_msg)

                short_err = (
                    error_msg[:300].replace("\n", " ") + "..."
                    if len(error_msg) > 300
                    else error_msg.replace("\n", " ")
                )
                await status_callback("log", "RUFF_ERR", short_err)

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
                # 最后一次尝试失败
                if run_dir:
                    save_artifact(run_dir, f"5_ruff_error_FINAL.txt", error_msg)

                short_err = error_msg[:300].replace("\n", " ") + "..."
                await status_callback("log", "RUFF_FAIL", short_err)

                await status_callback(
                    "log",
                    "WARN",
                    "Static check failed multiple times. Proceeding to Runtime anyway.",
                )

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

                    # --- [计时开始] LLM Runtime Fix ---
                    t_fix_start = time.time()

                    # 传入 matlab_code 作为参考
                    current_code = await generate_llm_response(
                        "6_runtime_fixer.md",
                        constraints=constraints_str,
                        blueprint_plan=blueprint_md,
                        error_summary=f"Runtime Error:\n{error}",
                        previous_code=current_code,
                        matlab_code=matlab_code,
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