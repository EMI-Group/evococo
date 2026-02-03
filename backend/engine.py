import asyncio
import json
import traceback
import os
import datetime
import uuid
import time
import re

# 注意：这里导入了 _load_prompt，请确保 generator.py 里没有把 _load_prompt 设为私有
from .generator import generate_llm_response

# 【修改点】导入新的执行器函数 execute_code_trial 和 cleanup_workspace
from .executor import (
    execute_code,
    execute_code_trial,
    check_syntax_with_ruff,
    cleanup_workspace,
)

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

# 【修改点】并行分支数量
NUM_BRANCHES = 3


# --- 数据加载辅助函数 ---


def load_rag_db():
    """从 JSON 文件加载 RAG 规则库"""
    if not os.path.exists(RULES_DB_PATH):
        print(f">>> [ERROR] Rules DB not found at {RULES_DB_PATH}")
        return []

    try:
        with open(RULES_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "rules" in data:
            return data["rules"]
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f">>> [ERROR] Failed to load rag_db.json: {e}")
        return []


def load_global_spec():
    """从 Markdown 文件加载全局规范"""
    if os.path.exists(GLOBAL_SPEC_PATH):
        try:
            with open(GLOBAL_SPEC_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f">>> [ERROR] Failed to load 0_global_spec.md: {e}")
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


# --- 核心逻辑：单分支生命周期 ---


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
    独立运行一个代码生成分支：Coder -> Static Fix -> Runtime Fix
    """
    branch_id = f"br{branch_idx}"
    # 【隔离关键】使用独立的 Session ID，防止文件冲突
    session_id = f"{base_session_id}_{branch_id}"

    # 增加一点 temperature 的扰动，保证多样性
    temp_offset = 0.05 * branch_idx

    log_prefix = f"[Branch {branch_idx}]"
    print(f">>> {log_prefix} Started flow...")

    current_code = ""

    try:
        # =================================================
        # STEP 4: CODER (Draft)
        # =================================================
        # 这里的 status_callback 如果频繁调用可能会导致 UI 混乱，
        # 所以在分支内部我们主要依靠 print log，或者发送不带 step_id 的 log

        current_code = await generate_llm_response(
            "4_coder.md",
            constraints=constraints_str,
            blueprint_plan=blueprint_md,
            asset_library=asset_lib_content,
            few_shot_examples=few_shot_content,
            temperature=0.7 + temp_offset,
        )
        current_code = current_code.replace("```python", "").replace("```", "").strip()

        if run_dir:
            save_artifact(run_dir, f"4_code_draft_{branch_id}.py", current_code)

        # =================================================
        # STEP 5: STATIC FIXER (Ruff Analysis)
        # =================================================
        MAX_STATIC_RETRIES = 2
        for i in range(MAX_STATIC_RETRIES + 1):
            is_valid, error_msg = check_syntax_with_ruff(current_code, session_id)

            # 如果 Ruff 没装或者报错
            if "not found" in error_msg and "ruff" in error_msg:
                print(f">>> {log_prefix} Ruff not found, skipping static check.")
                break

            if is_valid:
                break

            if i < MAX_STATIC_RETRIES:
                print(f">>> {log_prefix} Static Fix {i + 1}...")
                current_code = await generate_llm_response(
                    "5_static_fixer.md", error_log=error_msg, previous_code=current_code
                )
                current_code = (
                    current_code.replace("```python", "").replace("```", "").strip()
                )
            else:
                print(f">>> {log_prefix} Static Check Failed, proceeding anyway.")

        if run_dir:
            save_artifact(run_dir, f"5_code_static_final_{branch_id}.py", current_code)

        # =================================================
        # STEP 6: RUNTIME FIXER (The Tournament Trial)
        # =================================================
        MAX_RUNTIME_RETRIES = 2  # 赛马阶段不需要修太多次
        best_igd_in_branch = float("inf")
        best_history = []

        for attempt in range(MAX_RUNTIME_RETRIES + 1):
            print(f">>> {log_prefix} Runtime Attempt {attempt + 1}...")

            # 使用 execute_code_trial 获取详细指标 (IGD, History, NaN)
            report = execute_code_trial(current_code, session_id)

            # 保存现场
            if run_dir:
                save_artifact(
                    run_dir, f"6_code_exec_{branch_id}_try{attempt}.py", current_code
                )
                save_artifact(
                    run_dir,
                    f"6_log_{branch_id}_try{attempt}.txt",
                    f"IGD: {report['last_igd']}\nErr: {report['stderr']}\nOut: {report['stdout'][:500]}",
                )

            if report["success"]:
                # 运行成功
                if not report["has_nan"]:
                    # 完美运行 (无报错，无 NaN)
                    best_igd_in_branch = report["last_igd"]
                    best_history = report["igd_history"]
                    print(f">>> {log_prefix} Success! IGD={best_igd_in_branch}")

                    # 清理并返回成功结果
                    cleanup_workspace(session_id)
                    return {
                        "success": True,
                        "code": current_code,
                        "igd": best_igd_in_branch,
                        "igd_history": best_history,
                        "branch_idx": branch_idx,
                    }
                else:
                    print(f">>> {log_prefix} Success but NaN detected.")

            # 运行失败或有 NaN，尝试修复
            if attempt < MAX_RUNTIME_RETRIES:
                err_summary = report["stderr"]
                if report["has_nan"]:
                    err_summary = "Runtime Warning: NaN values detected in output metrics. Check division by zero or normalization."
                elif not report["success"] and not err_summary:
                    err_summary = f"Runtime Error: Execution failed. output: {report['stdout'][-200:]}"

                current_code = await generate_llm_response(
                    "6_runtime_fixer.md",
                    constraints=constraints_str,
                    blueprint_plan=blueprint_md,
                    error_summary=err_summary,
                    previous_code=current_code,
                    matlab_code=matlab_code,
                )
                current_code = (
                    current_code.replace("```python", "").replace("```", "").strip()
                )
            else:
                print(f">>> {log_prefix} Failed after max retries.")

        # 最终失败
        cleanup_workspace(session_id)
        return {
            "success": False,
            "code": current_code,
            "igd": float("inf"),
            "igd_history": [],
            "branch_idx": branch_idx,
        }

    except Exception as e:
        print(f">>> {log_prefix} Crashed: {e}")
        traceback.print_exc()
        cleanup_workspace(session_id)
        return {
            "success": False,
            "code": current_code,
            "igd": float("inf"),
            "igd_history": [],
            "branch_idx": branch_idx,
        }


# --- 主流程 Pipeline ---


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
            rule_entry = (
                f"ID: {r['id']}\nType: {tag}\nDesc: {desc}\nKeywords: {keywords}"
            )
            rules_context_list.append(rule_entry)

        all_rules_desc = "\n---\n".join(rules_context_list)

        rag_str = await generate_llm_response(
            "2_rag_selector.md", rules_context=all_rules_desc, analyst_report=ir_md
        )

        rag_json = extract_json(rag_str)
        selected_bug_numbers = rag_json.get("selected_bug_numbers", [])

        # 匹配规则逻辑
        def _bug_no_from_id(s: str):
            m = re.search(r"\bbug\b\s*#\s*(\d+)", str(s), flags=re.I)
            return int(m.group(1)) if m else None

        no_to_rule = {}
        for r in rag_db:
            no = _bug_no_from_id(r.get("id", ""))
            if no is not None:
                no_to_rule[no] = r

        matched_rules_list = []
        seen_ids = set()

        # 添加被选中的规则
        for no in selected_bug_numbers:
            r = no_to_rule.get(no)
            if r:
                rid = r.get("id", "")
                if rid and rid not in seen_ids:
                    instruction = r.get("instruction", r.get("description", ""))
                    matched_rules_list.append(f"[{rid}] {instruction}")
                    seen_ids.add(rid)

        # 添加通用规则
        for r in rag_db:
            if r.get("always_apply", False):
                rid = r.get("id", "")
                if rid and rid not in seen_ids:
                    instruction = r.get("instruction", r.get("description", ""))
                    matched_rules_list.append(f"[{rid}] {instruction}")
                    seen_ids.add(rid)

        matched_rules_text = "\n".join(matched_rules_list)
        if run_dir:
            save_artifact(run_dir, "2_rag_selection.json", rag_json)

        await status_callback(
            "step_done",
            "RAG Done",
            f"Found {len(selected_bug_numbers)} Rules",
            step_id="rag",
            extra_data=selected_bug_numbers,
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
        # STEP 4-6: THE TOURNAMENT (Parallel Execution)
        # =================================================
        print(f">>> [DEBUG] Starting {NUM_BRANCHES} parallel branches...")
        await status_callback(
            "step_start",
            "Step 4-6: Tournament",
            f"Racing {NUM_BRANCHES} candidates in parallel...",
            step_id="tournament",
            icon="fa-flag-checkered",
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

        # 并发等待所有分支运行完成
        results = await asyncio.gather(*tasks)
        # results format: [{"success": T/F, "code": "...", "igd": 0.1, "branch_idx": 0, "igd_history": [...]}, ...]

        # =================================================
        # STEP 7: SELECTOR (LLM Judge)
        # =================================================
        await status_callback(
            "step_start",
            "Step 7: Judge",
            "Selecting Best Implementation...",
            step_id="selector",
            icon="fa-gavel",
        )

        # 1. 整理候选人数据
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
                    "code_snippet": res["code"],  # LLM 需要代码来判断张量化程度
                }
            )

        # 保存候选列表供调试
        if run_dir:
            save_artifact(run_dir, "7_candidates_raw.json", candidates_data)

        candidates_json = json.dumps(candidates_data, indent=2)
        final_code = ""

        # 2. 调用 LLM 裁判 (除非全军覆没)
        if not valid_candidates_exist:
            print(">>> All branches failed. Picking the first one for debug.")
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
                print(">>> Asking Judge LLM to select best code...")
                selected_code = await generate_llm_response(
                    "7_selector.md", candidates_list=candidates_json
                )

                # 清理
                final_code = (
                    selected_code.replace("```python", "").replace("```", "").strip()
                )

                if run_dir:
                    save_artifact(run_dir, "7_judge_response.md", selected_code)

                await status_callback(
                    "step_done",
                    "Selector Done",
                    "Winner Selected",
                    step_id="selector",
                    is_success=True,
                )

            except Exception as e:
                print(f">>> Judge Failed: {e}. Fallback to Greedy.")
                # 兜底：选 IGD 最小的
                best_res = min(
                    [r for r in results if r["success"]],
                    key=lambda x: x["igd"],
                    default=results[0],
                )
                final_code = best_res["code"]

        # =================================================
        # FINISH
        # =================================================
        await status_callback("result_code", "Final Optimized Code", final_code)

        if run_dir:
            save_artifact(run_dir, "FINAL_OUTPUT.py", final_code)

        await status_callback("step_done", "Success", "Pipeline Complete!", step_id="finish", is_success=True)

    except Exception as e:
        print(">>> [ERROR] Pipeline Crashed:")
        traceback.print_exc()
        if run_dir:
            save_artifact(run_dir, "FATAL_ERROR.txt", traceback.format_exc())
        await status_callback("fatal", "System Error", str(e))