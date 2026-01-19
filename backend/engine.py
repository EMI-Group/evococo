import asyncio
import json
import traceback
import os
import datetime
from .generator import generate_llm_response
from .executor import execute_code

# --- 配置部分 ---

# RAG 数据库 (保持不变)
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
    """
    仅用于 RAG Step，因为我们需要解析出具体的 ID 列表。
    对于 Analyst 和 Architect 的 Markdown 输出，不再使用此函数。
    """
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

    # 1. 初始化运行目录
    try:
        run_dir = ensure_history_dir()
    except Exception:
        run_dir = None

    if run_dir:
        save_artifact(run_dir, "0_input.m", matlab_code)

    await status_callback(
        "init",
        "Pipeline Initialized",
        f"Session: {os.path.basename(run_dir) if run_dir else 'Temp'}",
    )

    try:
        # --- STEP 1: ANALYST (输出改为 Markdown) ---
        print(">>> [DEBUG] Step 1: Analyst loading 1_analyst.md")
        await status_callback(
            "step_start",
            "Agent Analyst",
            "Generating Logic Report (Markdown)...",
            step_id="analyst",
            icon="fa-magnifying-glass",
        )

        # 生成 Markdown 格式的分析报告
        ir_md = await generate_llm_response("1_analyst.md", matlab_code=matlab_code)

        # [SAVE] 保存为 .md
        if run_dir:
            save_artifact(run_dir, "1_analyst_ir.md", ir_md)

        # 前端展示：直接展示 Markdown 文本
        await status_callback("result_ir", "", ir_md)
        await status_callback(
            "step_done", "Analyst Finished", "Report Generated", step_id="analyst"
        )

        # --- STEP 2: RAG (输入改为 Markdown，输出保持 JSON 用于程序逻辑) ---
        print(">>> [DEBUG] Step 2: RAG loading 2_rag_selector.md")
        await status_callback(
            "step_start",
            "Knowledge Retrieval",
            "Scanning for Known Issues...",
            step_id="rag",
            icon="fa-book-open",
        )

        all_rules_desc = "\n".join([f"[ID: {r['id']}] {r['rule']}" for r in RAG_DB])

        # 将 Analyst 的 Markdown 报告传给 RAG
        rag_str = await generate_llm_response(
            "2_rag_selector.md",
            rules_context=all_rules_desc,
            analyst_report=ir_md,  # <--- 关键修改：传入 MD 文本
        )

        # RAG 的输出仍然需要是结构化的 (List of IDs)，以便 Engine 过滤 RAG_DB
        rag_json = extract_json(rag_str)
        selected_ids = rag_json.get("selected_rule_ids", [])

        # 提取匹配的具体规则文本
        matched_rules_text = "\n".join(
            [f"[{r['id']}] {r['rule']}" for r in RAG_DB if r["id"] in selected_ids]
        )

        if run_dir:
            save_artifact(run_dir, "2_rag_selection.json", rag_json)

        await status_callback(
            "step_done",
            "RAG Finished",
            f"Found {len(selected_ids)} Rules",
            step_id="rag",
            extra_data=selected_ids,
        )

        # --- STEP 3: ARCHITECT (输出改为 Markdown) ---
        print(">>> [DEBUG] Step 3: Architect loading 3_architect.md")
        await status_callback(
            "step_start",
            "Agent Architect",
            "Designing Tensor Blueprint...",
            step_id="architect",
            icon="fa-compass-drafting",
        )

        # 传入 Analyst 报告(MD) 和 RAG 规则(Text)
        blueprint_md = await generate_llm_response(
            "3_architect.md",
            rag_rules=matched_rules_text,
            analyst_report=ir_md,  # <--- 关键修改：传入 MD 文本
        )

        # [SAVE] 保存为 .md
        if run_dir:
            save_artifact(run_dir, "3_blueprint.md", blueprint_md)

        await status_callback("result_blueprint", "", blueprint_md)
        await status_callback(
            "step_done", "Architect Finished", "Blueprint Designed", step_id="architect"
        )

        # --- STEP 4: CODER ---
        print(">>> [DEBUG] Step 4: Coder loading 4_coder.md")

        # 构造约束条件：这里直接使用 Step 2 检索到的规则作为硬性约束
        # 因为 Blueprint 现在是 MD，不容易程序化提取 constraints，我们信任 RAG 的结果
        constraints_str = matched_rules_text if matched_rules_text else "None"

        last_error = ""

        for attempt in range(1, 4):
            is_fix = attempt > 1
            step_id = f"coder_{attempt}"
            await status_callback(
                "step_start",
                f"Coder (Try {attempt})",
                "Coding..." if not is_fix else "Fixing...",
                step_id=step_id,
                icon="fa-terminal",
            )

            # 生成代码
            # 注意：我们将 Markdown 格式的 blueprint 传给 prompt 中的 {blueprint_plan}
            code = await generate_llm_response(
                "4_coder.md",
                execution_mode="CORRECTION" if is_fix else "GENERATION",
                constraints=constraints_str,
                blueprint_plan=blueprint_md,  # <--- 关键修改：传入 MD 蓝图
                error_summary=f"Previous Error: {last_error}" if is_fix else "",
            )

            # 清理可能的代码块标记
            code = code.replace("```python", "").replace("```", "").strip()

            if run_dir:
                save_artifact(run_dir, f"4_code_try_{attempt}.py", code)

            await status_callback("result_code", "", code)

            # 执行代码
            print(f">>> [DEBUG] Executing code (Length: {len(code)})")
            success, output, error = execute_code(code)

            if run_dir:
                log_content = f"STDOUT:\n{output}\n\nSTDERR:\n{error}"
                save_artifact(
                    run_dir, f"4_execution_log_try_{attempt}.txt", log_content
                )

            if success:
                await status_callback(
                    "step_done",
                    "Verification Passed",
                    "Success!",
                    step_id=step_id,
                    is_success=True,
                )
                await status_callback("log", "STDOUT", output)
                return
            else:
                last_error = error
                await status_callback(
                    "step_done",
                    "Verification Failed",
                    "Retrying...",
                    step_id=step_id,
                    is_success=False,
                )
                await status_callback("log", "STDERR", error)

    except Exception as e:
        print(">>> [ERROR] Pipeline Crashed:")
        traceback.print_exc()
        if run_dir:
            save_artifact(run_dir, "FATAL_ERROR.txt", traceback.format_exc())
        await status_callback("fatal", "System Error", f"{str(e)}")