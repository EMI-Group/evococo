import asyncio
import json
import traceback
import os
import datetime
from .generator import generate_llm_response
from .executor import execute_code

# --- 配置部分 ---

# RAG 数据库 (模拟)
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
# 将历史记录存到根目录下的 run_history 文件夹（与 backend 平级）
HISTORY_DIR = os.path.join(PROJECT_ROOT, "run_history")


# --- 辅助函数：文件保存 ---


def ensure_history_dir():
    """确保 history 目录存在，并返回当前运行的独立时间戳目录"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    # 创建带时间戳的运行目录，例如: backend/history/20260113_123045
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(HISTORY_DIR, timestamp)
    os.makedirs(run_dir)
    return run_dir


def save_artifact(run_dir, filename, content):
    """保存中间产物到文件 (支持 JSON 自动格式化)"""
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


# --- 核心 Pipeline ---


async def run_pipeline(matlab_code: str, status_callback):
    print(">>> [DEBUG] run_pipeline started!")

    # 1. 初始化运行目录
    try:
        run_dir = ensure_history_dir()
        print(f">>> [INFO] Run artifacts will be saved to: {run_dir}")
    except Exception as e:
        print(f">>> [ERROR] Failed to create history dir: {e}")
        run_dir = None

    # 保存原始输入
    if run_dir:
        save_artifact(run_dir, "0_input.m", matlab_code)

    await status_callback(
        "init",
        "Pipeline Initialized",
        f"Session: {os.path.basename(run_dir) if run_dir else 'Temp'}",
    )

    try:
        # --- STEP 1: ANALYST ---
        print(">>> [DEBUG] Step 1: Analyst loading 1_analyst.md")
        await status_callback(
            "step_start",
            "Agent Analyst",
            "Extracting IR...",
            step_id="analyst",
            icon="fa-magnifying-glass",
        )

        # 构造 global_spec
        global_spec = "Target Framework: EvoX (PyTorch based). Hardware: GPU optimized."

        ir_str = await generate_llm_response(
            "1_analyst.md", matlab_code=matlab_code, global_spec=global_spec
        )
        ir_json = extract_json(ir_str)

        # [SAVE]
        if run_dir:
            save_artifact(run_dir, "1_analyst_ir.json", ir_json)

        print(f">>> [DEBUG] Analyst done. Keys: {ir_json.keys()}")

        await status_callback("result_ir", "", json.dumps(ir_json, indent=2))
        await status_callback(
            "step_done", "Analyst Finished", "IR Ready", step_id="analyst"
        )

        # --- STEP 2: RAG ---
        print(">>> [DEBUG] Step 2: RAG loading 2_rag_selector.md")
        await status_callback(
            "step_start",
            "Knowledge Retrieval",
            "Checking Knowledge Base...",
            step_id="rag",
            icon="fa-book-open",
        )

        all_rules = "\n".join([f"[ID: {r['id']}] {r['rule']}" for r in RAG_DB])

        rag_str = await generate_llm_response(
            "2_rag_selector.md", rules_context=all_rules, ir_json=json.dumps(ir_json)
        )

        rag_json = extract_json(rag_str)
        selected_ids = rag_json.get("selected_rule_ids", [])
        matched_rules_text = "\n".join(
            [f"[{r['id']}] {r['rule']}" for r in RAG_DB if r["id"] in selected_ids]
        )

        # [SAVE]
        if run_dir:
            save_artifact(run_dir, "2_rag_selection.json", rag_json)

        await status_callback(
            "step_done",
            "RAG Finished",
            f"Rules: {len(selected_ids)}",
            step_id="rag",
            extra_data=selected_ids,
        )

        # --- STEP 3: ARCHITECT ---
        print(">>> [DEBUG] Step 3: Architect loading 3_architect.md")
        await status_callback(
            "step_start",
            "Agent Architect",
            "Designing Blueprint...",
            step_id="architect",
            icon="fa-compass-drafting",
        )

        blueprint_str = await generate_llm_response(
            "3_architect.md", rag_rules=matched_rules_text, ir_json=json.dumps(ir_json)
        )
        blueprint_json = extract_json(blueprint_str)

        # [SAVE]
        if run_dir:
            save_artifact(run_dir, "3_blueprint.json", blueprint_json)

        await status_callback(
            "result_blueprint", "", json.dumps(blueprint_json, indent=2)
        )
        await status_callback(
            "step_done", "Architect Finished", "Blueprint Ready", step_id="architect"
        )

        # --- STEP 4: CODER ---
        print(">>> [DEBUG] Step 4: Coder loading 4_coder.md")
        constraints = "; ".join(
            [
                f"{c['rule_id']}: {c.get('code_snippet_requirement', 'Follow Rule')}"
                for c in blueprint_json.get("hard_constraints", [])
            ]
        )

        last_error = ""  # 初始化变量

        for attempt in range(1, 4):
            is_fix = attempt > 1
            step_id = f"coder_{attempt}"
            await status_callback(
                "step_start",
                f"Coder (Try {attempt})",
                "Generating..." if not is_fix else "Fixing...",
                step_id=step_id,
                icon="fa-terminal",
            )

            # 统一调用逻辑
            code = await generate_llm_response(
                "4_coder.md",
                execution_mode="CORRECTION" if is_fix else "GENERATION",
                constraints=constraints,
                blueprint_json=json.dumps(blueprint_json),
                error_summary=f"Previous Error: {last_error}" if is_fix else "",
            )

            code = code.replace("```python", "").replace("```", "").strip()

            # [SAVE] 每次生成的代码都存下来，方便对比
            if run_dir:
                save_artifact(run_dir, f"4_code_try_{attempt}.py", code)

            await status_callback("result_code", "", code)

            # 执行
            print(f">>> [DEBUG] Executing code (Length: {len(code)})")
            success, output, error = execute_code(code)

            # [SAVE] 保存执行结果日志
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
                print(">>> [DEBUG] Pipeline Success!")
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
        # [SAVE] 致命错误也存下来
        if run_dir:
            save_artifact(run_dir, "FATAL_ERROR.txt", traceback.format_exc())

        await status_callback("fatal", "System Error", f"{str(e)}")


def extract_json(text):
    """辅助函数：从 LLM 回复中提取 JSON"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[0]
        return json.loads(text.strip())
    except:
        # 暴力查找 { }
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {}