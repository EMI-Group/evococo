import asyncio
import json
import traceback
from .generator import generate_llm_response
from .executor import execute_code

# RAG 数据库
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


async def run_pipeline(matlab_code: str, status_callback):
    print(">>> [DEBUG] run_pipeline started!")
    await status_callback("init", "Pipeline Initialized", "Engine started...")

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

        ir_str = await generate_llm_response("1_analyst.md", matlab_code=matlab_code)
        ir_json = extract_json(ir_str)
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

        # ！！！关键修复：修正文件名 "2_rag.md" -> "2_rag_selector.md"
        rag_str = await generate_llm_response(
            "2_rag_selector.md", rules_context=all_rules, ir_json=json.dumps(ir_json)
        )

        rag_json = extract_json(rag_str)
        selected_ids = rag_json.get("selected_rule_ids", [])
        matched_rules_text = "\n".join(
            [f"[{r['id']}] {r['rule']}" for r in RAG_DB if r["id"] in selected_ids]
        )

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
            await status_callback("result_code", "", code)

            # 执行
            print(f">>> [DEBUG] Executing code (Length: {len(code)})")
            success, output, error = execute_code(code)

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