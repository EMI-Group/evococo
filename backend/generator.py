import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 1. Load environment variables
load_dotenv()

# 2. Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
# 如果没有配置模型，默认使用 gemini-2.5-flash，你可以根据需要修改
MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

# 3. Initialize OpenAI Async Client
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename, **kwargs):
    """
    Reads and populates the Markdown prompt template.
    """
    path = os.path.join(PROMPT_DIR, filename)
    if not os.path.exists(path):
        print(f">>> [ERROR] Prompt file not found: {path}")
        return f"Error: Prompt file {filename} not found."

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # ！！！关键修复：使用 replace 而不是 format
    # 因为 Prompt 里包含 JSON 的大括号 {}，format 函数会把它们当成变量占位符处理导致 Crash
    for key, value in kwargs.items():
        content = content.replace(f"{{{key}}}", str(value))

    return content


async def generate_llm_response(matlab_code: str, error_log: str = None, **kwargs):
    """
    Generates Code/JSON via the OpenAI-compatible API.
    支持动态参数传递给 _load_prompt (如 rules_context, ir_json 等)
    """
    try:
        # 决定使用哪个 prompt 文件
        # 如果传入了 prompt_file 参数（在 engine.py 调用时会用到），则使用它
        # 默认情况（coder 第一步）使用 system_prompt.md (或者是你逻辑里定义的默认值)
        # 注意：你的 engine.py 里是直接传文件名给这个函数的第一个参数的
        # 为了兼容你的 engine.py 写法： generate_llm_response("1_analyst.md", matlab_code=...)
        # 这里需要适配一下参数逻辑。

        # 既然你的 engine.py 是这样调用的： await generate_llm_response("1_analyst.md", matlab_code=code)
        # 这里的函数签名需要稍微调整一下，或者在 engine.py 里改调用方式。
        # 为了不改动 engine.py 太多，我假设第一个参数其实是 prompt_filename

        # 修正后的逻辑：
        # 函数签名应该是 def generate_llm_response(prompt_filename: str, **kwargs):
        pass
    except Exception as e:
        pass


# 重新定义正确的函数，以匹配你的 engine.py 调用方式
async def generate_llm_response(prompt_filename: str, **kwargs):
    """
    Args:
        prompt_filename (str): e.g., "1_analyst.md"
        **kwargs: Variables to inject (e.g., matlab_code, rules_context)
    """
    try:
        # A. Prepare Prompt
        prompt_content = _load_prompt(prompt_filename, **kwargs)

        # 特殊处理：如果是 coder 修复模式，需要追加 repair_info
        # 你的 engine.py 里可能会传入 execution_mode="CORRECTION" 和 error_summary
        if kwargs.get("execution_mode") == "CORRECTION" and kwargs.get("error_summary"):
            repair_instruction = _load_prompt(
                "repair_info.md", error_log=kwargs["error_summary"]
            )
            prompt_content += "\n\n" + repair_instruction

        # B. Call API
        # print(f">>> [DEBUG] Sending prompt to LLM (Length: {len(prompt_content)})") # 调试用

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=TEMPERATURE,
            stream=False,
        )

        # C. Clean Code
        content = response.choices[0].message.content
        # 去除 markdown 标记
        cleaned_content = (
            content.replace("```python", "")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        return cleaned_content

    except Exception as e:
        print(f"LLM Call Error: {e}")
        return f"{{ 'error': 'API Call Failed: {str(e)}' }}"