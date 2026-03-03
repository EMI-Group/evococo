import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 1. Load environment variables
load_dotenv()

# 2. Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
# Default model configuration
MODEL_NAME = os.getenv("OPENAI_MODEL")
# MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.0-flash-exp")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

# 3. Initialize OpenAI Async Client
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename, **kwargs):
    """
    Reads and populates the Markdown prompt template.
    Uses .replace() instead of .format() to avoid conflicts with JSON braces.
    """
    path = os.path.join(PROMPT_DIR, filename)
    if not os.path.exists(path):
        print(f">>> [ERROR] Prompt file not found: {path}")
        return f"Error: Prompt file {filename} not found."

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace placeholders {key} in Prompt
    for key, value in kwargs.items():
        # Convert value to string, handle possible None
        val_str = str(value) if value is not None else ""
        content = content.replace(f"{{{key}}}", val_str)

    return content


async def generate_llm_response(prompt_filename: str, **kwargs):
    """
    Generates text response via the OpenAI-compatible API.

    Args:
        prompt_filename (str): Filename in the prompts/ directory (e.g., "1_analyst.md")
        **kwargs: Variables to inject into the prompt (e.g., matlab_code, analyst_report)
    """
    try:
        # A. Prepare Prompt
        prompt_content = _load_prompt(prompt_filename, **kwargs)

        # B. Call API
        # print(f">>> [DEBUG] Sending prompt: {prompt_filename} (Length: {len(prompt_content)})")

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=TEMPERATURE,
            stream=False,
        )

        # C. Clean Response
        content = response.choices[0].message.content

        # [Step 6 Exclusive] Remove <analysis> tags and their content, keep only the subsequent code
        if "<analysis>" in content:
            # Find the position of </analysis> tag, take the content after it
            content = content.split("</analysis>")[-1].strip()

        # Remove any Markdown code block markers, keep only the content
        if "```markdown" in content:
            content = content.replace("```markdown", "").replace("```", "").strip()
        elif "```json" in content:
            content = content.replace("```json", "").replace("```", "").strip()
        elif "```python" in content:
            content = content.replace("```python", "").replace("```", "").strip()
        # Handle bare ```
        elif "```" in content:
            content = content.replace("```", "").strip()

        return content

    except Exception as e:
        print(f"LLM Call Error: {e}")
        return f"Error generating response: {str(e)}"