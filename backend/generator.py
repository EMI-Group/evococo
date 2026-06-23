import os
import json
import re
import asyncio
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import REASONING_EFFORT, LLM_PROVIDERS, CONCURRENT_LLM_LIMIT

# 1. Load environment variables
load_dotenv()

# 2. Configuration
ACTIVE_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "litellm")
if ACTIVE_PROVIDER not in LLM_PROVIDERS:
    raise ValueError(f"Unknown LLM provider: {ACTIVE_PROVIDER}")

provider_config = LLM_PROVIDERS[ACTIVE_PROVIDER]

API_KEY = os.getenv(provider_config["api_key_env"])
BASE_URL = provider_config["base_url"]
MODEL_NAME = provider_config["model"]
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

# 3. Initialize OpenAI Async Client
client = AsyncOpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL,
    timeout=300.0,
    max_retries=3
)

# Concurrency Semaphore for rate-limiting
_sem = None

def _get_semaphore():
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(CONCURRENT_LLM_LIMIT)
    return _sem

# LLM Run Statistics
_run_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "duration": 0.0
}

def reset_llm_stats():
    global _run_stats
    _run_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "duration": 0.0
    }

def get_llm_stats():
    return _run_stats

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


async def generate_llm_response(
    prompt_filename: str, response_model: type[BaseModel] = None, **kwargs
):
    """
    Generates text response via the OpenAI-compatible API.

    Args:
        prompt_filename (str): Filename in the prompts/ directory (e.g., "1_analyst.md")
        response_model (type[BaseModel], optional): Pydantic model to enforce and parse structured output.
        **kwargs: Variables to inject into the prompt (e.g., matlab_code, analyst_report)
    """
    try:
        # A. Prepare Prompt
        prompt_content = _load_prompt(prompt_filename, **kwargs)

        # B. Call API
        # print(f">>> [DEBUG] Sending prompt: {prompt_filename} (Length: {len(prompt_content)})")

        kwargs_api = {}
        if response_model:
            # We enforce JSON object returned to assist Pydantic structure mapping
            kwargs_api["response_format"] = {"type": "json_object"}

        async with _get_semaphore():
            start_time = time.perf_counter()
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=TEMPERATURE,
                stream=False,
                extra_body={"reasoning_effort": REASONING_EFFORT},
                **kwargs_api,
            )
            duration = time.perf_counter() - start_time
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        print(f">排队过滤后 [LLM Usage] Prompt: {prompt_filename} | Time: {duration:.2f}s | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total: {prompt_tokens + completion_tokens}")

        global _run_stats
        _run_stats["prompt_tokens"] += prompt_tokens
        _run_stats["completion_tokens"] += completion_tokens
        _run_stats["duration"] += duration


        # C. Clean Response
        content = response.choices[0].message.content

        # Pydantic Structured Output Mode
        if response_model:
            # Find JSON payload robustly handling stray markdown wrap
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            json_str = match.group(1) if match else content
            try:
                data = json.loads(json_str)
                return response_model(**data)
            except Exception as e:
                print(f">>> [ERROR] Failed to parse JSON to Pydantic: {e}")
                raise

        # [Default Mode] Remove <analysis> tags and their content, keep only the subsequent code
        if "<analysis>" in content:
            # Find the position of </analysis> tag, take the content after it
            content = content.split("</analysis>")[-1].strip()

        # Remove any Markdown code block markers, keep only the content
        # Note: Added robust regex stripping for clean code/text extraction
        content = re.sub(r"^```[a-zA-Z]*\s*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
        content = content.replace("```", "").strip()

        return content

    except Exception as e:
        print(f"LLM Call Error: {e}")
        return f"Error generating response: {str(e)}"
