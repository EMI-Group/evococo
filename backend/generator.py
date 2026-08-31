import os
import asyncio
import json
import re
import time
from functools import lru_cache

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import ACTIVE_LLM_PROVIDER, LLM_PROVIDERS, REASONING_EFFORT

ACTIVE_PROVIDER = ACTIVE_LLM_PROVIDER
if ACTIVE_PROVIDER not in LLM_PROVIDERS:
    raise ValueError(f"Unknown LLM provider: {ACTIVE_PROVIDER}")

provider_config = LLM_PROVIDERS[ACTIVE_PROVIDER]

API_KEY = os.getenv(provider_config["api_key_env"])
BASE_URL = provider_config["base_url"]
MODEL_NAME = provider_config["model"]
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.2))
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", 15.0))
LLM_READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT", 600.0))
LLM_NETWORK_RETRY_INTERVAL = float(os.getenv("LLM_NETWORK_RETRY_INTERVAL", 15.0))

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


@lru_cache(maxsize=1)
def _get_client() -> AsyncOpenAI:
    if not API_KEY or API_KEY.startswith("your_"):
        key_name = provider_config["api_key_env"]
        raise ValueError(
            f"Missing API key for provider '{ACTIVE_PROVIDER}'. Set {key_name} in .env."
        )
    return AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=httpx.Timeout(LLM_READ_TIMEOUT, connect=LLM_CONNECT_TIMEOUT),
        max_retries=10,
    )


async def _wait_for_provider_network():
    """Pause DeepSeek calls while the provider endpoint is unreachable."""
    if not ACTIVE_PROVIDER.startswith("deepseek"):
        return

    attempt = 0
    while True:
        try:
            timeout = httpx.Timeout(10.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as probe:
                await probe.get(BASE_URL)
            if attempt:
                print(">>> [NETWORK] DeepSeek endpoint reachable; resuming LLM call.")
            return
        except httpx.TransportError as exc:
            attempt += 1
            if attempt == 1 or attempt % 4 == 0:
                print(
                    f">>> [NETWORK] DeepSeek endpoint unavailable ({exc}); "
                    f"retrying in {LLM_NETWORK_RETRY_INTERVAL:.0f}s."
                )
            await asyncio.sleep(LLM_NETWORK_RETRY_INTERVAL)


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
    prompt_filename: str,
    response_model: type[BaseModel] = None,
    metrics_out: dict = None,
    **kwargs,
):
    """
    Generates text response via the OpenAI-compatible API.

    Args:
        prompt_filename (str): Filename in the prompts/ directory (e.g., "1_analyst.md")
        response_model (type[BaseModel], optional): Pydantic model to enforce and parse structured output.
        metrics_out (dict, optional): Dict to record token usage and latency metrics.
        **kwargs: Variables to inject into the prompt (e.g., matlab_code, analyst_report)
    """
    try:
        # A. Prepare Prompt
        prompt_content = _load_prompt(prompt_filename, **kwargs)

        # B. Call API
        kwargs_api = {}
        if response_model:
            # We enforce JSON object returned to assist Pydantic structure mapping
            kwargs_api["response_format"] = {"type": "json_object"}
        if REASONING_EFFORT:
            kwargs_api["extra_body"] = {"reasoning_effort": REASONING_EFFORT}

        await _wait_for_provider_network()
        start_time = time.perf_counter()
        response = await _get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=TEMPERATURE,
            stream=False,
            **kwargs_api,
        )
        latency = time.perf_counter() - start_time

        # Extract usage metrics
        usage = getattr(response, "usage", None)
        if metrics_out is not None:
            metrics_out["prompt_tokens"] = (
                getattr(usage, "prompt_tokens", 0) if usage else 0
            )
            metrics_out["completion_tokens"] = (
                getattr(usage, "completion_tokens", 0) if usage else 0
            )
            metrics_out["total_tokens"] = (
                getattr(usage, "total_tokens", 0) if usage else 0
            )
            metrics_out["latency"] = latency

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
