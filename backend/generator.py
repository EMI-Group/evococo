import asyncio
import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import ACTIVE_LLM_PROVIDER, LLM_PROVIDERS, REASONING_EFFORT
from .storage import load_analyst_cache, save_analyst_cache

ACTIVE_PROVIDER = ACTIVE_LLM_PROVIDER
if ACTIVE_PROVIDER not in LLM_PROVIDERS:
    raise ValueError(f"Unknown LLM provider: {ACTIVE_PROVIDER}")

provider_config = LLM_PROVIDERS[ACTIVE_PROVIDER]

API_KEY = os.getenv(provider_config["api_key_env"])
BASE_URL = provider_config["base_url"]
MODEL_NAME = provider_config["model"]
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "15.0"))
LLM_READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT", "600.0"))
LLM_NETWORK_RETRY_INTERVAL = float(os.getenv("LLM_NETWORK_RETRY_INTERVAL", "15.0"))
GEMINI_EXPLICIT_CACHE = os.getenv("GEMINI_EXPLICIT_CACHE", "0").strip() == "1"
ANALYST_RESULT_CACHE = os.getenv("ANALYST_RESULT_CACHE", "1").strip() != "0"

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
logger = logging.getLogger(__name__)


def _analyst_cache_key(
    messages: list[dict], temperature: float, reasoning_effort: str | None = None
) -> str:
    """Hash request identity without storing endpoint details or credentials."""
    identity = {
        "version": 1,
        "provider": ACTIVE_PROVIDER,
        "base_url": BASE_URL,
        "model": MODEL_NAME,
        "temperature": temperature,
        "reasoning_effort": (
            REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        ),
        "messages": messages,
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_messages(prompt_content: str, prompt_suffix: str = "") -> list[dict]:
    """Cache the shared Coder prefix only on an opted-in Gemini gateway."""
    if (
        GEMINI_EXPLICIT_CACHE
        and ACTIVE_PROVIDER == "custom"
        and "gemini" in MODEL_NAME.lower()
        and prompt_suffix
    ):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_content,
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
            },
            {"role": "user", "content": prompt_suffix},
        ]
    return [{"role": "user", "content": prompt_content + prompt_suffix}]


def _extract_cache_usage(usage) -> tuple[int | None, int | None]:
    """Read provider cache counters, preserving absent counters as unknown."""
    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    uncached = getattr(usage, "prompt_cache_miss_tokens", None)
    if cached is None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if cached is not None and uncached is None and prompt_tokens is not None:
        uncached = prompt_tokens - cached
    return cached, uncached


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
        raise FileNotFoundError(f"Prompt file not found: {path}")

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
    response_model: type[BaseModel] | None = None,
    metrics_out: dict | None = None,
    temperature: float | None = None,
    prompt_suffix: str = "",
    reasoning_effort: str | None = None,
    **kwargs,
):
    """
    Generates text response via the OpenAI-compatible API.

    Args:
        prompt_filename (str): Filename in the prompts/ directory (e.g., "1_analyst.md")
        response_model (type[BaseModel], optional): Pydantic model to enforce and parse structured output.
        metrics_out (dict, optional): Dict to record token usage and latency metrics.
        temperature (float, optional): Override the default API temperature.
        prompt_suffix (str, optional): Branch-specific instructions after the shared prompt.
        reasoning_effort (str, optional): Per-call override of the global reasoning setting.
        **kwargs: Variables to inject into the prompt (e.g., matlab_code, analyst_report)
    """
    try:
        # A. Prepare Prompt
        prompt_content = _load_prompt(prompt_filename, **kwargs)
        messages = _build_messages(prompt_content, prompt_suffix)
        request_temperature = TEMPERATURE if temperature is None else temperature
        request_effort = (
            REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        )
        if metrics_out is not None:
            metrics_out["reasoning_effort"] = request_effort
        cache_key = None
        if prompt_filename == "1_analyst.md" and response_model is None:
            if metrics_out is not None:
                metrics_out["local_cache_hit"] = False
            if ANALYST_RESULT_CACHE:
                cache_key = _analyst_cache_key(
                    messages, request_temperature, request_effort
                )
                cached_report = await asyncio.to_thread(load_analyst_cache, cache_key)
                if cached_report is not None:
                    if metrics_out is not None:
                        metrics_out.update(
                            {
                                "local_cache_hit": True,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                                "latency": 0.0,
                                "cached_input_tokens": None,
                                "uncached_input_tokens": None,
                            }
                        )
                    return cached_report

        # B. Call API
        kwargs_api = {}
        if response_model:
            # We enforce JSON object returned to assist Pydantic structure mapping
            kwargs_api["response_format"] = {"type": "json_object"}
        if request_effort:
            kwargs_api["extra_body"] = {"reasoning_effort": request_effort}

        await _wait_for_provider_network()
        start_time = time.perf_counter()
        response = await _get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=request_temperature,
            stream=False,
            **kwargs_api,
        )
        latency = time.perf_counter() - start_time

        # Extract usage metrics
        usage = getattr(response, "usage", None)
        cached_tokens, uncached_tokens = _extract_cache_usage(usage)
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
            metrics_out["cached_input_tokens"] = cached_tokens
            metrics_out["uncached_input_tokens"] = uncached_tokens

        if cached_tokens is not None:
            logger.info(
                "LLM input cache [%s]: cached=%s, uncached=%s",
                prompt_filename,
                cached_tokens,
                uncached_tokens,
            )

        # C. Clean Response
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty LLM response for {prompt_filename}")

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
        if not content:
            raise ValueError(f"Empty LLM response for {prompt_filename}")

        if (
            cache_key is not None
            and getattr(response.choices[0], "finish_reason", None) == "stop"
        ):
            await asyncio.to_thread(save_analyst_cache, cache_key, content)
        return content

    # Pipeline, branch and judge boundaries decide whether a failure can recover.
    # Never let an API error become an apparently valid report, blueprint or code.
    except Exception as e:
        print(f"LLM Call Error: {e}")
        raise
