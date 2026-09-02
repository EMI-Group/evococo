"""Shared helpers for the EvoCoCo experiment runner scripts.

Centralizes the common path, environment, and file I/O logic used by
``batch_translate.py``, ``batch_one_shot.py``, and ``one_shot.py`` so the
runners stay small and consistent.  Only the standard library is imported at
module level; third-party packages (e.g. ``dotenv``) are imported lazily
inside the functions that need them.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

MATLAB_EXTENSIONS: tuple[str, ...] = (".m", ".txt")

EXCLUDED_INPUT_FILES: frozenset[str] = frozenset(
    {"analysis.txt", "dryrun_output.txt", "references_and_copyrights.md"}
)


def project_root() -> Path:
    """Return the repository root (the parent of the experiments directory)."""
    return Path(__file__).resolve().parent.parent


def ensure_repo_root_on_path() -> None:
    """Append the repository root to ``sys.path`` so ``backend`` imports resolve."""
    root = str(project_root())
    if root not in sys.path:
        sys.path.append(root)


def load_dotenv_from_root() -> None:
    """Load the ``.env`` file from the repository root (idempotent)."""
    from dotenv import load_dotenv

    load_dotenv(project_root() / ".env")


def setup_litellm_env() -> None:
    """Export LiteLLM configuration as OpenAI env vars (must run before importing one_shot).

    ``one_shot`` constructs an ``AsyncOpenAI`` client at import time, so this
    mapping has to happen before that import.
    """
    if not os.getenv("OPENAI_API_KEY"):
        litellm_key = os.getenv("LITELLM_API_KEY")
        if litellm_key:
            os.environ["OPENAI_API_KEY"] = litellm_key
    if not os.getenv("OPENAI_BASE_URL"):
        litellm_url = os.getenv("LITELLM_BASE_URL")
        if litellm_url:
            os.environ["OPENAI_BASE_URL"] = litellm_url
    if not os.getenv("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = "gemini/gemini-3-flash-preview"


def set_win32_event_loop_policy() -> None:
    """Use the proactor event loop policy on Windows (required for subprocesses)."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def is_matlab_source(name: str) -> bool:
    """Return True if ``name`` looks like a MATLAB source file (.m or .txt)."""
    return name.endswith(MATLAB_EXTENSIONS)


def read_matlab_source(path: Path) -> str:
    """Read a MATLAB source file, or concatenate all sources under a directory.

    For directories, every MATLAB file is appended using the separator
    ``"\\n\\n--- {file} ---\\n"`` (sorted by file name), matching the historical
    runner output exactly.
    """
    if path.is_dir():
        parts = []
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if is_matlab_source(file):
                    fpath = Path(root) / file
                    parts.append(
                        f"\n\n--- {file} ---\n{fpath.read_text(encoding='utf-8')}"
                    )
        return "".join(parts)
    return path.read_text(encoding="utf-8")


def algorithm_name(path: Path) -> str:
    """Derive an algorithm name: directory name for folders, stem for files."""
    if path.is_dir():
        return path.name
    return path.stem


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` as pretty-printed, UTF-8 JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
