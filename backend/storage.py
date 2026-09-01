"""Data-loading and artifact I/O helpers for the EvoCoCo pipeline.

Extracted from backend/engine.py to keep the engine focused on orchestration.
All helpers preserve their original fallback behaviors:
- load_rag_db returns [] when the DB file is missing or unparseable.
- load_global_spec / load_resource return "" when the file is missing/unreadable.
- save_artifact is best-effort: failures are logged and skipped.
"""

import datetime
import json
import logging
import os
from pathlib import Path

from .config import (
    GLOBAL_SPEC_PATH,
    HISTORY_DIR,
    MAX_RETAINED_WORKSPACES,
    PROMPTS_DIR,
    RULES_DB_PATH,
)
from .executor import cleanup_old_workspaces

logger = logging.getLogger(__name__)


def load_rag_db():
    """Load RAG rule database from JSON file"""
    path = Path(RULES_DB_PATH)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "rules" in data:
            return data["rules"]
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001 - intentional: unparseable DB falls back to no rules
        logger.warning("Failed to load RAG database from %s: %s", RULES_DB_PATH, exc)
        return []


def load_global_spec():
    """Load global specification from Markdown file"""
    path = Path(GLOBAL_SPEC_PATH)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:  # noqa: BLE001 - intentional: unreadable spec falls back to ""
            logger.warning(
                "Failed to load global spec from %s: %s", GLOBAL_SPEC_PATH, exc
            )
    return ""


def load_resource(filename):
    """Load resource files (SDK, Examples, Reference Context)"""
    path = Path(PROMPTS_DIR) / filename
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:  # noqa: BLE001 - intentional: unreadable resource falls back to ""
            logger.warning("Failed to load resource %s: %s", filename, exc)
    return ""


def ensure_history_dir(algo_name="UnknownAlgo"):
    """Ensure history directory exists, and return independent timestamp directory for current run"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    timestamp = (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .strftime("%Y%m%d_%H%M%S")
    )
    run_dir = os.path.join(HISTORY_DIR, f"{timestamp}_{algo_name}")
    os.makedirs(run_dir)

    # Trigger global history cleanup
    cleanup_old_workspaces(HISTORY_DIR, max_retained=MAX_RETAINED_WORKSPACES)

    return run_dir


def save_artifact(run_dir, filename, content):
    """Save intermediate artifacts (best-effort: failures are logged and skipped)"""
    path = os.path.join(run_dir, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(content, (dict, list)):
                f.write(json.dumps(content, indent=2, ensure_ascii=False))
            else:
                f.write(str(content))
    except Exception as exc:  # noqa: BLE001 - intentional: best-effort artifact saving
        logger.warning("Failed to save artifact %s: %s", path, exc)
