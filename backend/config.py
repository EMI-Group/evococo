"""
Global Configuration for EvoCoCo
"""

import os
import shutil
from dotenv import load_dotenv

# --- Path Configurations (Pre-defined for env loading) ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
ENV_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, ".env.example")

# Auto-copy .env.example if .env doesn't exist
if not os.path.exists(ENV_PATH) and os.path.exists(ENV_EXAMPLE_PATH):
    shutil.copy(ENV_EXAMPLE_PATH, ENV_PATH)
    print(f"\n⚠️ Created {ENV_PATH} from template. Please update your API keys in it.\n")

# Load environment variables FIRST before setting config defaults
load_dotenv(ENV_PATH)

# --- Tournament Engine Settings ---
NUM_BRANCHES = 6

# --- LLM Generation Settings ---
# Reasoning effort for supported models (e.g., o1/o3-mini).
# Standard Options: "low", "medium", "high"
# Special Option: "minimal" (Exclusive to Gemini 3 Flash models via LiteLLM for absolute minimum thinking)
REASONING_EFFORT = "minimal"
LLM_PROVIDERS = {
    "zhipu": {
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        "model": os.getenv("ZHIPU_MODEL", "GLM-5.1"),
        "api_key_env": "ZHIPU_API_KEY"
    },
    "deepseek-v4-pro": {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("DEEPSEEK_MODEL_PRO", "deepseek-v4-pro"),
        "api_key_env": "DEEPSEEK_API_KEY"
    },
    "gemini": {
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        "model": os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        "api_key_env": "GEMINI_API_KEY"
    },
    "custom": {
        "base_url": os.getenv("CUSTOM_BASE_URL", "http://localhost:4000/v1"),
        "model": os.getenv("CUSTOM_MODEL", "custom-model"),
        "api_key_env": "CUSTOM_API_KEY"
    }
}
# Tensorization strategies to inject for each branch
STRATEGIES_SHORT = [
    "BROADCASTING (No Loops)",
    "EINSUM OPTIMIZATION",
    "MASKED OPS (No If/Else)",
    "IN-PLACE EFFICIENCY",
    "ADVANCED OPS (cdist)",
    "JIT-COMPLIANT PEELING",
]

STRATEGIES_FULL = [
    "STRATEGY: BROADCASTING EXPERT. Use standard PyTorch broadcasting (unsqueeze, expand) for all matrix operations. Strictly NO for-loops.",
    "STRATEGY: EINSUM OPTIMIZATION. Use `torch.einsum` for all matrix multiplications and dimension reductions. It is cleaner and faster.",
    "STRATEGY: MASKED OPERATIONS. Avoid `if/else` logic. Use `torch.where`, `torch.masked_fill` to handle conditional logic on tensors.",
    "STRATEGY: IN-PLACE EFFICIENCY. Minimize memory overhead. Use in-place operations (`add_`, `mul_`) where possible.",
    "STRATEGY: ADVANCED OPS. Use high-level PyTorch functions like `torch.cdist`, `torch.linalg.norm` instead of manual formulas.",
    "STRATEGY: JIT-COMPLIANT PEELING. Strictly avoid CPU-GPU sync. NO `.item()` or `.tolist()`. Pre-allocate fixed-size tensor buffers (e.g. `torch.empty`) and use tensor slice indexing `buffer[offset:offset+N] = ...` instead of Python dynamically growing `lists`.",
]

# --- Static Checker (Ruff) Settings ---
IGNORE_RUFF_CODES = [
    "E501",
    "E402",
    "E722",
    "E731",
    "E741",
    "E701",
    "E702",
    "E703",
    "I",
]

# --- Cleanup Settings ---
MAX_RETAINED_WORKSPACES = 1000

# --- Path Configurations ---
BASE_WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "temp_workspace")
HISTORY_DIR = os.path.join(PROJECT_ROOT, "run_history")
PROMPTS_DIR = os.path.join(BACKEND_DIR, "prompts")
DATABASE_DIR = os.path.join(BACKEND_DIR, "database")
RULES_DB_PATH = os.path.join(DATABASE_DIR, "rag_db.json")
GLOBAL_SPEC_PATH = os.path.join(PROMPTS_DIR, "0_global_spec.md")
