"""
Global Configuration for EvoCoder
"""

import os

# --- Tournament Engine Settings ---
NUM_BRANCHES = 6

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
MAX_RETAINED_WORKSPACES = 50

# --- Path Configurations ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
BASE_WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "temp_workspace")
HISTORY_DIR = os.path.join(PROJECT_ROOT, "run_history")
PROMPTS_DIR = os.path.join(BACKEND_DIR, "prompts")
DATABASE_DIR = os.path.join(BACKEND_DIR, "database")
RULES_DB_PATH = os.path.join(DATABASE_DIR, "rag_db.json")
GLOBAL_SPEC_PATH = os.path.join(PROMPTS_DIR, "0_global_spec.md")
