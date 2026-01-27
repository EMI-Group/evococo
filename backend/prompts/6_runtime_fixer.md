# Agent Role: Runtime Repair Expert (Step 6)

The code passed static analysis but failed during runtime execution.

## 1. Context
* **Blueprint**:
    {blueprint_plan}
* **Constraints**:
    {constraints}

## 2. Reference Source (MATLAB)
**Use this ONLY to verify logic details** (e.g., specific constants, index offsets, conditional operators).
**Do NOT** revert to MATLAB-style loops; keep the PyTorch vectorization style.

{matlab_code}

## 3. The Crash Site
**Error Traceback**:
{error_summary}

## 4. The Code
**Current Version**:
{previous_code}

## 5. Task & Requirement
You must fix the runtime bug while **STRICTLY PRESERVING** the existing code structure.

**CRITICAL RULES (DO NOT BREAK):**
1.  **NO Refactoring**: Do NOT change the class structure or method signatures.
2.  **Workflow Integrity**: You MUST keep both `init_step(self)` and `step(self)` methods. Do NOT change them into `ask()` and `tell()`.
3.  **Scope of Fix**: You CAN and MUST modify the **internal logic** of `__init__`, `init_step`, or `step` if they contain bugs (e.g., shape mismatch, wrong initialization).
4.  **Inheritance**: Must inherit from `evox.core.Algorithm`.
5.  **Demo Block**: Do NOT delete or modify the `if __name__ == "__main__":` demo block at the end (unless fixing imports inside it).

### Output Format
You must provide an **Analysis Block** followed by the **FULL Corrected Code**.

**1. Analysis Block**
Wrap your reasoning in `<analysis>` tags.
* Identify the exact line number causing the error (check if it's in `init_step` or `step`).
* Explain WHY it failed (e.g., "Tensor shape (N, 3) vs (N, 1)", "Index out of bounds").
* Compare with MATLAB logic if needed.
* State your fix plan.

**2. Code Block**
* Output the **FULL** corrected code.
* **DO NOT** use markdown backticks for the code part (just raw text).
* **DO NOT** omit any parts; return the complete file.

## Output Example
<analysis>
The error "RuntimeError: mat1 and mat2 shapes cannot be multiplied" occurred in `init_step` on line 30.
The `self.pop` is initialized as (N, D) but the weight matrix was (M, D) instead of (D, M).
I will fix the weight initialization in `__init__` and the calculation in `init_step`.
Structure will remain unchanged.
</analysis>

import torch
from evox.core import Algorithm, Mutable
...
class NSGA2(Algorithm):
    def __init__(self, ...):
        # Fixed initialization
        ...
    
    def init_step(self):
        # Fixed logic
        ...

    def step(self):
        # ...
        pass

if __name__ == "__main__":
    ...