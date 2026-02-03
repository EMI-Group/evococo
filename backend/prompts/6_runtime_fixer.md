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
**Error Traceback / Validation Failure**:
{error_summary}

*(Note: If the error says "Validation Failed", it means the code ran but produced invalid results, e.g., NaNs or no output. Check for division by zero, empty masks, or incorrect tensor shapes.)*

## 4. The Code
**Current Version**:
{previous_code}

## 5. Task & Requirement
You must fix the runtime bug inside `class <YourAlgoName>`.

**CRITICAL IMMUTABILITY RULE (READ CAREFULLY):**
The `if __name__ == "__main__":` block at the end of the file is the **GOLDEN STANDARD**.
* It defines the **Expected Interface** (constructor arguments, property access).
* **You are STRICTLY FORBIDDEN from modifying the code inside `if __name__ == "__main__":`**.
* If the error is in the main block (e.g., `TypeError: __init__() takes 4 arguments but 5 were given`), **you MUST fix your Class definition** to match the main block's call signature. Do NOT change the call signature in the main block.
* **Output the FULL file**, including the untouched main block.

**OTHER CRITICAL RULES:**
1. **NO Refactoring**: Do NOT change the class structure or method signatures.
2. **Workflow Integrity**: You MUST keep both `init_step(self)` and `step(self)` methods. Do NOT change them into `ask()` and `tell()`.
3. **Inheritance**: Must inherit from `evox.core.Algorithm`.
4. **NO NEW CONTROL-FLOW ESCAPES (STRICT)**:
    - **FORBIDDEN**: You MUST NOT introduce any new `break`, `continue`, `return`, or `raise` statements to fix logic errors.
    - **Loop Fixing Rule**: If a `while` loop might get stuck (e.g., `mask` is empty or infinite loop detected), **DO NOT use `break`**.
        - **CORRECT FIX**: Implement a "Deadlock Breaker". If no individuals are selected in a round, force select **ALL remaining individuals** (e.g., `if mask.sum() == 0: mask = remaining_indices`).
    - **Reasoning**: Using `break` drops valid individuals and corrupts population size. Using "Deadlock Breaker" preserves data integrity and ensures the algorithm proceeds.

5. **Scope of Fix**:
   - Modify internal logic of `__init__`, `init_step`, or `step` to fix runtime issues (shape, device, dtype, NaNs).
   - Prefer **tensor-safe and vectorized fixes**.

6. **Semantic Preservation**:
   - Do NOT alter algorithmic intent. Only fix the minimal root cause of the crash.
   - Do NOT silence errors by skipping computation steps. All originally intended computations must still occur.

### Output Format
You must provide an **Analysis Block** followed by the **FULL Corrected Code**.

**1. Analysis Block** Wrap your reasoning in `<analysis>` tags.
* Identify the exact line number causing the error.
* Explain WHY it failed (e.g., "Tensor shape mismatch", "Device mismatch", "Deadlock condition").
* State your fix plan (e.g., "Implement Deadlock Breaker instead of break").
* Explicitly confirm: **"No new break/continue/early-return/raise statements were added."**

**2. Code Block**
* Output the **FULL** corrected code.
* **DO NOT** use markdown backticks for the code part (just raw text).
* **DO NOT** omit any parts; return the complete file including the verification block.