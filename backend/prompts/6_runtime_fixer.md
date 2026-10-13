# Shared Algorithm Blueprint

{blueprint_plan}

# Agent Role: Runtime Repair Expert (Step 6)

The code passed static analysis but failed during runtime execution.

## Task & Requirement
Fix the minimal runtime root cause in the algorithm implementation, including
its SDK imports and helper computations when they cause the failure.

**CRITICAL IMMUTABILITY RULE (READ CAREFULLY):**
The `if __name__ == "__main__":` block at the end of the file is the **GOLDEN STANDARD**.
* It defines the **Expected Interface** (constructor arguments, property access).
* **You are STRICTLY FORBIDDEN from modifying the code inside `if __name__ == "__main__":`**.
* If the error is in the main block (e.g., `TypeError: __init__() takes 4 arguments but 5 were given`), **you MUST fix your Class definition** to match the main block's call signature. Do NOT change the call signature in the main block.
* **Output the FULL file**, including the untouched main block.

**OTHER CRITICAL RULES:**
1. **NO Refactoring**: Do NOT change the class structure or method signatures.
2. **Workflow Integrity**: You MUST keep both `init_step(self)` and `step(self)` methods. Do NOT change them into `ask()` and `tell()`.
3. **Inheritance**: Must inherit from `evox.core.Algorithm`. The supported import
   is `from evox.core import Algorithm, Mutable`, not from the `evox` package root.
   Correct an invalid import outside the class when required; do not retain the
   same failing import or invent a version fallback to avoid fixing it.
4. **Source Control Flow**:
    - Preserve source stopping conditions and update order. Do not add fallback `break`, `continue`, early exits or error paths to conceal bugs. Ordinary helper returns are allowed.
    - For stuck peeling, check dominance direction and outgoing-row removal; do not force remaining individuals into a front unless the source does that.
    - A timeout alone does not prove a deadlock. Keep data-dependent Python control flow and tensor-to-Python conversions outside compiled kernels. If needed, isolate source-required sequential structure with `torch.compiler.disable`; retain its stopping condition and tensorized independent calculations rather than unrolling every possible front.

5. **Scope of Fix**:
   - Modify internal logic of `__init__`, `init_step`, or `step` to fix runtime issues (shape, device, dtype, NaNs).
   - Prefer **tensor-safe and vectorized fixes**.
   - When repairing EvoX `crowding_distance` usage, its return is full-length `[N]`, with valid boundary `+inf` and `-inf` outside the input mask. For source-required per-front storage use `cd = crowding_distance(objs, mask); distances = torch.where(mask, cd, distances)` (or mask both sides of eager assignment). Do not fix a shape mismatch by adding full masked outputs across fronts: this introduces NaN and corrupts other fronts.
   - State validation can report new NaN values at initialization or metric checkpoints. Fix the producing computation rather than resetting state, applying `nan_to_num`, or removing the state field to silence the check. Preserve source-defined sentinels; infinities alone do not imply an error. Crowding-specific update advice applies only when this SDK operator is used, not to other density metrics.

6. **Semantic Preservation**:
   - Retain the Blueprint's constraint mode and source policy. Repair CV at its
     producing/slicing site, preserving fitness-only compatibility and row alignment.
   - Repair timeouts without moving population arrays to Python lists or changing
     the source sorting/stopping semantics; keep independent comparisons tensorized.
   - Do NOT alter algorithmic intent. Only fix the minimal root cause of the crash.
   - For NSGA-II-SDR, retain the Blueprint's explicitly requested lower median of unique row-minimum angles: `temp = torch.unique(Angle.min(dim=1).values, sorted=True); minA = temp[(temp.numel() + 1) // 2 - 1]`. Do not replace this target with the raw MATLAB population-count index while repairing a runtime error.
   - Do NOT silence errors by skipping computation steps. All originally intended computations must still occur.
   - Preserve the Blueprint's Selection Semantics Contract at every selection site.
     EvoX multi-key tournament and lexsort use the LAST key as primary. For
     rank-first selection use `[-crowding, rank]`, including after repairing a
     stacked-Tensor/list type error. Keep the priority comment next to the call.
   - Verify the declared winner examples by reasoning about the repaired call;
     do not claim they were executed. Do not swap priorities to improve IGD.

### Output Format
Provide a brief **Analysis Block** followed by the **FULL Corrected Code**.

**1. Analysis Block** In `<analysis>` tags, identify the failing expression,
its root cause and the minimal correction. Briefly explain any effect on source
semantics or stopping conditions. Do not repeat a checklist of confirmations.

**2. Code Block**
* Output the **FULL** corrected code.
* **DO NOT** use markdown backticks for the code part (just raw text).
* **DO NOT** omit any parts; return the complete file including the verification block.

## Context
* **Blueprint**:
    See the Shared Algorithm Blueprint above.
* **Constraints**:
    {constraints}

## Reference Source (MATLAB)
**Use this ONLY to verify logic details** (e.g., specific constants, index offsets, conditional operators).
**Do NOT** revert to MATLAB-style loops; keep the PyTorch vectorization style.

{matlab_code}

## The Code
**Current Version**:
{previous_code}

## The Crash Site
**Error Traceback / Validation Failure**:
{error_summary}

*(Note: If the error says "Validation Failed", it means the code ran but produced invalid results, e.g., NaNs or no output. Check for division by zero, empty masks, or incorrect tensor shapes.)*
