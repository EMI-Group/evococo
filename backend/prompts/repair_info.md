# ⚠️ CRITICAL REPAIR TASK

The code you generated previously failed to execute.

## Error Traceback
{error_log}

## Mandatory Repair Rules (READ CAREFULLY)

1.  **PRESERVE ALGORITHM IDENTITY**: 
    * You are implementing the **exact algorithm** defined in the **Blueprint** provided earlier.
    * **DO NOT** switch algorithms (e.g., if Blueprint says "NSGA-II/SDR", do **NOT** implement "NSGA-III").
    * **DO NOT** change the Class Name or the fundamental logic flow.

2.  **Fix Strategy**:
    * Analyze the traceback above.
    * **Only** modify the specific lines causing the error (e.g., shape mismatch, wrong API, or import error).
    * **Keep the rest of the code intact.**

3.  **Strict Constraints**:
    * **NO XML**: Ignore any previous references to XML; follow the Markdown Blueprint.
    * **Mutable Semantics**: Continue to use `self.pop = Mutable(...)`. Never use `.value`.
    * **Tensor Shapes**: Adhere strictly to the defined Tensor Map.

## Output
Output **ONLY** the corrected, fully executable Python code string.