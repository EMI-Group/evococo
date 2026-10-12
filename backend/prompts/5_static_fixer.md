# Shared Algorithm Blueprint

{blueprint_plan}

# Agent Role: Static Fixer (Step 5)

The Python code generated in Step 4 has static analysis errors (detected by Ruff).

## Task
1.  **Fix ONLY the Reported Errors**:
    * If it says "undefined name", define it or import it.
    * If it says "syntax error", fix the syntax.
    * If it says "import not used", remove it (unless it's needed for registration).
2.  **STABILITY**: Do NOT change any logic, tensor shapes, or class names. Just fix the static errors.
3.  Preserve all declared selection priorities, directions and key-order comments.
    Fixing an import or type error must not change which candidate wins.

## Output
Output ONLY the fully corrected Python code string.

## Selection Semantics to Preserve
{selection_semantics}

The current blueprint (including the per-site Selection Semantics Contract):
See the Shared Algorithm Blueprint above.

## The Code
{previous_code}

## Error Report
{error_log}
