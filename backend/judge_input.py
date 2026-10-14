"""Compact judge-only views while retaining the original tested source elsewhere."""

import ast
import copy
from collections import Counter


def _split_demo(code: str) -> tuple[str, str, str] | None:
    """Identify a trailing main block with an unambiguous local algo constructor."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    if not tree.body:
        return None
    main = tree.body[-1]
    if not (
        isinstance(main, ast.If)
        and not main.orelse
        and isinstance(main.test, ast.Compare)
        and isinstance(main.test.left, ast.Name)
        and main.test.left.id == "__name__"
        and len(main.test.ops) == 1
        and isinstance(main.test.ops[0], ast.Eq)
        and len(main.test.comparators) == 1
        and isinstance(main.test.comparators[0], ast.Constant)
        and main.test.comparators[0].value == "__main__"
    ):
        return None
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    normalized = copy.deepcopy(main)
    constructors = [
        node.value.func
        for node in ast.walk(normalized)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "algo"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in classes
    ]
    if len(constructors) != 1:
        return None
    class_name = constructors[0].id
    constructors[0].id = "__ALGORITHM_CLASS__"
    body = "".join(code.splitlines(keepends=True)[: main.lineno - 1])
    return body, ast.unparse(normalized), class_name


def compact_judge_candidates(candidates: list[dict]) -> dict:
    """Omit failed source and share equivalent demos among successful candidates."""
    demos = [
        _split_demo(candidate["code_snippet"]) if candidate["success"] else None
        for candidate in candidates
    ]
    counts = Counter(demo[1] for demo in demos if demo is not None)
    shared = []
    indices = {}
    views = []
    for candidate, demo in zip(candidates, demos, strict=True):
        view = candidate.copy()
        if not candidate["success"]:
            view.pop("code_snippet", None)
        elif demo is not None and counts[demo[1]] > 1:
            body, block, class_name = demo
            if block not in indices:
                indices[block] = len(shared)
                shared.append(block)
            view.update(
                code_snippet=body,
                demo_ref=indices[block],
                demo_class_name=class_name,
            )
        views.append(view)
    return {"shared_demo_blocks": shared, "candidates": views}
