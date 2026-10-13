"""Read-only hints about generated host scalar extraction and compilation scope."""

import ast


def review_tensor_sync(code: str) -> dict:
    """Report call sites, not runtime synchronization counts or inferred tensor types."""
    report = {
        "analysis_kind": "static_call_sites_not_runtime_sync_counts",
        "item_sites": [],
        "scalar_cast_sites": [],
        "tensor_condition_sites": [],
        "whole_step_disabled": [],
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        report["syntax_error"] = str(error)
        return report

    imports = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def qualified(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified(node.value)}.{node.attr}"
        return ""

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.in_function = 0

        def record(self, kind, node):
            report[kind].append(
                {
                    "line": node.lineno,
                    "scope": ".".join(self.scope),
                    "expression": ast.unparse(node)[:160],
                }
            )

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            self.in_function += 1
            if node.name == "step":
                for decorator in node.decorator_list:
                    target = (
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                    if qualified(target) in {
                        "torch.compiler.disable",
                        "torch._dynamo.disable",
                    }:
                        self.record("whole_step_disabled", node)
            self.generic_visit(node)
            self.in_function -= 1
            self.scope.pop()

        def visit_Call(self, node):
            if self.in_function:
                if isinstance(node.func, ast.Attribute) and node.func.attr == "item":
                    self.record("item_sites", node)
                if qualified(node.func) in {
                    "float",
                    "int",
                    "bool",
                    "builtins.float",
                    "builtins.int",
                    "builtins.bool",
                }:
                    self.record("scalar_cast_sites", node)
            self.generic_visit(node)

        def visit_If(self, node):
            self.review_condition(node)
            self.generic_visit(node)

        def visit_While(self, node):
            self.review_condition(node)
            self.generic_visit(node)

        def review_condition(self, node):
            if self.in_function and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"any", "all", "sum", "min", "max"}
                for call in ast.walk(node.test)
            ):
                self.record("tensor_condition_sites", node.test)

    Visitor().visit(tree)
    return report
