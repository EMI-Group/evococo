import os
import ast
import re
import argparse
import asyncio
import json
import uuid

# Add evocoder root to sys.path to allow importing backend
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.executor import check_syntax_with_ruff, execute_code_trial

BOILERPLATE = """
# === INJECTED BENCHMARK HARNESS ===
if __name__ == '__main__':
    import torch
    import evox
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cpu")
    
    prob = DTLZ2(m=3)
    pf = prob.pf()
    
    try:
        # Standard EvoCoder Blueprint signature
        algo = {CLASS_NAME}(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    except TypeError:
        try:
            # Fallback 1: NSGA-II-SDR_strong.py style (Problem provided)
            algo = {CLASS_NAME}(problem=prob, pop_size=100)
        except TypeError:
            # Fallback 2: Basic generic
            algo = {CLASS_NAME}()

    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = workflow.step

    for i in range(50):
        jit_state_step()
        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            # Filter NaNs for valid metric calculation
            if fit is not None:
                fit = fit[~torch.any(torch.isnan(fit), dim=1)]
                if len(fit) > 0:
                    print(f"IGD: {igd(fit, pf)}") # Standardized keyword so parser can catch it!
"""

def test_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

async def main():
    parser = argparse.ArgumentParser(description="Evaluate generated algorithm codes.")
    parser.add_argument("-d", "--dir", type=str, required=True, help="Directory containing .py scripts to benchmark")
    args = parser.parse_args()

    results = []

    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a directory.")
        return

    py_files = [f for f in os.listdir(args.dir) if f.endswith(".py") and f != "zero_shot.py"]

    if not py_files:
        print(f"No valid .py files to benchmark in {args.dir}")
        return

    print("=" * 60)
    print(f"{'File':<30} | {'Syntax':<6} | {'Static':<6} | {'Exec':<6} | {'Optim':<6}")
    print("-" * 60)

    for py_file in py_files:
        path = os.path.join(args.dir, py_file)
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        # Metric 1: Syntax
        syntax_pass = test_syntax(code)
        
        # Metric 2: Static
        static_pass = False
        if syntax_pass:
            static_pass, _ = await check_syntax_with_ruff(code, session_id=f"bench_{uuid.uuid4().hex[:6]}")

        # Metric 3 & 4: Execution & Optimization
        exec_pass = False
        optim_pass = False

        if syntax_pass:
            modified_code = code
            if "if __name__ ==" not in code.replace(' ', ''):
                # Need to inject harness
                m = re.search(r"class\s+([A-Za-z0-9_]+)\s*\(", code)
                if m:
                    class_name = m.group(1)
                    modified_code += BOILERPLATE.replace("{CLASS_NAME}", class_name)
                
            report = await execute_code_trial(modified_code, session_id=f"run_{uuid.uuid4().hex[:6]}", filename=py_file)
            
            exec_pass = report.get("success", False)
            
            # Override with strict 20% improvement condition for convergence
            igds = report.get("igd_history", [])
            optim_pass = len(igds) >= 2 and igds[-1] < igds[0] * 0.8
            
            if exec_pass and report.get("has_nan", False):
                exec_pass = False # Marked failure if NaNs heavily persist

        results.append({
            "file": py_file,
            "syntax": syntax_pass,
            "static": static_pass,
            "exec": exec_pass,
            "optim": optim_pass
        })

        print(f"{py_file[:28]:<30} | {str(syntax_pass):<6} | {str(static_pass):<6} | {str(exec_pass):<6} | {str(optim_pass):<6}")

    print("=" * 60)

    with open(os.path.join(args.dir, "benchmark_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Summary JSON saved to {os.path.join(args.dir, 'benchmark_report.json')}")

if __name__ == "__main__":
    asyncio.run(main())
