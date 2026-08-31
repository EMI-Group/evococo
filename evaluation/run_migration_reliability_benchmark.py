"""Evaluate migration reliability of generated EvoX algorithms."""

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

    import time
    start_time = time.time()
    for i in range(100):
        jit_state_step()
        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            # Filter NaNs for valid metric calculation
            if fit is not None:
                fit = fit[~torch.any(torch.isnan(fit), dim=1)]
                if len(fit) > 0:
                    print(f"IGD: {igd(fit, pf)}") # Standardized keyword so parser can catch it!
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")
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
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent benchmark workers")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    report_file = os.path.join(args.dir, "benchmark_report.json")
    results = []
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    processed_files = {r["file"] for r in results}

    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a directory.")
        return

    py_files = [f for f in os.listdir(args.dir) if f.endswith(".py") and f != "zero_shot.py"]

    if not py_files:
        print(f"No valid .py files to benchmark in {args.dir}")
        return

    print("=" * 90)
    print(f"{'File':<30} | {'Syntax':<6} | {'Static':<6} | {'Exec':<6} | {'Optim':<6} | {'IGD':<8} | {'Time(s)'}")
    print("-" * 90)

    sem = asyncio.Semaphore(args.workers)
    write_lock = asyncio.Lock()

    async def evaluate_single_file(py_file):
        if py_file in processed_files:
            return
            
        async with sem:
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
            final_igd = float("inf")
            exec_time = -1.0

            if syntax_pass:
                # Prepend global torch.compile bypass and set threads to 1 to avoid JIT compilation overhead, hangs, and CPU thrashing
                modified_code = "import torch\ntorch.set_num_threads(1)\ntorch.compile = lambda fn, *args, **kwargs: fn\n" + code
                if not re.search(r'''if\s+__name__\s*==\s*["']__main__["']\s*:''', code):
                    # Need to inject harness
                    m = re.search(r"class\s+([A-Za-z0-9_]+)\s*\(", code)
                    if m:
                        class_name = m.group(1)
                        modified_code += BOILERPLATE.replace("{CLASS_NAME}", class_name)
                    
                report = await execute_code_trial(modified_code, session_id=f"run_{uuid.uuid4().hex[:6]}", filename=py_file)
                
                exec_pass = report.get("success", False)
                final_igd = report.get("last_igd", float("inf"))
                
                # Extract execution time independently without modifying executor.py
                exec_time = -1.0
                stdout_str = report.get("stdout", "")
                t_match = re.search(r"Execution time.*?:\s*([0-9]*[.]?[0-9]+)s", stdout_str, re.IGNORECASE)
                if t_match:
                    try:
                        exec_time = float(t_match.group(1))
                    except ValueError:
                        pass
                
                # Override with absolute IGD threshold (< 0.25) for convergence on DTLZ2
                igds = report.get("igd_history", [])
                optim_pass = len(igds) >= 2 and igds[-1] < 0.25
                
                if exec_pass and report.get("has_nan", False):
                    exec_pass = False # Marked failure if NaNs heavily persist
                    
                # If it successfully converged, it executed properly
                if optim_pass and final_igd != float("inf"):
                    exec_pass = True

            async with write_lock:
                results.append({
                    "file": py_file,
                    "syntax": syntax_pass,
                    "static": static_pass,
                    "exec": exec_pass,
                    "optim": optim_pass,
                    "final_igd": final_igd,
                    "exec_time": exec_time
                })

                igd_str = f"{final_igd:.4f}" if final_igd != float("inf") else "inf"
                time_str = f"{exec_time:.2f}" if exec_time >= 0 else "N/A"
                print(f"{py_file[:28]:<30} | {str(syntax_pass):<6} | {str(static_pass):<6} | {str(exec_pass):<6} | {str(optim_pass):<6} | {igd_str:<8} | {time_str}")
                
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    tasks = [evaluate_single_file(f) for f in py_files]
    await asyncio.gather(*tasks)

    print("=" * 90)
    print(f"Summary JSON saved to {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
