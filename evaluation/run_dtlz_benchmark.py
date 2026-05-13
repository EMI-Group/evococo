import argparse
import sys
import os
import time
import importlib.util
import torch
import csv

import evox
from evomo.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from evox.workflows import StdWorkflow
from evox.metrics import igd

def get_problem(name):
    d_map = {
        "DTLZ1": 7,
        "DTLZ2": 12,
        "DTLZ3": 12,
        "DTLZ4": 12,
        "DTLZ5": 12,
        "DTLZ6": 12,
        "DTLZ7": 21
    }
    d = d_map[name]
    cls = getattr(evox.problems.numerical, name)
    return cls(d=d, m=3), d

def main():
    parser = argparse.ArgumentParser(description="Run an algorithm on a DTLZ problem.")
    parser.add_argument("--algo_file", type=str, required=True, help="Path to the algorithm .py file")
    parser.add_argument("--problem", type=str, required=True, choices=[f"DTLZ{i}" for i in range(1, 8)], help="Problem name")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()

    # Set up device and reproducibility
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Initialize problem
    prob, n_vars = get_problem(args.problem)
    pf = prob.pf()

    # Dynamically load the algorithm module
    module_name = os.path.basename(args.algo_file).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, args.algo_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Find the algorithm class
    algo_class = getattr(module, module_name.replace("-", "").replace("_", ""))
    # If the class name has hyphens removed but is still different, we can search for a subclass of evox.core.Algorithm
    if not hasattr(module, module_name.replace("-", "").replace("_", "")):
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, evox.core.Algorithm) and attr.__name__ != "Algorithm":
                algo_class = attr
                break
    
    # Initialize algorithm
    try:
        algo = algo_class(pop_size=100, n_objs=3, lb=-torch.zeros(n_vars), ub=torch.ones(n_vars))
    except TypeError:
        try:
            algo = algo_class(problem=prob, pop_size=100)
        except TypeError:
            algo = algo_class()

    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = workflow.step

    # Warm-up (not timed)
    jit_state_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed Execution
    start_time = time.perf_counter()
    for _ in range(100):
        jit_state_step()
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exec_time = time.perf_counter() - start_time

    # Evaluate IGD
    fit = workflow.algorithm.fit
    if fit is not None:
        fit = fit[~torch.any(torch.isnan(fit), dim=1)]
        if len(fit) > 0:
            final_igd = igd(fit, pf).item()
        else:
            final_igd = float('inf')
    else:
        final_igd = float('inf')

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write to CSV
    file_exists = os.path.isfile(args.output)
    with open(args.output, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Algorithm", "Problem", "Seed", "IGD", "Execution_Time_s"])
        writer.writerow([module_name, args.problem, args.seed, final_igd, exec_time])

    print(f"[{module_name} | {args.problem} | Seed {args.seed}] IGD: {final_igd:.4f}, Time: {exec_time:.2f}s")

if __name__ == "__main__":
    main()
