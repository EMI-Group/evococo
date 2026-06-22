import os
import re
import sys
import json
import tempfile
import subprocess
import matplotlib.pyplot as plt
import numpy as np

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
                    print(f"IGD: {igd(fit, pf)}")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")
"""

def classify_error(code_path):
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        import ast
        try:
            ast.parse(code)
        except SyntaxError as se:
            return "Syntax Error", str(se)

        modified_code = "import torch\ntorch.set_num_threads(1)\ntorch.compile = lambda fn, *args, **kwargs: fn\n" + code
        if "if __name__ ==" not in code.replace(' ', ''):
            m = re.search(r"class\s+([A-Za-z0-9_]+)\s*\(", code)
            if m:
                class_name = m.group(1)
                modified_code += BOILERPLATE.replace("{CLASS_NAME}", class_name)
            else:
                return "Unknown Class Name", "Could not find Algorithm class definition."

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(modified_code)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=12
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        stdout = res.stdout
        stderr = res.stderr

        if res.returncode == 0:
            igds = []
            for line in stdout.split("\n"):
                if "IGD:" in line:
                    try:
                        igds.append(float(line.split("IGD:")[-1].strip()))
                    except:
                        pass
            if igds and igds[-1] < 0.25:
                return "Success", ""
            else:
                last_igd_val = igds[-1] if igds else float('inf')
                return "Poor Convergence", f"Last IGD: {last_igd_val:.4f}"

        err_msg = stderr.lower()
        if "shape" in err_msg or "size" in err_msg or "dimension" in err_msg or "matmul" in err_msg or "broadcast" in err_msg:
            return "Shape Mismatch", stderr.strip().split("\n")[-1]
        elif "evox" in err_msg or "mutable" in err_msg or "parameter" in err_msg or "state" in err_msg or "typeerror" in err_msg:
            return "EvoX API Misuse", stderr.strip().split("\n")[-1]
        else:
            match = re.search(r"(\w+Error):", stderr)
            err_type = match.group(1) if match else "Runtime Crash"
            return f"Runtime Crash ({err_type})", stderr.strip().split("\n")[-1]

    except subprocess.TimeoutExpired:
        return "Timeout / Hang", "Script execution took longer than 12 seconds."
    except Exception as e:
        return "Unknown Error", str(e)

def main():
    configs = {
        "Full EvoCoCo": "./experiments/ablation_subset_results/results_full_combined",
        "w/o Architect": "./experiments/ablation_subset_results/results_no_architect",
        "w/o Runtime Fixer": "./experiments/ablation_subset_results/results_no_runtime_fixer",
        "w/o RAG Rules": "./experiments/ablation_subset_results/results_no_rag",
        "w/o Multi-Branch Selection": "./experiments/ablation_subset_results/results_single_branch",
    }

    stats = {}
    detailed_observations = {}

    for name, path in configs.items():
        report_path = os.path.join(path, "benchmark_report.json")
        if not os.path.exists(report_path):
            print(f"Skipping {name}, report file not found at {report_path}")
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = len(data)
        if total == 0:
            print(f"Skipping {name}, report is empty.")
            continue

        syntax_pass = sum(1 for item in data if item["syntax"])
        static_pass = sum(1 for item in data if item["static"])
        exec_pass = sum(1 for item in data if item["exec"])
        optim_pass = sum(1 for item in data if item["optim"])
        
        valid_igds = [item["final_igd"] for item in data if item["optim"] and item["final_igd"] != float("inf")]
        avg_igd = np.mean(valid_igds) if valid_igds else float('nan')

        stats[name] = {
            "Total": total,
            "Syntax Success (%)": syntax_pass / total * 100,
            "Static Pass (%)": static_pass / total * 100,
            "Exec Success (%)": exec_pass / total * 100,
            "Optim Success (%)": optim_pass / total * 100,
            "Avg IGD": avg_igd,
        }

        shape_mismatches = 0
        evox_api_misuses = 0
        static_issues = total - static_pass
        poor_convergences = 0
        runtime_crashes = {}

        for item in data:
            py_file = item["file"]
            file_path = os.path.join(path, py_file)

            if not os.path.exists(file_path):
                continue

            if item.get("optim", False):
                label, detail = "Success", ""
            elif item.get("exec", False):
                label, detail = "Poor Convergence", ""
            else:
                label, detail = classify_error(file_path)

            if label == "Shape Mismatch":
                shape_mismatches += 1
                runtime_crashes["Shape Mismatch"] = runtime_crashes.get("Shape Mismatch", 0) + 1
            elif label == "EvoX API Misuse":
                evox_api_misuses += 1
                runtime_crashes["EvoX API Misuse"] = runtime_crashes.get("EvoX API Misuse", 0) + 1
            elif label == "Poor Convergence":
                poor_convergences += 1
            elif "Runtime Crash" in label:
                crash_type = label.replace("Runtime Crash (", "").replace(")", "")
                runtime_crashes[crash_type] = runtime_crashes.get(crash_type, 0) + 1

        detailed_observations[name] = {
            "Shape Mismatch Count": shape_mismatches,
            "EvoX API Misuse Count": evox_api_misuses,
            "Static Issues Count": static_issues,
            "Poor Convergence Count": poor_convergences,
            "Runtime Crash Subtypes": runtime_crashes,
        }

    if not stats:
        print("No reports found to aggregate.")
        return

    # Write LaTeX Table
    print("\n" + "="*50)
    print("=== LaTeX Table Code (Subset) ===")
    print("="*50)
    print("\\begin{table*}[t]")
    print("  \\centering")
    print("  \\caption{Ablation analysis on selected subset of 12 algorithms.}")
    print("  \\label{tab:ablation_subset_results}")
    print("  \\begin{tabular}{lccccc}")
    print("    \\hline")
    print("    \\textbf{Variant} & \\textbf{Syntax (\\%)} & \\textbf{Static Pass (\\%)} & \\textbf{Exec (\\%)} & \\textbf{Optimized (\\%)} & \\textbf{Avg IGD} \\\\")
    print("    \\hline")
    for name, val in stats.items():
        igd_str = f"{val['Avg IGD']:.4f}" if not np.isnan(val['Avg IGD']) else "N/A"
        print(f"    {name:<26} & {val['Syntax Success (%)']:.1f}\\% & {val['Static Pass (%)']:.1f}\\% & {val['Exec Success (%)']:.1f}\\% & {val['Optim Success (%)']:.1f}\\% & {igd_str} \\\\")
    print("    \\hline")
    print("  \\end{tabular}")
    print("\\end{table*}")

    # Write Detailed Observations Report to MD
    report_file = "./experiments/benchmark_results_final/ablation_subset_table.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Subset Ablation Experiment Observations & Metrics (12 Algorithms)\n\n")
        f.write("## 1. Quantitative Metrics Table\n\n")
        f.write("| Variant | Syntax Success (%) | Static Pass (%) | Exec Success (%) | Optimized (%) | Avg IGD |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for name, val in stats.items():
            igd_str = f"{val['Avg IGD']:.4f}" if not np.isnan(val['Avg IGD']) else "N/A"
            f.write(f"| {name} | {val['Syntax Success (%)']:.1f}% | {val['Static Pass (%)']:.1f}% | {val['Exec Success (%)']:.1f}% | {val['Optim Success (%)']:.1f}% | {igd_str} |\n")
        
        f.write("\n## 2. Detailed Observations by Variant\n\n")
        for name, obs in detailed_observations.items():
            f.write(f"### {name}\n")
            f.write(f"- **Static Issues**: {obs['Static Issues Count']} runs\n")
            f.write(f"- **Poor Convergence**: {obs['Poor Convergence Count']} runs\n")
            f.write(f"- **EvoX API Misuse**: {obs['EvoX API Misuse Count']} occurrences\n")
            f.write(f"- **Shape Mismatch**: {obs['Shape Mismatch Count']} occurrences\n")
            
            f.write("- **Runtime Crash Subtype Breakdown**:\n")
            subtypes = obs['Runtime Crash Subtypes']
            if subtypes:
                for k, v in subtypes.items():
                    f.write(f"  - *{k}*: {v} occurrence(s)\n")
            else:
                f.write("  - None\n")
            f.write("\n")

    print(f"\nDetailed report saved to {report_file}")

    # Plot results
    categories = ["Syntax", "Static Pass", "Exec", "Optimized"]
    x = np.arange(len(categories))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (name, val) in enumerate(stats.items()):
        scores = [
            val["Syntax Success (%)"],
            val["Static Pass (%)"],
            val["Exec Success (%)"],
            val["Optim Success (%)"]
        ]
        ax.bar(x + (i - 1.5) * width, scores, width, label=name)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Ablation Study (Subset): Success Rates on Different Pipeline Stages")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="lower left")
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_out = "./experiments/ablation_subset_results/ablation_subset_results.png"
    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
    plt.savefig(plot_out, dpi=300)
    print(f"Comparison chart saved to {plot_out}")

if __name__ == "__main__":
    main()
