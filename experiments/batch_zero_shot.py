import os
import argparse
import asyncio
import sys

# Import the translation function from the existing zero_shot baseline
from zero_shot import zero_shot_translate, MODEL_NAME

async def main():
    parser = argparse.ArgumentParser(description="Batch Zero-Shot MATLAB to Python Translation Baseline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing MATLAB (.m/.txt) files")
    parser.add_argument("--output_dir", type=str, default="experiments/baselines", help="Directory to save generated Python files")
    parser.add_argument("--repeats", type=int, default=5, help="Number of times to run zero-shot for each algorithm")
    parser.add_argument("--mode", type=str, choices=["weak", "strong"], default="strong", help="Baseline mode: 'weak' or 'strong'")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    repeats = args.repeats
    mode = args.mode

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Gather all algorithm files
    m_files = []
    for f in os.listdir(input_dir):
        path = os.path.join(input_dir, f)
        if os.path.isdir(path) or f.endswith('.m') or f.endswith('.txt'):
            m_files.append(f)

    if not m_files:
        print(f"No algorithms found in {input_dir}.")
        return

    print("=======================================")
    print(" [Batch Baseline: Zero-Shot Translation] ")
    print(f" Mode:    {mode.upper()} Baseline")
    print(f" Model:   {MODEL_NAME}")
    print(f" Input:   {input_dir}")
    print(f" Output:  {output_dir}")
    print(f" Repeats: {repeats}")
    print(f" Total Algorithms: {len(m_files)}")
    print("=======================================\n")

    total_tasks = len(m_files) * repeats
    current_task = 0

    for m_file in sorted(m_files):
        file_path = os.path.join(input_dir, m_file)
        
        # Read MATLAB code
        matlab_code = ""
        if os.path.isdir(file_path):
            algo_name = m_file
            for root, _, files in os.walk(file_path):
                for f in sorted(files):
                    if f.endswith('.m') or f.endswith('.txt'):
                        with open(os.path.join(root, f), "r", encoding="utf-8") as file_obj:
                            matlab_code += f"\n\n--- {f} ---\n{file_obj.read()}"
        else:
            algo_name = os.path.splitext(m_file)[0]
            with open(file_path, "r", encoding="utf-8") as file_obj:
                matlab_code = file_obj.read()

        print(f"Processing Algorithm: {algo_name}")

        for i in range(1, repeats + 1):
            current_task += 1
            out_name = f"{algo_name}_zero_shot_{mode}_run{i}.py"
            out_path = os.path.join(output_dir, out_name)
            
            # Skip if already generated (helpful for resuming)
            if os.path.exists(out_path):
                print(f"  [{current_task}/{total_tasks}] Run {i} already exists, skipping.")
                continue

            print(f"  [{current_task}/{total_tasks}] Generating Run {i}...")
            
            python_code = await zero_shot_translate(matlab_code, mode)
            
            if python_code:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(python_code)
                print(f"    -> Saved to {out_name}")
            else:
                print(f"    -> Failed to generate code for Run {i}.")

    print("\nBatch zero-shot translation completed.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
