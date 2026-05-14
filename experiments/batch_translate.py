import os
import sys
import asyncio
import argparse

# Add parent directory to path to allow importing backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import run_pipeline

async def process_file(file_path, num_repeats, output_dir=None):
    matlab_code = ""
    if os.path.isdir(file_path):
        for root, _, files in os.walk(file_path):
            for file in sorted(files):
                if file.endswith('.m') or file.endswith('.txt'):
                    fpath = os.path.join(root, file)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        matlab_code += f"\n\n--- {file} ---\n{f.read()}"
        algo_name = os.path.basename(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            matlab_code = f.read()
        algo_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n{'='*50}\nProcessing Algorithm: {algo_name}\n{'='*50}")
    
    for i in range(num_repeats):
        run_idx = i + 1
        
        if output_dir:
            out_file = os.path.join(output_dir, f"{algo_name}_run{run_idx}.py")
            if os.path.exists(out_file):
                print(f"\n--- Run {run_idx}/{num_repeats} for {algo_name} already exists, skipping ---")
                continue
                
        print(f"\n--- Run {run_idx}/{num_repeats} for {algo_name} ---")
        
        final_code_captured = []
        
        async def custom_callback(type_, title, message, step_id=None, extra_data=None, is_success=None, icon=None):
            if type_ == "log":
                # To avoid too much spam, we can just print the log. You can comment this out if too verbose.
                print(f"  [{title}] {message}")
            elif type_ == "step_start":
                print(f"\n  >>> Starting {title}: {message}")
            elif type_ == "step_done":
                print(f"  <<< Finished {title} (Success: {is_success})")
            elif type_ == "fatal":
                print(f"  !!! FATAL ERROR: {title} - {message}")
            elif type_ == "result_code":
                final_code_captured.append(message)
                
        try:
            await run_pipeline(matlab_code, custom_callback)
            print(f"--- Run {run_idx} completed ---")
            
            if final_code_captured and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_file = os.path.join(output_dir, f"{algo_name}_run{run_idx}.py")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(final_code_captured[0])
                print(f"Saved result to {out_file}")
                
        except Exception as e:
            print(f"--- Run {run_idx} failed with exception: {e} ---")

async def main():
    parser = argparse.ArgumentParser(description="Batch translate MATLAB algorithms to Python using EvoCoder.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing MATLAB (.m) files")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save the final generated Python files")
    parser.add_argument("--repeats", type=int, default=5, help="Number of times to run the pipeline for each algorithm")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    m_files = []
    for f in os.listdir(input_dir):
        path = os.path.join(input_dir, f)
        if os.path.isdir(path) or f.endswith('.m') or f.endswith('.txt'):
            m_files.append(f)
            
    if not m_files:
        print(f"No algorithms (.m/.txt files or folders) found in {input_dir}.")
        return

    print(f"Found {len(m_files)} algorithm(s) to process. Outputs will be saved to {output_dir}")
    
    for m_file in m_files:
        file_path = os.path.join(input_dir, m_file)
        await process_file(file_path, args.repeats, output_dir)

    print("\nAll batch translations completed.")
    print(f"You can find the generated code files in '{output_dir}', and detailed logs in 'run_history'.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
