import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# 1. Load dotenv and export Litellm configuration as OpenAI env vars BEFORE importing one_shot
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

if not os.getenv("OPENAI_API_KEY"):
    litellm_key = os.getenv("LITELLM_API_KEY")
    if litellm_key:
        os.environ["OPENAI_API_KEY"] = litellm_key
if not os.getenv("OPENAI_BASE_URL"):
    litellm_url = os.getenv("LITELLM_BASE_URL")
    if litellm_url:
        os.environ["OPENAI_BASE_URL"] = litellm_url
if not os.getenv("OPENAI_MODEL"):
    os.environ["OPENAI_MODEL"] = "gemini/gemini-3-flash-preview"

# Ensure experiments directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from one_shot import one_shot_translate, MODEL_NAME

async def worker(sem, m_file, run_idx, input_dir, output_dir, stats):
    async with sem:
        file_path = os.path.join(input_dir, m_file)
        algo_name = os.path.splitext(m_file)[0]
        out_name = f"{algo_name}_run{run_idx}.py"
        out_path = os.path.join(output_dir, out_name)
        
        if os.path.exists(out_path):
            print(f"[Skipped] {out_name} already exists.")
            stats["skipped"] += 1
            return

        print(f"[Generating] {algo_name} Run {run_idx}...")
        
        with open(file_path, "r", encoding="utf-8") as file_obj:
            matlab_code = file_obj.read()
            
        for attempt in range(5):
            try:
                python_code = await one_shot_translate(matlab_code)
                if python_code and "class" in python_code:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(python_code)
                    print(f"[Success] {out_name} saved on attempt {attempt+1}.")
                    stats["success"] += 1
                    return
                else:
                    print(f"[Warning] {out_name} attempt {attempt+1} got empty or invalid response.")
            except Exception as e:
                print(f"[Error] {out_name} attempt {attempt+1} failed with exception: {e}")
            sleep_time = (attempt + 1) * 15
            print(f"Waiting {sleep_time}s before retry...")
            await asyncio.sleep(sleep_time)
            
        print(f"[Fatal] {out_name} failed after all attempts.")
        stats["failed"] += 1

async def main():
    parser = argparse.ArgumentParser(description="Batch One-Shot MATLAB to Python Translation Baseline")
    parser.add_argument("--input_dir", type=str, default="experiments/all_matlab_algorithms", help="Directory containing MATLAB files")
    parser.add_argument("--output_dir", type=str, default="experiments/baseline-48", help="Directory to save generated Python files")
    parser.add_argument("--repeats", type=int, default=5, help="Number of times to run translation for each algorithm")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum number of concurrent translations")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    repeats = args.repeats
    concurrency = args.concurrency

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Gather all algorithm files
    m_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt') and f not in ["analysis.txt", "dryrun_output.txt", "references_and_copyrights.md"]])

    if not m_files:
        print(f"No algorithms found in {input_dir}.")
        return

    print("=======================================")
    print(" [Batch Baseline: One-Shot Translation] ")
    print(f" Model:   {MODEL_NAME}")
    print(f" Input:   {input_dir}")
    print(f" Output:  {output_dir}")
    print(f" Repeats: {repeats}")
    print(f" Concurrency limit: {concurrency}")
    print(f" Total Algorithms: {len(m_files)}")
    print(f" Total Runs to execute: {len(m_files) * repeats}")
    print("=======================================\n")

    sem = asyncio.Semaphore(concurrency)
    stats = {"success": 0, "skipped": 0, "failed": 0}

    tasks = []
    for m_file in m_files:
        for i in range(1, repeats + 1):
            tasks.append(worker(sem, m_file, i, input_dir, output_dir, stats))

    await asyncio.gather(*tasks)

    print("\n=======================================")
    print(" Translation Finished Summary ")
    print(f" Success: {stats['success']}")
    print(f" Skipped: {stats['skipped']}")
    print(f" Failed:  {stats['failed']}")
    print("=======================================")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
