"""Batch One-Shot MATLAB to Python Translation Baseline."""

import argparse
import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path

from _common import (
    EXCLUDED_INPUT_FILES,
    ensure_repo_root_on_path,
    load_dotenv_from_root,
    read_matlab_source,
    set_win32_event_loop_policy,
    setup_litellm_env,
)

# 1. Load dotenv and export Litellm configuration as OpenAI env vars BEFORE importing one_shot
ensure_repo_root_on_path()
load_dotenv_from_root()
setup_litellm_env()

# This import must happen after the environment mapping above because one_shot
# initializes its provider configuration at import time.
one_shot_module = importlib.import_module("one_shot")
MODEL_NAME = one_shot_module.MODEL_NAME
one_shot_translate = one_shot_module.one_shot_translate

logger = logging.getLogger(__name__)


async def worker(
    sem: asyncio.Semaphore,
    m_file: str,
    run_idx: int,
    input_dir: str,
    output_dir: str,
    stats: dict,
) -> None:
    async with sem:
        file_path = Path(input_dir) / m_file
        algo_name = os.path.splitext(m_file)[0]
        out_name = f"{algo_name}_run{run_idx}.py"
        out_path = Path(output_dir) / out_name

        if out_path.exists():
            print(f"[Skipped] {out_name} already exists.")
            stats["skipped"] += 1
            return

        print(f"[Generating] {algo_name} Run {run_idx}...")

        matlab_code = await asyncio.to_thread(read_matlab_source, file_path)

        for attempt in range(5):
            try:
                python_code = await one_shot_translate(matlab_code)
                if python_code and "class" in python_code:
                    await asyncio.to_thread(
                        out_path.write_text, python_code, encoding="utf-8"
                    )
                    print(f"[Success] {out_name} saved on attempt {attempt + 1}.")
                    stats["success"] += 1
                    return
                else:
                    print(
                        f"[Warning] {out_name} attempt {attempt + 1} got empty or invalid response."
                    )
            except Exception as e:
                # Intentional broad catch for the retry loop: transient network/API
                # errors are retried up to 5 attempts (logger.exception documents it).
                logger.exception("[Error] %s attempt %d failed", out_name, attempt + 1)
                print(
                    f"[Error] {out_name} attempt {attempt + 1} failed with exception: {e}"
                )
            sleep_time = (attempt + 1) * 15
            print(f"Waiting {sleep_time}s before retry...")
            await asyncio.sleep(sleep_time)

        print(f"[Fatal] {out_name} failed after all attempts.")
        stats["failed"] += 1


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch One-Shot MATLAB to Python Translation Baseline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="experiments/all_matlab_algorithms",
        help="Directory containing MATLAB files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/baseline-48",
        help="Directory to save generated Python files",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of times to run translation for each algorithm",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent translations",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    repeats = args.repeats
    concurrency = args.concurrency

    if not Path(input_dir).is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Gather all algorithm files
    m_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.endswith(".txt") and f not in EXCLUDED_INPUT_FILES
        ]
    )

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
    set_win32_event_loop_policy()
    asyncio.run(main())
