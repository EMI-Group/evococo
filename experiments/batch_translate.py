import os
import sys
import asyncio
import argparse
import json
import time
from datetime import datetime

# Add parent directory to path to allow importing backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import run_pipeline
from backend.config import NUM_BRANCHES, REASONING_EFFORT
from backend.generator import ACTIVE_PROVIDER, MODEL_NAME


async def process_file(file_path, num_repeats, output_dir=None, repeat_concurrency=1):
    matlab_code = ""
    if os.path.isdir(file_path):
        for root, _, files in os.walk(file_path):
            for file in sorted(files):
                if file.endswith(".m") or file.endswith(".txt"):
                    fpath = os.path.join(root, file)
                    with open(fpath, "r", encoding="utf-8") as f:
                        matlab_code += f"\n\n--- {file} ---\n{f.read()}"
        algo_name = os.path.basename(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            matlab_code = f.read()
        algo_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n{'=' * 50}\nProcessing Algorithm: {algo_name}\n{'=' * 50}")

    repeat_semaphore = asyncio.Semaphore(repeat_concurrency)

    async def process_run(run_idx):
        async with repeat_semaphore:
            # History directories currently use second-resolution timestamps.
            # Stagger concurrent repeats so each EvoCoCo pipeline gets an
            # independent run_history directory and executor session namespace.
            if repeat_concurrency > 1:
                await asyncio.sleep((run_idx - 1) * 1.1)
            await run_one(run_idx)

    async def run_one(run_idx):
        run_started_at = datetime.now().astimezone()
        run_timer = time.perf_counter()

        if output_dir:
            out_file = os.path.join(output_dir, f"{algo_name}_run{run_idx}.py")
            stats_file = os.path.join(
                output_dir, f"{algo_name}_run{run_idx}_stats.json"
            )
            if os.path.exists(out_file) and os.path.exists(stats_file):
                print(
                    f"\n--- Run {run_idx}/{num_repeats} for {algo_name} already exists, skipping ---"
                )
                return

        print(f"\n--- Run {run_idx}/{num_repeats} for {algo_name} ---")

        final_code_captured = []

        async def custom_callback(
            type_,
            title,
            message,
            step_id=None,
            extra_data=None,
            is_success=None,
            icon=None,
        ):
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
            stats_data = await run_pipeline(matlab_code, custom_callback)
            print(f"--- Run {run_idx} completed ---")

            run_finished_at = datetime.now().astimezone()
            if stats_data is None:
                stats_data = {}
            stats_data["batch_run"] = {
                "status": "completed"
                if final_code_captured
                else "completed_without_code",
                "algorithm": algo_name,
                "run_index": run_idx,
                "source_path": os.path.abspath(file_path),
                "started_at": run_started_at.isoformat(),
                "finished_at": run_finished_at.isoformat(),
                "wall_time_seconds": time.perf_counter() - run_timer,
            }

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

                if final_code_captured:
                    out_file = os.path.join(output_dir, f"{algo_name}_run{run_idx}.py")
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(final_code_captured[0])
                    print(f"Saved result to {out_file}")

                if stats_data:
                    import json

                    with open(stats_file, "w", encoding="utf-8") as f:
                        json.dump(stats_data, f, indent=2, ensure_ascii=False)
                    print(f"Saved stats JSON to {stats_file}")

        except Exception as e:
            print(f"--- Run {run_idx} failed with exception: {e} ---")
            if output_dir:
                import json

                os.makedirs(output_dir, exist_ok=True)
                run_finished_at = datetime.now().astimezone()
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "batch_run": {
                                "status": "failed",
                                "algorithm": algo_name,
                                "run_index": run_idx,
                                "source_path": os.path.abspath(file_path),
                                "started_at": run_started_at.isoformat(),
                                "finished_at": run_finished_at.isoformat(),
                                "wall_time_seconds": time.perf_counter() - run_timer,
                                "error": str(e),
                            }
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

    await asyncio.gather(
        *(process_run(run_idx) for run_idx in range(1, num_repeats + 1))
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Batch translate MATLAB algorithms to Python using EvoCoder."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing MATLAB (.m) files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save the final generated Python files",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of times to run the pipeline for each algorithm",
    )
    parser.add_argument(
        "--repeat-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent repeats for each algorithm (algorithms remain sequential)",
    )
    args = parser.parse_args()

    if args.repeat_concurrency < 1:
        parser.error("--repeat-concurrency must be at least 1")

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    m_files = []
    for f in os.listdir(input_dir):
        path = os.path.join(input_dir, f)
        if os.path.isdir(path) or f.endswith(".m") or f.endswith(".txt"):
            m_files.append(f)

    if not m_files:
        print(f"No algorithms (.m/.txt files or folders) found in {input_dir}.")
        return

    m_files = sorted(m_files)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "batch_manifest.json")
    manifest = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(output_dir),
        "active_provider": ACTIVE_PROVIDER,
        "model": MODEL_NAME,
        "reasoning_effort_requested": REASONING_EFFORT,
        "algorithms": [os.path.splitext(name)[0] for name in m_files],
        "algorithm_count": len(m_files),
        "repeats_per_algorithm": args.repeats,
        "expected_runs": len(m_files) * args.repeats,
        "repeat_concurrency": args.repeat_concurrency,
        "evo_branches_per_pipeline": NUM_BRANCHES,
        "maximum_parallel_evo_branches": args.repeat_concurrency * NUM_BRANCHES,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        f"Found {len(m_files)} algorithm(s) to process. "
        f"Repeats={args.repeats}, repeat concurrency={args.repeat_concurrency}, "
        f"model={MODEL_NAME}, reasoning_effort={REASONING_EFFORT}. "
        f"Outputs will be saved to {output_dir}"
    )

    for m_file in m_files:
        file_path = os.path.join(input_dir, m_file)
        await process_file(
            file_path,
            args.repeats,
            output_dir,
            repeat_concurrency=args.repeat_concurrency,
        )

    aggregate_tokens = {"prompt": 0, "completion": 0, "total": 0}
    aggregate_llm_time = 0.0
    aggregate_wall_time = 0.0
    run_status_counts = {}
    stats_files = [
        name for name in os.listdir(output_dir) if name.endswith("_stats.json")
    ]
    for name in stats_files:
        try:
            with open(os.path.join(output_dir, name), "r", encoding="utf-8") as f:
                stats = json.load(f)
            tokens = stats.get("total_tokens", {})
            for key in aggregate_tokens:
                aggregate_tokens[key] += int(tokens.get(key, 0) or 0)
            aggregate_llm_time += float(stats.get("total_llm_time", 0.0) or 0.0)
            batch_run = stats.get("batch_run", {})
            aggregate_wall_time += float(batch_run.get("wall_time_seconds", 0.0) or 0.0)
            status = batch_run.get("status", "unknown")
            run_status_counts[status] = run_status_counts.get(status, 0) + 1
        except (OSError, ValueError, TypeError):
            run_status_counts["unreadable_stats"] = (
                run_status_counts.get("unreadable_stats", 0) + 1
            )

    manifest.update(
        {
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "stats_file_count": len(stats_files),
            "run_status_counts": run_status_counts,
            "aggregate_tokens": aggregate_tokens,
            "aggregate_llm_time_seconds": aggregate_llm_time,
            "aggregate_wall_time_seconds": aggregate_wall_time,
        }
    )

    if MODEL_NAME == "deepseek-v4-pro":
        input_cost = aggregate_tokens["prompt"] / 1_000_000 * 0.435
        output_cost = aggregate_tokens["completion"] / 1_000_000 * 0.87
        manifest["estimated_cost"] = {
            "currency": "USD",
            "effective_date": "2026-08-13",
            "accounting": "all_prompt_tokens_assumed_cache_miss",
            "input_cache_miss_usd_per_1m": 0.435,
            "output_usd_per_1m": 0.87,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nAll batch translations completed.")
    print(
        f"You can find the generated code files in '{output_dir}', and detailed logs in 'run_history'."
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
