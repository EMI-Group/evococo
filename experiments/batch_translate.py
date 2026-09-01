"""Batch translate MATLAB algorithms to Python using EvoCoder."""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from _common import (
    MATLAB_EXTENSIONS,
    algorithm_name,
    ensure_repo_root_on_path,
    read_json,
    read_matlab_source,
    set_win32_event_loop_policy,
    write_json,
)

ensure_repo_root_on_path()

from backend.config import NUM_BRANCHES, REASONING_EFFORT
from backend.engine import run_pipeline
from backend.generator import ACTIVE_PROVIDER, MODEL_NAME

logger = logging.getLogger(__name__)


async def process_file(
    file_path: Path,
    num_repeats: int,
    output_dir: Path | None = None,
    repeat_concurrency: int = 1,
) -> None:
    matlab_code = await asyncio.to_thread(read_matlab_source, file_path)
    algo_name = algorithm_name(file_path)
    print(f"\n{'=' * 50}\nProcessing Algorithm: {algo_name}\n{'=' * 50}")

    repeat_semaphore = asyncio.Semaphore(repeat_concurrency)

    async def process_run(run_idx: int) -> None:
        async with repeat_semaphore:
            # History directories currently use second-resolution timestamps.
            # Stagger concurrent repeats so each EvoCoCo pipeline gets an
            # independent run_history directory and executor session namespace.
            if repeat_concurrency > 1:
                await asyncio.sleep((run_idx - 1) * 1.1)
            await run_one(run_idx)

    async def run_one(run_idx: int) -> None:
        run_started_at = datetime.now().astimezone()
        run_timer = time.perf_counter()

        if output_dir:
            out_file = output_dir / f"{algo_name}_run{run_idx}.py"
            stats_file = output_dir / f"{algo_name}_run{run_idx}_stats.json"
            if out_file.exists() and stats_file.exists():
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
                output_dir.mkdir(parents=True, exist_ok=True)

                if final_code_captured:
                    out_file = output_dir / f"{algo_name}_run{run_idx}.py"
                    await asyncio.to_thread(
                        out_file.write_text, final_code_captured[0], encoding="utf-8"
                    )
                    print(f"Saved result to {out_file}")

                await asyncio.to_thread(write_json, stats_file, stats_data)
                print(f"Saved stats JSON to {stats_file}")

        except Exception as e:
            # Intentional broad fallback: a failed pipeline run is recorded as
            # failed stats and the batch continues (logger.exception documents it).
            logger.exception("--- Run %d failed ---", run_idx)
            print(f"--- Run {run_idx} failed with exception: {e} ---")
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                run_finished_at = datetime.now().astimezone()
                await asyncio.to_thread(
                    write_json,
                    stats_file,
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
                )

    await asyncio.gather(
        *(process_run(run_idx) for run_idx in range(1, num_repeats + 1))
    )


async def main() -> None:
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

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    m_files = []
    for f in input_dir.iterdir():
        if f.is_dir() or f.name.endswith(MATLAB_EXTENSIONS):
            m_files.append(f.name)

    if not m_files:
        print(f"No algorithms (.m/.txt files or folders) found in {input_dir}.")
        return

    m_files = sorted(m_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "batch_manifest.json"
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
    await asyncio.to_thread(write_json, manifest_path, manifest)

    print(
        f"Found {len(m_files)} algorithm(s) to process. "
        f"Repeats={args.repeats}, repeat concurrency={args.repeat_concurrency}, "
        f"model={MODEL_NAME}, reasoning_effort={REASONING_EFFORT}. "
        f"Outputs will be saved to {output_dir}"
    )

    for m_file in m_files:
        file_path = input_dir / m_file
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
            stats = await asyncio.to_thread(read_json, output_dir / name)
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
    await asyncio.to_thread(write_json, manifest_path, manifest)

    print("\nAll batch translations completed.")
    print(
        f"You can find the generated code files in '{output_dir}', and detailed logs in 'run_history'."
    )


if __name__ == "__main__":
    set_win32_event_loop_policy()
    asyncio.run(main())
