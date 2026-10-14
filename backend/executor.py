import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from .config import (
    BASE_WORKSPACE_DIR,
    IGNORE_RUFF_CODES,
    MAX_RETAINED_WORKSPACES,
)

os.makedirs(BASE_WORKSPACE_DIR, exist_ok=True)


async def _run_subprocess(cmd, *, cwd, timeout):
    """Reap the child before returning, including on timeout or cancellation."""
    # Popen also works on Windows selector loops used by some ASGI servers.
    # Shield process creation so cancellation cannot lose a just-started child.
    spawning = asyncio.create_task(
        asyncio.to_thread(
            subprocess.Popen,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
    )
    try:
        process = await asyncio.shield(spawning)
    except asyncio.CancelledError:
        process = await spawning
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.communicate)
        raise

    communicating = asyncio.create_task(asyncio.to_thread(process.communicate))
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communicating), timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await communicating
        raise
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def _write_text_file(path: str, content: str) -> None:
    """Write text content to a file (run via asyncio.to_thread in async contexts)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def setup_workspace(session_id: str) -> str:
    """Create isolated workspace directory"""
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_path


def cleanup_workspace(session_id: str):
    """Retain the workspace and remove entries beyond the configured limit."""
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    if os.path.exists(workspace_path):
        try:
            print(f">>> [DEBUG] Workspace kept at: {workspace_path}")

            # Global cleanup: to avoid filling up the disk
            cleanup_old_workspaces(
                BASE_WORKSPACE_DIR, max_retained=MAX_RETAINED_WORKSPACES
            )
        except Exception as e:  # noqa: BLE001 - best-effort cleanup, non-fatal
            print(f"Error cleaning up workspace {session_id}: {e}")


def cleanup_old_workspaces(base_dir: str, max_retained: int = MAX_RETAINED_WORKSPACES):
    """Keep only the 'max_retained' most recent directories in base_dir"""
    try:
        if not os.path.exists(base_dir):
            return

        dirs = []
        for d in os.listdir(base_dir):
            path = os.path.join(base_dir, d)
            if os.path.isdir(path):
                dirs.append(path)

        # Sort by modification time, oldest first
        dirs.sort(key=lambda x: os.path.getmtime(x))

        if len(dirs) > max_retained:
            dirs_to_delete = dirs[:-max_retained]
            for d in dirs_to_delete:
                shutil.rmtree(d, ignore_errors=True)
                print(f">>> [DEBUG] Deleted old workspace: {d}")
    except Exception as e:  # noqa: BLE001 - best-effort cleanup, non-fatal
        print(f">>> [DEBUG] Error in global workspace cleanup: {e}")


async def check_syntax_with_ruff(
    code: str, session_id: str | None = None
) -> tuple[bool, str]:
    """Use Ruff for static code analysis"""
    is_temp_session = False
    if not session_id:
        session_id = f"check_{str(uuid.uuid4())[:8]}"
        is_temp_session = True

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, "temp_check.py")

    try:
        await asyncio.to_thread(_write_text_file, file_path, code)

        python_dir = os.path.dirname(sys.executable)
        ruff_executable = os.path.join(python_dir, "ruff")
        if not os.path.exists(ruff_executable):
            ruff_executable = "ruff"

        cmd = [
            ruff_executable,
            "check",
            file_path,
            "--select",
            "E,F,I",
            "--ignore",
            ",".join(IGNORE_RUFF_CODES),
            "--output-format",
            "full",
            "--no-cache",
        ]

        try:
            result = await _run_subprocess(cmd, cwd=workspace, timeout=5)
            stdout_str = result.stdout.decode(errors="replace") if result.stdout else ""
            stderr_str = result.stderr.decode(errors="replace") if result.stderr else ""
            returncode = result.returncode
        except asyncio.TimeoutError:
            return False, "System Error: Ruff check timed out."

        if returncode == 0:
            return True, ""
        else:
            raw_output = stdout_str + "\n" + stderr_str
            clean_error = raw_output.replace(file_path, "script.py").strip()
            if not clean_error:
                clean_error = f"Ruff failed (Exit Code: {returncode}), check install."
            return False, clean_error

    except Exception as e:  # noqa: BLE001 - report error as string, never raise
        return False, f"Static Check Error: {e!s}"
    finally:
        if is_temp_session:
            cleanup_workspace(session_id)


async def execute_code(
    code_str: str,
    session_id: str | None = None,
    filename="algo_script.py",
    *,
    validate_fitness: bool = False,
    supports_constraints: bool = False,
    objective_mode: str = "multi",
    objective_direction: str = "min",
):
    """Run generated code in an isolated workspace and return its output."""
    if not session_id:
        session_id = f"exec_{str(uuid.uuid4())[:8]}"

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, filename)

    try:
        await asyncio.to_thread(_write_text_file, file_path, code_str)
    except Exception as e:  # noqa: BLE001 - write failure reported as error tuple
        return False, "", f"System Error: Failed to write file - {e!s}"

    try:
        try:
            if validate_fitness:
                runner = os.path.join(os.path.dirname(__file__), "trial_runner.py")
                cmd = [sys.executable, runner, filename]
                if supports_constraints:
                    cmd.append("--constraints")
                if objective_mode == "single":
                    cmd.append("--single-objective")
                    if objective_direction == "max":
                        cmd.append("--maximize")
            else:
                cmd = [sys.executable, filename]
            # Scale the per-problem budget with the 100-step verification profile.
            result = await _run_subprocess(
                cmd, cwd=workspace, timeout=240 if supports_constraints else 120
            )
            stdout_str = result.stdout.decode(errors="replace") if result.stdout else ""
            stderr_str = result.stderr.decode(errors="replace") if result.stderr else ""
            returncode = result.returncode
        except asyncio.TimeoutError:
            return (
                False,
                "",
                "Runtime Error: Execution timed out (Possible infinite loop).",
            )

        if returncode == 0:
            return True, stdout_str, stderr_str
        else:
            return False, stdout_str, stderr_str

    except Exception as e:  # noqa: BLE001 - runtime failure reported as error tuple
        return False, "", f"Runtime Error: {e!s}\n{traceback.format_exc()}"


def parse_igd_from_stdout(stdout: str) -> list[float]:
    """Extract IGD values from standard output"""
    igds = []
    matches = re.findall(
        r"\bIGD:\s*([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan))(?![\w.])",
        stdout,
        re.IGNORECASE,
    )
    for val_str in matches:
        try:
            if "nan" in val_str.lower():
                val = float("nan")
            else:
                val = float(val_str)
            igds.append(val)
        except (TypeError, ValueError):
            continue
    return igds


def parse_exec_time_from_stdout(stdout: str) -> float:
    """Extract Execution time from standard output"""
    match = re.search(r"Execution time for Gen 2-\d+.*?:\s*([0-9]*[.]?[0-9]+)s", stdout)
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            pass
    return -1.0


async def execute_code_trial(
    code_str: str,
    session_id: str,
    filename="algo_script.py",
    *,
    supports_constraints: bool = False,
    objective_mode: str = "multi",
    objective_direction: str = "min",
) -> dict:
    """Advanced trial run function: execute code and analyze convergence metrics"""
    start_time = time.time()
    single = objective_mode == "single"
    if objective_mode not in {"single", "multi"} or objective_direction not in {
        "min",
        "max",
    }:
        raise ValueError("Invalid objective mode or direction")
    success, output, error = await execute_code(
        code_str,
        session_id,
        filename,
        validate_fitness=True,
        supports_constraints=supports_constraints,
        **(
            {"objective_mode": "single", "objective_direction": objective_direction}
            if single
            else {}
        ),
    )
    duration = time.time() - start_time

    igds = parse_igd_from_stdout(output)
    constraint_history = []
    if single:
        igds = []
        generations = (
            {"Sphere": [], "ConstrainedSphere": []}
            if supports_constraints
            else {None: []}
        )
        for line in output.splitlines():
            if line.startswith("SINGLE_OBJECTIVE_METRICS: "):
                try:
                    row = json.loads(line.removeprefix("SINGLE_OBJECTIVE_METRICS: "))
                    value = row["best_fitness"]
                    problem = row["problem"] if supports_constraints else None
                    if (
                        problem not in generations
                        or row["direction"] != objective_direction
                        or type(row["generation"]) is not int
                        or type(value) not in (int, float)
                        or not math.isfinite(value)
                    ):
                        raise ValueError("Invalid single-objective metric values")
                    generations[problem].append(row["generation"])
                    if problem == "ConstrainedSphere":
                        feasible_best = row["feasible_best_fitness"]
                        if (
                            type(row["population"]) is not int
                            or row["population"] <= 0
                            or type(row["feasible_count"]) is not int
                            or not 0 <= row["feasible_count"] <= row["population"]
                            or type(row["mean_cv"]) not in (int, float)
                            or not math.isfinite(row["mean_cv"])
                            or row["mean_cv"] < 0
                            or (row["feasible_count"] == 0) != (feasible_best is None)
                            or (
                                feasible_best is not None
                                and (
                                    type(feasible_best) not in (int, float)
                                    or not math.isfinite(feasible_best)
                                )
                            )
                        ):
                            raise ValueError(
                                "Invalid single-objective constraint metrics"
                            )
                        constraint_history.append(row)
                    else:
                        igds.append(value)
                except (ValueError, TypeError, KeyError) as exc:
                    success = False
                    error += f"\nRuntime validation failed: malformed single-objective metrics ({exc})."
        if success and any(
            steps != list(range(5, 101, 5)) for steps in generations.values()
        ):
            success = False
            error += "\nRuntime validation failed: single-objective checkpoints must cover steps 5-100."
    for line in output.splitlines():
        if not single and line.startswith("CONSTRAINT_METRICS: "):
            try:
                record = json.loads(line.removeprefix("CONSTRAINT_METRICS: "))
                if (
                    record["problem"] != "MW4"
                    or type(record["population"]) is not int
                    or record["population"] <= 0
                    or type(record["feasible_count"]) is not int
                    or not 0 <= record["feasible_count"] <= record["population"]
                    or not math.isfinite(record["mean_cv"])
                    or record["mean_cv"] < 0
                    or type(record["generation"]) is not int
                    or record["generation"] <= 0
                    or (record["feasible_count"] == 0)
                    != (record["feasible_igd"] is None)
                    or (
                        record["feasible_igd"] is not None
                        and (
                            not math.isfinite(record["feasible_igd"])
                            or record["feasible_igd"] < 0
                        )
                    )
                ):
                    raise ValueError("Invalid constrained metric values")
                constraint_history.append(record)
            except (ValueError, TypeError, KeyError) as exc:
                success = False
                error += f"\nRuntime validation failed: malformed constraint metrics ({exc})."
    if supports_constraints and success and not constraint_history:
        success = False
        error += "\nRuntime validation failed: no constrained verification metrics were produced."
    has_nan = any(math.isnan(v) for v in igds)
    has_nonfinite = any(not math.isfinite(v) for v in igds)

    last_igd = float("inf")
    if igds:
        last_igd = igds[-1]

    is_converging = False
    if len(igds) >= 2 and not has_nonfinite and igds[-1] < igds[0]:
        is_converging = True
    if single and objective_direction == "max":
        is_converging = len(igds) >= 2 and not has_nonfinite and igds[-1] > igds[0]

    exec_time = parse_exec_time_from_stdout(output)

    return {
        "success": success,
        "stdout": output,
        "stderr": error,
        **(
            {
                "metric_name": "best_fitness",
                "metric_direction": objective_direction,
                "best_fitness": last_igd,
                "fitness_history": igds,
                "last_metric": last_igd,
            }
            if single
            else {"last_igd": last_igd, "igd_history": igds}
        ),
        "has_nan": has_nan,
        "has_nonfinite": has_nonfinite,
        "is_converging": is_converging,
        "duration": duration,
        "exec_time": exec_time,
        "constraint_history": constraint_history,
    }
