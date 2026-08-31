import os
import asyncio
import sys
import uuid
import re
import math
import time
import shutil
import traceback
import subprocess

from .config import (
    IGNORE_RUFF_CODES,
    BASE_WORKSPACE_DIR,
    MAX_RETAINED_WORKSPACES,
)

if not os.path.exists(BASE_WORKSPACE_DIR):
    os.makedirs(BASE_WORKSPACE_DIR)


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
        except Exception as e:
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
    except Exception as e:
        print(f">>> [DEBUG] Error in global workspace cleanup: {e}")


async def check_syntax_with_ruff(code: str, session_id: str = None) -> tuple[bool, str]:
    """Use Ruff for static code analysis"""
    is_temp_session = False
    if not session_id:
        session_id = f"check_{str(uuid.uuid4())[:8]}"
        is_temp_session = True

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, "temp_check.py")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

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

        def run_ruff():
            return subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workspace,
                timeout=5,
            )

        try:
            result = await asyncio.to_thread(run_ruff)
            stdout_str = result.stdout.decode(errors="replace") if result.stdout else ""
            stderr_str = result.stderr.decode(errors="replace") if result.stderr else ""
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            return False, "System Error: Ruff check timed out."

        if returncode == 0:
            return True, ""
        else:
            raw_output = stdout_str + "\n" + stderr_str
            clean_error = raw_output.replace(file_path, "script.py").strip()
            if not clean_error:
                clean_error = f"Ruff failed (Exit Code: {returncode}), check install."
            return False, clean_error

    except Exception as e:
        return False, f"Static Check Error: {str(e)}"
    finally:
        if is_temp_session:
            cleanup_workspace(session_id)


async def execute_code(
    code_str: str, session_id: str = None, filename="algo_script.py"
):
    """Run generated code in an isolated workspace and return its output."""
    if not session_id:
        session_id = f"exec_{str(uuid.uuid4())[:8]}"

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, filename)

    try:
        # Note: can use aiofiles or to_thread here if needed, but simple file write is usually fast enough
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_str)
    except Exception as e:
        return False, "", f"System Error: Failed to write file - {str(e)}"

    try:

        def run_script():
            return subprocess.run(
                [sys.executable, filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workspace,
                timeout=60,
            )

        try:
            result = await asyncio.to_thread(run_script)
            stdout_str = result.stdout.decode(errors="replace") if result.stdout else ""
            stderr_str = result.stderr.decode(errors="replace") if result.stderr else ""
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            return (
                False,
                "",
                "Runtime Error: Execution timed out (Possible infinite loop).",
            )

        if returncode == 0:
            return True, stdout_str, stderr_str
        else:
            return False, stdout_str, stderr_str

    except Exception as e:
        return False, "", f"Runtime Error: {str(e)}\n{traceback.format_exc()}"


def parse_igd_from_stdout(stdout: str) -> list[float]:
    """Extract IGD values from standard output"""
    igds = []
    matches = re.findall(r"IGD:\s*([+-]?([0-9]*[.])?[0-9]+|nan)", stdout, re.IGNORECASE)
    for m in matches:
        val_str = m[0]
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
    match = re.search(r"Execution time for Gen 2-50.*?:\s*([0-9]*[.]?[0-9]+)s", stdout)
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            pass
    return -1.0


async def execute_code_trial(
    code_str: str, session_id: str, filename="algo_script.py"
) -> dict:
    """Advanced trial run function: execute code and analyze convergence metrics"""
    start_time = time.time()
    success, output, error = await execute_code(code_str, session_id, filename)
    duration = time.time() - start_time

    igds = parse_igd_from_stdout(output)
    has_nan = "nan" in output.lower()

    last_igd = float("inf")
    if igds:
        valid_igds = [v for v in igds if not math.isnan(v)]
        if valid_igds:
            last_igd = valid_igds[-1]

    is_converging = False
    if len(igds) >= 2 and igds[-1] < igds[0]:
        is_converging = True

    exec_time = parse_exec_time_from_stdout(output)

    return {
        "success": success,
        "stdout": output,
        "stderr": error,
        "last_igd": last_igd,
        "igd_history": igds,
        "has_nan": has_nan,
        "is_converging": is_converging,
        "duration": duration,
        "exec_time": exec_time,
    }
