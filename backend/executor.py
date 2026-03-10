import os
import asyncio
import sys
import uuid
import re
import math
import time
import shutil

# --- Configuration section ---

# Ruff ignore rule configuration
IGNORE_RUFF_CODES = [
    "E501",
    "E402",
    "E722",
    "E731",
    "E741",
    "E701",
    "E702",
    "E703",
    "I",
]

# --- Path configuration ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
BASE_WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "temp_workspace")

if not os.path.exists(BASE_WORKSPACE_DIR):
    os.makedirs(BASE_WORKSPACE_DIR)


def setup_workspace(session_id: str) -> str:
    """Create isolated workspace directory"""
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_path


def cleanup_workspace(session_id: str):
    """
    Clean up workspace directory
    [Modified] Retains up to 50 global debug records to prevent disk bombs.
    """
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    if os.path.exists(workspace_path):
        try:
            print(f">>> [DEBUG] Workspace kept at: {workspace_path}")

            # Global cleanup: to avoid filling up the disk
            cleanup_old_workspaces(BASE_WORKSPACE_DIR, max_retained=50)
        except Exception as e:
            print(f"Error cleaning up workspace {session_id}: {e}")


def cleanup_old_workspaces(base_dir: str, max_retained: int = 50):
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

        cmd = [
            "ruff",
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

        # Use asyncio.create_subprocess_exec to avoid blocking the event loop
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
            # Decode output
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return False, "System Error: Ruff check timed out."

        if process.returncode == 0:
            return True, ""
        else:
            raw_output = stdout_str + "\n" + stderr_str
            clean_error = raw_output.replace(file_path, "script.py").strip()
            if not clean_error:
                clean_error = (
                    f"Ruff failed (Exit Code: {process.returncode}), check install."
                )
            return False, clean_error

    except Exception as e:
        return False, f"Static Check Error: {str(e)}"
    finally:
        if is_temp_session:
            cleanup_workspace(session_id)


async def execute_code(
    code_str: str, session_id: str = None, filename="algo_script.py"
):
    """
    Base execution function: run code and return output
    [Modified] Force CPU execution to prevent parallel freeze
    """
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

    # === [Key modification] Force CPU mode ===
    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU, PyTorch fallback to CPU
    # env["OMP_NUM_THREADS"] = "1"  # Limit CPU threads to prevent 100% usage
    # env["MKL_NUM_THREADS"] = "1"
    # ==============================

    try:
        # Use asyncio.create_subprocess_exec to avoid event loop blocking
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            filename,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
        )

        try:
            # 30 seconds timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return (
                False,
                "",
                "Runtime Error: Execution timed out (Possible infinite loop).",
            )

        if process.returncode == 0:
            return True, stdout_str, stderr_str
        else:
            return False, stdout_str, stderr_str

    except Exception as e:
        return False, "", f"Runtime Error: {str(e)}"


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
        except:  # noqa: E722
            continue
    return igds


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

    return {
        "success": success,
        "stdout": output,
        "stderr": error,
        "last_igd": last_igd,
        "igd_history": igds,
        "has_nan": has_nan,
        "is_converging": is_converging,
        "duration": duration,
    }
