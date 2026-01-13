import os
import subprocess
import sys

# Define sandbox path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(BASE_DIR, "backend", "temp_workspace")


def execute_code(code_str: str, filename="temp_algo.py"):
    """
    Writes the code string to a file and executes it within the sandbox directory.

    Args:
        code_str (str): The Python code to run.
        filename (str): The temporary filename.

    Returns:
        tuple: (success (bool), stdout (str), stderr (str))
    """
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    file_path = os.path.join(WORKSPACE_DIR, filename)

    # 1. Write code to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_str)
    except Exception as e:
        return False, "", f"System Error: Failed to write file - {str(e)}"

    # 2. Run code using the current Python interpreter
    # Timeout is set to 30 seconds to prevent infinite loops
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORKSPACE_DIR,  # Ensure execution happens inside the temp dir
        )

        if result.returncode == 0:
            return True, result.stdout, ""
        else:
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Runtime Error: Execution timed out (Possible infinite loop)."
    except Exception as e:
        return False, "", f"Runtime Error: {str(e)}"