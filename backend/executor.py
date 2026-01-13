import os
import subprocess
import sys

# --- 路径配置修改 ---

# 获取当前文件 (executor.py) 所在的目录 -> .../evocoder/backend
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录 (往上跳一级) -> .../evocoder
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# [修改点] 将工作区移到项目根目录下的 temp_workspace (与 backend 平级)
WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "temp_workspace")


def execute_code(code_str: str, filename="temp_algo.py"):
    """
    Writes the code string to a file and executes it within the sandbox directory.
    """
    # 自动创建目录（如果不存在）
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    file_path = os.path.join(WORKSPACE_DIR, filename)

    # 1. Write code to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_str)
    except Exception as e:
        return False, "", f"System Error: Failed to write file - {str(e)}"

    # 2. Run code
    try:
        # 注意：这里 cwd 变了，引用其他模块可能需要注意，但对于独立脚本通常没问题
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORKSPACE_DIR,  # 在根目录的 temp_workspace 下运行
        )

        if result.returncode == 0:
            return True, result.stdout, ""
        else:
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Runtime Error: Execution timed out (Possible infinite loop)."
    except Exception as e:
        return False, "", f"Runtime Error: {str(e)}"