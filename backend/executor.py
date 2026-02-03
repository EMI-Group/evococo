import os
import subprocess
import sys
import shutil
import uuid
import re
import math
import time

# --- 配置部分 ---

# Ruff 忽略规则配置
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

# --- 路径配置 ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
BASE_WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "temp_workspace")

if not os.path.exists(BASE_WORKSPACE_DIR):
    os.makedirs(BASE_WORKSPACE_DIR)


def setup_workspace(session_id: str) -> str:
    """创建隔离的工作目录"""
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_path


def cleanup_workspace(session_id: str):
    """清理工作目录"""
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    if os.path.exists(workspace_path):
        try:
            shutil.rmtree(workspace_path)
        except Exception as e:
            print(f"Error cleaning up workspace {session_id}: {e}")


def check_syntax_with_ruff(code: str, session_id: str = None) -> tuple[bool, str]:
    """使用 Ruff 进行静态代码分析"""
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=workspace,
        )

        if result.returncode == 0:
            return True, ""
        else:
            raw_output = result.stdout + "\n" + result.stderr
            clean_error = raw_output.replace(file_path, "script.py").strip()
            if not clean_error:
                clean_error = (
                    f"Ruff failed (Exit Code: {result.returncode}), check install."
                )
            return False, clean_error

    except subprocess.TimeoutExpired:
        return False, "System Error: Ruff check timed out."
    except Exception as e:
        return False, f"Static Check Error: {str(e)}"
    finally:
        if is_temp_session:
            cleanup_workspace(session_id)


def execute_code(code_str: str, session_id: str = None, filename="algo_script.py"):
    """
    基础执行函数：运行代码并返回输出
    """
    if not session_id:
        session_id = f"exec_{str(uuid.uuid4())[:8]}"

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_str)
    except Exception as e:
        return False, "", f"System Error: Failed to write file - {str(e)}"

    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=30,  # 30秒超时，防止死循环
            cwd=workspace,
        )

        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Runtime Error: Execution timed out (Possible infinite loop)."
    except Exception as e:
        return False, "", f"Runtime Error: {str(e)}"


def parse_igd_from_stdout(stdout: str) -> list[float]:
    """
    从标准输出中提取 IGD 值
    假设格式类似于: "Gen 10 IGD: 0.1234"
    """
    igds = []
    # 正则匹配浮点数，忽略 NaN
    matches = re.findall(r"IGD:\s*([+-]?([0-9]*[.])?[0-9]+|nan)", stdout, re.IGNORECASE)
    for m in matches:
        val_str = m[0]
        try:
            if "nan" in val_str.lower():
                val = float("nan")
            else:
                val = float(val_str)
            igds.append(val)
        except:
            continue
    return igds


def execute_code_trial(
    code_str: str, session_id: str, filename="algo_script.py"
) -> dict:
    """
    高级试跑函数：执行代码并分析收敛性指标
    """
    start_time = time.time()
    success, output, error = execute_code(code_str, session_id, filename)
    duration = time.time() - start_time

    igds = parse_igd_from_stdout(output)

    # 检查是否有 NaN
    has_nan = "nan" in output.lower()

    # 获取最后的 IGD 值
    last_igd = float("inf")
    if igds:
        # 过滤掉 NaN 后的最后一个值
        valid_igds = [v for v in igds if not math.isnan(v)]
        if valid_igds:
            last_igd = valid_igds[-1]

    # 简单的收敛性判断
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