import os
import subprocess
import sys
import shutil
import uuid

# --- 配置部分 ---

# Ruff 忽略规则配置
# 仅检测关键语法和逻辑错误，忽略格式化和风格问题
IGNORE_RUFF_CODES = [
    "E501",  # line too long (行太长，不影响运行)
    "E402",  # import not at top (模板代码中常有中间import，忽略)
    "E722",  # bare except (裸except，虽然不好但能跑)
    "E731",  # lambda assignment (lambda赋值)
    "E741",  # ambiguous variable name (变量名歧义，如 l, O, I)
    "E701",
    "E702",
    "E703",  # multiple statements in one line (单行多语句)
    "I",  # ignore all isort errors (忽略导入排序建议)
]

# --- 路径配置 ---

# 获取当前文件 (executor.py) 所在的目录 -> .../evocoder/backend
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录 (往上跳一级) -> .../evocoder
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# 基础工作区路径移到项目根目录下的 temp_workspace
BASE_WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "temp_workspace")

# 确保基础工作区存在
if not os.path.exists(BASE_WORKSPACE_DIR):
    os.makedirs(BASE_WORKSPACE_DIR)


def setup_workspace(session_id: str) -> str:
    """
    为当前 Session 创建独立的子工作目录
    例如: .../evocoder/temp_workspace/a1b2c3d4/
    """
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_path


def cleanup_workspace(session_id: str):
    """
    清理 Session 的工作目录
    """
    workspace_path = os.path.join(BASE_WORKSPACE_DIR, session_id)
    if os.path.exists(workspace_path):
        try:
            shutil.rmtree(workspace_path)
        except Exception as e:
            print(f"Error cleaning up workspace {session_id}: {e}")


def check_syntax_with_ruff(code: str, session_id: str = None) -> tuple[bool, str]:
    """
    使用 Ruff 进行静态代码分析
    配置: 开启 E,F,I 检测，但忽略 IGNORE_RUFF_CODES 中的规则
    返回: (is_valid: bool, error_message: str)
    """
    # 如果没提供 session_id，创建一个临时的用于检查，查完即删
    is_temp_session = False
    if not session_id:
        session_id = f"check_{str(uuid.uuid4())[:8]}"
        is_temp_session = True

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, "temp_check.py")

    try:
        # 1. 写入代码到临时文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 2. 构建 Ruff 命令
        # --select E,F,I: 启用 Pycodestyle(E), Pyflakes(F), Isort(I)
        # --ignore ...: 忽略指定的规则
        cmd = [
            "ruff",
            "check",
            file_path,
            "--select",
            "E,F,I",
            "--ignore",
            ",".join(IGNORE_RUFF_CODES),
            "--output-format",
            "text",
            "--no-cache",
        ]

        # 注意：需要确保系统已安装 ruff (pip install ruff)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=workspace,  # 在该目录下运行
        )

        if result.returncode == 0:
            return True, ""
        else:
            # 数据清洗：将绝对路径替换为通用文件名，避免 LLM 被路径干扰
            # 同时也去除可能的空行
            clean_error = result.stdout.replace(file_path, "script.py").strip()
            return False, clean_error

    except FileNotFoundError:
        return (
            False,
            "System Error: 'ruff' not found. Please install it via `pip install ruff`.",
        )
    except Exception as e:
        return False, f"Static Check Error: {str(e)}"
    finally:
        # 如果是专门为此检查创建的临时 session，检查完就清理掉
        if is_temp_session:
            cleanup_workspace(session_id)


def execute_code(code_str: str, session_id: str = None, filename="algo_script.py"):
    """
    在隔离的 Session 目录中执行代码
    """
    # 如果没提供 session_id，创建一个临时的
    # 默认不自动清理 exec session，方便调试
    if not session_id:
        session_id = f"exec_{str(uuid.uuid4())[:8]}"

    workspace = setup_workspace(session_id)
    file_path = os.path.join(workspace, filename)

    # 1. 写入代码文件
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_str)
    except Exception as e:
        return False, "", f"System Error: Failed to write file - {str(e)}"

    # 2. 运行代码
    try:
        # 使用当前环境的 Python 解释器
        # cwd=workspace: 确保代码运行时的“当前目录”是该 session 目录
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=30,  # 防止死循环
            cwd=workspace,  # 【关键】隔离运行环境
        )

        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Runtime Error: Execution timed out (Possible infinite loop)."
    except Exception as e:
        return False, "", f"Runtime Error: {str(e)}"