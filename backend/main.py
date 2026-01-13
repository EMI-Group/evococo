from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import traceback

# 导入你的 engine 模块
from .engine import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "EvoCoder Backend is Running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> [WS] Client Connected")  # 连接成功日志

    try:
        # ！！！关键修复：必须用 while True 保持连接一直处于监听状态
        while True:
            # 等待接收前端消息
            data = await websocket.receive_text()

            # 收到消息后立即打印，用于调试
            print(f">>> [WS] Received Data: {data[:100]}...")

            try:
                payload = json.loads(data)
                matlab_code = payload.get("code", "")
            except json.JSONDecodeError:
                print(">>> [WS Error] Invalid JSON received")
                continue

            # 定义回调函数，用于把处理进度发回给前端
            async def send_update(
                type_,
                title,
                message,
                step_id=None,
                extra_data=None,
                is_success=None,
                icon=None,
            ):
                response_data = {
                    "type": type_,
                    "title": title,
                    "message": message,
                    "step_id": step_id,
                    "extra_data": extra_data,
                    "is_success": is_success,
                    "icon": icon,
                }
                # 过滤 None 虽然不是必须的，但可以保持干净，这里直接发送
                await websocket.send_text(json.dumps(response_data))

            # 执行核心 Pipeline
            if matlab_code:
                print(">>> [DEBUG] Starting run_pipeline...")
                await run_pipeline(matlab_code, send_update)
            else:
                print(">>> [WS Warning] Received empty code.")
                await send_update("log", "Warning", "Code is empty.")

    except WebSocketDisconnect:
        print(">>> [WS] Client Disconnected")
    except Exception as e:
        print(">>> [WS Fatal Error]")
        traceback.print_exc()
        # 尝试通知前端发生了致命错误（如果连接还活着）
        try:
            await websocket.send_text(json.dumps({"type": "fatal", "message": str(e)}))
        except:
            pass