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
        while True:
            # 等待接收前端消息
            data = await websocket.receive_text()

            # 收到消息后立即打印
            print(f">>> [WS] Received Data Length: {len(data)}")

            try:
                payload = json.loads(data)
                matlab_code = payload.get("code", "")
            except json.JSONDecodeError:
                print(">>> [WS Error] Invalid JSON received")
                continue

            # --- 定义健壮的回调函数 ---
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

                try:
                    # 检查连接状态再发送
                    # 注意：websocket.client_state 只能粗略检查，try-except 才是最稳的
                    await websocket.send_text(json.dumps(response_data))
                except (WebSocketDisconnect, RuntimeError) as e:
                    # 如果连接已断开，打印日志但不要抛出异常，防止打断后续清理工作
                    print(
                        f">>> [WS Warning] Connection closed, failed to send update: {title}"
                    )
                    # 这里可以选择抛出一个自定义异常来停止 pipeline，或者默默忽略
                    # 为了防止 pipeline 继续跑空车，我们这里选择静默忽略，让外层循环处理
                    pass
                except Exception as e:
                    print(f">>> [WS Send Error] {str(e)}")

            # --- 执行核心 Pipeline ---
            if matlab_code:
                print(">>> [DEBUG] Starting run_pipeline...")
                try:
                    await run_pipeline(matlab_code, send_update)
                except (WebSocketDisconnect, RuntimeError):
                    print(">>> [WS] Pipeline stopped due to disconnection.")
                    break  # 退出 while 循环
            else:
                print(">>> [WS Warning] Received empty code.")
                await send_update("log", "Warning", "Code is empty.")

    except WebSocketDisconnect:
        print(">>> [WS] Client Disconnected (Normal Close)")
    except Exception as e:
        # 只捕获真正的意外错误
        print(">>> [WS Fatal Error]")
        traceback.print_exc()