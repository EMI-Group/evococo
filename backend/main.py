import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import traceback

# Import your engine module
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
    print(">>> [WS] Client Connected")  # Connection success log

    try:
        while True:
            # Wait to receive frontend message
            data = await websocket.receive_text()

            # Print immediately upon receiving message
            print(f">>> [WS] Received Data Length: {len(data)}")

            try:
                payload = json.loads(data)
                matlab_code = payload.get("code", "")
            except json.JSONDecodeError:
                print(">>> [WS Error] Invalid JSON received")
                continue

            # --- Define robust callback function ---
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
                    # Check connection status before sending
                    # Note: websocket.client_state only roughly checks, try-except is the most robust
                    await websocket.send_text(json.dumps(response_data))
                except (WebSocketDisconnect, RuntimeError) as e:  # noqa: F841
                    # If connection is disconnected, print log but do not raise exception to avoid interrupting cleanup
                    print(
                        f">>> [WS Warning] Connection closed, failed to send update: {title}"
                    )
                    # You can choose to raise a custom exception here to stop pipeline, or ignore silently
                    # To prevent pipeline from running empty, we choose to silently ignore and let outer loop handle
                    pass
                except Exception as e:
                    print(f">>> [WS Send Error] {str(e)}")

            # --- Execute core Pipeline ---
            if matlab_code:
                print(">>> [DEBUG] Starting run_pipeline...")
                try:
                    await run_pipeline(matlab_code, send_update)
                except (WebSocketDisconnect, RuntimeError):
                    print(">>> [WS] Pipeline stopped due to disconnection.")
                    break  # Exit while loop
            else:
                print(">>> [WS Warning] Received empty code.")
                await send_update("log", "Warning", "Code is empty.")

    except WebSocketDisconnect:
        print(">>> [WS] Client Disconnected (Normal Close)")
    except Exception as e:  # noqa: F841
        # Only catch genuine unexpected errors
        print(">>> [WS Fatal Error]")
        traceback.print_exc()
