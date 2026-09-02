import asyncio
import json
import logging
import sys
import traceback

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import your engine module
from .engine import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("evococo.backend.main")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "EvoCoCo Backend is Running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS Client Connected")

    # Hoisted out of the loop: define the status callback once (captures websocket)
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
        except (WebSocketDisconnect, RuntimeError):
            # Log and ignore send failures; outer loop handles disconnect
            logger.warning("Connection closed, failed to send update: %s", title)
        except Exception:
            # Deliberately swallow any send failure so the pipeline never crashes
            logger.exception("Failed to send update: %s", title)

    try:
        while True:
            # Wait to receive frontend message
            data = await websocket.receive_text()

            # Log immediately upon receiving message
            logger.info("WS Received data length: %s", len(data))

            try:
                payload = json.loads(data)
                matlab_code = payload.get("code", "")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
                continue

            # --- Execute core Pipeline ---
            if matlab_code:
                logger.info("Starting run_pipeline...")
                try:
                    await run_pipeline(matlab_code, send_update)
                except (WebSocketDisconnect, RuntimeError):
                    logger.warning("Pipeline stopped due to disconnection.")
                    break  # Exit while loop
            else:
                logger.warning("Received empty code.")
                await send_update("log", "Warning", "Code is empty.")

    except WebSocketDisconnect:
        logger.info("WS Client Disconnected (Normal Close)")
    except Exception:  # noqa: BLE001
        # Only catch genuine unexpected errors
        logger.error("WS Fatal Error")
        traceback.print_exc()
