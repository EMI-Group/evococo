import asyncio
import json
import logging
import os
import secrets
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# Import your engine module
from .engine import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("evococo.backend.main")

WS_TOKEN = os.getenv("EVOCOCO_WS_TOKEN", "").strip() or secrets.token_urlsafe(32)
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.getenv("EVOCOCO_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app):
    logger.info(
        "Open http://localhost:8000/#token=%s (adjust the port if needed). "
        "Keep this connection token private.",
        quote(WS_TOKEN, safe=""),
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_PATH, headers={"Cache-Control": "no-store"})


@app.get("/health")
def read_root():
    return {"status": "EvoCoCo Backend is Running"}


async def authenticate_websocket(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    scheme = "https" if websocket.url.scheme == "wss" else "http"
    same_origin = f"{scheme}://{websocket.url.netloc}"
    # Local files have an opaque ("null") origin. They still need the secret.
    # Origin-less clients must authenticate as well; no HTTP endpoint reveals it.
    if origin not in (None, "null", same_origin) and origin not in ALLOWED_ORIGINS:
        # Complete the handshake so browsers receive 1008 instead of an opaque
        # HTTP 403 / close 1006 and can stop retrying. No application data is read.
        await websocket.accept()
        await websocket.close(code=1008, reason="Origin is not allowed")
        return False

    await websocket.accept()
    try:
        payload = await asyncio.wait_for(websocket.receive_json(), timeout=10)
    except WebSocketDisconnect:
        return False
    except (asyncio.TimeoutError, ValueError, KeyError):
        await websocket.close(code=1008, reason="Connection token required")
        return False

    token = payload.get("token") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "auth"
        or not isinstance(token, str)
        or not secrets.compare_digest(
            token.encode("utf-8", errors="surrogatepass"),
            WS_TOKEN.encode("utf-8", errors="surrogatepass"),
        )
    ):
        await websocket.close(code=1008, reason="Invalid connection token")
        return False

    await websocket.send_json({"type": "auth_ok"})
    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not await authenticate_websocket(websocket):
        return
    logger.info("WS Client Authenticated")

    pipeline_task = None

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

        await websocket.send_text(json.dumps(response_data))

    try:
        while True:
            # Wait to receive frontend message
            data = await websocket.receive_text()

            # Log immediately upon receiving message
            logger.info("WS Received data length: %s", len(data))

            try:
                payload = json.loads(data)
                matlab_code = (
                    payload.get("code", "") if isinstance(payload, dict) else ""
                )
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
                continue

            # --- Execute core Pipeline ---
            if isinstance(matlab_code, str) and matlab_code.strip():
                if pipeline_task is not None and not pipeline_task.done():
                    await send_update(
                        "log", "Warning", "A pipeline is already running."
                    )
                    continue
                if pipeline_task is not None:
                    await pipeline_task
                logger.info("Starting run_pipeline...")
                # Keep receiving while the run is active so disconnects cancel it
                # even when it is waiting on an LLM or a child process.
                pipeline_task = asyncio.create_task(
                    run_pipeline(matlab_code, send_update)
                )
            else:
                logger.warning("Received empty code.")
                await send_update("log", "Warning", "Code is empty.")

    except WebSocketDisconnect:
        logger.info("WS Client Disconnected (Normal Close)")
    except Exception:  # noqa: BLE001
        # Only catch genuine unexpected errors
        logger.error("WS Fatal Error")
        traceback.print_exc()
    finally:
        if pipeline_task is not None:
            if not pipeline_task.done():
                pipeline_task.cancel()
            await asyncio.gather(pipeline_task, return_exceptions=True)
