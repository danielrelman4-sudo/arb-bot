from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from bot_bridge import BotBridge
from routes.analytics import router as analytics_router

LOGGER = logging.getLogger(__name__)

# Global state
bridge = BotBridge(
    host=os.getenv("BOT_HOST", "127.0.0.1"),
    port=int(os.getenv("BOT_PORT", "9120")),
)
ws_clients: list[WebSocket] = []


def _broadcast_state(state: dict) -> None:
    """Called by BotBridge when new state arrives from bot."""
    for ws in list(ws_clients):
        try:
            asyncio.get_event_loop().create_task(ws.send_json(state))
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    bridge._on_state = _broadcast_state
    asyncio.create_task(_reconnect_loop())
    yield
    await bridge.disconnect()


app = FastAPI(title="Arb Bot Dashboard", lifespan=lifespan)
app.include_router(analytics_router)


async def _reconnect_loop() -> None:
    """Keep trying to connect to the bot."""
    while True:
        if not bridge.connected:
            await bridge.connect()
        await asyncio.sleep(5)


# --- WebSocket endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        # Send last known state immediately
        if bridge.last_state:
            await ws.send_json(bridge.last_state)
        while True:
            # Keep connection alive, listen for pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# --- REST endpoints ---

@app.get("/api/state")
async def get_state():
    return bridge.last_state or {"type": "state", "mode": "disconnected"}


@app.post("/api/config/{key}")
async def update_config(key: str, body: dict):
    result = await bridge.send_config_update(key, body.get("value"))
    return result


@app.post("/api/kill-switch")
async def kill_switch(body: dict):
    result = await bridge.send_kill_switch(body.get("activate", False))
    return result


# Serve React build
frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = frontend_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dir / "index.html"))
