from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

LOGGER = logging.getLogger(__name__)

OnConfigUpdate = Callable[[str, Any], Awaitable[dict[str, Any]]]


class ControlSocket:
    """TCP server for dashboard communication.

    Pushes state snapshots to connected clients and accepts
    config update / kill switch commands.
    """

    def __init__(
        self,
        port: int = 9120,
        host: str = "127.0.0.1",
        on_config_update: OnConfigUpdate | None = None,
        kill_switch_path: str = ".kill_switch",
    ) -> None:
        self._port = port
        self._host = host
        self._on_config_update = on_config_update
        self._kill_switch_path = kill_switch_path
        self._server: asyncio.AbstractServer | None = None
        self._clients: list[asyncio.StreamWriter] = []

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._port
        )
        LOGGER.info("control socket listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        for writer in list(self._clients):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def push_state(self, state: dict[str, Any]) -> None:
        line = json.dumps(state, default=str) + "\n"
        data = line.encode("utf-8")
        dead: list[asyncio.StreamWriter] = []
        for writer in self._clients:
            try:
                writer.write(data)
                await writer.drain()
            except Exception:
                dead.append(writer)
        for writer in dead:
            self._clients.remove(writer)
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._clients.append(writer)
        peer = writer.get_extra_info("peername")
        LOGGER.info("control socket client connected: %s", peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resp = await self._handle_message(msg)
                if resp is not None:
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            if writer in self._clients:
                self._clients.remove(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            LOGGER.info("control socket client disconnected: %s", peer)

    async def _handle_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        msg_type = msg.get("type", "")

        if msg_type == "update_config":
            key = msg.get("key", "")
            value = msg.get("value")
            if not key:
                return {"type": "update_config_ack", "ok": False, "error": "missing key"}
            if self._on_config_update is None:
                return {"type": "update_config_ack", "ok": False, "error": "no handler"}
            try:
                result = await self._on_config_update(key, value)
                return {"type": "update_config_ack", "key": key, **result}
            except Exception as exc:
                return {"type": "update_config_ack", "ok": False, "key": key, "error": str(exc)}

        if msg_type == "kill_switch":
            activate = msg.get("activate", False)
            try:
                if activate:
                    Path(self._kill_switch_path).touch()
                else:
                    Path(self._kill_switch_path).unlink(missing_ok=True)
                return {"type": "kill_switch_ack", "active": activate, "ok": True}
            except Exception as exc:
                return {"type": "kill_switch_ack", "ok": False, "error": str(exc)}

        return None
