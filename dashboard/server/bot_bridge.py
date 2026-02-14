from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class BotBridge:
    """TCP client that connects to the bot's control socket."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9120,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._on_state = on_state
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._listen_task: asyncio.Task | None = None
        self._connected = False
        self._last_state: dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_state(self) -> dict[str, Any]:
        return dict(self._last_state)

    async def connect(self) -> None:
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self._host, self._port
            )
            self._connected = True
            self._listen_task = asyncio.create_task(self._listen_loop())
            LOGGER.info("connected to bot at %s:%d", self._host, self._port)
        except (ConnectionRefusedError, OSError) as exc:
            LOGGER.warning("cannot connect to bot: %s", exc)
            self._connected = False

    async def disconnect(self) -> None:
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False

    async def send_config_update(self, key: str, value: Any) -> dict[str, Any]:
        if not self._connected or self._writer is None or self._reader is None:
            return {"ok": False, "error": "not connected"}
        msg = json.dumps({"type": "update_config", "key": key, "value": value})
        self._writer.write((msg + "\n").encode("utf-8"))
        await self._writer.drain()
        # Read response (next non-state line)
        while True:
            line = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
            if not line:
                return {"ok": False, "error": "connection closed"}
            data = json.loads(line)
            if data.get("type") == "update_config_ack":
                return data
            # It was a state push, store it and keep reading
            self._last_state = data
            if self._on_state:
                self._on_state(data)

    async def send_kill_switch(self, activate: bool) -> dict[str, Any]:
        if not self._connected or self._writer is None or self._reader is None:
            return {"ok": False, "error": "not connected"}
        msg = json.dumps({"type": "kill_switch", "activate": activate})
        self._writer.write((msg + "\n").encode("utf-8"))
        await self._writer.drain()
        while True:
            line = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
            if not line:
                return {"ok": False, "error": "connection closed"}
            data = json.loads(line)
            if data.get("type") == "kill_switch_ack":
                return data
            self._last_state = data
            if self._on_state:
                self._on_state(data)

    async def _listen_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "state":
                    self._last_state = data
                    if self._on_state:
                        self._on_state(data)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._connected = False
