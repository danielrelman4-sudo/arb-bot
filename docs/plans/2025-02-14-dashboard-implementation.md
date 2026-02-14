# Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a sidecar web dashboard that shows live PnL, positions, trade feed, analytics, and allows hot-reload parameter tuning for the arb bot.

**Architecture:** Sidecar process (FastAPI + React) communicates with the bot via a TCP control socket on localhost:9120. Bot pushes state snapshots every cycle; dashboard sends config update commands. Historical data read from existing SQLite databases.

**Tech Stack:** Python 3.9 (asyncio TCP server, FastAPI, Uvicorn, aiosqlite), React 18 + TypeScript + Tailwind CSS + Recharts + Vite.

---

## Task 1: Bot-Side Control Socket

Add a thin TCP server to the bot that pushes state and accepts config commands. Zero new dependencies (stdlib only).

**Files:**
- Create: `arb_bot/control_socket.py`
- Modify: `arb_bot/engine.py` (add socket startup + state serialization)
- Modify: `arb_bot/config.py` (add `control_socket_port` setting)
- Test: `arb_bot/tests/test_control_socket.py`

### Step 1: Add config setting

Add to `AppSettings` in `arb_bot/config.py`:

```python
# After the existing paper_monte_carlo_seed field (~line 695):
control_socket_port: int = 9120
```

Add env var loading in `load_settings()` (~line 812, before the closing paren of `AppSettings(`):

```python
control_socket_port=_as_int(os.getenv("ARB_CONTROL_SOCKET_PORT"), 9120),
```

Commit: `feat(dashboard): add ARB_CONTROL_SOCKET_PORT config setting`

### Step 2: Write control_socket.py tests

Create `arb_bot/tests/test_control_socket.py`:

```python
from __future__ import annotations

import asyncio
import json
import pytest

from arb_bot.control_socket import ControlSocket


@pytest.fixture
def free_port():
    """Find a free port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_push_state_to_connected_client(free_port: int) -> None:
    cs = ControlSocket(port=free_port)
    await cs.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        state = {"type": "state", "cycle": 1, "mode": "live"}
        await cs.push_state(state)
        await asyncio.sleep(0.05)
        line = await asyncio.wait_for(reader.readline(), timeout=1.0)
        msg = json.loads(line)
        assert msg["type"] == "state"
        assert msg["cycle"] == 1
        writer.close()
        await writer.wait_closed()
    finally:
        await cs.stop()


@pytest.mark.asyncio
async def test_config_update_round_trip(free_port: int) -> None:
    results: list[dict] = []

    async def handler(key: str, value: object) -> dict:
        results.append({"key": key, "value": value})
        return {"ok": True, "old": 10.0, "new": value}

    cs = ControlSocket(port=free_port, on_config_update=handler)
    await cs.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        cmd = json.dumps({"type": "update_config", "key": "ARB_DAILY_LOSS_CAP_USD", "value": 15.0})
        writer.write((cmd + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=1.0)
        ack = json.loads(line)
        assert ack["ok"] is True
        assert ack["new"] == 15.0
        assert len(results) == 1
        writer.close()
        await writer.wait_closed()
    finally:
        await cs.stop()


@pytest.mark.asyncio
async def test_kill_switch_command(free_port: int, tmp_path) -> None:
    kill_file = tmp_path / ".kill_switch"

    cs = ControlSocket(port=free_port, kill_switch_path=str(kill_file))
    await cs.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        cmd = json.dumps({"type": "kill_switch", "activate": True})
        writer.write((cmd + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=1.0)
        ack = json.loads(line)
        assert ack["ok"] is True
        assert kill_file.exists()
        writer.close()
        await writer.wait_closed()
    finally:
        await cs.stop()


@pytest.mark.asyncio
async def test_no_crash_on_disconnected_client(free_port: int) -> None:
    cs = ControlSocket(port=free_port)
    await cs.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        # Should not raise even though client disconnected
        await cs.push_state({"type": "state", "cycle": 1})
        assert cs.client_count == 0
    finally:
        await cs.stop()


@pytest.mark.asyncio
async def test_multiple_clients(free_port: int) -> None:
    cs = ControlSocket(port=free_port)
    await cs.start()
    try:
        r1, w1 = await asyncio.open_connection("127.0.0.1", free_port)
        r2, w2 = await asyncio.open_connection("127.0.0.1", free_port)
        await asyncio.sleep(0.05)
        assert cs.client_count == 2
        await cs.push_state({"type": "state", "cycle": 5})
        await asyncio.sleep(0.05)
        line1 = await asyncio.wait_for(r1.readline(), timeout=1.0)
        line2 = await asyncio.wait_for(r2.readline(), timeout=1.0)
        assert json.loads(line1)["cycle"] == 5
        assert json.loads(line2)["cycle"] == 5
        w1.close(); w2.close()
        await w1.wait_closed(); await w2.wait_closed()
    finally:
        await cs.stop()
```

Run: `python3 -m pytest arb_bot/tests/test_control_socket.py -v`
Expected: FAIL (module not found)

### Step 3: Implement control_socket.py

Create `arb_bot/control_socket.py`:

```python
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
```

Run: `python3 -m pytest arb_bot/tests/test_control_socket.py -v`
Expected: All 5 tests PASS

Run: `python3 -m pytest arb_bot/tests/ -x -q`
Expected: All tests pass (no regressions)

Commit: `feat(dashboard): add ControlSocket TCP server for dashboard IPC`

### Step 4: Wire control socket into engine

Modify `arb_bot/engine.py`:

**In `__init__` (after existing initialization, ~line 200):**
```python
from arb_bot.control_socket import ControlSocket

# Add at end of __init__:
self._control_socket: ControlSocket | None = None
```

**In `run_forever` (at the start of the try block, before the while loop, ~line 250):**
```python
if self._settings.control_socket_port > 0:
    self._control_socket = ControlSocket(
        port=self._settings.control_socket_port,
        on_config_update=self._handle_config_update,
        kill_switch_path=self._settings.risk.kill_switch_file,
    )
    await self._control_socket.start()
```

**In `run_forever` finally block (before `await self.aclose()`):**
```python
if self._control_socket is not None:
    await self._control_socket.stop()
```

**After `run_once` returns (at the end of run_once, before the return):**
Add a `_push_dashboard_state` call. Find where `run_once` builds the CycleReport and add:

```python
if self._control_socket is not None and self._control_socket.client_count > 0:
    await self._push_dashboard_state(report)
```

**Add new methods to ArbEngine:**
```python
async def _push_dashboard_state(self, report: CycleReport) -> None:
    state = self._build_state_snapshot(report)
    try:
        await self._control_socket.push_state(state)
    except Exception:
        LOGGER.debug("control socket push failed", exc_info=True)

def _build_state_snapshot(self, report: CycleReport) -> dict:
    import time as _time
    state = self._state
    return {
        "type": "state",
        "ts": _time.time(),
        "cycle": getattr(self, "_cycle_count", 0),
        "mode": "live" if self._settings.live_mode else "paper",
        "uptime_seconds": _time.time() - self._start_ts if hasattr(self, "_start_ts") else 0,
        "cash_by_venue": dict(state.cash_by_venue),
        "locked_capital_by_venue": dict(state.locked_capital_by_venue),
        "open_positions": [],
        "last_cycle": {
            "duration_ms": int((report.ended_at - report.started_at).total_seconds() * 1000),
            "quotes_count": report.quotes_count,
            "opportunities_found": report.opportunities_count,
            "near_opportunities": report.near_opportunities_count,
            "decisions": [
                {
                    "action": d.action,
                    "kind": str(d.opportunity.kind.value) if d.opportunity else "",
                    "match_key": getattr(d.opportunity, "match_key", ""),
                    "reason": d.reason,
                    "edge": d.metrics.get("detected_edge_per_contract", 0),
                    "fill_prob": d.metrics.get("fill_probability", 0),
                }
                for d in report.decisions
            ],
        },
        "stream_status": {},
        "daily_pnl": 0.0,
        "daily_trades": 0,
        "config_snapshot": self._config_snapshot(),
    }

def _config_snapshot(self) -> dict[str, object]:
    """Extract key tunable settings for the dashboard."""
    s = self._settings
    return {
        "ARB_LIVE_MODE": s.live_mode,
        "ARB_DAILY_LOSS_CAP_USD": s.risk.daily_loss_cap_usd if hasattr(s.risk, "daily_loss_cap_usd") else 0,
        "ARB_MAX_DOLLARS_PER_TRADE": s.sizing.max_dollars_per_trade,
        "ARB_MAX_CONTRACTS_PER_TRADE": s.sizing.max_contracts_per_trade,
        "ARB_MAX_BANKROLL_FRACTION_PER_TRADE": s.sizing.max_bankroll_fraction_per_trade,
        "ARB_MARKET_COOLDOWN_SECONDS": s.risk.market_cooldown_seconds,
        "ARB_LANE_STRUCTURAL_BUCKET_ENABLED": s.lanes.structural_bucket.enabled,
        "ARB_LANE_CROSS_ENABLED": s.lanes.cross_venue.enabled,
        "ARB_LANE_STRUCTURAL_PARITY_ENABLED": s.lanes.structural_parity.enabled,
        "ARB_MAX_BUCKET_LEGS": s.strategy.max_bucket_legs,
        "ARB_MAX_BUCKET_CONSECUTIVE_FAILURES": s.strategy.max_consecutive_bucket_failures,
    }

async def _handle_config_update(self, key: str, value: object) -> dict:
    """Handle a config update from the dashboard."""
    NOT_HOT_RELOADABLE = {
        "KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_PRIVATE_KEY_PEM",
        "ARB_STRUCTURAL_RULES_PATH", "ARB_CONTROL_SOCKET_PORT",
    }
    if key in NOT_HOT_RELOADABLE:
        return {"ok": False, "error": "requires_restart"}

    snapshot = self._config_snapshot()
    if key not in snapshot:
        return {"ok": False, "error": f"unknown key: {key}"}

    old = snapshot[key]
    # Apply the update by setting the env var and rebuilding settings
    import os
    os.environ[key] = str(value)
    try:
        from arb_bot.config import load_settings
        new_settings = load_settings()
        self._settings = new_settings
        return {"ok": True, "old": old, "new": value}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
```

Also add `self._start_ts = time.time()` and `self._cycle_count = 0` in `__init__`, and `self._cycle_count += 1` at the start of `run_once`.

Run: `python3 -m pytest arb_bot/tests/ -x -q`
Expected: All tests pass

Commit: `feat(dashboard): wire ControlSocket into ArbEngine with state push + config update`

---

## Task 2: Dashboard Server — Project Setup + Bot Bridge

Set up the FastAPI server and the TCP client that connects to the bot.

**Files:**
- Create: `dashboard/server/main.py`
- Create: `dashboard/server/bot_bridge.py`
- Create: `dashboard/server/requirements.txt`

### Step 1: Create requirements.txt

Create `dashboard/server/requirements.txt`:

```
fastapi>=0.109.0
uvicorn>=0.27.0
aiosqlite>=0.19.0
websockets>=12.0
```

Install: `python3 -m pip install fastapi uvicorn aiosqlite`

### Step 2: Create bot_bridge.py

Create `dashboard/server/bot_bridge.py`:

```python
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
```

### Step 3: Create main.py (FastAPI app)

Create `dashboard/server/main.py`:

```python
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from bot_bridge import BotBridge

LOGGER = logging.getLogger(__name__)

# Global state
bridge = BotBridge()
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
```

Commit: `feat(dashboard): add FastAPI server with BotBridge TCP client`

---

## Task 3: Dashboard Server — SQLite Analytics Routes

Add read-only SQLite queries for historical analytics data.

**Files:**
- Create: `dashboard/server/db_reader.py`
- Create: `dashboard/server/routes/__init__.py`
- Create: `dashboard/server/routes/analytics.py`
- Modify: `dashboard/server/main.py` (mount analytics router)

### Step 1: Create db_reader.py

Create `dashboard/server/db_reader.py`:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _find_db(name: str) -> str | None:
    """Search common locations for the SQLite database."""
    candidates = [
        Path(name),
        Path("data") / name,
        Path("..") / "data" / name,
        Path("..") / ".." / name,
        Path("..") / ".." / "data" / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


class DbReader:
    """Read-only access to the bot's SQLite databases."""

    def __init__(
        self,
        analytics_path: str | None = None,
        orders_path: str | None = None,
    ) -> None:
        self._analytics_path = analytics_path or _find_db("analytics.db")
        self._orders_path = orders_path or _find_db("arb_orders.db")

    def _analytics_conn(self) -> sqlite3.Connection:
        if not self._analytics_path:
            raise FileNotFoundError("analytics.db not found")
        conn = sqlite3.connect(self._analytics_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _orders_conn(self) -> sqlite3.Connection:
        if not self._orders_path:
            raise FileNotFoundError("arb_orders.db not found")
        conn = sqlite3.connect(self._orders_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Analytics queries ---

    def daily_pnl(self, limit: int = 30) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT date, SUM(realized_pnl) as pnl, COUNT(*) as trades, "
                "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
                "AVG(slippage) as avg_slippage, AVG(fill_rate) as avg_fill_rate "
                "FROM fills GROUP BY date ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def pnl_by_lane(self) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT kind, SUM(realized_pnl) as pnl, COUNT(*) as trades, "
                "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins "
                "FROM fills GROUP BY kind ORDER BY pnl DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def pnl_by_bucket(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT market_key as group_id, SUM(realized_pnl) as pnl, "
                "COUNT(*) as trades, "
                "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
                "AVG(realized_pnl) as avg_pnl "
                "FROM fills WHERE kind = 'structural_bucket' "
                "GROUP BY market_key ORDER BY pnl DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def equity_curve(self) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT timestamp, realized_pnl, kind, market_key "
                "FROM fills ORDER BY timestamp ASC"
            ).fetchall()
            curve = []
            cumulative = 0.0
            high_water = 0.0
            for r in rows:
                cumulative += r["realized_pnl"]
                high_water = max(high_water, cumulative)
                curve.append({
                    "ts": r["timestamp"],
                    "pnl": round(cumulative, 4),
                    "trade_pnl": round(r["realized_pnl"], 4),
                    "high_water": round(high_water, 4),
                    "drawdown": round(cumulative - high_water, 4),
                    "kind": r["kind"],
                })
            return curve
        finally:
            conn.close()

    # --- Order queries ---

    def open_positions(self) -> list[dict[str, Any]]:
        conn = self._orders_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._orders_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM order_intents ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
```

### Step 2: Create analytics route

Create `dashboard/server/routes/__init__.py`: (empty file)

Create `dashboard/server/routes/analytics.py`:

```python
from __future__ import annotations

from fastapi import APIRouter

from db_reader import DbReader

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
db = DbReader()


@router.get("/daily-pnl")
async def daily_pnl(limit: int = 30):
    return db.daily_pnl(limit)


@router.get("/by-lane")
async def pnl_by_lane():
    return db.pnl_by_lane()


@router.get("/by-bucket")
async def pnl_by_bucket(limit: int = 50):
    return db.pnl_by_bucket(limit)


@router.get("/equity-curve")
async def equity_curve():
    return db.equity_curve()


@router.get("/positions")
async def positions():
    return db.open_positions()


@router.get("/recent-trades")
async def recent_trades(limit: int = 50):
    return db.recent_trades(limit)
```

### Step 3: Mount router in main.py

Add to `dashboard/server/main.py`:

```python
from routes.analytics import router as analytics_router
app.include_router(analytics_router)
```

Commit: `feat(dashboard): add SQLite analytics routes for historical data`

---

## Task 4: Frontend — Project Scaffold + Layout

Set up the React + TypeScript + Tailwind project with routing and the shell layout.

**Files:**
- Create: `dashboard/frontend/` (entire Vite project)

### Step 1: Scaffold Vite + React + TypeScript

```bash
cd dashboard
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install react-router-dom recharts
```

### Step 2: Configure Tailwind

Update `dashboard/frontend/src/index.css`:

```css
@import "tailwindcss";
```

Update `dashboard/frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': { target: 'ws://localhost:8080', ws: true },
    },
  },
})
```

### Step 3: Create types and hooks

Create `dashboard/frontend/src/lib/types.ts`:

```typescript
export interface BotState {
  type: "state";
  ts: number;
  cycle: number;
  mode: "live" | "paper" | "disconnected";
  uptime_seconds: number;
  cash_by_venue: Record<string, number>;
  locked_capital_by_venue: Record<string, number>;
  open_positions: Position[];
  last_cycle: CycleInfo;
  stream_status: Record<string, string>;
  daily_pnl: number;
  daily_trades: number;
  daily_loss_cap_remaining: number;
  consecutive_failures: number;
  config_snapshot: Record<string, number | boolean | string>;
}

export interface CycleInfo {
  duration_ms: number;
  quotes_count: number;
  opportunities_found: number;
  near_opportunities: number;
  decisions: Decision[];
}

export interface Decision {
  action: string;
  kind: string;
  match_key: string;
  reason: string;
  edge: number;
  fill_prob: number;
}

export interface Position {
  market_id: string;
  side: string;
  contracts: number;
  entry: number;
  expected_profit: number;
  opened_at: number;
}

export interface DailyPnl {
  date: string;
  pnl: number;
  trades: number;
  wins: number;
  avg_slippage: number;
  avg_fill_rate: number;
}

export interface EquityPoint {
  ts: number;
  pnl: number;
  trade_pnl: number;
  high_water: number;
  drawdown: number;
  kind: string;
}

export interface LanePnl {
  kind: string;
  pnl: number;
  trades: number;
  wins: number;
}

export interface BucketPnl {
  group_id: string;
  pnl: number;
  trades: number;
  wins: number;
  avg_pnl: number;
}
```

Create `dashboard/frontend/src/hooks/useWebSocket.ts`:

```typescript
import { useEffect, useRef, useState, useCallback } from "react";
import type { BotState, Decision } from "../lib/types";

const EMPTY_STATE: BotState = {
  type: "state",
  ts: 0,
  cycle: 0,
  mode: "disconnected",
  uptime_seconds: 0,
  cash_by_venue: {},
  locked_capital_by_venue: {},
  open_positions: [],
  last_cycle: { duration_ms: 0, quotes_count: 0, opportunities_found: 0, near_opportunities: 0, decisions: [] },
  stream_status: {},
  daily_pnl: 0,
  daily_trades: 0,
  daily_loss_cap_remaining: 0,
  consecutive_failures: 0,
  config_snapshot: {},
};

export function useWebSocket() {
  const [state, setState] = useState<BotState>(EMPTY_STATE);
  const [connected, setConnected] = useState(false);
  const [tradeFeed, setTradeFeed] = useState<Decision[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws`;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as BotState;
          setState(data);
          if (data.last_cycle?.decisions?.length) {
            setTradeFeed((prev) => [...data.last_cycle.decisions, ...prev].slice(0, 200));
          }
        } catch {}
      };
    }

    connect();
    return () => { wsRef.current?.close(); };
  }, []);

  const sendPing = useCallback(() => {
    wsRef.current?.send("ping");
  }, []);

  return { state, connected, tradeFeed, sendPing };
}
```

Create `dashboard/frontend/src/hooks/useApi.ts`:

```typescript
import { useState, useEffect, useCallback } from "react";

export function useApi<T>(url: string, defaultValue: T) {
  const [data, setData] = useState<T>(defaultValue);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(url);
      const json = await res.json();
      setData(json);
    } catch {
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => { refresh(); }, [refresh]);

  return { data, loading, refresh };
}

export async function postConfig(key: string, value: number | boolean | string) {
  const res = await fetch(`/api/config/${key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  return res.json();
}

export async function postKillSwitch(activate: boolean) {
  const res = await fetch("/api/kill-switch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activate }),
  });
  return res.json();
}
```

### Step 4: Create App shell with routing

Replace `dashboard/frontend/src/App.tsx`:

```tsx
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { useWebSocket } from "./hooks/useWebSocket";
import Dashboard from "./pages/Dashboard";
import Analytics from "./pages/Analytics";
import Config from "./pages/Config";
import System from "./pages/System";

function App() {
  const { state, connected, tradeFeed } = useWebSocket();

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-950 text-gray-100">
        {/* Sidebar */}
        <nav className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-1">
          <h1 className="text-lg font-bold mb-4 text-emerald-400">Arb Bot</h1>
          <NavLink to="/" end className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>Dashboard</NavLink>
          <NavLink to="/analytics" className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>Analytics</NavLink>
          <NavLink to="/config" className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>Config</NavLink>
          <NavLink to="/system" className={({ isActive }) =>
            `px-3 py-2 rounded text-sm ${isActive ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`
          }>System</NavLink>
          <div className="mt-auto">
            <div className={`text-xs px-3 py-1 rounded ${connected ? "text-emerald-400" : "text-red-400"}`}>
              {connected ? "Connected" : "Disconnected"}
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<Dashboard state={state} tradeFeed={tradeFeed} />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/config" element={<Config state={state} />} />
            <Route path="/system" element={<System state={state} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
```

### Step 5: Create placeholder pages

Create `dashboard/frontend/src/pages/Dashboard.tsx`:
```tsx
import type { BotState, Decision } from "../lib/types";

export default function Dashboard({ state, tradeFeed }: { state: BotState; tradeFeed: Decision[] }) {
  return <div><h2 className="text-xl font-bold mb-4">Dashboard</h2><p>Coming in Task 5</p></div>;
}
```

Create `dashboard/frontend/src/pages/Analytics.tsx`:
```tsx
export default function Analytics() {
  return <div><h2 className="text-xl font-bold mb-4">Analytics</h2><p>Coming in Task 6</p></div>;
}
```

Create `dashboard/frontend/src/pages/Config.tsx`:
```tsx
import type { BotState } from "../lib/types";
export default function Config({ state }: { state: BotState }) {
  return <div><h2 className="text-xl font-bold mb-4">Config</h2><p>Coming in Task 7</p></div>;
}
```

Create `dashboard/frontend/src/pages/System.tsx`:
```tsx
import type { BotState } from "../lib/types";
export default function System({ state }: { state: BotState }) {
  return <div><h2 className="text-xl font-bold mb-4">System</h2><p>Coming in Task 7</p></div>;
}
```

Run: `cd dashboard/frontend && npm run build`
Expected: Build succeeds

Commit: `feat(dashboard): scaffold React frontend with routing, types, WebSocket hook`

---

## Task 5: Frontend — Dashboard Page (Live PnL + Trade Feed)

Build the primary Dashboard page with header stats, PnL chart, positions table, and scrolling trade feed.

**Files:**
- Create: `dashboard/frontend/src/components/PnlChart.tsx`
- Create: `dashboard/frontend/src/components/PositionsTable.tsx`
- Create: `dashboard/frontend/src/components/TradeFeed.tsx`
- Create: `dashboard/frontend/src/components/StatusBadge.tsx`
- Modify: `dashboard/frontend/src/pages/Dashboard.tsx`

### Step 1: Build components and wire the page

Implement `StatusBadge`, `PnlChart` (Recharts line chart), `PositionsTable`, `TradeFeed`, and the full `Dashboard` page with the header bar, 2-column top layout, and scrolling feed.

Key component details:
- **StatusBadge**: Green "LIVE" or yellow "PAPER" or red "DISCONNECTED" pill
- **PnlChart**: Recharts `LineChart` showing cumulative daily PnL from `tradeFeed`, updates live
- **PositionsTable**: Table from `state.open_positions` showing market, side, contracts, entry, expected profit
- **TradeFeed**: Scrolling list of `tradeFeed` decisions with timestamp, action badge (FILLED/SKIPPED/DETECTED), kind, match_key, reason

Run: `cd dashboard/frontend && npm run build`
Expected: Build succeeds

Commit: `feat(dashboard): implement Dashboard page with live PnL, positions, trade feed`

---

## Task 6: Frontend — Analytics Page (Equity Curve + Attribution)

Build the Analytics page with equity curve, daily PnL bars, lane attribution, and per-bucket table.

**Files:**
- Create: `dashboard/frontend/src/components/EquityCurve.tsx`
- Modify: `dashboard/frontend/src/pages/Analytics.tsx`

### Step 1: Build Analytics page

Use `useApi` hook to fetch from `/api/analytics/equity-curve`, `/api/analytics/daily-pnl`, `/api/analytics/by-lane`, `/api/analytics/by-bucket`.

Key component details:
- **EquityCurve**: Recharts `AreaChart` with cumulative PnL line, high water mark dashed line, drawdown shaded area
- **Daily PnL**: Recharts `BarChart`, green bars for positive days, red for negative
- **Lane attribution**: Recharts `PieChart` or horizontal `BarChart` with toggle
- **Bucket table**: HTML table with sorting (click column headers), filter dropdown for lane

Run: `cd dashboard/frontend && npm run build`
Expected: Build succeeds

Commit: `feat(dashboard): implement Analytics page with equity curve, PnL attribution`

---

## Task 7: Frontend — Config + System Pages

Build the Config page (parameter editor) and System page (health + kill switch).

**Files:**
- Create: `dashboard/frontend/src/components/ConfigSection.tsx`
- Create: `dashboard/frontend/src/components/KillSwitch.tsx`
- Modify: `dashboard/frontend/src/pages/Config.tsx`
- Modify: `dashboard/frontend/src/pages/System.tsx`

### Step 1: Build Config page

Use `state.config_snapshot` to render current values. Group params into collapsible sections. Each param: label, current value, input field, Apply button. Apply calls `postConfig(key, value)`.

Config sections:
- Safety Rails: `ARB_DAILY_LOSS_CAP_USD`, `ARB_MAX_CONSECUTIVE_FAILURES`
- Sizing: `ARB_MAX_DOLLARS_PER_TRADE`, `ARB_MAX_CONTRACTS_PER_TRADE`, `ARB_MAX_BANKROLL_FRACTION_PER_TRADE`
- Lanes: `ARB_LANE_STRUCTURAL_BUCKET_ENABLED`, `ARB_LANE_CROSS_ENABLED`, `ARB_LANE_STRUCTURAL_PARITY_ENABLED`
- Bucket Quality: `ARB_MAX_BUCKET_LEGS`, `ARB_MAX_BUCKET_CONSECUTIVE_FAILURES`
- Cooldowns: `ARB_MARKET_COOLDOWN_SECONDS`

### Step 2: Build System page

- Stream status: colored badges from `state.stream_status`
- Cycle info: `state.last_cycle.duration_ms`, `quotes_count`
- Uptime: formatted from `state.uptime_seconds`
- Consecutive failures: `state.consecutive_failures`
- Kill switch: Big red button with confirmation dialog, calls `postKillSwitch(true)`

Run: `cd dashboard/frontend && npm run build`
Expected: Build succeeds

Final commit: `feat(dashboard): implement Config and System pages`

---

## Summary

| Task | Description | Bot Changes | Dashboard Changes |
|------|-------------|-------------|-------------------|
| 1 | Control Socket | `control_socket.py` (new), `engine.py`, `config.py` | None |
| 2 | Server Setup | None | `main.py`, `bot_bridge.py` |
| 3 | Analytics Routes | None | `db_reader.py`, `routes/analytics.py` |
| 4 | Frontend Scaffold | None | Vite project, routing, hooks, types |
| 5 | Dashboard Page | None | PnlChart, PositionsTable, TradeFeed |
| 6 | Analytics Page | None | EquityCurve, daily/lane/bucket views |
| 7 | Config + System | None | ConfigSection, KillSwitch |

Total new files: ~20. Bot-side changes: 1 new module (~120 lines) + minor engine.py wiring.
