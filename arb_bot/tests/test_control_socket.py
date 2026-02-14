from __future__ import annotations

import asyncio
import json
import socket

import pytest

from arb_bot.control_socket import ControlSocket


@pytest.fixture
def free_port():
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_push_state_to_connected_client(free_port: int) -> None:
    async def _inner():
        cs = ControlSocket(port=free_port)
        await cs.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            await asyncio.sleep(0.05)  # let server accept handler register client
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

    _run(_inner())


def test_config_update_round_trip(free_port: int) -> None:
    async def _inner():
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

    _run(_inner())


def test_kill_switch_command(free_port: int, tmp_path) -> None:
    async def _inner():
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

    _run(_inner())


def test_no_crash_on_disconnected_client(free_port: int) -> None:
    async def _inner():
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

    _run(_inner())


def test_multiple_clients(free_port: int) -> None:
    async def _inner():
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
            w1.close()
            w2.close()
            await w1.wait_closed()
            await w2.wait_closed()
        finally:
            await cs.stop()

    _run(_inner())
