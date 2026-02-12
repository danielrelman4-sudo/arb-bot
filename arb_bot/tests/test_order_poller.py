"""Tests for Phase 0D: Post-submission order status polling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from arb_bot.config import RiskSettings
from arb_bot.models import OrderState, OrderStatus
from arb_bot.order_poller import PollResult, poll_order_until_terminal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(state: OrderState, filled: int = 0, remaining: int = 0) -> OrderStatus:
    return OrderStatus(
        order_id="ORD-1",
        state=state,
        filled_contracts=filled,
        remaining_contracts=remaining,
    )


class FakePollAdapter:
    """Adapter that returns a configurable sequence of OrderStatus responses."""

    def __init__(self) -> None:
        self.venue = "test"
        self._statuses: list[OrderStatus | None] = []
        self._status_idx = 0
        self._cancel_result = True
        self._cancel_call_count = 0

    def set_statuses(self, *statuses: OrderStatus | None) -> None:
        self._statuses = list(statuses)
        self._status_idx = 0

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        if not self._statuses:
            return None
        idx = min(self._status_idx, len(self._statuses) - 1)
        self._status_idx += 1
        return self._statuses[idx]

    async def cancel_order(self, order_id: str) -> bool:
        self._cancel_call_count += 1
        return self._cancel_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPollResult:
    def test_defaults(self) -> None:
        r = PollResult(order_id="X", final_status=None)
        assert r.timed_out is False
        assert r.cancelled_on_timeout is False


class TestPollOrderUntilTerminal:
    def test_immediately_filled(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(_status(OrderState.FILLED, filled=10))

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=1.0)
        )

        assert result.final_status is not None
        assert result.final_status.state is OrderState.FILLED
        assert result.timed_out is False
        assert result.cancelled_on_timeout is False

    def test_transitions_open_to_filled(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(
            _status(OrderState.OPEN),
            _status(OrderState.OPEN),
            _status(OrderState.FILLED, filled=5),
        )

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=2.0)
        )

        assert result.final_status is not None
        assert result.final_status.state is OrderState.FILLED
        assert result.timed_out is False

    def test_cancelled_is_terminal(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(_status(OrderState.CANCELLED))

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=1.0)
        )

        assert result.final_status.state is OrderState.CANCELLED
        assert result.timed_out is False

    def test_expired_is_terminal(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(_status(OrderState.EXPIRED))

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=1.0)
        )

        assert result.final_status.state is OrderState.EXPIRED
        assert result.timed_out is False

    def test_timeout_triggers_cancel(self) -> None:
        adapter = FakePollAdapter()
        # Always return OPEN — will never reach terminal.
        adapter.set_statuses(_status(OrderState.OPEN))
        adapter._cancel_result = True

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(
                adapter, "ORD-1",
                poll_interval=0.01, timeout=0.05,
                cancel_on_timeout=True,
            )
        )

        assert result.timed_out is True
        assert result.cancelled_on_timeout is True
        assert adapter._cancel_call_count == 1

    def test_timeout_without_cancel(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(_status(OrderState.OPEN))

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(
                adapter, "ORD-1",
                poll_interval=0.01, timeout=0.05,
                cancel_on_timeout=False,
            )
        )

        assert result.timed_out is True
        assert result.cancelled_on_timeout is False
        assert adapter._cancel_call_count == 0

    def test_timeout_cancel_fails(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(_status(OrderState.OPEN))
        adapter._cancel_result = False

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(
                adapter, "ORD-1",
                poll_interval=0.01, timeout=0.05,
                cancel_on_timeout=True,
            )
        )

        assert result.timed_out is True
        assert result.cancelled_on_timeout is False

    def test_adapter_returns_none_exits_immediately(self) -> None:
        """If adapter doesn't support get_order_status, return immediately."""
        adapter = FakePollAdapter()
        adapter.set_statuses(None)

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=1.0)
        )

        assert result.final_status is None
        assert result.timed_out is False

    def test_error_during_poll_retries(self) -> None:
        adapter = FakePollAdapter()
        call_count = 0

        async def _flaky_status(order_id: str) -> OrderStatus | None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("transient")
            return _status(OrderState.FILLED, filled=5)

        adapter.get_order_status = _flaky_status  # type: ignore[assignment]

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=2.0)
        )

        assert result.final_status is not None
        assert result.final_status.state is OrderState.FILLED
        assert call_count == 3

    def test_partially_filled_keeps_polling(self) -> None:
        adapter = FakePollAdapter()
        adapter.set_statuses(
            _status(OrderState.PARTIALLY_FILLED, filled=3, remaining=7),
            _status(OrderState.PARTIALLY_FILLED, filled=5, remaining=5),
            _status(OrderState.FILLED, filled=10, remaining=0),
        )

        result = asyncio.get_event_loop().run_until_complete(
            poll_order_until_terminal(adapter, "ORD-1", poll_interval=0.01, timeout=2.0)
        )

        assert result.final_status.state is OrderState.FILLED
        assert result.final_status.filled_contracts == 10


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestPollingConfig:
    def test_defaults(self) -> None:
        r = RiskSettings()
        assert r.order_poll_interval_seconds == 1.0
        assert r.order_poll_timeout_seconds == 15.0
        assert r.cancel_on_poll_timeout is True

    def test_custom(self) -> None:
        r = RiskSettings(
            order_poll_interval_seconds=0.5,
            order_poll_timeout_seconds=30.0,
            cancel_on_poll_timeout=False,
        )
        assert r.order_poll_interval_seconds == 0.5
        assert r.order_poll_timeout_seconds == 30.0
        assert r.cancel_on_poll_timeout is False
