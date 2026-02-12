"""Tests for Phase 0C: Cancel/replace for unfilled legs."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arb_bot.models import OrderState, OrderStatus


# ---------------------------------------------------------------------------
# OrderStatus / OrderState model tests
# ---------------------------------------------------------------------------


class TestOrderStatusModel:
    def test_order_state_values(self) -> None:
        assert OrderState.OPEN.value == "open"
        assert OrderState.FILLED.value == "filled"
        assert OrderState.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderState.CANCELLED.value == "cancelled"
        assert OrderState.EXPIRED.value == "expired"
        assert OrderState.UNKNOWN.value == "unknown"

    def test_order_status_defaults(self) -> None:
        status = OrderStatus(order_id="X", state=OrderState.OPEN)
        assert status.order_id == "X"
        assert status.state is OrderState.OPEN
        assert status.filled_contracts == 0
        assert status.remaining_contracts == 0
        assert status.average_price is None
        assert status.raw == {}

    def test_order_status_with_fields(self) -> None:
        status = OrderStatus(
            order_id="ORD-1",
            state=OrderState.PARTIALLY_FILLED,
            filled_contracts=3,
            remaining_contracts=7,
            average_price=0.45,
            raw={"exchange_status": "partial"},
        )
        assert status.filled_contracts == 3
        assert status.remaining_contracts == 7
        assert status.average_price == 0.45


# ---------------------------------------------------------------------------
# ExchangeAdapter base defaults
# ---------------------------------------------------------------------------


class TestBaseAdapterDefaults:
    def test_cancel_order_returns_false(self) -> None:
        from arb_bot.exchanges.base import ExchangeAdapter

        class Stub(ExchangeAdapter):
            venue = "stub"
            async def fetch_quotes(self): return []
            async def place_pair_order(self, plan): raise NotImplementedError
            async def place_single_order(self, leg): raise NotImplementedError

        adapter = Stub()
        result = asyncio.get_event_loop().run_until_complete(adapter.cancel_order("x"))
        assert result is False

    def test_get_order_status_returns_none(self) -> None:
        from arb_bot.exchanges.base import ExchangeAdapter

        class Stub(ExchangeAdapter):
            venue = "stub"
            async def fetch_quotes(self): return []
            async def place_pair_order(self, plan): raise NotImplementedError
            async def place_single_order(self, leg): raise NotImplementedError

        adapter = Stub()
        result = asyncio.get_event_loop().run_until_complete(adapter.get_order_status("x"))
        assert result is None


# ---------------------------------------------------------------------------
# Kalshi cancel_order / get_order_status
# ---------------------------------------------------------------------------


class TestKalshiCancelReplace:
    def _make_adapter(self):
        from arb_bot.exchanges.kalshi import KalshiAdapter
        from arb_bot.config import KalshiSettings

        settings = KalshiSettings(
            enabled=True,
            api_base_url="https://trading-api.kalshi.com/trade-api/v2",
            key_id="test-key",
            private_key_path="",
        )
        adapter = KalshiAdapter(settings)
        # Set a fake private key so credential checks pass.
        adapter._private_key = MagicMock()
        return adapter

    def test_cancel_order_success(self) -> None:
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(return_value={})

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("ORD-123")
        )
        assert result is True
        adapter._private_request.assert_awaited_once_with(
            "DELETE", "/portfolio/orders/ORD-123"
        )

    def test_cancel_order_failure(self) -> None:
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(side_effect=Exception("not found"))

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("ORD-123")
        )
        assert result is False

    def test_cancel_order_no_credentials(self) -> None:
        adapter = self._make_adapter()
        adapter._private_key = None

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("ORD-123")
        )
        assert result is False

    def test_get_order_status_resting(self) -> None:
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(return_value={
            "order": {
                "order_id": "ORD-1",
                "status": "resting",
                "filled_count": 0,
                "remaining_count": 10,
                "average_price": None,
            }
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("ORD-1")
        )
        assert status is not None
        assert status.state is OrderState.OPEN
        assert status.filled_contracts == 0
        assert status.remaining_contracts == 10

    def test_get_order_status_executed(self) -> None:
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(return_value={
            "order": {
                "order_id": "ORD-2",
                "status": "executed",
                "filled_count": 10,
                "remaining_count": 0,
                "average_price": 45,
            }
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("ORD-2")
        )
        assert status is not None
        assert status.state is OrderState.FILLED
        assert status.filled_contracts == 10
        assert status.average_price == pytest.approx(0.45)

    def test_get_order_status_cancelled(self) -> None:
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(return_value={
            "order": {
                "order_id": "ORD-3",
                "status": "canceled",
                "filled_count": 0,
                "remaining_count": 5,
            }
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("ORD-3")
        )
        assert status is not None
        assert status.state is OrderState.CANCELLED

    def test_get_order_status_partial_unknown(self) -> None:
        """Unknown status with fills → PARTIALLY_FILLED."""
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(return_value={
            "order": {
                "order_id": "ORD-4",
                "status": "some_new_status",
                "filled_count": 3,
                "remaining_count": 7,
            }
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("ORD-4")
        )
        assert status is not None
        assert status.state is OrderState.PARTIALLY_FILLED

    def test_get_order_status_api_error(self) -> None:
        adapter = self._make_adapter()
        adapter._private_request = AsyncMock(side_effect=Exception("timeout"))

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("ORD-X")
        )
        assert status is None

    def test_get_order_status_no_credentials(self) -> None:
        adapter = self._make_adapter()
        adapter._private_key = None

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("ORD-X")
        )
        assert status is None


# ---------------------------------------------------------------------------
# Polymarket cancel_order / get_order_status
# ---------------------------------------------------------------------------


class TestPolymarketCancelReplace:
    def _make_adapter(self):
        from arb_bot.exchanges.polymarket import PolymarketAdapter
        from arb_bot.config import PolymarketSettings

        settings = PolymarketSettings(
            enabled=True,
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            private_key="",
        )
        adapter = PolymarketAdapter(settings)
        adapter._live_client = MagicMock()
        return adapter

    def test_cancel_order_success(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.cancel = MagicMock(return_value={"canceled": True})

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("POLY-1")
        )
        assert result is True

    def test_cancel_order_not_canceled(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.cancel = MagicMock(
            return_value={"not_canceled": True}
        )

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("POLY-1")
        )
        assert result is False

    def test_cancel_order_exception(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.cancel = MagicMock(side_effect=Exception("fail"))

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("POLY-1")
        )
        assert result is False

    def test_cancel_order_no_client(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client = None

        result = asyncio.get_event_loop().run_until_complete(
            adapter.cancel_order("POLY-1")
        )
        assert result is False

    def test_get_order_status_live(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.get_order = MagicMock(return_value={
            "id": "POLY-1",
            "status": "live",
            "size_matched": "0",
            "original_size": "10",
            "price": "0.50",
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("POLY-1")
        )
        assert status is not None
        assert status.state is OrderState.OPEN
        assert status.filled_contracts == 0
        assert status.remaining_contracts == 10

    def test_get_order_status_matched(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.get_order = MagicMock(return_value={
            "id": "POLY-2",
            "status": "matched",
            "size_matched": "5",
            "original_size": "5",
            "price": "0.45",
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("POLY-2")
        )
        assert status is not None
        assert status.state is OrderState.FILLED
        assert status.filled_contracts == 5
        assert status.remaining_contracts == 0

    def test_get_order_status_partial_fill(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.get_order = MagicMock(return_value={
            "id": "POLY-3",
            "status": "live",
            "size_matched": "3",
            "original_size": "10",
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("POLY-3")
        )
        assert status is not None
        assert status.state is OrderState.PARTIALLY_FILLED
        assert status.filled_contracts == 3
        assert status.remaining_contracts == 7

    def test_get_order_status_cancelled(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.get_order = MagicMock(return_value={
            "id": "POLY-4",
            "status": "canceled",
            "size_matched": "2",
            "original_size": "10",
        })

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("POLY-4")
        )
        assert status is not None
        assert status.state is OrderState.CANCELLED

    def test_get_order_status_api_error(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client.get_order = MagicMock(side_effect=Exception("timeout"))

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("POLY-X")
        )
        assert status is None

    def test_get_order_status_no_client(self) -> None:
        adapter = self._make_adapter()
        adapter._live_client = None

        status = asyncio.get_event_loop().run_until_complete(
            adapter.get_order_status("POLY-X")
        )
        assert status is None
