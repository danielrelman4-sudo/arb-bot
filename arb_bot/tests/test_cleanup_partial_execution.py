"""Tests for auto-cancel cleanup on partial multi-leg execution.

Validates that when a multi-leg trade partially fills, unfilled orders
with order IDs are cancelled via the exchange adapter.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from arb_bot.models import (
    LegExecutionResult,
    MultiLegExecutionResult,
    PlannedLegExecutionResult,
    Side,
    TradeLegPlan,
)


def _make_leg_result(
    venue: str,
    market_id: str,
    success: bool,
    order_id: str | None = None,
    contracts: int = 10,
) -> PlannedLegExecutionResult:
    return PlannedLegExecutionResult(
        leg=TradeLegPlan(
            venue=venue,
            market_id=market_id,
            side=Side.YES,
            contracts=contracts,
            limit_price=0.50,
        ),
        result=LegExecutionResult(
            success=success,
            order_id=order_id,
            requested_contracts=contracts,
            filled_contracts=contracts if success else 0,
            average_price=0.50 if success else None,
            raw={},
        ),
    )


def _make_engine_with_mock_exchanges(venues: dict[str, AsyncMock]):
    """Create a minimal object that has _cleanup_partial_execution bound."""
    from arb_bot.engine import ArbEngine

    engine = object.__new__(ArbEngine)
    engine._exchanges = venues
    return engine


class TestCleanupPartialExecution:
    """Tests for _cleanup_partial_execution method."""

    def test_cancels_unfilled_orders_with_order_id(self):
        """Failed legs with order IDs should be cancelled."""
        mock_adapter = AsyncMock()
        mock_adapter.cancel_order = AsyncMock(return_value=True)

        engine = _make_engine_with_mock_exchanges({"venue0": mock_adapter})

        execution = MultiLegExecutionResult(
            legs=(
                _make_leg_result("venue0", "m1", success=True, order_id="ord-1"),
                _make_leg_result("venue0", "m2", success=False, order_id="ord-2"),
            ),
            error="one or more legs failed",
        )

        asyncio.get_event_loop().run_until_complete(
            engine._cleanup_partial_execution(execution)
        )

        # ord-2 should be cancelled (failed but has order_id)
        mock_adapter.cancel_order.assert_called_once_with("ord-2")

    def test_skips_legs_without_order_id(self):
        """Failed legs without order IDs should not trigger cancel."""
        mock_adapter = AsyncMock()
        mock_adapter.cancel_order = AsyncMock(return_value=True)

        engine = _make_engine_with_mock_exchanges({"venue0": mock_adapter})

        execution = MultiLegExecutionResult(
            legs=(
                _make_leg_result("venue0", "m1", success=True, order_id="ord-1"),
                _make_leg_result("venue0", "m2", success=False, order_id=None),
            ),
            error="one or more legs failed",
        )

        asyncio.get_event_loop().run_until_complete(
            engine._cleanup_partial_execution(execution)
        )

        mock_adapter.cancel_order.assert_not_called()

    def test_skips_successful_legs(self):
        """Successfully filled legs should not be cancelled."""
        mock_adapter = AsyncMock()
        mock_adapter.cancel_order = AsyncMock(return_value=True)

        engine = _make_engine_with_mock_exchanges({"venue0": mock_adapter})

        execution = MultiLegExecutionResult(
            legs=(
                _make_leg_result("venue0", "m1", success=True, order_id="ord-1"),
                _make_leg_result("venue0", "m2", success=True, order_id="ord-2"),
            ),
            error=None,
        )

        asyncio.get_event_loop().run_until_complete(
            engine._cleanup_partial_execution(execution)
        )

        mock_adapter.cancel_order.assert_not_called()

    def test_cancel_exception_does_not_propagate(self):
        """Cancel failures should be logged but not raise."""
        mock_adapter = AsyncMock()
        mock_adapter.cancel_order = AsyncMock(side_effect=Exception("network error"))

        engine = _make_engine_with_mock_exchanges({"venue0": mock_adapter})

        execution = MultiLegExecutionResult(
            legs=(
                _make_leg_result("venue0", "m1", success=True, order_id="ord-1"),
                _make_leg_result("venue0", "m2", success=False, order_id="ord-2"),
            ),
            error="one or more legs failed",
        )

        # Should not raise despite cancel_order throwing
        asyncio.get_event_loop().run_until_complete(
            engine._cleanup_partial_execution(execution)
        )
        mock_adapter.cancel_order.assert_called_once_with("ord-2")

    def test_multi_venue_cancels(self):
        """Cancels should go to the correct venue adapter."""
        mock_kalshi = AsyncMock()
        mock_kalshi.cancel_order = AsyncMock(return_value=True)
        mock_poly = AsyncMock()
        mock_poly.cancel_order = AsyncMock(return_value=True)

        engine = _make_engine_with_mock_exchanges({
            "kalshi": mock_kalshi,
            "polymarket": mock_poly,
        })

        execution = MultiLegExecutionResult(
            legs=(
                _make_leg_result("kalshi", "k1", success=True, order_id="k-ord-1"),
                _make_leg_result("polymarket", "p1", success=False, order_id="p-ord-2"),
            ),
            error="one or more legs failed",
        )

        asyncio.get_event_loop().run_until_complete(
            engine._cleanup_partial_execution(execution)
        )

        mock_kalshi.cancel_order.assert_not_called()
        mock_poly.cancel_order.assert_called_once_with("p-ord-2")

    def test_unknown_venue_skipped(self):
        """If venue adapter is missing, cancel is skipped gracefully."""
        engine = _make_engine_with_mock_exchanges({})  # No adapters

        execution = MultiLegExecutionResult(
            legs=(
                _make_leg_result("unknown", "m1", success=False, order_id="ord-1"),
            ),
            error="one or more legs failed",
        )

        # Should not raise
        asyncio.get_event_loop().run_until_complete(
            engine._cleanup_partial_execution(execution)
        )
