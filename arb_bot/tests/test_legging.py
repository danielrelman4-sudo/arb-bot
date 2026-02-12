"""Tests for Phase 0B: Legging risk controls (sequential leg execution)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch

import pytest

from arb_bot.config import RiskSettings
from arb_bot.models import (
    BinaryQuote,
    ExecutionStyle,
    LegExecutionResult,
    MultiLegExecutionResult,
    OpportunityKind,
    PairExecutionResult,
    Side,
    TradeLegPlan,
    TradePlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_plan(
    limit_a: float = 0.45,
    limit_b: float = 0.50,
) -> TradePlan:
    return TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="K1",
                side=Side.YES,
                contracts=5,
                limit_price=limit_a,
            ),
            TradeLegPlan(
                venue="polymarket",
                market_id="P1",
                side=Side.NO,
                contracts=5,
                limit_price=limit_b,
            ),
        ),
        contracts=5,
        capital_required=4.75,
        capital_required_by_venue={"kalshi": 2.25, "polymarket": 2.50},
        expected_profit=0.25,
        edge_per_contract=0.05,
    )


def _intra_plan() -> TradePlan:
    return TradePlan(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.YES,
                contracts=10,
                limit_price=0.40,
            ),
            TradeLegPlan(
                venue="kalshi",
                market_id="M2",
                side=Side.NO,
                contracts=10,
                limit_price=0.55,
            ),
        ),
        contracts=10,
        capital_required=9.5,
        capital_required_by_venue={"kalshi": 9.5},
        expected_profit=0.50,
        edge_per_contract=0.05,
    )


def _ok_result(order_id: str = "ORD-1", contracts: int = 5, price: float = 0.45) -> LegExecutionResult:
    return LegExecutionResult(
        success=True,
        order_id=order_id,
        requested_contracts=contracts,
        filled_contracts=contracts,
        average_price=price,
        raw={},
    )


def _fail_result(error: str = "timeout") -> LegExecutionResult:
    return LegExecutionResult(
        success=False,
        order_id=None,
        requested_contracts=5,
        filled_contracts=0,
        average_price=None,
        raw={"error": error},
    )


def _quote(venue: str, market_id: str, yes_price: float = 0.45, no_price: float = 0.55) -> BinaryQuote:
    return BinaryQuote(
        venue=venue,
        market_id=market_id,
        yes_buy_price=yes_price,
        no_buy_price=no_price,
        yes_buy_size=100.0,
        no_buy_size=100.0,
    )


class FakeAdapter:
    """Minimal fake ExchangeAdapter for testing sequential execution."""

    def __init__(self, venue: str) -> None:
        self.venue = venue
        self._order_results: list[LegExecutionResult] = []
        self._quote_results: list[BinaryQuote | None] = []
        self._order_call_count = 0
        self._quote_call_count = 0
        self._order_delay: float = 0.0

    def set_order_results(self, *results: LegExecutionResult) -> None:
        self._order_results = list(results)

    def set_quote_results(self, *results: BinaryQuote | None) -> None:
        self._quote_results = list(results)

    async def place_single_order(self, leg: TradeLegPlan) -> LegExecutionResult:
        if self._order_delay > 0:
            await asyncio.sleep(self._order_delay)
        idx = min(self._order_call_count, len(self._order_results) - 1)
        self._order_call_count += 1
        return self._order_results[idx]

    async def place_pair_order(self, plan: TradePlan) -> PairExecutionResult:
        raise NotImplementedError

    async def fetch_quotes(self) -> list[BinaryQuote]:
        raise NotImplementedError

    async def fetch_quote(self, market_id: str) -> BinaryQuote | None:
        idx = min(self._quote_call_count, len(self._quote_results) - 1)
        self._quote_call_count += 1
        return self._quote_results[idx]

    async def get_available_cash(self) -> float | None:
        return None

    async def aclose(self) -> None:
        pass


class FakeEngine:
    """Minimal shim that holds the fields the execution methods need.

    We bind all relevant ArbEngine methods at init so internal calls like
    ``self._check_leg_quote_drift`` resolve correctly.
    """

    def __init__(
        self,
        exchanges: dict[str, FakeAdapter],
        risk_settings: RiskSettings | None = None,
    ) -> None:
        self._exchanges = exchanges
        self._settings = _FakeSettings(risk_settings or RiskSettings())

        # Bind ArbEngine methods onto this instance so self-dispatch works.
        import types
        from arb_bot.engine import ArbEngine

        for name in (
            "_execute_live_plan",
            "_execute_sequential_legs",
            "_execute_parallel_legs",
            "_check_leg_quote_drift",
        ):
            method = getattr(ArbEngine, name)
            setattr(self, name, types.MethodType(method, self))

    async def execute_sequential(self, plan: TradePlan) -> MultiLegExecutionResult:
        return await self._execute_sequential_legs(plan)  # type: ignore[attr-defined]

    async def execute_parallel(self, plan: TradePlan) -> MultiLegExecutionResult:
        return await self._execute_parallel_legs(plan)  # type: ignore[attr-defined]

    async def check_drift(self, leg: TradeLegPlan, tolerance: float) -> bool:
        return await self._check_leg_quote_drift(leg, tolerance)  # type: ignore[attr-defined]


@dataclass
class _FakeSettings:
    risk: RiskSettings


# ---------------------------------------------------------------------------
# Tests: _check_leg_quote_drift
# ---------------------------------------------------------------------------


class TestCheckLegQuoteDrift:
    @pytest.fixture()
    def engine(self) -> FakeEngine:
        kalshi = FakeAdapter("kalshi")
        poly = FakeAdapter("polymarket")
        return FakeEngine({"kalshi": kalshi, "polymarket": poly})

    def test_within_tolerance(self, engine: FakeEngine) -> None:
        leg = TradeLegPlan(venue="kalshi", market_id="K1", side=Side.YES, contracts=5, limit_price=0.45)
        engine._exchanges["kalshi"].set_quote_results(_quote("kalshi", "K1", yes_price=0.46))
        result = asyncio.get_event_loop().run_until_complete(engine.check_drift(leg, 0.03))
        assert result is True

    def test_exceeds_tolerance(self, engine: FakeEngine) -> None:
        leg = TradeLegPlan(venue="kalshi", market_id="K1", side=Side.YES, contracts=5, limit_price=0.45)
        engine._exchanges["kalshi"].set_quote_results(_quote("kalshi", "K1", yes_price=0.50))
        result = asyncio.get_event_loop().run_until_complete(engine.check_drift(leg, 0.03))
        assert result is False

    def test_no_side_uses_correct_price(self, engine: FakeEngine) -> None:
        leg = TradeLegPlan(venue="polymarket", market_id="P1", side=Side.NO, contracts=5, limit_price=0.50)
        engine._exchanges["polymarket"].set_quote_results(_quote("polymarket", "P1", no_price=0.52))
        result = asyncio.get_event_loop().run_until_complete(engine.check_drift(leg, 0.03))
        assert result is True

    def test_no_side_exceeds_tolerance(self, engine: FakeEngine) -> None:
        leg = TradeLegPlan(venue="polymarket", market_id="P1", side=Side.NO, contracts=5, limit_price=0.50)
        engine._exchanges["polymarket"].set_quote_results(_quote("polymarket", "P1", no_price=0.55))
        result = asyncio.get_event_loop().run_until_complete(engine.check_drift(leg, 0.03))
        assert result is False

    def test_missing_quote_returns_false(self, engine: FakeEngine) -> None:
        leg = TradeLegPlan(venue="kalshi", market_id="K1", side=Side.YES, contracts=5, limit_price=0.45)
        engine._exchanges["kalshi"].set_quote_results(None)
        result = asyncio.get_event_loop().run_until_complete(engine.check_drift(leg, 0.03))
        assert result is False

    def test_missing_adapter_returns_false(self, engine: FakeEngine) -> None:
        leg = TradeLegPlan(venue="unknown", market_id="X1", side=Side.YES, contracts=5, limit_price=0.45)
        result = asyncio.get_event_loop().run_until_complete(engine.check_drift(leg, 0.03))
        assert result is False


# ---------------------------------------------------------------------------
# Tests: _execute_sequential_legs
# ---------------------------------------------------------------------------


class TestSequentialLegs:
    def _engine(self, risk: RiskSettings | None = None) -> tuple[FakeEngine, FakeAdapter, FakeAdapter]:
        kalshi = FakeAdapter("kalshi")
        poly = FakeAdapter("polymarket")
        eng = FakeEngine({"kalshi": kalshi, "polymarket": poly}, risk)
        return eng, kalshi, poly

    def test_both_legs_succeed(self) -> None:
        eng, kalshi, poly = self._engine()
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))
        poly.set_quote_results(_quote("polymarket", "P1", no_price=0.50))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert result.success
        assert result.error is None
        assert len(result.legs) == 2
        assert result.legs[0].result.success
        assert result.legs[1].result.success
        assert result.filled_contracts == 5

    def test_first_leg_fails_skips_second(self) -> None:
        eng, kalshi, poly = self._engine()
        kalshi.set_order_results(_fail_result("rejected"))
        poly.set_order_results(_ok_result("B", 5, 0.50))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert not result.success
        assert len(result.legs) == 2
        assert not result.legs[0].result.success
        assert not result.legs[1].result.success
        assert result.legs[1].result.raw["error"] == "prior leg failed"
        # Poly adapter should never have been called.
        assert poly._order_call_count == 0

    def test_second_leg_fails_after_first_fills(self) -> None:
        eng, kalshi, poly = self._engine()
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_fail_result("insufficient funds"))
        poly.set_quote_results(_quote("polymarket", "P1", no_price=0.50))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert not result.success
        assert len(result.legs) == 2
        assert result.legs[0].result.success
        assert not result.legs[1].result.success

    def test_quote_drift_aborts_second_leg(self) -> None:
        eng, kalshi, poly = self._engine()
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))
        # Price moved to 0.60 — way beyond the 0.03 default tolerance from limit of 0.50.
        poly.set_quote_results(_quote("polymarket", "P1", no_price=0.60))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert not result.success
        assert "drift" in (result.error or "").lower()
        assert result.legs[0].result.success
        assert not result.legs[1].result.success
        assert "drift" in result.legs[1].result.raw.get("error", "")
        # Poly order should NOT have been placed.
        assert poly._order_call_count == 0

    def test_time_window_exceeded_aborts_second_leg(self) -> None:
        # Use a very short time window so the test runs fast.
        risk = RiskSettings(leg_max_time_window_seconds=0.0)
        eng, kalshi, poly = self._engine(risk)
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))
        poly.set_quote_results(_quote("polymarket", "P1", no_price=0.50))
        # Small delay so perf_counter advances past 0.
        kalshi._order_delay = 0.01

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert not result.success
        assert "time window" in (result.error or "").lower() or "aborted" in (result.error or "").lower()
        assert result.legs[0].result.success
        assert not result.legs[1].result.success
        assert poly._order_call_count == 0

    def test_quote_drift_within_tolerance_proceeds(self) -> None:
        eng, kalshi, poly = self._engine()
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))
        # Price moved slightly — within 0.03 tolerance.
        poly.set_quote_results(_quote("polymarket", "P1", no_price=0.52))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert result.success
        assert result.error is None
        assert poly._order_call_count == 1

    def test_exception_in_first_leg_skips_second(self) -> None:
        eng, kalshi, poly = self._engine()

        async def _raise(_leg: TradeLegPlan) -> LegExecutionResult:
            raise ConnectionError("network error")

        kalshi.place_single_order = _raise  # type: ignore[assignment]
        poly.set_order_results(_ok_result("B", 5, 0.50))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_sequential(plan))

        assert not result.success
        assert len(result.legs) == 2
        assert not result.legs[0].result.success
        assert "network error" in result.legs[0].result.raw.get("error", "")
        assert not result.legs[1].result.success
        assert poly._order_call_count == 0


# ---------------------------------------------------------------------------
# Tests: _execute_parallel_legs (existing behavior preserved)
# ---------------------------------------------------------------------------


class TestParallelLegs:
    def test_both_legs_succeed(self) -> None:
        kalshi = FakeAdapter("kalshi")
        poly = FakeAdapter("polymarket")
        eng = FakeEngine({"kalshi": kalshi, "polymarket": poly})
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_parallel(plan))

        assert result.success
        assert len(result.legs) == 2

    def test_one_leg_fails(self) -> None:
        kalshi = FakeAdapter("kalshi")
        poly = FakeAdapter("polymarket")
        eng = FakeEngine({"kalshi": kalshi, "polymarket": poly})
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_fail_result("timeout"))

        plan = _cross_plan()
        result = asyncio.get_event_loop().run_until_complete(eng.execute_parallel(plan))

        assert not result.success
        assert result.legs[0].result.success
        assert not result.legs[1].result.success


# ---------------------------------------------------------------------------
# Tests: routing logic (sequential vs parallel)
# ---------------------------------------------------------------------------


class TestExecutionRouting:
    """Verify _execute_live_plan routes correctly based on settings and plan shape."""

    def _engine(self, risk: RiskSettings | None = None) -> tuple[FakeEngine, FakeAdapter, FakeAdapter]:
        kalshi = FakeAdapter("kalshi")
        poly = FakeAdapter("polymarket")
        eng = FakeEngine({"kalshi": kalshi, "polymarket": poly}, risk)
        return eng, kalshi, poly

    def test_cross_venue_uses_sequential_by_default(self) -> None:
        eng, kalshi, poly = self._engine()
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))
        poly.set_quote_results(_quote("polymarket", "P1", no_price=0.50))

        plan = _cross_plan()

        from arb_bot.engine import ArbEngine
        result = asyncio.get_event_loop().run_until_complete(
            ArbEngine._execute_live_plan(eng, plan)  # type: ignore[arg-type]
        )

        assert result.success
        # Sequential: kalshi called first, then poly — poly should have had
        # a quote drift check (quote_call_count >= 1).
        assert poly._quote_call_count >= 1

    def test_sequential_disabled_uses_parallel(self) -> None:
        risk = RiskSettings(sequential_legs=False)
        eng, kalshi, poly = self._engine(risk)
        kalshi.set_order_results(_ok_result("A", 5, 0.45))
        poly.set_order_results(_ok_result("B", 5, 0.50))

        plan = _cross_plan()

        from arb_bot.engine import ArbEngine
        result = asyncio.get_event_loop().run_until_complete(
            ArbEngine._execute_live_plan(eng, plan)  # type: ignore[arg-type]
        )

        assert result.success
        # Parallel: no quote drift check.
        assert poly._quote_call_count == 0

    def test_same_venue_multi_leg_uses_parallel(self) -> None:
        """Even with sequential_legs=True, single-venue plans use parallel."""
        eng, kalshi, _ = self._engine()
        kalshi.set_order_results(_ok_result("A", 10, 0.40), _ok_result("B", 10, 0.55))

        plan = _intra_plan()

        from arb_bot.engine import ArbEngine
        result = asyncio.get_event_loop().run_until_complete(
            ArbEngine._execute_live_plan(eng, plan)  # type: ignore[arg-type]
        )

        assert result.success
        # No quote checks — parallel execution.
        assert kalshi._quote_call_count == 0


# ---------------------------------------------------------------------------
# Tests: config defaults
# ---------------------------------------------------------------------------


class TestLeggingConfig:
    def test_default_sequential_legs_enabled(self) -> None:
        r = RiskSettings()
        assert r.sequential_legs is True

    def test_default_drift_tolerance(self) -> None:
        r = RiskSettings()
        assert r.leg_quote_drift_tolerance == 0.03

    def test_default_max_time_window(self) -> None:
        r = RiskSettings()
        assert r.leg_max_time_window_seconds == 10.0

    def test_custom_values(self) -> None:
        r = RiskSettings(
            sequential_legs=False,
            leg_quote_drift_tolerance=0.05,
            leg_max_time_window_seconds=5.0,
        )
        assert r.sequential_legs is False
        assert r.leg_quote_drift_tolerance == 0.05
        assert r.leg_max_time_window_seconds == 5.0
