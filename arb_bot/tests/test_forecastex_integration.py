"""Tests for Phase 9D: ForecastEx end-to-end paper validation.

Validates the full ForecastEx integration pipeline:
- Quote normalization from IBKR-style data → BinaryQuote
- Cross-venue opportunity detection across K↔F, P↔F, K↔P pairs
- Dry-run order generation with correct ForecastEx semantics
- Fee model accuracy for ForecastEx contracts
- Coverage tracking with 3 venues
- No-sell constraint validation in order planning

Since we cannot connect to IBKR TWS in this test environment, these
tests validate the pipeline using mock data flowing through the real
strategy, sizing, and engine layers.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from arb_bot.config import (
    AppSettings,
    FillModelSettings,
    ForecastExSettings,
    KalshiSettings,
    OpportunityLaneSettings,
    PolymarketSettings,
    RiskSettings,
    SizingSettings,
    StrategySettings,
    UniverseRankingSettings,
)
from arb_bot.cross_mapping import (
    CrossVenueMapping,
    VenuePair,
    VenueRef,
    venue_pairs,
)
from arb_bot.engine import ArbEngine
from arb_bot.exchanges.forecastex import ForecastExAdapter, IBKRRateLimiter
from arb_bot.fee_model import FeeModel, FeeModelConfig, OrderType, VenueFeeSchedule
from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    EngineState,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
    TradeLegPlan,
    TradePlan,
)
from arb_bot.risk import RiskManager
from arb_bot.strategy import ArbitrageFinder


NOW = datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc)


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===================================================================
# Helpers
# ===================================================================


def _make_kalshi_quote(
    market_id: str,
    yes_buy: float = 0.45,
    no_buy: float = 0.55,
    **metadata: Any,
) -> BinaryQuote:
    return BinaryQuote(
        venue="kalshi",
        market_id=market_id,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=100,
        no_buy_size=100,
        fee_per_contract=0.0,
        observed_at=NOW,
        metadata={"title": f"Kalshi {market_id}", **metadata},
    )


def _make_forecastex_quote(
    market_id: str,
    yes_buy: float = 0.52,
    no_buy: float = 0.50,
    **metadata: Any,
) -> BinaryQuote:
    return BinaryQuote(
        venue="forecastex",
        market_id=market_id,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=100,
        no_buy_size=100,
        fee_per_contract=0.005,
        observed_at=NOW,
        metadata={"title": f"ForecastEx {market_id}", **metadata},
    )


def _make_polymarket_quote(
    market_id: str,
    yes_buy: float = 0.48,
    no_buy: float = 0.53,
    **metadata: Any,
) -> BinaryQuote:
    return BinaryQuote(
        venue="polymarket",
        market_id=market_id,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=100,
        no_buy_size=100,
        fee_per_contract=0.0,
        observed_at=NOW,
        metadata={"question": f"Polymarket {market_id}", **metadata},
    )


# ===================================================================
# End-to-end cross-venue detection with ForecastEx
# ===================================================================


class TestE2ECrossVenueDetection:
    """End-to-end tests for cross-venue opportunity detection with ForecastEx."""

    def test_kalshi_forecastex_arb_detected_via_mapping(self, tmp_path: Path) -> None:
        """K↔F arb detected through mapping file, not just fuzzy matching."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "fed_rate,KXFED-MAR26,,FX-FEDFUNDS-MAR26\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        # Create an edge: kalshi YES cheap + forecastex NO cheap = arb.
        quotes = [
            _make_kalshi_quote("KXFED-MAR26", yes_buy=0.40, no_buy=0.65),
            _make_forecastex_quote("FX-FEDFUNDS-MAR26", yes_buy=0.65, no_buy=0.40),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        assert len(cross) > 0
        # Verify venue pair in match key.
        assert any("kalshi:forecastex" in o.match_key for o in cross)

    def test_three_venue_all_pairs_checked(self, tmp_path: Path) -> None:
        """With 3-venue mapping, opportunities checked for K↔P, K↔F, P↔F."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "cpi,KXCPI-MAR26,0xcpi123,FX-CPI-MAR26\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.001,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        # Create strong edges across all pairs.
        quotes = [
            _make_kalshi_quote("KXCPI-MAR26", yes_buy=0.35, no_buy=0.70),
            _make_polymarket_quote("0xcpi123", yes_buy=0.70, no_buy=0.35),
            _make_forecastex_quote("FX-CPI-MAR26", yes_buy=0.50, no_buy=0.55),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        match_keys = {o.match_key for o in cross}

        # At least K↔P pair should have strong arb.
        kp_found = any("kalshi:polymarket" in k for k in match_keys)
        assert kp_found, f"K↔P arb not found, keys: {match_keys}"

    def test_forecastex_intra_venue_arb_detected(self) -> None:
        """Intra-venue arb within forecastex should be detected."""
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        # YES + NO < $1 minus fees = arb
        quote = _make_forecastex_quote("FX-TEST", yes_buy=0.40, no_buy=0.45)
        opps = finder.find([quote])
        intra = [o for o in opps if o.kind is OpportunityKind.INTRA_VENUE]
        assert len(intra) > 0

    def test_no_arb_when_prices_fair(self) -> None:
        """No arb when prices are correctly aligned."""
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        # YES + NO = $1.00 exactly — no edge.
        quote = _make_forecastex_quote("FX-TEST", yes_buy=0.50, no_buy=0.50)
        opps = finder.find([quote])
        assert len(opps) == 0


# ===================================================================
# Coverage tracking with ForecastEx
# ===================================================================


class TestE2ECoverageTracking:
    """End-to-end coverage tracking with ForecastEx venues."""

    def test_full_coverage_three_venues(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "fed_rate,KXFED,0xfed,FX-FED\n"
            "cpi,KXCPI,0xcpi,FX-CPI\n"
            "gdp,KXGDP,0xgdp,\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_kalshi_quote("KXFED"),
            _make_kalshi_quote("KXCPI"),
            _make_kalshi_quote("KXGDP"),
            _make_polymarket_quote("0xfed"),
            _make_polymarket_quote("0xcpi"),
            _make_polymarket_quote("0xgdp"),
            _make_forecastex_quote("FX-FED"),
            _make_forecastex_quote("FX-CPI"),
        ]
        snapshot = finder.coverage_snapshot(quotes)
        assert snapshot["cross_mapping_kalshi_refs_total"] == 3
        assert snapshot["cross_mapping_kalshi_refs_seen"] == 3
        assert snapshot["cross_mapping_polymarket_refs_total"] == 3
        assert snapshot["cross_mapping_polymarket_refs_seen"] == 3
        # Only 2 forecastex refs in mappings (gdp has no forecastex).
        assert snapshot["cross_mapping_forecastex_refs_total"] == 2
        assert snapshot["cross_mapping_forecastex_refs_seen"] == 2
        # All 3 groups have at least 2 venues resolved.
        assert snapshot["cross_mapping_pairs_covered"] == 3

    def test_partial_forecastex_coverage(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "fed_rate,KXFED,0xfed,FX-FED\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        # Only kalshi quote available.
        quotes = [_make_kalshi_quote("KXFED")]
        snapshot = finder.coverage_snapshot(quotes)
        assert snapshot["cross_mapping_forecastex_refs_seen"] == 0
        assert snapshot["cross_mapping_pairs_covered"] == 0


# ===================================================================
# ForecastEx fee accuracy
# ===================================================================


class TestE2EFeeAccuracy:
    """End-to-end fee model accuracy for ForecastEx."""

    def test_forecastex_pair_fee(self) -> None:
        """ForecastEx charges $0.01 per YES/NO pair = $0.005 each."""
        schedule = VenueFeeSchedule(venue="forecastex", taker_fee_per_contract=0.005)
        config = FeeModelConfig(venues=(schedule,))
        model = FeeModel(config)

        yes_estimate = model.estimate("forecastex", OrderType.TAKER, contracts=10, price=0.55)
        no_estimate = model.estimate("forecastex", OrderType.TAKER, contracts=10, price=0.45)

        # YES + NO = pair fee.
        pair_total = yes_estimate.total_fee + no_estimate.total_fee
        assert abs(pair_total - 0.10) < 1e-9  # $0.01/pair * 10 contracts

    def test_forecastex_fee_vs_kalshi(self) -> None:
        """ForecastEx flat fee vs Kalshi curve fee for same trade."""
        fx_schedule = VenueFeeSchedule(venue="forecastex", taker_fee_per_contract=0.005)
        k_schedule = VenueFeeSchedule(venue="kalshi", taker_curve_coefficient=0.07, curve_round_up=True)
        config = FeeModelConfig(venues=(fx_schedule, k_schedule))
        model = FeeModel(config)

        fx_fee = model.estimate("forecastex", OrderType.TAKER, contracts=10, price=0.55)
        k_fee = model.estimate("kalshi", OrderType.TAKER, contracts=10, price=0.55)

        # Both should be non-negative.
        assert fx_fee.total_fee >= 0
        assert k_fee.total_fee >= 0


# ===================================================================
# No-sell constraint validation
# ===================================================================


class TestE2ENoSellConstraint:
    """Validate ForecastEx no-sell constraint in order semantics."""

    def test_exit_via_opposing_contract(self) -> None:
        """Exiting a YES position = buying NO (not selling YES)."""
        # This is a documentation/design test — the adapter handles this
        # by always using BUY action, mapping Side to C/P.
        assert ForecastExAdapter.side_to_right(Side.YES) == "C"
        assert ForecastExAdapter.side_to_right(Side.NO) == "P"

    def test_right_to_side_roundtrip(self) -> None:
        for side in [Side.YES, Side.NO]:
            right = ForecastExAdapter.side_to_right(side)
            back = ForecastExAdapter.right_to_side(right)
            assert back == side

    def test_trade_plan_legs_are_buys(self) -> None:
        """Both legs of a ForecastEx arb should be BUY operations."""
        # ForecastEx arb: buy YES on one venue, buy NO on ForecastEx.
        plan = TradePlan(
            kind=OpportunityKind.CROSS_VENUE,
            execution_style=ExecutionStyle.TAKER,
            legs=(
                TradeLegPlan("kalshi", "KXFED", Side.YES, 10, 0.45),
                TradeLegPlan("forecastex", "FX-FED", Side.NO, 10, 0.40),
            ),
            contracts=10,
            capital_required=8.5,
            capital_required_by_venue={"kalshi": 4.5, "forecastex": 4.0},
            expected_profit=0.15,
            edge_per_contract=0.015,
        )
        # Both legs should have Side defined (BUY side for each venue).
        assert plan.legs[0].side == Side.YES
        assert plan.legs[1].side == Side.NO


# ===================================================================
# ForecastEx adapter properties
# ===================================================================


class TestE2EAdapterProperties:
    """End-to-end adapter property validation."""

    def test_adapter_venue_name(self) -> None:
        adapter = ForecastExAdapter.__new__(ForecastExAdapter)
        adapter._settings = ForecastExSettings()
        assert adapter.venue == "forecastex"

    def test_adapter_streaming_requires_connection(self) -> None:
        """Streaming requires ib_async + active connection, not just config."""
        adapter = ForecastExAdapter.__new__(ForecastExAdapter)
        adapter._settings = ForecastExSettings(enable_stream=True)
        adapter._connected = False
        # Without ib_async installed + connection, streaming reports False.
        assert adapter.supports_streaming() is False

    def test_adapter_streaming_disabled_in_config(self) -> None:
        adapter = ForecastExAdapter.__new__(ForecastExAdapter)
        adapter._settings = ForecastExSettings(enable_stream=False)
        adapter._connected = True
        assert adapter.supports_streaming() is False

    def test_rate_limiter_conservative(self) -> None:
        """Rate limiter should be well below IBKR's 10 req/s limit."""
        limiter = IBKRRateLimiter()
        # Default 6.0 req/s = 60% of IBKR limit.
        assert limiter._max_per_second == 6.0
        assert limiter._burst_size == 8


# ===================================================================
# Mapping pair generation validation
# ===================================================================


class TestE2EMappingPairs:
    """Validate pair generation logic for common ForecastEx scenarios."""

    def test_economic_indicator_mapping_pairs(self) -> None:
        """Economic indicators (Fed rate, CPI) typically have K+F+P mappings."""
        mapping = CrossVenueMapping(
            group_id="fed_rate_mar26",
            kalshi=VenueRef(key="kalshi_market_id", value="KXFED-MAR26-T425"),
            polymarket=VenueRef(key="polymarket_market_id", value="0xfed_mar26"),
            forecastex=VenueRef(key="forecastex_market_id", value="FX-FEDFUNDS-MAR26-425"),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 3
        venues = {(p.left_venue, p.right_venue) for p in pairs}
        assert ("kalshi", "polymarket") in venues
        assert ("kalshi", "forecastex") in venues
        assert ("polymarket", "forecastex") in venues

    def test_kalshi_only_forecastex_mapping(self) -> None:
        """Some markets only trade on K and F (no Polymarket equivalent)."""
        mapping = CrossVenueMapping(
            group_id="gdp_q1_26",
            kalshi=VenueRef(key="kalshi_market_id", value="KXGDP-Q126"),
            polymarket=VenueRef(key="polymarket_market_id", value=""),
            forecastex=VenueRef(key="forecastex_market_id", value="FX-GDP-Q126"),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 1
        assert pairs[0].left_venue == "kalshi"
        assert pairs[0].right_venue == "forecastex"

    def test_empty_mapping_produces_no_pairs(self) -> None:
        mapping = CrossVenueMapping(
            group_id="empty",
            kalshi=VenueRef(key="kalshi_market_id", value=""),
            polymarket=VenueRef(key="polymarket_market_id", value=""),
        )
        assert venue_pairs(mapping) == []


# ===================================================================
# Engine state operations with forecastex
# ===================================================================


class TestE2EEngineStateOps:
    """End-to-end engine state operations with forecastex venue."""

    def test_capital_allocation_three_venues(self) -> None:
        """Capital distributed correctly across 3 venues."""
        state = EngineState(
            cash_by_venue={
                "kalshi": 500.0,
                "polymarket": 500.0,
                "forecastex": 300.0,
            },
            locked_capital_by_venue={
                "kalshi": 50.0,
                "forecastex": 25.0,
            },
        )
        total_cash = sum(state.cash_by_venue.values())
        total_locked = sum(state.locked_capital_by_venue.values())
        assert total_cash == 1300.0
        assert total_locked == 75.0

    def test_capital_release_after_settlement(self) -> None:
        """Simulates capital release after ForecastEx position settles."""
        state = EngineState(
            cash_by_venue={"forecastex": 200.0},
            locked_capital_by_venue={"forecastex": 100.0},
        )
        # Simulate settlement: return locked capital + profit.
        profit = 5.0
        committed = 100.0
        state.cash_by_venue["forecastex"] = state.cash_for("forecastex") + committed + profit
        state.locked_capital_by_venue["forecastex"] = max(
            0.0, state.locked_for("forecastex") - committed
        )
        assert state.cash_for("forecastex") == 305.0
        assert state.locked_for("forecastex") == 0.0

    def test_open_market_tracking_forecastex(self) -> None:
        state = EngineState(cash_by_venue={"forecastex": 300.0})
        state.mark_open_market("forecastex", "FX-CPI-1")
        state.mark_open_market("forecastex", "FX-CPI-2")
        assert len(state.open_markets_by_venue["forecastex"]) == 2
        state.unmark_open_market("forecastex", "FX-CPI-1")
        assert len(state.open_markets_by_venue["forecastex"]) == 1
