"""Tests for Phase 9C: Engine multi-venue wiring.

Validates that the engine correctly initializes, tracks bankroll, and
aggregates quotes from all three venues (Kalshi, Polymarket, ForecastEx).
"""

from __future__ import annotations

import types
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
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
from arb_bot.engine import ArbEngine
from arb_bot.fee_model import FeeModel, FeeModelConfig, VenueFeeSchedule
from arb_bot.models import BinaryQuote, EngineState


NOW = datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc)


def _settings(
    *,
    kalshi_enabled: bool = True,
    polymarket_enabled: bool = False,
    forecastex_enabled: bool = False,
    default_bankroll: float = 1000.0,
    bankroll_by_venue: dict[str, float] | None = None,
) -> AppSettings:
    return AppSettings(
        live_mode=False,
        run_once=True,
        poll_interval_seconds=60,
        dry_run=True,
        paper_strict_simulation=True,
        paper_position_lifetime_seconds=600,
        paper_dynamic_lifetime_enabled=False,
        paper_dynamic_lifetime_resolution_fraction=0.02,
        paper_dynamic_lifetime_min_seconds=60,
        paper_dynamic_lifetime_max_seconds=600,
        stream_mode=False,
        stream_recompute_cooldown_ms=0,
        default_bankroll_usd=default_bankroll,
        bankroll_by_venue=bankroll_by_venue or {},
        log_level="INFO",
        strategy=StrategySettings(),
        lanes=OpportunityLaneSettings(),
        sizing=SizingSettings(),
        risk=RiskSettings(),
        universe=UniverseRankingSettings(),
        fill_model=FillModelSettings(),
        kalshi=KalshiSettings(enabled=kalshi_enabled),
        polymarket=PolymarketSettings(enabled=polymarket_enabled),
        forecastex=ForecastExSettings(enabled=forecastex_enabled),
        paper_rolling_settlement_enabled=False,
        paper_rolling_settlement_min_hold_seconds=120,
    )


# ===================================================================
# Engine build_exchanges
# ===================================================================


class TestBuildExchanges:
    """Tests for _build_exchanges with ForecastEx."""

    def test_only_kalshi(self) -> None:
        settings = _settings(kalshi_enabled=True)
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        exchanges = engine._build_exchanges()
        assert "kalshi" in exchanges
        assert "forecastex" not in exchanges
        assert len(exchanges) == 1

    def test_kalshi_and_forecastex(self) -> None:
        settings = _settings(kalshi_enabled=True, forecastex_enabled=True)
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        exchanges = engine._build_exchanges()
        assert "kalshi" in exchanges
        assert "forecastex" in exchanges
        assert len(exchanges) == 2

    def test_all_three_venues(self) -> None:
        settings = _settings(
            kalshi_enabled=True,
            polymarket_enabled=True,
            forecastex_enabled=True,
        )
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        exchanges = engine._build_exchanges()
        assert "kalshi" in exchanges
        assert "polymarket" in exchanges
        assert "forecastex" in exchanges
        assert len(exchanges) == 3

    def test_only_forecastex(self) -> None:
        settings = _settings(
            kalshi_enabled=False,
            polymarket_enabled=False,
            forecastex_enabled=True,
        )
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        exchanges = engine._build_exchanges()
        assert "forecastex" in exchanges
        assert len(exchanges) == 1

    def test_no_exchanges_raises(self) -> None:
        settings = _settings(
            kalshi_enabled=False,
            polymarket_enabled=False,
            forecastex_enabled=False,
        )
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        with pytest.raises(ValueError, match="No exchanges enabled"):
            engine._build_exchanges()

    def test_forecastex_adapter_has_correct_venue(self) -> None:
        settings = _settings(forecastex_enabled=True)
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        exchanges = engine._build_exchanges()
        assert exchanges["forecastex"].venue == "forecastex"


# ===================================================================
# Initial cash map with forecastex
# ===================================================================


class TestInitialCashMap:
    """Tests for bankroll initialization with 3 venues."""

    def test_default_bankroll_all_venues(self) -> None:
        settings = _settings(
            kalshi_enabled=True,
            polymarket_enabled=True,
            forecastex_enabled=True,
            default_bankroll=500.0,
        )
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        engine._exchanges = engine._build_exchanges()
        cash_map = engine._initial_cash_map()
        assert cash_map["kalshi"] == 500.0
        assert cash_map["polymarket"] == 500.0
        assert cash_map["forecastex"] == 500.0

    def test_per_venue_override(self) -> None:
        settings = _settings(
            kalshi_enabled=True,
            forecastex_enabled=True,
            default_bankroll=500.0,
            bankroll_by_venue={"forecastex": 200.0, "kalshi": 800.0},
        )
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        engine._exchanges = engine._build_exchanges()
        cash_map = engine._initial_cash_map()
        assert cash_map["kalshi"] == 800.0
        assert cash_map["forecastex"] == 200.0

    def test_forecastex_only_bankroll(self) -> None:
        settings = _settings(
            kalshi_enabled=False,
            forecastex_enabled=True,
            default_bankroll=1000.0,
        )
        engine = ArbEngine.__new__(ArbEngine)
        engine._settings = settings
        engine._exchanges = engine._build_exchanges()
        cash_map = engine._initial_cash_map()
        assert "forecastex" in cash_map
        assert cash_map["forecastex"] == 1000.0
        assert "kalshi" not in cash_map


# ===================================================================
# EngineState with forecastex
# ===================================================================


class TestEngineStateThreeVenue:
    """Tests for EngineState tracking 3 venues."""

    def test_cash_for_forecastex(self) -> None:
        state = EngineState(cash_by_venue={
            "kalshi": 500.0,
            "polymarket": 500.0,
            "forecastex": 300.0,
        })
        assert state.cash_for("forecastex") == 300.0

    def test_cash_for_missing_venue_returns_zero(self) -> None:
        state = EngineState(cash_by_venue={"kalshi": 500.0})
        assert state.cash_for("forecastex") == 0.0

    def test_locked_capital_per_venue(self) -> None:
        state = EngineState(
            cash_by_venue={"kalshi": 500.0, "forecastex": 300.0},
            locked_capital_by_venue={"kalshi": 100.0, "forecastex": 50.0},
        )
        assert state.locked_for("kalshi") == 100.0
        assert state.locked_for("forecastex") == 50.0

    def test_mark_open_market_forecastex(self) -> None:
        state = EngineState(cash_by_venue={"forecastex": 300.0})
        state.mark_open_market("forecastex", "FX-CPI-MAR26")
        assert "FX-CPI-MAR26" in state.open_markets_by_venue.get("forecastex", set())

    def test_unmark_open_market_forecastex(self) -> None:
        state = EngineState(
            cash_by_venue={"forecastex": 300.0},
            open_markets_by_venue={"forecastex": {"FX-CPI-MAR26"}},
        )
        state.unmark_open_market("forecastex", "FX-CPI-MAR26")
        assert "FX-CPI-MAR26" not in state.open_markets_by_venue.get("forecastex", set())


# ===================================================================
# Fee model with forecastex schedule
# ===================================================================


class TestFeeModelForecastEx:
    """Tests for ForecastEx fee schedule in the fee model."""

    def test_forecastex_fee_schedule(self) -> None:
        """ForecastEx charges $0.005 per contract (half of $0.01 per pair)."""
        schedule = VenueFeeSchedule(
            venue="forecastex",
            taker_fee_per_contract=0.005,
        )
        config = FeeModelConfig(venues=(schedule,))
        model = FeeModel(config)
        assert model.get_schedule("forecastex") is not None
        assert model.get_schedule("forecastex").taker_fee_per_contract == 0.005

    def test_forecastex_fee_estimate(self) -> None:
        from arb_bot.fee_model import OrderType
        schedule = VenueFeeSchedule(
            venue="forecastex",
            taker_fee_per_contract=0.005,
        )
        config = FeeModelConfig(venues=(schedule,))
        model = FeeModel(config)
        estimate = model.estimate("forecastex", OrderType.TAKER, contracts=10)
        assert abs(estimate.total_fee - 0.05) < 1e-9

    def test_all_three_venues_in_fee_model(self) -> None:
        schedules = (
            VenueFeeSchedule(venue="kalshi", taker_curve_coefficient=0.07),
            VenueFeeSchedule(venue="polymarket", taker_fee_per_contract=0.0),
            VenueFeeSchedule(venue="forecastex", taker_fee_per_contract=0.005),
        )
        config = FeeModelConfig(venues=schedules)
        model = FeeModel(config)
        assert model.get_schedule("kalshi") is not None
        assert model.get_schedule("polymarket") is not None
        assert model.get_schedule("forecastex") is not None


# ===================================================================
# Quote aggregation with 3 venues
# ===================================================================


def _make_quote(
    venue: str,
    market_id: str,
    yes_buy: float = 0.45,
    no_buy: float = 0.55,
    **metadata: Any,
) -> BinaryQuote:
    return BinaryQuote(
        venue=venue,
        market_id=market_id,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=100,
        no_buy_size=100,
        fee_per_contract=0.005 if venue == "forecastex" else 0.0,
        metadata=metadata,
    )


class TestQuoteAggregation:
    """Tests for aggregating quotes from 3 venues."""

    def test_three_venue_quotes_merge(self) -> None:
        quotes = [
            _make_quote("kalshi", "KXFED-MAR26"),
            _make_quote("polymarket", "0xfed123"),
            _make_quote("forecastex", "FX-FEDFUNDS"),
        ]
        by_venue: dict[str, list[BinaryQuote]] = {}
        for q in quotes:
            by_venue.setdefault(q.venue, []).append(q)
        assert len(by_venue) == 3
        assert len(by_venue["forecastex"]) == 1

    def test_forecastex_quote_fee(self) -> None:
        q = _make_quote("forecastex", "FX-CPI")
        assert q.fee_per_contract == 0.005


# ===================================================================
# Config env var integration
# ===================================================================


class TestConfigForecastExIntegration:
    """Tests for ForecastEx configuration in AppSettings."""

    def test_forecastex_settings_in_app_settings(self) -> None:
        settings = _settings(forecastex_enabled=True)
        assert settings.forecastex.enabled is True
        assert settings.forecastex.port == 7496

    def test_forecastex_disabled_by_default(self) -> None:
        settings = _settings()
        assert settings.forecastex.enabled is False

    def test_forecastex_fee_config(self) -> None:
        settings = _settings(forecastex_enabled=True)
        assert settings.forecastex.fee_per_contract == 0.005
        assert settings.forecastex.payout_per_contract == 1.0

    def test_forecastex_contract_type_config(self) -> None:
        settings = _settings(forecastex_enabled=True)
        assert settings.forecastex.contract_type == "forecast"

    def test_bankroll_by_venue_includes_forecastex(self) -> None:
        settings = _settings(
            forecastex_enabled=True,
            bankroll_by_venue={"forecastex": 250.0},
        )
        assert settings.bankroll_by_venue["forecastex"] == 250.0
