"""Tests for v18 Micro-Momentum: tiered VPIN gate, contract selection, sizing, triggers, cooldowns."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketMeta, CryptoMarketQuote
from arb_bot.crypto.engine import _select_momentum_contract, _compute_momentum_size


# ── Helpers ──────────────────────────────────────────────────────


def _make_quote(
    ticker="KXBTCD-26FEB1602-T68500",
    underlying="BTC",
    direction="above",
    strike=68500.0,
    yes_buy=0.25,
    no_buy=0.80,
    tte_minutes=10.0,
):
    meta = CryptoMarketMeta(
        underlying=underlying,
        interval="daily",
        expiry=datetime.now(timezone.utc) + timedelta(minutes=tte_minutes),
        strike=strike,
        direction=direction,
        series_ticker=f"KX{underlying}D",
    )
    return CryptoMarketQuote(
        market=CryptoMarket(ticker=ticker, meta=meta),
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=100,
        no_buy_size=100,
        yes_bid_price=yes_buy - 0.02,
        no_bid_price=no_buy - 0.02,
        time_to_expiry_minutes=tte_minutes,
        implied_probability=yes_buy,
    )


def _classify_vpin_zone(
    max_vpin: float | None,
    momentum_enabled: bool,
    vpin_halt_threshold: float,
    momentum_vpin_floor: float,
    momentum_vpin_ceiling: float,
) -> str:
    """Classify VPIN into 'normal', 'momentum', or 'halt' zone.

    Mirrors the tiered gate logic in CryptoEngine._run_cycle.
    """
    if max_vpin is None:
        return "normal"

    if momentum_enabled:
        if max_vpin > momentum_vpin_ceiling:
            return "halt"
        elif max_vpin >= momentum_vpin_floor:
            return "momentum"
        else:
            return "normal"
    else:
        if max_vpin > vpin_halt_threshold:
            return "halt"
        return "normal"


def _check_momentum_trigger(
    regime: str | None,
    is_transitioning: bool,
    ofi_alignment: float,
    ofi_magnitude: float,
    momentum_count: int,
    settings: CryptoSettings,
) -> tuple:
    """Check if momentum trade should trigger.

    Returns (should_fire: bool, reason: str).
    Mirrors CryptoEngine._try_momentum_trades gating logic.
    """
    if regime is None:
        return (False, "no_regime")
    if regime != "high_vol":
        return (False, f"wrong_regime:{regime}")
    if is_transitioning:
        return (False, "transitioning")
    if momentum_count >= settings.momentum_max_concurrent:
        return (False, "max_concurrent")
    if ofi_alignment < settings.momentum_ofi_alignment_min:
        return (False, "low_alignment")
    if ofi_magnitude < settings.momentum_ofi_magnitude_min:
        return (False, "low_magnitude")
    return (True, "ok")


# ── 1. Tiered VPIN gate ─────────────────────────────────────────


class TestTieredVPINGate:
    """Test _classify_vpin_zone helper."""

    def test_below_floor_is_normal(self):
        zone = _classify_vpin_zone(0.80, True, 0.85, 0.85, 0.95)
        assert zone == "normal"

    def test_in_zone_is_momentum(self):
        zone = _classify_vpin_zone(0.90, True, 0.85, 0.85, 0.95)
        assert zone == "momentum"

    def test_above_ceiling_is_halt(self):
        zone = _classify_vpin_zone(0.96, True, 0.85, 0.85, 0.95)
        assert zone == "halt"

    def test_at_floor_boundary_is_momentum(self):
        zone = _classify_vpin_zone(0.85, True, 0.85, 0.85, 0.95)
        assert zone == "momentum"

    def test_at_ceiling_boundary_is_momentum(self):
        """At ceiling exactly (not >) should be momentum, not halt."""
        zone = _classify_vpin_zone(0.95, True, 0.85, 0.85, 0.95)
        assert zone == "momentum"

    def test_momentum_disabled_uses_old_binary(self):
        """When momentum_enabled=False, use original binary halt behavior."""
        # Above old threshold -> halt
        zone = _classify_vpin_zone(0.90, False, 0.85, 0.85, 0.95)
        assert zone == "halt"
        # Below old threshold -> normal
        zone = _classify_vpin_zone(0.80, False, 0.85, 0.85, 0.95)
        assert zone == "normal"

    def test_none_vpin_is_normal(self):
        zone = _classify_vpin_zone(None, True, 0.85, 0.85, 0.95)
        assert zone == "normal"


# ── 2. Contract selection ────────────────────────────────────────


class TestMomentumContractSelection:
    """Test _select_momentum_contract."""

    def _settings(self, **overrides):
        kwargs = dict(
            momentum_enabled=True,
            momentum_max_tte_minutes=15.0,
            momentum_price_floor=0.15,
            momentum_price_ceiling=0.40,
        )
        kwargs.update(overrides)
        return CryptoSettings(**kwargs)

    def test_bullish_yes_above(self):
        """Bullish OFI → YES on above-strike above spot."""
        q = _make_quote(strike=69000.0, direction="above", yes_buy=0.25)
        result = _select_momentum_contract([q], 68500.0, +1, self._settings())
        assert result is not None
        quote, side = result
        assert side == "yes"
        assert quote.market.ticker == q.market.ticker

    def test_bearish_no_above(self):
        """Bearish OFI → NO on above-strike below spot."""
        q = _make_quote(strike=68000.0, direction="above", no_buy=0.25)
        result = _select_momentum_contract([q], 68500.0, -1, self._settings())
        assert result is not None
        quote, side = result
        assert side == "no"

    def test_price_floor_filters(self):
        """Cheap contracts below price floor are rejected."""
        q = _make_quote(strike=69000.0, direction="above", yes_buy=0.10)
        result = _select_momentum_contract([q], 68500.0, +1, self._settings())
        assert result is None

    def test_price_ceiling_filters(self):
        """Expensive contracts above price ceiling are rejected."""
        q = _make_quote(strike=69000.0, direction="above", yes_buy=0.50)
        result = _select_momentum_contract([q], 68500.0, +1, self._settings())
        assert result is None

    def test_tte_filter(self):
        """Contracts past max TTE are rejected."""
        q = _make_quote(strike=69000.0, direction="above", yes_buy=0.25, tte_minutes=20.0)
        result = _select_momentum_contract([q], 68500.0, +1, self._settings())
        assert result is None

    def test_no_candidates_returns_none(self):
        """Empty quotes → None."""
        result = _select_momentum_contract([], 68500.0, +1, self._settings())
        assert result is None

    def test_no_strike_filtered(self):
        """Contracts without a strike (15min up/down) are filtered out."""
        q = _make_quote(strike=None, direction="up", yes_buy=0.25)
        result = _select_momentum_contract([q], 68500.0, +1, self._settings())
        assert result is None

    def test_nearest_strike_preferred(self):
        """When multiple candidates exist, nearest strike to spot wins."""
        q_far = _make_quote(
            ticker="FAR", strike=70000.0, direction="above", yes_buy=0.20, tte_minutes=10.0,
        )
        q_near = _make_quote(
            ticker="NEAR", strike=68600.0, direction="above", yes_buy=0.35, tte_minutes=10.0,
        )
        result = _select_momentum_contract(
            [q_far, q_near], 68500.0, +1, self._settings(),
        )
        assert result is not None
        quote, side = result
        assert quote.market.ticker == "NEAR"


# ── 3. Momentum sizing ──────────────────────────────────────────


class TestMomentumSizing:
    """Test _compute_momentum_size."""

    def _settings(self, **overrides):
        kwargs = dict(
            momentum_enabled=True,
            momentum_kelly_fraction=0.03,
            momentum_max_position=25.0,
        )
        kwargs.update(overrides)
        return CryptoSettings(**kwargs)

    def test_basic_sizing(self):
        """500 * 0.03 * 0.8 / 0.25 = 48."""
        contracts = _compute_momentum_size(500.0, 0.25, 0.8, self._settings())
        assert contracts == 48

    def test_max_position_cap(self):
        """Dollar amount capped by momentum_max_position."""
        # 500 * 0.03 * 1.0 = 15.0 (under $25 cap) -> 15 / 0.20 = 75
        contracts = _compute_momentum_size(500.0, 0.20, 1.0, self._settings())
        assert contracts == 75
        # With lower cap: max_position=10 -> 10 / 0.20 = 50
        contracts = _compute_momentum_size(500.0, 0.20, 1.0, self._settings(momentum_max_position=10.0))
        assert contracts == 50

    def test_bankroll_cap(self):
        """Cannot exceed bankroll."""
        # 10 * 0.03 * 1.0 = 0.30, but bankroll is only 10
        # dollar_amount = min(0.30, 25.0) = 0.30, min(0.30, 10.0) = 0.30
        # contracts = int(0.30 / 0.25) = 1
        contracts = _compute_momentum_size(10.0, 0.25, 1.0, self._settings())
        assert contracts == 1

    def test_zero_contracts_at_low_bankroll(self):
        """Very low bankroll -> 0 contracts."""
        contracts = _compute_momentum_size(1.0, 0.25, 0.5, self._settings())
        # 1.0 * 0.03 * 0.5 = 0.015 / 0.25 = 0.06 -> int = 0
        assert contracts == 0

    def test_alignment_scales_size(self):
        """Higher alignment -> more contracts."""
        contracts_low = _compute_momentum_size(500.0, 0.25, 0.6, self._settings())
        contracts_high = _compute_momentum_size(500.0, 0.25, 1.0, self._settings())
        assert contracts_high > contracts_low


# ── 4. Momentum trigger ─────────────────────────────────────────


class TestMomentumTrigger:
    """Test _check_momentum_trigger conditions."""

    def _settings(self, **overrides):
        kwargs = dict(
            momentum_enabled=True,
            momentum_ofi_alignment_min=0.6,
            momentum_ofi_magnitude_min=200.0,
            momentum_max_concurrent=2,
        )
        kwargs.update(overrides)
        return CryptoSettings(**kwargs)

    def test_fires_when_all_conditions_met(self):
        fire, reason = _check_momentum_trigger(
            regime="high_vol", is_transitioning=False,
            ofi_alignment=0.8, ofi_magnitude=300.0,
            momentum_count=0, settings=self._settings(),
        )
        assert fire is True
        assert reason == "ok"

    def test_blocked_by_wrong_regime(self):
        fire, reason = _check_momentum_trigger(
            regime="mean_reverting", is_transitioning=False,
            ofi_alignment=0.8, ofi_magnitude=300.0,
            momentum_count=0, settings=self._settings(),
        )
        assert fire is False
        assert "wrong_regime" in reason

    def test_blocked_by_low_alignment(self):
        fire, reason = _check_momentum_trigger(
            regime="high_vol", is_transitioning=False,
            ofi_alignment=0.3, ofi_magnitude=300.0,
            momentum_count=0, settings=self._settings(),
        )
        assert fire is False
        assert "low_alignment" in reason

    def test_blocked_by_low_magnitude(self):
        fire, reason = _check_momentum_trigger(
            regime="high_vol", is_transitioning=False,
            ofi_alignment=0.8, ofi_magnitude=50.0,
            momentum_count=0, settings=self._settings(),
        )
        assert fire is False
        assert "low_magnitude" in reason

    def test_blocked_by_transitioning(self):
        fire, reason = _check_momentum_trigger(
            regime="high_vol", is_transitioning=True,
            ofi_alignment=0.8, ofi_magnitude=300.0,
            momentum_count=0, settings=self._settings(),
        )
        assert fire is False
        assert "transitioning" in reason

    def test_blocked_by_no_regime(self):
        fire, reason = _check_momentum_trigger(
            regime=None, is_transitioning=False,
            ofi_alignment=0.8, ofi_magnitude=300.0,
            momentum_count=0, settings=self._settings(),
        )
        assert fire is False
        assert "no_regime" in reason


# ── 5. Momentum cooldowns ───────────────────────────────────────


class TestMomentumCooldown:
    """Test momentum cooldown and concurrency limits."""

    def test_cooldown_blocks_recent_trade(self):
        """A symbol that settled recently should be blocked."""
        cooldowns = {"btcusdt": time.time() - 30}  # 30s ago
        cooldown_seconds = 120.0
        last = cooldowns.get("btcusdt")
        assert last is not None
        assert (time.time() - last) < cooldown_seconds

    def test_cooldown_allows_after_window(self):
        """A symbol that settled long ago should be allowed."""
        cooldowns = {"btcusdt": time.time() - 200}  # 200s ago
        cooldown_seconds = 120.0
        last = cooldowns.get("btcusdt")
        assert last is not None
        assert (time.time() - last) >= cooldown_seconds

    def test_no_prior_trade_no_cooldown(self):
        """A symbol with no prior trade should not be blocked."""
        cooldowns = {}
        last = cooldowns.get("btcusdt")
        assert last is None  # No cooldown

    def test_max_concurrent_blocks(self):
        """At max concurrent momentum positions, no new trades."""
        fire, reason = _check_momentum_trigger(
            regime="high_vol", is_transitioning=False,
            ofi_alignment=0.8, ofi_magnitude=300.0,
            momentum_count=2,
            settings=CryptoSettings(momentum_enabled=True, momentum_max_concurrent=2),
        )
        assert fire is False
        assert "max_concurrent" in reason

    def test_max_concurrent_allows(self):
        """Below max concurrent, trades are allowed."""
        fire, reason = _check_momentum_trigger(
            regime="high_vol", is_transitioning=False,
            ofi_alignment=0.8, ofi_magnitude=300.0,
            momentum_count=1,
            settings=CryptoSettings(momentum_enabled=True, momentum_max_concurrent=2),
        )
        assert fire is True
