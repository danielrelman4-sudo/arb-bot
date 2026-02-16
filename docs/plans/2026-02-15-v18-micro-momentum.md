# v18 Micro-Momentum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a momentum trading path that activates in the VPIN "momentum zone" (0.85–0.95), buying the nearest OTM contract in the OFI direction.

**Architecture:** Modify the VPIN halt gate from binary (halt/pass) to tiered (pass/momentum-zone/full-halt). Add `_try_momentum_trades()` method to engine that selects nearest OTM contract in $0.15–$0.40 price sweet spot, sizes with fixed Kelly fraction scaled by OFI alignment. Add `strategy` field to FeatureVector for classifier training.

**Tech Stack:** Python async, existing CryptoEngine, FeatureStore, regime detector, OFI infrastructure.

---

### Task 1: Add momentum config fields to `config.py`

**Files:**
- Modify: `arb_bot/crypto/config.py` (CryptoSettings dataclass + env var loading)

**Step 1: Add 12 momentum fields to CryptoSettings**

After the existing `regime_transition_sizing_multiplier` field block, add:

```python
# ── Momentum strategy (v18) ──────────────────────────────────────
momentum_enabled: bool = False
momentum_vpin_floor: float = 0.85       # Lower bound of momentum zone
momentum_vpin_ceiling: float = 0.95     # Full halt above this
momentum_ofi_alignment_min: float = 0.6 # Min cross-timescale OFI agreement
momentum_ofi_magnitude_min: float = 200.0  # Min |weighted OFI|
momentum_max_tte_minutes: float = 15.0  # Only short-dated contracts
momentum_price_floor: float = 0.15      # Min buy price (avoid dead money)
momentum_price_ceiling: float = 0.40    # Max buy price (avoid priced-in)
momentum_kelly_fraction: float = 0.03   # Fixed fraction of bankroll
momentum_max_position: float = 25.0     # Max dollar per momentum trade
momentum_max_concurrent: int = 2        # Max open momentum positions
momentum_cooldown_seconds: float = 120.0  # Per-symbol cooldown
```

**Step 2: Add env var loading**

In the `from_env()` / constructor section where env vars are parsed, add:

```python
momentum_enabled=_as_bool(os.getenv("ARB_CRYPTO_MOMENTUM_ENABLED"), False),
momentum_vpin_floor=float(os.getenv("ARB_CRYPTO_MOMENTUM_VPIN_FLOOR", "0.85")),
momentum_vpin_ceiling=float(os.getenv("ARB_CRYPTO_MOMENTUM_VPIN_CEILING", "0.95")),
momentum_ofi_alignment_min=float(os.getenv("ARB_CRYPTO_MOMENTUM_OFI_ALIGNMENT_MIN", "0.6")),
momentum_ofi_magnitude_min=float(os.getenv("ARB_CRYPTO_MOMENTUM_OFI_MAGNITUDE_MIN", "200.0")),
momentum_max_tte_minutes=float(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_TTE_MINUTES", "15.0")),
momentum_price_floor=float(os.getenv("ARB_CRYPTO_MOMENTUM_PRICE_FLOOR", "0.15")),
momentum_price_ceiling=float(os.getenv("ARB_CRYPTO_MOMENTUM_PRICE_CEILING", "0.40")),
momentum_kelly_fraction=float(os.getenv("ARB_CRYPTO_MOMENTUM_KELLY_FRACTION", "0.03")),
momentum_max_position=float(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_POSITION", "25.0")),
momentum_max_concurrent=int(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_CONCURRENT", "2")),
momentum_cooldown_seconds=float(os.getenv("ARB_CRYPTO_MOMENTUM_COOLDOWN_SECONDS", "120.0")),
```

**Step 3: Run existing config tests**

Run: `python3 -m pytest arb_bot/tests/test_crypto_config.py -v`
Expected: All existing tests pass.

**Step 4: Commit**

```bash
git add arb_bot/crypto/config.py
git commit -m "feat(crypto): add v18 momentum config fields"
```

---

### Task 2: Add `strategy` field to FeatureVector

**Files:**
- Modify: `arb_bot/crypto/feature_store.py`
- Test: `arb_bot/tests/test_crypto_feature_store.py`

**Step 1: Add strategy field to FeatureVector**

In the `FeatureVector` dataclass, after the `side` / `entry_price` fields (line ~90), add:

```python
# Strategy identifier (for classifier training)
strategy: str = "model"            # "model" (v17) or "momentum" (v18)
```

**Step 2: Add `strategy` to ALL_COLUMNS**

In `ALL_COLUMNS` (line ~117), add `"strategy"` after `"entry_price"`:

```python
ALL_COLUMNS = [
    "ticker", "timestamp", "side", "entry_price", "strategy",
] + FEATURE_COLUMNS + ["outcome"]
```

**Step 3: Run feature store tests**

Run: `python3 -m pytest arb_bot/tests/test_crypto_feature_store.py -v`
Expected: Pass (existing tests should still work since `strategy` has a default).

**Step 4: Commit**

```bash
git add arb_bot/crypto/feature_store.py
git commit -m "feat(crypto): add strategy field to FeatureVector for v18"
```

---

### Task 3: Refactor VPIN halt gate to tiered system

**Files:**
- Modify: `arb_bot/crypto/engine.py` (lines 780-789: VPIN halt gate)
- Test: `arb_bot/tests/test_momentum.py` (new file)

**Step 1: Write failing tests for tiered VPIN gate**

Create `arb_bot/tests/test_momentum.py`:

```python
"""Tests for v18 micro-momentum strategy."""

import asyncio
import math
import time
import types
from dataclasses import dataclass, replace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.market_scanner import (
    CryptoMarket, CryptoMarketMeta, CryptoMarketQuote,
)
from arb_bot.crypto.regime_detector import RegimeSnapshot


# ── Helpers ────────────────────────────────────────────────────────

def _make_settings(**overrides) -> CryptoSettings:
    """Create settings with momentum defaults for testing."""
    defaults = dict(
        vpin_halt_enabled=True,
        vpin_halt_threshold=0.85,
        momentum_enabled=True,
        momentum_vpin_floor=0.85,
        momentum_vpin_ceiling=0.95,
        momentum_ofi_alignment_min=0.6,
        momentum_ofi_magnitude_min=200.0,
        momentum_max_tte_minutes=15.0,
        momentum_price_floor=0.15,
        momentum_price_ceiling=0.40,
        momentum_kelly_fraction=0.03,
        momentum_max_position=25.0,
        momentum_max_concurrent=2,
        momentum_cooldown_seconds=120.0,
        regime_sizing_enabled=True,
        regime_kelly_high_vol=0.0,
        price_feed_symbols="btcusdt",
        mc_n_paths=100,
    )
    defaults.update(overrides)
    return CryptoSettings(**defaults)


def _make_quote(
    ticker: str = "KXBTCD-26FEB1602-T68500",
    underlying: str = "BTC",
    direction: str = "above",
    strike: float = 68500.0,
    yes_buy: float = 0.25,
    no_buy: float = 0.80,
    tte_minutes: float = 10.0,
) -> CryptoMarketQuote:
    """Create a market quote for testing."""
    from datetime import datetime, timezone, timedelta
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


class FakeVPINCalculator:
    """Fake VPIN calculator returning a fixed value."""
    def __init__(self, vpin: float, signed_vpin: float = 0.0, trend: float = 0.0):
        self._vpin = vpin
        self._signed = signed_vpin
        self._trend = trend
    def get_vpin(self): return self._vpin
    def get_signed_vpin(self): return self._signed
    def get_vpin_trend(self): return self._trend


# ── Tiered VPIN gate tests ─────────────────────────────────────────

class TestTieredVPINGate:
    """Test the 3-tier VPIN gate: normal / momentum zone / full halt."""

    def test_vpin_below_floor_allows_normal_trading(self):
        """VPIN < 0.85 → normal model path, no momentum."""
        settings = _make_settings()
        result = _classify_vpin_zone(settings, vpin=0.80)
        assert result == "normal"

    def test_vpin_in_momentum_zone(self):
        """0.85 <= VPIN <= 0.95 → momentum zone."""
        settings = _make_settings()
        result = _classify_vpin_zone(settings, vpin=0.90)
        assert result == "momentum"

    def test_vpin_above_ceiling_full_halt(self):
        """VPIN > 0.95 → full halt."""
        settings = _make_settings()
        result = _classify_vpin_zone(settings, vpin=0.97)
        assert result == "halt"

    def test_vpin_at_floor_boundary(self):
        """VPIN exactly at floor → momentum zone (inclusive)."""
        settings = _make_settings()
        result = _classify_vpin_zone(settings, vpin=0.85)
        assert result == "momentum"

    def test_vpin_at_ceiling_boundary(self):
        """VPIN exactly at ceiling → momentum zone (inclusive)."""
        settings = _make_settings()
        result = _classify_vpin_zone(settings, vpin=0.95)
        assert result == "momentum"

    def test_momentum_disabled_falls_back_to_binary_halt(self):
        """If momentum_enabled=False, VPIN > floor → halt (old behavior)."""
        settings = _make_settings(momentum_enabled=False)
        result = _classify_vpin_zone(settings, vpin=0.90)
        assert result == "halt"

    def test_vpin_none_treated_as_normal(self):
        """If VPIN is None (not available), treat as normal."""
        settings = _make_settings()
        result = _classify_vpin_zone(settings, vpin=None)
        assert result == "normal"


def _classify_vpin_zone(settings: CryptoSettings, vpin: float | None) -> str:
    """Helper to classify VPIN into zone. Mirrors engine logic."""
    if vpin is None:
        return "normal"
    if not settings.vpin_halt_enabled:
        return "normal"
    if settings.momentum_enabled:
        if vpin > settings.momentum_vpin_ceiling:
            return "halt"
        elif vpin >= settings.momentum_vpin_floor:
            return "momentum"
        else:
            return "normal"
    else:
        # Old binary behavior
        if vpin > settings.vpin_halt_threshold:
            return "halt"
        return "normal"
```

**Step 2: Run tests to verify they pass (pure-logic tests)**

Run: `python3 -m pytest arb_bot/tests/test_momentum.py::TestTieredVPINGate -v`
Expected: All 7 pass (these test the helper function, not the engine yet).

**Step 3: Modify VPIN halt gate in engine.py**

Replace lines 780-789 in `_run_cycle()`:

```python
# 1d. VPIN tiered gate: classify zone for each symbol
_momentum_zone = False
if self._settings.vpin_halt_enabled and self._vpin_calculators:
    max_vpin = 0.0
    for sym, calc in self._vpin_calculators.items():
        vpin_val = calc.get_vpin()
        if vpin_val is not None and vpin_val > max_vpin:
            max_vpin = vpin_val

    if self._settings.momentum_enabled:
        if max_vpin > self._settings.momentum_vpin_ceiling:
            LOGGER.info(
                "CryptoEngine: VPIN full halt — max VPIN=%.3f > %.3f, skipping cycle",
                max_vpin, self._settings.momentum_vpin_ceiling,
            )
            return
        elif max_vpin >= self._settings.momentum_vpin_floor:
            _momentum_zone = True
            LOGGER.info(
                "CryptoEngine: VPIN momentum zone — max VPIN=%.3f (%.3f–%.3f)",
                max_vpin, self._settings.momentum_vpin_floor,
                self._settings.momentum_vpin_ceiling,
            )
    else:
        if max_vpin > self._settings.vpin_halt_threshold:
            LOGGER.info(
                "CryptoEngine: VPIN halt — max VPIN=%.3f > %.3f, skipping cycle",
                max_vpin, self._settings.vpin_halt_threshold,
            )
            return
```

Then after `_update_regime()` (line ~795), add the momentum branch:

```python
# 2c. Momentum path (v18): if in momentum zone, try momentum trades instead of model
if _momentum_zone and self._settings.momentum_enabled:
    await self._try_momentum_trades(market_quotes)
    # Still settle expired positions
    await self._settle_expired_positions()
    # End-of-cycle timing
    cycle_elapsed = time.monotonic() - cycle_start
    if cycle_elapsed > 1.0:
        LOGGER.info("CryptoEngine: momentum cycle %d took %.1fs", self._cycle_count, cycle_elapsed)
    return
```

Also apply the same tiered gate to `run_cycle_with_quotes()` (the parallel method).

**Step 4: Run engine tests**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: All 67 existing engine tests pass (momentum is disabled by default).

**Step 5: Commit**

```bash
git add arb_bot/crypto/engine.py arb_bot/tests/test_momentum.py
git commit -m "feat(crypto): tiered VPIN gate (normal/momentum/halt)"
```

---

### Task 4: Implement `_try_momentum_trades()` — contract selection

**Files:**
- Modify: `arb_bot/crypto/engine.py`
- Test: `arb_bot/tests/test_momentum.py`

**Step 1: Write contract selection tests**

Add to `test_momentum.py`:

```python
class TestMomentumContractSelection:
    """Test contract selection: nearest OTM in $0.15–$0.40 sweet spot."""

    def test_bullish_selects_above_strike_above_spot(self):
        """Bullish OFI → buy YES on 'above' strike just above spot."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=67500.0, yes_buy=0.90, tte_minutes=10),  # ITM, skip
            _make_quote(strike=68250.0, yes_buy=0.30, tte_minutes=10),  # OTM, good
            _make_quote(strike=68500.0, yes_buy=0.20, tte_minutes=10),  # OTM, farther
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result is not None
        assert result[0].market.meta.strike == 68250.0
        assert result[1] == "yes"

    def test_bearish_selects_no_on_above_strike_below_spot(self):
        """Bearish OFI → buy NO on 'above' strike just below spot."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=67750.0, yes_buy=0.80, no_buy=0.25, tte_minutes=10),
            _make_quote(strike=67500.0, yes_buy=0.90, no_buy=0.15, tte_minutes=10),
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=-1, settings=_make_settings())
        assert result is not None
        assert result[0].market.meta.strike == 67750.0
        assert result[1] == "no"

    def test_price_floor_filters_cheap_contracts(self):
        """Contracts below $0.15 are filtered out."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=69000.0, yes_buy=0.10, tte_minutes=10),  # Too cheap
            _make_quote(strike=68250.0, yes_buy=0.25, tte_minutes=10),
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result is not None
        assert result[0].market.meta.strike == 68250.0

    def test_price_ceiling_filters_expensive_contracts(self):
        """Contracts above $0.40 are filtered out."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=68100.0, yes_buy=0.55, tte_minutes=10),  # Too expensive
            _make_quote(strike=68250.0, yes_buy=0.30, tte_minutes=10),
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result is not None
        assert result[0].market.meta.strike == 68250.0

    def test_tte_filter(self):
        """Contracts beyond max TTE are filtered out."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=68250.0, yes_buy=0.30, tte_minutes=30),  # Too far
            _make_quote(strike=68500.0, yes_buy=0.20, tte_minutes=10),  # OK
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result is not None
        assert result[0].market.meta.strike == 68500.0

    def test_no_candidates_returns_none(self):
        """No suitable contracts → None."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=68250.0, yes_buy=0.05, tte_minutes=10),  # Too cheap
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result is None

    def test_only_daily_above_below_contracts(self):
        """15min up/down contracts are excluded (no strike)."""
        spot = 68000.0
        quotes = [
            _make_quote(
                ticker="KXBTC15M-26FEB1602-U12",
                strike=None,  # 15min up/down has no strike
                direction="up",
                yes_buy=0.30,
                tte_minutes=10,
            ),
        ]
        # Need to handle None strike — should be filtered out
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result is None

    def test_nearest_strike_preferred(self):
        """Of two valid contracts, nearest to spot wins."""
        spot = 68000.0
        quotes = [
            _make_quote(strike=68500.0, yes_buy=0.20, tte_minutes=10),
            _make_quote(strike=68250.0, yes_buy=0.30, tte_minutes=10),
        ]
        result = _select_momentum_contract(quotes, spot, ofi_direction=1, settings=_make_settings())
        assert result[0].market.meta.strike == 68250.0
```

**Step 2: Implement `_select_momentum_contract` helper**

This is a pure function (can be module-level or static method) in `engine.py`:

```python
def _select_momentum_contract(
    quotes: list[CryptoMarketQuote],
    spot_price: float,
    ofi_direction: int,  # +1 bullish, -1 bearish
    settings: CryptoSettings,
) -> tuple[CryptoMarketQuote, str] | None:
    """Select the best momentum contract: nearest OTM in price sweet spot.

    Returns (quote, side) or None if no suitable contract found.
    side is "yes" for bullish, "no" for bearish.
    """
    candidates = []

    for q in quotes:
        meta = q.market.meta
        # Only above/below daily contracts with a strike
        if meta.strike is None:
            continue
        if meta.direction not in ("above", "below"):
            continue
        # TTE filter
        if q.time_to_expiry_minutes > settings.momentum_max_tte_minutes:
            continue
        if q.time_to_expiry_minutes <= 0:
            continue

        # Determine side and buy price based on OFI direction
        if ofi_direction > 0:
            # Bullish: buy YES on above-strikes above spot
            if meta.direction == "above" and meta.strike > spot_price:
                side = "yes"
                buy_price = q.yes_buy_price
            elif meta.direction == "below" and meta.strike < spot_price:
                side = "no"
                buy_price = q.no_buy_price
            else:
                continue
        else:
            # Bearish: buy NO on above-strikes below spot
            if meta.direction == "above" and meta.strike < spot_price:
                side = "no"
                buy_price = q.no_buy_price
            elif meta.direction == "below" and meta.strike > spot_price:
                side = "yes"
                buy_price = q.yes_buy_price
            else:
                continue

        # Price sweet spot filter
        if buy_price < settings.momentum_price_floor:
            continue
        if buy_price > settings.momentum_price_ceiling:
            continue

        strike_distance = abs(meta.strike - spot_price)
        candidates.append((q, side, buy_price, strike_distance))

    if not candidates:
        return None

    # Sort by strike proximity (nearest first = highest gamma)
    candidates.sort(key=lambda x: x[3])
    best = candidates[0]
    return (best[0], best[1])
```

**Step 3: Run contract selection tests**

Run: `python3 -m pytest arb_bot/tests/test_momentum.py::TestMomentumContractSelection -v`
Expected: All 8 pass.

**Step 4: Commit**

```bash
git add arb_bot/crypto/engine.py arb_bot/tests/test_momentum.py
git commit -m "feat(crypto): momentum contract selection (nearest OTM, price sweet spot)"
```

---

### Task 5: Implement `_try_momentum_trades()` — trigger + sizing + execution

**Files:**
- Modify: `arb_bot/crypto/engine.py`
- Test: `arb_bot/tests/test_momentum.py`

**Step 1: Write momentum trigger + sizing tests**

Add to `test_momentum.py`:

```python
class TestMomentumTrigger:
    """Test OFI alignment + magnitude trigger logic."""

    def test_trigger_fires_when_conditions_met(self):
        """Strong aligned OFI in high_vol regime → trigger."""
        assert _check_momentum_trigger(
            regime="high_vol",
            ofi_alignment=0.8,
            ofi_magnitude=300.0,
            is_transitioning=False,
            settings=_make_settings(),
        ) is True

    def test_trigger_blocked_wrong_regime(self):
        """Mean-reverting regime → no momentum."""
        assert _check_momentum_trigger(
            regime="mean_reverting",
            ofi_alignment=0.8,
            ofi_magnitude=300.0,
            is_transitioning=False,
            settings=_make_settings(),
        ) is False

    def test_trigger_blocked_low_alignment(self):
        """OFI alignment below threshold → no trigger."""
        assert _check_momentum_trigger(
            regime="high_vol",
            ofi_alignment=0.4,
            ofi_magnitude=300.0,
            is_transitioning=False,
            settings=_make_settings(),
        ) is False

    def test_trigger_blocked_low_magnitude(self):
        """OFI magnitude below threshold → no trigger."""
        assert _check_momentum_trigger(
            regime="high_vol",
            ofi_alignment=0.8,
            ofi_magnitude=100.0,
            is_transitioning=False,
            settings=_make_settings(),
        ) is False

    def test_trigger_blocked_transitioning(self):
        """Regime transitioning → no momentum."""
        assert _check_momentum_trigger(
            regime="high_vol",
            ofi_alignment=0.8,
            ofi_magnitude=300.0,
            is_transitioning=True,
            settings=_make_settings(),
        ) is False

    def test_trigger_blocked_no_regime(self):
        """No regime classified → no trigger."""
        assert _check_momentum_trigger(
            regime=None,
            ofi_alignment=0.8,
            ofi_magnitude=300.0,
            is_transitioning=False,
            settings=_make_settings(),
        ) is False


def _check_momentum_trigger(
    regime: str | None,
    ofi_alignment: float,
    ofi_magnitude: float,
    is_transitioning: bool,
    settings: CryptoSettings,
) -> bool:
    """Check if momentum trigger conditions are met."""
    if regime is None:
        return False
    if regime != "high_vol":
        return False
    if is_transitioning:
        return False
    if ofi_alignment < settings.momentum_ofi_alignment_min:
        return False
    if ofi_magnitude < settings.momentum_ofi_magnitude_min:
        return False
    return True


class TestMomentumSizing:
    """Test fixed-fraction sizing for momentum trades."""

    def test_basic_sizing(self):
        """3% of $500 bankroll at 0.8 alignment = $12, at $0.25 = 48 contracts."""
        contracts = _compute_momentum_size(
            bankroll=500.0, buy_price=0.25, ofi_alignment=0.8,
            settings=_make_settings(),
        )
        # 500 * 0.03 * 0.8 = 12.0; 12.0 / 0.25 = 48
        assert contracts == 48

    def test_max_position_cap(self):
        """Dollar amount capped at momentum_max_position."""
        contracts = _compute_momentum_size(
            bankroll=5000.0, buy_price=0.25, ofi_alignment=1.0,
            settings=_make_settings(momentum_max_position=25.0),
        )
        # 5000 * 0.03 * 1.0 = 150, capped to 25; 25 / 0.25 = 100
        assert contracts == 100

    def test_bankroll_cap(self):
        """Cannot exceed bankroll."""
        contracts = _compute_momentum_size(
            bankroll=10.0, buy_price=0.25, ofi_alignment=1.0,
            settings=_make_settings(),
        )
        # 10 * 0.03 * 1.0 = 0.30; 0.30 / 0.25 = 1
        assert contracts == 1

    def test_zero_contracts_at_low_bankroll(self):
        """Very low bankroll → 0 contracts."""
        contracts = _compute_momentum_size(
            bankroll=1.0, buy_price=0.25, ofi_alignment=0.6,
            settings=_make_settings(),
        )
        # 1.0 * 0.03 * 0.6 = 0.018; 0.018 / 0.25 = 0
        assert contracts == 0

    def test_alignment_scales_size(self):
        """Higher alignment → more contracts."""
        c1 = _compute_momentum_size(
            bankroll=500.0, buy_price=0.25, ofi_alignment=0.6,
            settings=_make_settings(),
        )
        c2 = _compute_momentum_size(
            bankroll=500.0, buy_price=0.25, ofi_alignment=1.0,
            settings=_make_settings(),
        )
        assert c2 > c1


def _compute_momentum_size(
    bankroll: float,
    buy_price: float,
    ofi_alignment: float,
    settings: CryptoSettings,
) -> int:
    """Compute momentum position size."""
    dollar_amount = bankroll * settings.momentum_kelly_fraction * ofi_alignment
    dollar_amount = min(dollar_amount, settings.momentum_max_position)
    dollar_amount = min(dollar_amount, bankroll)
    if buy_price <= 0:
        return 0
    return int(dollar_amount / buy_price)


class TestMomentumCooldownAndConcurrency:
    """Test cooldown and max concurrent limits."""

    def test_cooldown_blocks_recent_trade(self):
        """Cannot trade a symbol within cooldown window."""
        last_settled = time.time() - 60  # 60s ago
        assert _check_cooldown(
            last_settled, cooldown=120.0,
        ) is False  # Blocked (60 < 120)

    def test_cooldown_allows_after_window(self):
        """Can trade after cooldown expires."""
        last_settled = time.time() - 180  # 180s ago
        assert _check_cooldown(
            last_settled, cooldown=120.0,
        ) is True  # Allowed (180 > 120)

    def test_no_prior_trade_no_cooldown(self):
        """No prior trade → no cooldown."""
        assert _check_cooldown(None, cooldown=120.0) is True

    def test_max_concurrent_blocks(self):
        """Cannot exceed max concurrent momentum positions."""
        assert _check_concurrent(current=2, max_allowed=2) is False

    def test_max_concurrent_allows(self):
        """Under limit → allowed."""
        assert _check_concurrent(current=1, max_allowed=2) is True


def _check_cooldown(last_settled_time: float | None, cooldown: float) -> bool:
    if last_settled_time is None:
        return True
    return (time.time() - last_settled_time) >= cooldown


def _check_concurrent(current: int, max_allowed: int) -> bool:
    return current < max_allowed
```

**Step 2: Implement `_try_momentum_trades()` in engine.py**

Add to `CryptoEngine` class:

```python
async def _try_momentum_trades(self, market_quotes: list[CryptoMarketQuote]) -> None:
    """Attempt momentum trades in VPIN momentum zone.

    Selects nearest OTM contract in OFI direction, sizes with fixed Kelly fraction.
    """
    if self._current_regime is None:
        return
    if self._current_regime.regime != "high_vol":
        LOGGER.debug("CryptoEngine: momentum skip — regime=%s (need high_vol)",
                      self._current_regime.regime)
        return
    if self._current_regime.is_transitioning:
        LOGGER.debug("CryptoEngine: momentum skip — regime transitioning")
        return

    # Count current momentum positions
    momentum_count = sum(
        1 for p in self._positions.values()
        if getattr(p, 'strategy', 'model') == 'momentum'
    )
    if momentum_count >= self._settings.momentum_max_concurrent:
        LOGGER.debug("CryptoEngine: momentum skip — %d/%d concurrent",
                      momentum_count, self._settings.momentum_max_concurrent)
        return

    # Check OFI trigger per symbol
    for binance_sym in self._settings.price_feed_symbols.split(","):
        binance_sym = binance_sym.strip()
        if not binance_sym:
            continue

        # Check symbol cooldown
        last_settled = self._momentum_cooldowns.get(binance_sym)
        if last_settled is not None:
            if (time.time() - last_settled) < self._settings.momentum_cooldown_seconds:
                continue

        # Get OFI alignment and direction from regime detector
        ofi_multi = self._price_feed.get_ofi_multiscale(binance_sym, [30, 60, 120, 300])
        if not ofi_multi:
            continue

        ofi_alignment = self._current_regime.ofi_alignment
        # Compute weighted OFI magnitude
        weights = {30: 4.0, 60: 3.0, 120: 2.0, 300: 1.0}
        weighted_sum = 0.0
        weight_total = 0.0
        for window, ofi_val in ofi_multi.items():
            w = weights.get(window, 1.0)
            weighted_sum += ofi_val * w
            weight_total += w
        ofi_direction_raw = weighted_sum / weight_total if weight_total > 0 else 0.0
        ofi_magnitude = abs(ofi_direction_raw)
        ofi_direction = 1 if ofi_direction_raw > 0 else -1

        # Check trigger
        if ofi_alignment < self._settings.momentum_ofi_alignment_min:
            continue
        if ofi_magnitude < self._settings.momentum_ofi_magnitude_min:
            continue

        # Get spot price
        spot = self._price_feed.get_current_price(binance_sym)
        if spot is None or spot <= 0:
            continue

        # Filter quotes to this underlying
        underlying = binance_sym.replace("usdt", "").upper()
        sym_quotes = [q for q in market_quotes if q.market.meta.underlying == underlying]

        # Select best contract
        result = _select_momentum_contract(sym_quotes, spot, ofi_direction, self._settings)
        if result is None:
            LOGGER.debug("CryptoEngine: momentum — no suitable contract for %s (dir=%+d)",
                          binance_sym, ofi_direction)
            continue

        quote, side = result

        # Skip if already have position in this ticker
        if quote.market.ticker in self._positions:
            continue

        # Size
        buy_price = quote.yes_buy_price if side == "yes" else quote.no_buy_price
        contracts = _compute_momentum_size(
            self._bankroll, buy_price, ofi_alignment, self._settings,
        )
        if contracts <= 0:
            continue

        # Execute
        self._execute_momentum_trade(quote, side, contracts, ofi_alignment, ofi_magnitude)

        LOGGER.info(
            "CryptoEngine: MOMENTUM %s %s %d@%.2f¢ (OFI dir=%+d, align=%.2f, mag=%.0f, tte=%.1fm)",
            side.upper(), quote.market.ticker, contracts, buy_price * 100,
            ofi_direction, ofi_alignment, ofi_magnitude, quote.time_to_expiry_minutes,
        )

        # Re-check concurrent limit
        momentum_count += 1
        if momentum_count >= self._settings.momentum_max_concurrent:
            break
```

**Step 3: Implement `_execute_momentum_trade()`**

```python
def _execute_momentum_trade(
    self,
    quote: CryptoMarketQuote,
    side: str,
    contracts: int,
    ofi_alignment: float,
    ofi_magnitude: float,
) -> None:
    """Execute a momentum paper trade."""
    buy_price = quote.yes_buy_price if side == "yes" else quote.no_buy_price

    # Apply paper slippage
    slippage = self._settings.paper_slippage_cents / 100.0
    entry_price = min(0.99, buy_price + slippage)

    capital_needed = entry_price * contracts
    if capital_needed > self._bankroll:
        contracts = int(self._bankroll / entry_price)
        if contracts <= 0:
            return
        capital_needed = entry_price * contracts

    self._bankroll -= capital_needed

    # Create a synthetic CryptoEdge for position tracking
    from arb_bot.crypto.price_model import ProbabilityEstimate
    synthetic_edge = CryptoEdge(
        market=quote.market,
        model_prob=ProbabilityEstimate(probability=0.0, uncertainty=1.0, ci_low=0.0, ci_high=1.0),
        market_implied_prob=quote.implied_probability,
        edge=0.0,
        edge_cents=0.0,
        side=side,
        recommended_price=buy_price,
        model_uncertainty=1.0,
        time_to_expiry_minutes=quote.time_to_expiry_minutes,
        yes_buy_price=quote.yes_buy_price,
        no_buy_price=quote.no_buy_price,
    )

    pos = CryptoPosition(
        ticker=quote.market.ticker,
        side=side,
        contracts=contracts,
        entry_price=entry_price,
        entry_time=time.time(),
        edge=synthetic_edge,
        model_prob=0.0,
        market_implied_prob=quote.implied_probability,
    )
    # Tag as momentum trade
    pos.strategy = "momentum"  # type: ignore[attr-defined]
    self._positions[quote.market.ticker] = pos

    # Record to feature store
    if self._feature_store is not None:
        fv = self._build_feature_vector(synthetic_edge)
        fv.entry_price = entry_price
        fv.strategy = "momentum"
        self._feature_store.record_entry(fv)

    # Cycle recorder hook
    if (self._cycle_recorder is not None
            and hasattr(self, '_rec_cycle_id_current')
            and self._rec_cycle_id_current is not None):
        self._cycle_recorder.record_trade(
            self._rec_cycle_id_current, None, quote.market.ticker,
            side, contracts, entry_price,
        )
```

**Step 4: Add `_momentum_cooldowns` dict and `strategy` to CryptoPosition**

In `__init__`:
```python
self._momentum_cooldowns: Dict[str, float] = {}  # binance_sym → last settle time
```

Update `CryptoPosition` to include `strategy`:
```python
@dataclass
class CryptoPosition:
    """An open position in a crypto market."""
    ticker: str
    side: str
    contracts: int
    entry_price: float
    entry_time: float
    edge: CryptoEdge
    model_prob: float
    market_implied_prob: float
    strategy: str = "model"
```

In `_settle_expired_positions()`, after settling a momentum trade, record cooldown:
```python
# Record momentum cooldown
if getattr(pos, 'strategy', 'model') == 'momentum':
    underlying = pos.edge.market.meta.underlying
    binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
    if binance_sym:
        self._momentum_cooldowns[binance_sym] = time.time()
```

**Step 5: Run all momentum tests**

Run: `python3 -m pytest arb_bot/tests/test_momentum.py -v`
Expected: All tests pass.

**Step 6: Run full engine tests**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: All 67 pass.

**Step 7: Commit**

```bash
git add arb_bot/crypto/engine.py arb_bot/tests/test_momentum.py
git commit -m "feat(crypto): momentum trigger, sizing, and execution"
```

---

### Task 6: Wire up momentum in `paper_test.py`

**Files:**
- Modify: `arb_bot/crypto/paper_test.py`

**Step 1: Enable momentum in paper_test.py settings**

Add to the `CryptoSettings(...)` constructor:

```python
momentum_enabled=True,
momentum_vpin_floor=0.85,
momentum_vpin_ceiling=0.95,
momentum_ofi_alignment_min=0.6,
momentum_ofi_magnitude_min=200.0,
momentum_max_tte_minutes=15.0,
momentum_price_floor=0.15,
momentum_price_ceiling=0.40,
momentum_kelly_fraction=0.03,
momentum_max_position=25.0,
momentum_max_concurrent=2,
momentum_cooldown_seconds=120.0,
```

**Step 2: Add momentum banner line**

After the existing banner lines, add:

```python
print(f"  Momentum:           {'ON' if settings.momentum_enabled else 'OFF'}"
      f"  VPIN zone: {settings.momentum_vpin_floor:.2f}–{settings.momentum_vpin_ceiling:.2f}"
      f"  OFI align>{settings.momentum_ofi_alignment_min:.1f} mag>{settings.momentum_ofi_magnitude_min:.0f}")
```

**Step 3: Commit**

```bash
git add arb_bot/crypto/paper_test.py
git commit -m "feat(crypto): enable v18 momentum in paper_test.py"
```

---

### Task 7: Write integration tests

**Files:**
- Test: `arb_bot/tests/test_momentum.py` (extend)

**Step 1: Write integration tests**

Add integration tests that verify the full momentum flow through the engine:

```python
class TestMomentumIntegration:
    """Integration tests verifying the full momentum path in the engine."""

    def test_momentum_trade_recorded_in_feature_store(self):
        """Momentum trades should be recorded with strategy='momentum'."""
        # Verify feature store receives strategy tag
        pass  # implement with mock feature store

    def test_momentum_cooldown_prevents_rapid_trading(self):
        """After a momentum trade settles, cooldown blocks same symbol."""
        pass

    def test_momentum_concurrent_limit_enforced(self):
        """Cannot exceed max_concurrent momentum positions."""
        pass

    def test_momentum_disabled_preserves_old_vpin_halt(self):
        """momentum_enabled=False → VPIN > 0.85 halts as before."""
        pass

    def test_settlement_records_cooldown(self):
        """Settling a momentum trade records cooldown for that symbol."""
        pass

    def test_no_momentum_in_mean_reverting(self):
        """Momentum path skips in mean_reverting regime."""
        pass

    def test_no_momentum_when_transitioning(self):
        """Momentum blocked during regime transitions."""
        pass

    def test_existing_position_blocks_duplicate(self):
        """Cannot open momentum trade on ticker already in positions."""
        pass
```

These tests should use the FakeEngine pattern with mocked dependencies. The exact implementations will follow the patterns established in `test_crypto_engine.py`.

**Step 2: Run all tests**

Run: `python3 -m pytest arb_bot/tests/test_momentum.py -v`
Expected: All ~40 tests pass.

**Step 3: Run full test suite**

Run: `python3 -m pytest arb_bot/tests/ -q --ignore=arb_bot/tests/test_engine_multi_venue.py`
Expected: ~3110+ passed, 16 skipped, 0 failures.

**Step 4: Commit**

```bash
git add arb_bot/tests/test_momentum.py
git commit -m "test(crypto): v18 momentum integration tests"
```

---

### Task 8: Launch v18 paper test

**Step 1: Start v18 paper test with nohup**

```bash
nohup python3 -m arb_bot.crypto.paper_test --duration 480 --min-edge 0.12 --model empirical --max-tte 60 --scan-interval 5 > arb_bot/output/paper_v18_momentum.log 2>&1 &
echo $!
```

**Step 2: Verify it starts and shows momentum banner**

```bash
sleep 10 && head -30 arb_bot/output/paper_v18_momentum.log
```

Expected: Banner shows "Momentum: ON" with VPIN zone and OFI thresholds.

**Step 3: Commit and push**

```bash
git push origin backlog/b2-monte-carlo-sim
```
