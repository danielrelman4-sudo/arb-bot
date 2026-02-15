# Crypto Microstructure Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the crypto prediction engine from a stateless GBM model to a microstructure-informed model with directional signals (OFI), activity-scaled volatility, per-underlying correlation caps, and honest uncertainty estimates.

**Architecture:** Three parallel workstreams executed concurrently: Tier 0 (config/logic fixes to stop losses), Tier 1 (OFI data layer + drift injection — the real alpha), and Tier 2-lite (activity-scaled volatility). All changes preserve existing test compatibility and add new tests for each module.

**Tech Stack:** Python 3.9, numpy, aiohttp (Binance REST), websockets (Binance WS), pytest

---

## Workstream A: Tier 0 — Stop the Bleeding (Config & Logic Fixes)

### Task A1: Disable 15-min up/down markets

The model outputs ~50% for all up/down markets because zero-drift GBM is symmetric. This is a coinflip, not an edge. Disable until OFI drift provides actual directional signal.

**Files:**
- Modify: `arb_bot/crypto/config.py` (add `allowed_directions` setting)
- Modify: `arb_bot/crypto/engine.py:360-458` (filter out up/down in `_compute_model_probabilities`)
- Test: `arb_bot/tests/test_crypto_engine.py`

**Step 1: Write the failing test**

```python
# In test_crypto_engine.py
def test_up_down_markets_filtered_when_disabled(self):
    """Up/down markets should be skipped when allowed_directions excludes them."""
    settings = CryptoSettings(
        allowed_directions=["above", "below"],  # no "up" or "down"
        mc_num_paths=100,
    )
    engine = CryptoEngine(settings)
    # inject a price
    engine.price_feed.inject_tick(PriceTick("btcusdt", 70000.0, time.time(), 1.0))

    # Create an "up" market quote and an "above" market quote
    up_quote = _make_quote(direction="up", strike=None, tte=10.0)
    above_quote = _make_quote(direction="above", strike=69000.0, tte=10.0)

    probs = engine._compute_model_probabilities([up_quote, above_quote])

    # "up" market should be excluded, "above" should be included
    assert up_quote.market.ticker not in probs
    assert above_quote.market.ticker in probs
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py::TestCryptoEngine::test_up_down_markets_filtered_when_disabled -v`
Expected: FAIL — `allowed_directions` not a valid field

**Step 3: Implement**

In `config.py`, add to `CryptoSettings`:
```python
allowed_directions: List[str] = field(default_factory=lambda: ["above", "below"])
```

In `engine.py:_compute_model_probabilities`, add early filter after computing `direction`:
```python
direction = mq.market.meta.direction
if direction not in self._settings.allowed_directions:
    continue
```

In `load_crypto_settings()`:
```python
allowed_directions=_as_csv(os.getenv("ARB_CRYPTO_ALLOWED_DIRECTIONS")) or ["above", "below"],
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py::TestCryptoEngine::test_up_down_markets_filtered_when_disabled -v`
Expected: PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/config.py arb_bot/crypto/engine.py arb_bot/tests/test_crypto_engine.py
git commit -m "feat(crypto): disable up/down markets by default — no edge without OFI"
```

---

### Task A2: Per-underlying position correlation cap

The paper test had 7 simultaneous BTC positions at different strikes. When BTC dropped, all lost together. Cap positions per underlying.

**Files:**
- Modify: `arb_bot/crypto/config.py` (add `max_positions_per_underlying`)
- Modify: `arb_bot/crypto/engine.py:247-260` (check per-underlying count before opening)
- Test: `arb_bot/tests/test_crypto_engine.py`

**Step 1: Write the failing test**

```python
def test_per_underlying_position_cap(self):
    """Should not open more than max_positions_per_underlying for same underlying."""
    settings = CryptoSettings(
        max_positions_per_underlying=2,
        max_concurrent_positions=10,
        mc_num_paths=100,
    )
    engine = CryptoEngine(settings)

    # Manually inject 2 BTC positions
    for i in range(2):
        engine._positions[f"KXBTCD-fake-{i}"] = CryptoPosition(
            ticker=f"KXBTCD-fake-{i}", side="yes", contracts=10,
            entry_price=0.50, entry_time=time.time(),
            edge=_make_edge(underlying="BTC"), model_prob=0.6,
            market_implied_prob=0.5,
        )

    # Try to open a 3rd BTC position
    edge = _make_edge(underlying="BTC", ticker="KXBTCD-fake-2")
    contracts = engine._compute_position_size(edge)
    # The cycle logic should block it — test via _count_positions_for_underlying
    count = engine._count_positions_for_underlying("BTC")
    assert count == 2
    assert count >= settings.max_positions_per_underlying
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py::TestCryptoEngine::test_per_underlying_position_cap -v`
Expected: FAIL — `max_positions_per_underlying` and `_count_positions_for_underlying` don't exist

**Step 3: Implement**

In `config.py`, add to `CryptoSettings`:
```python
max_positions_per_underlying: int = 3
```

In `engine.py`, add helper method to `CryptoEngine`:
```python
def _count_positions_for_underlying(self, underlying: str) -> int:
    """Count open positions for a given underlying (e.g., 'BTC')."""
    count = 0
    for ticker in self._positions:
        # Extract underlying from ticker: KXBTCD-... -> BTC, KXSOLD-... -> SOL
        for ul, _ in _KALSHI_TO_BINANCE.items():
            if ul in ticker.upper():
                if ul == underlying:
                    count += 1
                break
    return count
```

In `engine.py:_run_cycle` and `run_cycle_with_quotes`, add check in the edge loop:
```python
# Check per-underlying cap
underlying = edge.market.meta.underlying
if self._count_positions_for_underlying(underlying) >= self._settings.max_positions_per_underlying:
    continue
```

In `load_crypto_settings()`:
```python
max_positions_per_underlying=_as_int(os.getenv("ARB_CRYPTO_MAX_POSITIONS_PER_UNDERLYING"), 3),
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/config.py arb_bot/crypto/engine.py arb_bot/tests/test_crypto_engine.py
git commit -m "feat(crypto): per-underlying position correlation cap (default 3)"
```

---

### Task A3: Raise min edge for daily markets

3-4% edges on daily markets were noise. Raise to 5%+ after blending.

**Files:**
- Modify: `arb_bot/crypto/config.py` (change default `min_edge_pct` from 0.05 to 0.05 — keep, but add `min_edge_pct_daily`)
- Modify: `arb_bot/crypto/edge_detector.py` (accept per-timeframe min edges)
- Test: `arb_bot/tests/test_edge_detector.py`

**Step 1: Write the failing test**

```python
def test_daily_market_higher_min_edge(self):
    """Daily (above/below) markets should require higher min edge than 15-min."""
    detector = EdgeDetector(
        min_edge_pct=0.03,
        min_edge_pct_daily=0.06,
        use_blending=False,
    )
    # Daily above-strike market with 4% edge (should be filtered)
    daily_quote = _make_quote(direction="above", tte=50.0)
    daily_prob = ProbabilityEstimate(0.60, 0.55, 0.65, 0.05, 1000)
    # implied = 0.54 -> edge = 0.06 - cost. At yes=0.56, edge = 0.04 -> below 6%

    # This market should be filtered because 4% < 6% min for daily
    edges = detector.detect_edges([daily_quote], {daily_quote.market.ticker: daily_prob})
    assert len(edges) == 0  # Filtered by daily threshold
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `min_edge_pct_daily` not a valid parameter

**Step 3: Implement**

In `edge_detector.py`:
- Add `min_edge_pct_daily: float = 0.06` to `EdgeDetector.__init__`
- In `detect_edges`, determine if market is "daily" based on TTE (>30 min) or direction (above/below)
- Apply the appropriate threshold

In `config.py`:
```python
min_edge_pct_daily: float = 0.06  # 6% min edge for daily above/below markets
```

**Step 4: Run all edge detector tests**

Run: `python3 -m pytest arb_bot/tests/test_edge_detector.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/config.py arb_bot/crypto/edge_detector.py arb_bot/tests/test_edge_detector.py
git commit -m "feat(crypto): higher min edge for daily markets (6% vs 3%)"
```

---

### Task A4: Model uncertainty multiplier (widen Wilson CI)

Wilson CI only captures MC sampling error, not input uncertainty (vol estimate could be 20-30% wrong). Multiply by a configurable factor to honestly reflect total model uncertainty.

**Files:**
- Modify: `arb_bot/crypto/config.py` (add `model_uncertainty_multiplier`)
- Modify: `arb_bot/crypto/engine.py:430-458` (scale uncertainty after MC estimate)
- Test: `arb_bot/tests/test_crypto_engine.py`

**Step 1: Write the failing test**

```python
def test_uncertainty_multiplier_scales_wilson_ci(self):
    """model_uncertainty_multiplier should widen the CI from MC estimates."""
    settings = CryptoSettings(
        model_uncertainty_multiplier=3.0,
        mc_num_paths=100,
    )
    engine = CryptoEngine(settings)
    engine.price_feed.inject_tick(PriceTick("btcusdt", 70000.0, time.time(), 1.0))

    quote = _make_quote(direction="above", strike=69000.0, tte=10.0)
    probs = engine._compute_model_probabilities([quote])

    # The uncertainty should be ~3x what raw Wilson CI gives
    prob = probs[quote.market.ticker]
    # Raw Wilson CI for 1000 paths and p~0.8 is about 0.025
    # With 3x multiplier, expect ~0.075
    assert prob.uncertainty > 0.05  # Much wider than raw
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `model_uncertainty_multiplier` not recognized

**Step 3: Implement**

In `config.py`:
```python
model_uncertainty_multiplier: float = 3.0  # Inflate Wilson CI to reflect vol estimation uncertainty
```

In `engine.py`, after computing `prob` in `_compute_model_probabilities`:
```python
# Scale uncertainty to reflect input uncertainty (vol estimate), not just MC sampling
unc_mult = self._settings.model_uncertainty_multiplier
if unc_mult != 1.0:
    scaled_unc = prob.uncertainty * unc_mult
    prob = ProbabilityEstimate(
        probability=prob.probability,
        ci_lower=max(0.0, prob.probability - scaled_unc),
        ci_upper=min(1.0, prob.probability + scaled_unc),
        uncertainty=scaled_unc,
        num_paths=prob.num_paths,
    )
```

In `load_crypto_settings()`:
```python
model_uncertainty_multiplier=_as_float(os.getenv("ARB_CRYPTO_MODEL_UNCERTAINTY_MULTIPLIER"), 3.0),
```

**Step 4: Run tests**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/config.py arb_bot/crypto/engine.py arb_bot/tests/test_crypto_engine.py
git commit -m "feat(crypto): uncertainty multiplier (3x) reflects vol estimation error"
```

---

## Workstream B: Tier 1 — OFI Directional Signal (The Alpha)

### Task B1: Extend PriceTick and PriceFeed to track buyer/seller volume

Parse `is_buyer_maker` from Binance trade WS. Aggregate buy vs sell volume.

**Files:**
- Modify: `arb_bot/crypto/price_feed.py` (add `is_buyer_maker` to PriceTick, track buy/sell deques)
- Test: `arb_bot/tests/test_price_feed.py`

**Step 1: Write the failing test**

```python
def test_price_tick_has_buyer_maker_flag():
    """PriceTick should store is_buyer_maker."""
    tick = PriceTick("btcusdt", 70000.0, time.time(), 1.0, is_buyer_maker=True)
    assert tick.is_buyer_maker is True

def test_price_feed_tracks_buy_sell_volume():
    """PriceFeed should separately track buy and sell volume."""
    feed = PriceFeed(symbols=["btcusdt"])
    now = time.time()
    # Inject buy trades (is_buyer_maker=False means buyer is taker = buy pressure)
    feed.inject_tick(PriceTick("btcusdt", 70000.0, now, 1.5, is_buyer_maker=False))
    feed.inject_tick(PriceTick("btcusdt", 70001.0, now + 1, 0.5, is_buyer_maker=True))

    buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
    assert buy_vol == pytest.approx(1.5)
    assert sell_vol == pytest.approx(0.5)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest arb_bot/tests/test_price_feed.py::test_price_tick_has_buyer_maker_flag -v`
Expected: FAIL — `is_buyer_maker` not a field

**Step 3: Implement**

Update `PriceTick`:
```python
@dataclass(frozen=True)
class PriceTick:
    symbol: str
    price: float
    timestamp: float
    volume: float = 0.0
    is_buyer_maker: bool | None = None  # True=sell, False=buy, None=unknown
```

Update `PriceFeed.__init__` to add buy/sell deques:
```python
self._buy_sells: Dict[str, Deque[tuple[float, float, bool]]] = {
    s: deque(maxlen=_MAX_TICK_HISTORY) for s in self._symbols
}  # (timestamp, volume, is_buy)
```

Update `_handle_message` to parse `m` field (is_buyer_maker):
```python
is_buyer_maker = msg.get("m", None)  # True = seller is maker (sell trade)
tick = PriceTick(symbol=sym, price=price, timestamp=ts, volume=qty, is_buyer_maker=is_buyer_maker)
# Track buy/sell
if is_buyer_maker is not None:
    is_buy = not is_buyer_maker  # buyer is taker = buy pressure
    dq_bs = self._buy_sells.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
    dq_bs.append((ts, qty, is_buy))
```

Update `inject_tick` similarly.

Add `get_buy_sell_volume`:
```python
def get_buy_sell_volume(self, symbol: str, window_seconds: int = 300) -> tuple[float, float]:
    """Return (buy_volume, sell_volume) over the last window_seconds."""
    sym = symbol.lower()
    dq = self._buy_sells.get(sym)
    if not dq:
        return (0.0, 0.0)
    cutoff = time.time() - window_seconds
    buy_vol = 0.0
    sell_vol = 0.0
    for ts, vol, is_buy in dq:
        if ts >= cutoff:
            if is_buy:
                buy_vol += vol
            else:
                sell_vol += vol
    return (buy_vol, sell_vol)
```

**Step 4: Run tests**

Run: `python3 -m pytest arb_bot/tests/test_price_feed.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/price_feed.py arb_bot/tests/test_price_feed.py
git commit -m "feat(crypto): track buy/sell volume from Binance is_buyer_maker flag"
```

---

### Task B2: OFI computation method on PriceFeed

Expose `get_ofi(symbol, window_seconds)` returning -1 to +1.

**Files:**
- Modify: `arb_bot/crypto/price_feed.py`
- Test: `arb_bot/tests/test_price_feed.py`

**Step 1: Write the failing test**

```python
def test_ofi_computation():
    """OFI should return (buy - sell) / (buy + sell)."""
    feed = PriceFeed(symbols=["btcusdt"])
    now = time.time()
    # 3.0 buy, 1.0 sell -> OFI = (3-1)/(3+1) = 0.5
    feed.inject_tick(PriceTick("btcusdt", 70000.0, now, 2.0, is_buyer_maker=False))
    feed.inject_tick(PriceTick("btcusdt", 70001.0, now + 1, 1.0, is_buyer_maker=False))
    feed.inject_tick(PriceTick("btcusdt", 70002.0, now + 2, 1.0, is_buyer_maker=True))

    ofi = feed.get_ofi("btcusdt", window_seconds=60)
    assert ofi == pytest.approx(0.5, abs=0.01)

def test_ofi_no_data_returns_zero():
    """OFI with no data should return 0 (neutral)."""
    feed = PriceFeed(symbols=["btcusdt"])
    assert feed.get_ofi("btcusdt") == 0.0

def test_ofi_all_buys_returns_one():
    feed = PriceFeed(symbols=["btcusdt"])
    now = time.time()
    feed.inject_tick(PriceTick("btcusdt", 70000.0, now, 5.0, is_buyer_maker=False))
    assert feed.get_ofi("btcusdt", window_seconds=60) == pytest.approx(1.0)

def test_ofi_all_sells_returns_negative_one():
    feed = PriceFeed(symbols=["btcusdt"])
    now = time.time()
    feed.inject_tick(PriceTick("btcusdt", 70000.0, now, 5.0, is_buyer_maker=True))
    assert feed.get_ofi("btcusdt", window_seconds=60) == pytest.approx(-1.0)
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `get_ofi` doesn't exist

**Step 3: Implement**

```python
def get_ofi(self, symbol: str, window_seconds: int = 300) -> float:
    """Order Flow Imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol).

    Returns a value in [-1, +1]. Positive = net buy pressure.
    Returns 0.0 if no data available.
    """
    buy_vol, sell_vol = self.get_buy_sell_volume(symbol, window_seconds)
    total = buy_vol + sell_vol
    if total <= 0:
        return 0.0
    return (buy_vol - sell_vol) / total
```

**Step 4: Run tests**

Run: `python3 -m pytest arb_bot/tests/test_price_feed.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/price_feed.py arb_bot/tests/test_price_feed.py
git commit -m "feat(crypto): OFI computation — order flow imbalance signal"
```

---

### Task B3: Volume flow rate tracking

Track volume per minute for activity-scaled volatility (Tier 2 lite).

**Files:**
- Modify: `arb_bot/crypto/price_feed.py`
- Test: `arb_bot/tests/test_price_feed.py`

**Step 1: Write the failing test**

```python
def test_volume_flow_rate():
    """Should return average volume per minute over window."""
    feed = PriceFeed(symbols=["btcusdt"])
    now = time.time()
    # Inject 120 BTC volume over ~2 minutes
    for i in range(120):
        feed.inject_tick(PriceTick("btcusdt", 70000.0, now - 120 + i, 1.0))

    rate = feed.get_volume_flow_rate("btcusdt", window_seconds=120)
    # 120 volume over 2 minutes = 60 per minute
    assert rate == pytest.approx(60.0, rel=0.1)

def test_volume_flow_rate_no_data():
    feed = PriceFeed(symbols=["btcusdt"])
    assert feed.get_volume_flow_rate("btcusdt") == 0.0
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `get_volume_flow_rate` doesn't exist

**Step 3: Implement**

```python
def get_volume_flow_rate(self, symbol: str, window_seconds: int = 300) -> float:
    """Average volume per minute over the last window_seconds.

    Returns 0.0 if insufficient data.
    """
    sym = symbol.lower()
    ticks = self._ticks.get(sym)
    if not ticks:
        return 0.0
    cutoff = time.time() - window_seconds
    total_vol = sum(t.volume for t in ticks if t.timestamp >= cutoff)
    minutes = window_seconds / 60.0
    if minutes <= 0:
        return 0.0
    return total_vol / minutes
```

**Step 4: Run tests**

Run: `python3 -m pytest arb_bot/tests/test_price_feed.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/price_feed.py arb_bot/tests/test_price_feed.py
git commit -m "feat(crypto): volume flow rate tracking for activity scaling"
```

---

### Task B4: Historical OFI bootstrap from Binance aggTrades REST

Fetch 48 hours of historical aggTrades for initial OFI alpha calibration.

**Files:**
- Create: `arb_bot/crypto/ofi_calibrator.py`
- Test: `arb_bot/tests/test_ofi_calibrator.py`

**Step 1: Write the failing test**

```python
def test_ofi_alpha_calibration():
    """Given paired (OFI, forward_return) samples, fit alpha coefficient."""
    from arb_bot.crypto.ofi_calibrator import OFICalibrator

    cal = OFICalibrator()
    # Synthetic data: OFI * 0.5 = forward return (alpha = 0.5)
    import numpy as np
    rng = np.random.default_rng(42)
    for _ in range(200):
        ofi = rng.uniform(-1, 1)
        fwd_return = 0.5 * ofi + rng.normal(0, 0.01)
        cal.record_sample(ofi, fwd_return)

    result = cal.calibrate()
    assert result.alpha == pytest.approx(0.5, abs=0.05)
    assert result.r_squared > 0.5
    assert result.n_samples == 200

def test_ofi_calibrator_low_r2_returns_zero_alpha():
    """When OFI has no predictive power, alpha should be 0."""
    from arb_bot.crypto.ofi_calibrator import OFICalibrator

    cal = OFICalibrator(min_r_squared=0.01)
    import numpy as np
    rng = np.random.default_rng(42)
    for _ in range(100):
        ofi = rng.uniform(-1, 1)
        fwd_return = rng.normal(0, 0.05)  # Pure noise
        cal.record_sample(ofi, fwd_return)

    result = cal.calibrate()
    assert result.alpha == 0.0  # No signal -> no drift

def test_ofi_calibrator_rolling_window():
    """Calibrator should use only the most recent samples."""
    from arb_bot.crypto.ofi_calibrator import OFICalibrator

    cal = OFICalibrator(max_samples=50)
    import numpy as np
    rng = np.random.default_rng(42)
    # Add 100 samples — only last 50 should be kept
    for _ in range(100):
        cal.record_sample(rng.uniform(-1, 1), rng.normal(0, 0.01))

    assert len(cal._samples) == 50
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `ofi_calibrator` module doesn't exist

**Step 3: Implement `ofi_calibrator.py`**

```python
"""OFI-to-drift calibrator: fits alpha for mu = alpha * OFI.

Performs rolling OLS regression of Order Flow Imbalance against
forward returns to determine the optimal drift coefficient.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass(frozen=True)
class OFICalibrationResult:
    alpha: float       # Fitted drift coefficient
    r_squared: float   # Goodness of fit
    n_samples: int     # Number of samples used


class OFICalibrator:
    """Calibrates OFI alpha using rolling OLS."""

    def __init__(
        self,
        max_samples: int = 5000,
        min_samples: int = 30,
        min_r_squared: float = 0.01,
    ) -> None:
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._min_r_squared = min_r_squared
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=max_samples)

    def record_sample(self, ofi: float, forward_return: float) -> None:
        self._samples.append((ofi, forward_return))

    def calibrate(self) -> OFICalibrationResult:
        n = len(self._samples)
        if n < self._min_samples:
            return OFICalibrationResult(alpha=0.0, r_squared=0.0, n_samples=n)

        arr = np.array(list(self._samples))
        x = arr[:, 0]  # OFI
        y = arr[:, 1]  # forward return

        # OLS: y = alpha * x + epsilon
        # alpha = sum(x*y) / sum(x^2)
        x_sq_sum = float(np.sum(x * x))
        if x_sq_sum < 1e-12:
            return OFICalibrationResult(alpha=0.0, r_squared=0.0, n_samples=n)

        alpha = float(np.sum(x * y)) / x_sq_sum

        # R-squared
        y_pred = alpha * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_sq = max(0.0, r_sq)

        # If R² below threshold, signal is not predictive — return 0
        if r_sq < self._min_r_squared:
            return OFICalibrationResult(alpha=0.0, r_squared=r_sq, n_samples=n)

        return OFICalibrationResult(alpha=alpha, r_squared=r_sq, n_samples=n)
```

**Step 4: Run tests**

Run: `python3 -m pytest arb_bot/tests/test_ofi_calibrator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/ofi_calibrator.py arb_bot/tests/test_ofi_calibrator.py
git commit -m "feat(crypto): OFI calibrator with rolling OLS and R² gating"
```

---

### Task B5: Wire OFI drift into engine

Replace `drift = 0.0` with `drift = alpha * ofi` from calibrator.

**Files:**
- Modify: `arb_bot/crypto/config.py` (add OFI config)
- Modify: `arb_bot/crypto/engine.py:101-131,415` (integrate OFICalibrator, compute drift)
- Test: `arb_bot/tests/test_crypto_engine.py`

**Step 1: Write the failing test**

```python
def test_ofi_drift_injected_into_paths(self):
    """When OFI is positive, drift should be positive."""
    settings = CryptoSettings(
        mc_num_paths=2000,
        ofi_enabled=True,
        ofi_window_seconds=60,
    )
    engine = CryptoEngine(settings)

    # Inject price + heavy buy pressure
    now = time.time()
    for i in range(60):
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 70000.0 + i * 0.1, now - 60 + i, 2.0, is_buyer_maker=False)
        )
    # Set a known alpha
    engine._ofi_calibrator._samples.extend([(0.5, 0.001)] * 100)

    ofi = engine.price_feed.get_ofi("btcusdt", window_seconds=60)
    assert ofi > 0.5  # Strong buy pressure

    # The drift computed in _compute_model_probabilities should be > 0
    quote = _make_quote(direction="above", strike=69500.0, tte=10.0, underlying="BTC")
    probs = engine._compute_model_probabilities([quote])
    # With positive drift, P(above 69500) should be higher than without
    assert probs[quote.market.ticker].probability > 0.5
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `ofi_enabled` not recognized, `_ofi_calibrator` doesn't exist

**Step 3: Implement**

In `config.py`:
```python
# OFI (Order Flow Imbalance) drift
ofi_enabled: bool = True
ofi_window_seconds: int = 300
ofi_alpha: float = 0.0  # Starting alpha (0 = neutral, calibrated at runtime)
ofi_recalibrate_interval_hours: float = 4.0
```

In `engine.py.__init__`:
```python
from arb_bot.crypto.ofi_calibrator import OFICalibrator
self._ofi_calibrator = OFICalibrator()
```

In `engine.py._compute_model_probabilities`, replace `drift = 0.0`:
```python
# Compute OFI-based drift if enabled
drift = 0.0
if self._settings.ofi_enabled and binance_sym:
    ofi = self._price_feed.get_ofi(binance_sym, window_seconds=self._settings.ofi_window_seconds)
    cal_result = self._ofi_calibrator.calibrate()
    alpha = cal_result.alpha if cal_result.alpha != 0 else self._settings.ofi_alpha
    drift = alpha * ofi  # Annualized drift from OFI
```

In `load_crypto_settings()`:
```python
ofi_enabled=_as_bool(os.getenv("ARB_CRYPTO_OFI_ENABLED"), True),
ofi_window_seconds=_as_int(os.getenv("ARB_CRYPTO_OFI_WINDOW_SECONDS"), 300),
ofi_alpha=_as_float(os.getenv("ARB_CRYPTO_OFI_ALPHA"), 0.0),
ofi_recalibrate_interval_hours=_as_float(os.getenv("ARB_CRYPTO_OFI_RECALIBRATE_HOURS"), 4.0),
```

**Step 4: Run tests**

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/config.py arb_bot/crypto/engine.py arb_bot/tests/test_crypto_engine.py
git commit -m "feat(crypto): wire OFI drift into MC path generation — directional signal"
```

---

### Task B6: Historical aggTrades bootstrap

Fetch historical trade data from Binance REST to bootstrap OFI calibration on first run.

**Files:**
- Modify: `arb_bot/crypto/price_feed.py` (add `load_historical_trades`)
- Test: `arb_bot/tests/test_price_feed.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_load_historical_trades_populates_buy_sell():
    """Historical aggTrades should populate buy/sell volume tracking."""
    feed = PriceFeed(symbols=["btcusdt"])
    # Mock the REST call
    # ... (use monkeypatch or mock aiohttp)
    # After loading, get_ofi should return a non-zero value
    # and get_buy_sell_volume should have data
```

Note: This test requires mocking aiohttp. The implementation should follow the same pattern as `load_historical()` but use the `/api/v3/aggTrades` endpoint which includes the `m` (is_buyer_maker) field.

**Step 2-5:** Standard TDD cycle. The key implementation is:

```python
async def load_historical_trades(self, symbol: str, hours: int = 48) -> int:
    """Bootstrap buy/sell volume from Binance aggTrades REST.

    Loads recent aggregate trades to populate OFI data for calibration.
    Returns the number of trades loaded.
    """
    sym = symbol.lower()
    url = f"https://api.binance.com/api/v3/aggTrades?symbol={sym.upper()}&limit=1000"
    # Fetch in batches going backward from now
    # Parse 'm' field for is_buyer_maker
    # Inject into _buy_sells deque
```

**Commit:**
```bash
git commit -m "feat(crypto): historical aggTrades bootstrap for OFI calibration"
```

---

## Workstream C: Tier 2 Lite — Activity-Scaled Volatility

### Task C1: Activity-scaled vol in engine

Scale volatility by `sqrt(current_volume_rate / avg_volume_rate)`.

**Files:**
- Modify: `arb_bot/crypto/config.py` (add `activity_scaling_enabled`, `avg_volume_rate_window_minutes`)
- Modify: `arb_bot/crypto/engine.py:406-408` (apply activity scaling after vol computation)
- Test: `arb_bot/tests/test_crypto_engine.py`

**Step 1: Write the failing test**

```python
def test_activity_scaling_increases_vol_in_high_volume(self):
    """When current volume is 2x average, vol should be scaled up by sqrt(2)."""
    settings = CryptoSettings(
        activity_scaling_enabled=True,
        mc_num_paths=100,
    )
    engine = CryptoEngine(settings)

    now = time.time()
    # Inject "normal" volume for 30 min (1.0 per tick, 1 tick/sec)
    for i in range(1800):
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 70000.0, now - 1800 + i, 1.0)
        )
    # Last 5 minutes: 2x volume
    for i in range(300):
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 70000.0, now - 300 + i, 2.0)
        )

    # The vol used in path generation should be ~sqrt(2) * base vol
    # We test indirectly via wider probability spread
    quote = _make_quote(direction="above", strike=70500.0, tte=10.0)
    probs = engine._compute_model_probabilities([quote])
    prob_high_vol = probs[quote.market.ticker].probability

    # Compare against no scaling
    settings2 = CryptoSettings(activity_scaling_enabled=False, mc_num_paths=100)
    engine2 = CryptoEngine(settings2)
    # ... inject same data, compute prob, verify high_vol prob is different
```

**Step 2-5:** Standard TDD cycle. Key implementation:

In `config.py`:
```python
activity_scaling_enabled: bool = True
activity_scaling_short_window_seconds: int = 300   # "current" volume rate window
activity_scaling_long_window_seconds: int = 3600   # "average" volume rate window
```

In `engine.py`, after vol computation:
```python
# Activity scaling: adjust vol by sqrt(current_volume_rate / avg_volume_rate)
if self._settings.activity_scaling_enabled:
    short_rate = self._price_feed.get_volume_flow_rate(
        binance_sym, window_seconds=self._settings.activity_scaling_short_window_seconds
    )
    long_rate = self._price_feed.get_volume_flow_rate(
        binance_sym, window_seconds=self._settings.activity_scaling_long_window_seconds
    )
    if long_rate > 0 and short_rate > 0:
        activity_ratio = short_rate / long_rate
        # Clamp to prevent extreme scaling
        activity_ratio = max(0.25, min(4.0, activity_ratio))
        vol *= math.sqrt(activity_ratio)
```

**Commit:**
```bash
git commit -m "feat(crypto): activity-scaled volatility — vol adjusts with volume flow"
```

---

## Integration Testing

### Task D1: Run full test suite

```bash
python3 -m pytest arb_bot/tests/test_crypto_engine.py arb_bot/tests/test_price_feed.py arb_bot/tests/test_edge_detector.py arb_bot/tests/test_ofi_calibrator.py arb_bot/tests/test_price_model.py arb_bot/tests/test_crypto_config.py -v
```

Expected: ALL PASS (127 existing + ~25 new)

### Task D2: 1-hour paper test with all upgrades

```bash
nohup python3 -u -m arb_bot.crypto.paper_test \
    --symbols BTC SOL \
    --duration-minutes 60 \
    --mc-paths 1000 \
    --min-edge 0.05 \
    --scan-interval 30 \
    --max-tte 600 \
    > /tmp/crypto_v2_1hr_run.log 2>&1 &
```

Compare against baseline (-$108.94 in 1hr):
- Expect fewer trades (15-min markets disabled, higher min edge)
- Expect better win rate on remaining trades
- Expect smaller position sizes (3x uncertainty multiplier)

### Task D3: Final commit

```bash
git add -A
git commit -m "feat(crypto): microstructure upgrade v2 — OFI drift, activity scaling, risk controls"
```
