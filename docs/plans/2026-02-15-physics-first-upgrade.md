# Physics-First Microstructure Upgrade Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full "Physics-First" strategy: power-law OFI impact, Hawkes self-exciting jump process, up/down reference-price fix, OFI data quality (WebSocket streaming), Kalshi execution awareness, and per-cycle CSV data logging for offline calibration.

**Architecture:** Six tasks executed sequentially. Each modifies a focused set of files with full TDD. Tasks 1-3 upgrade the model physics (power-law OFI, Hawkes jumps, reference-price fix). Task 4 adds WebSocket aggTrades streaming for OFI data quality. Task 5 adds CSV data logging. Task 6 updates the paper test to use all new features.

**Tech Stack:** Python 3.9, numpy, scipy (for nonlinear least squares in calibrator), pytest

---

## Task 1: Power-Law OFI Impact Model

Replace `drift = alpha * OFI` with `drift = alpha * sgn(OFI) * |OFI|^theta` where theta ≈ 0.5 (the square-root law from Kyle/Cont).

**Files:**
- Modify: `arb_bot/crypto/config.py` — add `ofi_impact_exponent` field
- Modify: `arb_bot/crypto/engine.py:476-486` — power-law drift computation
- Modify: `arb_bot/crypto/ofi_calibrator.py` — fit power-law model via scipy
- Create: `arb_bot/tests/test_crypto_ofi_power_law.py`

### Step 1: Write failing tests

```python
# arb_bot/tests/test_crypto_ofi_power_law.py
"""Tests for power-law OFI impact model."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.ofi_calibrator import OFICalibrator, OFICalibrationResult
from arb_bot.crypto.price_feed import PriceTick
from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketMeta, CryptoMarketQuote
from datetime import datetime, timedelta, timezone


def _make_settings(**overrides) -> CryptoSettings:
    defaults = dict(
        enabled=True, paper_mode=True, bankroll=1000.0,
        mc_num_paths=100, min_edge_pct=0.05, min_edge_cents=0.02,
        max_model_uncertainty=0.15, kelly_fraction_cap=0.10,
        max_position_per_market=100.0, max_concurrent_positions=10,
        scan_interval_seconds=0.01, paper_slippage_cents=0.0,
        confidence_level=0.95, mc_vol_window_minutes=30,
        price_feed_symbols=["btcusdt"], symbols=["KXBTC"],
        min_minutes_to_expiry=2, max_minutes_to_expiry=60,
        allowed_directions=["above", "below"],
    )
    defaults.update(overrides)
    return CryptoSettings(**defaults)


class TestPowerLawConfig:
    """Config field for OFI impact exponent."""

    def test_default_exponent_is_half(self):
        s = CryptoSettings()
        assert s.ofi_impact_exponent == 0.5

    def test_exponent_configurable(self):
        s = CryptoSettings(ofi_impact_exponent=0.7)
        assert s.ofi_impact_exponent == 0.7

    def test_exponent_one_gives_linear(self):
        """Exponent=1.0 should reproduce the old linear model."""
        s = CryptoSettings(ofi_impact_exponent=1.0)
        assert s.ofi_impact_exponent == 1.0


class TestPowerLawDrift:
    """Engine applies power-law transform to OFI before drift computation."""

    def test_power_law_dampens_extreme_ofi(self):
        """OFI=1.0 with theta=0.5 should produce drift = alpha * 1.0.
        OFI=0.25 with theta=0.5 should produce drift = alpha * 0.5.
        The ratio should be 2:1, not 4:1 (which linear would give)."""
        settings = _make_settings(
            ofi_enabled=True, ofi_alpha=1.0,
            ofi_impact_exponent=0.5,
        )
        engine = CryptoEngine(settings)
        # The _apply_ofi_impact method should exist
        drift_full = engine._apply_ofi_impact(1.0, alpha=1.0)
        drift_quarter = engine._apply_ofi_impact(0.25, alpha=1.0)
        # sqrt(1.0) / sqrt(0.25) = 1.0 / 0.5 = 2.0
        assert abs(drift_full / drift_quarter - 2.0) < 0.01

    def test_power_law_preserves_sign(self):
        """Negative OFI should produce negative drift."""
        settings = _make_settings(
            ofi_enabled=True, ofi_alpha=1.0,
            ofi_impact_exponent=0.5,
        )
        engine = CryptoEngine(settings)
        drift_neg = engine._apply_ofi_impact(-0.5, alpha=1.0)
        drift_pos = engine._apply_ofi_impact(0.5, alpha=1.0)
        assert drift_neg < 0
        assert drift_pos > 0
        assert abs(drift_neg) == pytest.approx(abs(drift_pos), abs=1e-10)

    def test_exponent_one_is_linear(self):
        """With theta=1.0, drift should equal alpha * ofi exactly."""
        settings = _make_settings(
            ofi_enabled=True, ofi_alpha=2.5,
            ofi_impact_exponent=1.0,
        )
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(0.7, alpha=2.5)
        assert drift == pytest.approx(2.5 * 0.7, abs=1e-10)

    def test_zero_ofi_gives_zero_drift(self):
        settings = _make_settings(ofi_enabled=True, ofi_alpha=1.0, ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        assert engine._apply_ofi_impact(0.0, alpha=1.0) == 0.0


class TestPowerLawCalibrator:
    """OFI calibrator fits power-law model via nonlinear least squares."""

    def test_calibrate_returns_theta(self):
        """Calibration result should include theta parameter."""
        cal = OFICalibrator(min_samples=5, min_r_squared=0.0)
        # Generate synthetic data: return = 0.5 * sgn(ofi) * |ofi|^0.5
        rng = np.random.default_rng(42)
        for _ in range(100):
            ofi = rng.uniform(-1, 1)
            ret = 0.5 * math.copysign(abs(ofi) ** 0.5, ofi) + rng.normal(0, 0.01)
            cal.record_sample(ofi, ret)
        result = cal.calibrate()
        assert hasattr(result, 'theta')
        # Should recover theta near 0.5
        assert 0.3 < result.theta < 0.7

    def test_calibrate_returns_alpha(self):
        """Should still return an alpha coefficient."""
        cal = OFICalibrator(min_samples=5, min_r_squared=0.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            ofi = rng.uniform(-1, 1)
            ret = 0.5 * math.copysign(abs(ofi) ** 0.5, ofi) + rng.normal(0, 0.01)
            cal.record_sample(ofi, ret)
        result = cal.calibrate()
        assert 0.3 < result.alpha < 0.8

    def test_insufficient_samples_returns_defaults(self):
        """With too few samples, theta should default to 0.5, alpha to 0."""
        cal = OFICalibrator(min_samples=30)
        for i in range(5):
            cal.record_sample(0.1 * i, 0.001 * i)
        result = cal.calibrate()
        assert result.alpha == 0.0
        assert result.theta == 0.5

    def test_noise_returns_default_theta(self):
        """Pure noise should fail R² gate and return default theta=0.5."""
        cal = OFICalibrator(min_samples=5, min_r_squared=0.01)
        rng = np.random.default_rng(99)
        for _ in range(100):
            ofi = rng.uniform(-1, 1)
            ret = rng.normal(0, 1.0)  # pure noise, no signal
            cal.record_sample(ofi, ret)
        result = cal.calibrate()
        assert result.theta == 0.5  # default
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest arb_bot/tests/test_crypto_ofi_power_law.py -v`
Expected: FAIL — `ofi_impact_exponent` not a valid field, `_apply_ofi_impact` doesn't exist, `theta` not on result

### Step 3: Implement

**config.py** — add field after `ofi_recalibrate_interval_hours`:

```python
ofi_impact_exponent: float = 0.5  # Power-law exponent (0.5 = square root law)
```

**ofi_calibrator.py** — replace OLS with power-law fit:

Add `theta` field to `OFICalibrationResult`:
```python
@dataclass(frozen=True)
class OFICalibrationResult:
    alpha: float
    theta: float       # Power-law exponent
    r_squared: float
    n_samples: int
```

Replace `calibrate()` body with scipy curve_fit:
```python
def calibrate(self) -> OFICalibrationResult:
    n = len(self._samples)
    if n < self._min_samples:
        return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=0.0, n_samples=n)

    data = np.array(list(self._samples))
    x = data[:, 0]  # OFI values
    y = data[:, 1]  # forward returns

    if np.all(np.abs(x) < 1e-12):
        return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=0.0, n_samples=n)

    # Power-law model: y = alpha * sgn(x) * |x|^theta
    def _power_law(x_arr, alpha, theta):
        return alpha * np.copysign(np.abs(x_arr) ** theta, x_arr)

    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            _power_law, x, y,
            p0=[0.1, 0.5],
            bounds=([−10.0, 0.1], [10.0, 1.5]),
            maxfev=2000,
        )
        alpha_fit, theta_fit = popt
    except Exception:
        # Fallback to OLS linear if scipy fails
        x_sq_sum = float(np.sum(x * x))
        if x_sq_sum < 1e-12:
            return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=0.0, n_samples=n)
        alpha_fit = float(np.sum(x * y)) / x_sq_sum
        theta_fit = 1.0  # linear fallback

    # R-squared (uncentered)
    y_pred = _power_law(x, alpha_fit, theta_fit)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum(y * y))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    if r_sq < self._min_r_squared:
        return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=r_sq, n_samples=n)

    return OFICalibrationResult(
        alpha=alpha_fit, theta=theta_fit,
        r_squared=r_sq, n_samples=n,
    )
```

**engine.py** — add `_apply_ofi_impact` method and update drift computation:

```python
def _apply_ofi_impact(self, ofi: float, alpha: float) -> float:
    """Apply power-law impact model: drift = alpha * sgn(ofi) * |ofi|^theta."""
    theta = self._settings.ofi_impact_exponent
    if ofi == 0.0 or alpha == 0.0:
        return 0.0
    return alpha * math.copysign(abs(ofi) ** theta, ofi)
```

Replace the drift computation block (lines 476-486):
```python
drift = 0.0
if self._settings.ofi_enabled and binance_sym:
    ofi = self._price_feed.get_ofi(
        binance_sym,
        window_seconds=self._settings.ofi_window_seconds,
    )
    cal_result = self._get_ofi_calibration()
    alpha = cal_result.alpha if cal_result.alpha != 0 else self._settings.ofi_alpha
    drift = self._apply_ofi_impact(ofi, alpha)
```

Update all code that constructs `OFICalibrationResult` to include `theta=0.5` default.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest arb_bot/tests/test_crypto_ofi_power_law.py -v`
Expected: PASS

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: PASS (existing tests unbroken — theta=0.5 default changes behavior but OFI drift tests used alpha=0 so drift is still 0)

### Step 5: Commit

```bash
git add arb_bot/crypto/config.py arb_bot/crypto/engine.py arb_bot/crypto/ofi_calibrator.py arb_bot/tests/test_crypto_ofi_power_law.py
git commit -m "feat(crypto): power-law OFI impact model (theta=0.5 square root law)"
```

---

## Task 2: Hawkes Self-Exciting Jump Process

Replace the static Merton jump intensity (λ=3/day constant) with a Hawkes self-exciting process where large returns spike the intensity, which then decays exponentially.

**Files:**
- Create: `arb_bot/crypto/hawkes.py` — HawkesIntensity tracker
- Modify: `arb_bot/crypto/config.py` — add Hawkes parameters
- Modify: `arb_bot/crypto/engine.py` — feed returns to Hawkes tracker, pass dynamic intensity to path generator
- Modify: `arb_bot/crypto/price_model.py` — accept dynamic intensity, simulate Hawkes within paths
- Create: `arb_bot/tests/test_crypto_hawkes.py`

### Step 1: Write failing tests

```python
# arb_bot/tests/test_crypto_hawkes.py
"""Tests for Hawkes self-exciting jump intensity process."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from arb_bot.crypto.hawkes import HawkesIntensity
from arb_bot.crypto.price_model import PriceModel


class TestHawkesIntensity:
    """Core Hawkes intensity tracker."""

    def test_baseline_intensity(self):
        """Without any shocks, intensity equals baseline mu."""
        h = HawkesIntensity(mu=3.0, alpha=0.5, beta=0.1)
        assert h.intensity(time.monotonic()) == pytest.approx(3.0)

    def test_shock_spikes_intensity(self):
        """Recording a shock should increase intensity above baseline."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.1)
        t0 = time.monotonic()
        h.record_shock(t0, magnitude=1.0)
        # Immediately after shock: intensity = mu + alpha * magnitude
        intensity = h.intensity(t0)
        assert intensity > 3.0
        assert intensity == pytest.approx(3.0 + 5.0 * 1.0)

    def test_intensity_decays_exponentially(self):
        """Intensity should decay back toward mu after a shock."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=1.0)
        t0 = 1000.0
        h.record_shock(t0, magnitude=1.0)
        # At t0: intensity = 3 + 5 = 8
        # At t0 + 1/beta = t0 + 1.0: intensity = 3 + 5*e^(-1) ≈ 3 + 1.84
        t1 = t0 + 1.0
        intensity = h.intensity(t1)
        expected = 3.0 + 5.0 * math.exp(-1.0)
        assert intensity == pytest.approx(expected, abs=0.01)

    def test_multiple_shocks_additive(self):
        """Multiple shocks should stack additively."""
        h = HawkesIntensity(mu=3.0, alpha=2.0, beta=0.5)
        t0 = 1000.0
        h.record_shock(t0, magnitude=1.0)
        h.record_shock(t0 + 0.5, magnitude=1.0)
        # At t0 + 0.5: first shock decayed by e^(-0.5*0.5), second is fresh
        t_eval = t0 + 0.5
        expected = 3.0 + 2.0 * math.exp(-0.5 * 0.5) + 2.0
        assert h.intensity(t_eval) == pytest.approx(expected, abs=0.01)

    def test_old_shocks_pruned(self):
        """Shocks older than max_history should be pruned."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=1.0, max_history_seconds=60.0)
        t0 = 1000.0
        h.record_shock(t0, magnitude=1.0)
        # 120 seconds later, shock should be pruned
        t_late = t0 + 120.0
        assert h.intensity(t_late) == pytest.approx(3.0, abs=0.01)

    def test_zero_magnitude_no_effect(self):
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=1.0)
        t0 = time.monotonic()
        h.record_shock(t0, magnitude=0.0)
        assert h.intensity(t0) == pytest.approx(3.0)

    def test_large_return_triggers_shock(self):
        """The record_return method should only trigger on |return| > threshold."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=1.0, return_threshold_sigma=4.0)
        t0 = time.monotonic()
        # Small return — no shock
        h.record_return(t0, log_return=0.001, realized_vol=0.01)
        assert h.intensity(t0) == pytest.approx(3.0)
        # Large return (5 sigma) — triggers shock
        h.record_return(t0 + 1.0, log_return=0.06, realized_vol=0.01)
        assert h.intensity(t0 + 1.0) > 3.0


class TestHawkesConfig:
    """Config parameters for Hawkes process."""

    def test_hawkes_fields_exist(self):
        from arb_bot.crypto.config import CryptoSettings
        s = CryptoSettings()
        assert hasattr(s, 'hawkes_enabled')
        assert hasattr(s, 'hawkes_alpha')
        assert hasattr(s, 'hawkes_beta')
        assert hasattr(s, 'hawkes_return_threshold_sigma')

    def test_hawkes_defaults(self):
        from arb_bot.crypto.config import CryptoSettings
        s = CryptoSettings()
        assert s.hawkes_enabled is True
        assert s.hawkes_alpha == 5.0
        assert s.hawkes_beta == pytest.approx(math.log(2) / 600.0)  # 10-min half-life
        assert s.hawkes_return_threshold_sigma == 4.0


class TestHawkesPathGeneration:
    """PriceModel generates paths with dynamic jump intensity."""

    def test_hawkes_paths_more_volatile_after_shock(self):
        """Paths generated with elevated Hawkes intensity should have
        wider terminal distribution than paths with baseline intensity."""
        model = PriceModel(num_paths=5000, seed=42)
        # Baseline: lambda=3/day
        paths_calm = model.generate_paths_jump_diffusion(
            current_price=70000, volatility=0.5, horizon_minutes=15,
            jump_intensity=3.0,
        )
        # Post-shock: lambda=20/day (elevated)
        paths_shock = model.generate_paths_jump_diffusion(
            current_price=70000, volatility=0.5, horizon_minutes=15,
            jump_intensity=20.0,
        )
        # The shocked paths should have higher variance
        var_calm = np.var(paths_calm)
        var_shock = np.var(paths_shock)
        assert var_shock > var_calm

    def test_engine_feeds_hawkes_intensity_to_paths(self):
        """After a large return, the engine should pass elevated
        jump_intensity to the path generator."""
        from arb_bot.crypto.config import CryptoSettings
        from arb_bot.crypto.engine import CryptoEngine
        from arb_bot.crypto.price_feed import PriceTick

        settings = CryptoSettings(
            enabled=True, paper_mode=True,
            mc_num_paths=100, hawkes_enabled=True,
            use_jump_diffusion=True,
            allowed_directions=["above", "below"],
            price_feed_symbols=["btcusdt"], symbols=["KXBTC"],
        )
        engine = CryptoEngine(settings)
        # Inject a calm price history
        now = time.time()
        for i in range(60):
            engine.price_feed.inject_tick(PriceTick(
                "btcusdt", 70000.0, now - 60 + i, 100.0,
            ))
        # Baseline intensity
        baseline = engine._hawkes.intensity(time.monotonic())
        assert baseline == pytest.approx(settings.mc_jump_intensity, abs=0.1)

        # Inject a crash (5% drop in one tick)
        engine.price_feed.inject_tick(PriceTick(
            "btcusdt", 66500.0, now + 1, 10000.0,
        ))
        engine._update_hawkes_from_returns()
        elevated = engine._hawkes.intensity(time.monotonic())
        assert elevated > baseline
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest arb_bot/tests/test_crypto_hawkes.py -v`
Expected: FAIL — `arb_bot.crypto.hawkes` doesn't exist

### Step 3: Implement

**arb_bot/crypto/hawkes.py:**

```python
"""Hawkes self-exciting jump intensity process.

Models volatility clustering: large returns spike the jump intensity,
which decays exponentially. Based on Filimonov & Sornette (2012).

    lambda(t) = mu + sum_{t_i < t} alpha * m_i * exp(-beta * (t - t_i))

where m_i is the magnitude of shock i.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple


@dataclass
class HawkesIntensity:
    """Track a self-exciting jump intensity process.

    Parameters
    ----------
    mu : float
        Baseline jump intensity (jumps per day).
    alpha : float
        Excitation amplitude — how much each shock increases intensity.
    beta : float
        Decay rate (per second). Half-life = ln(2) / beta.
    max_history_seconds : float
        Prune shocks older than this (default 30 min).
    return_threshold_sigma : float
        Only returns exceeding this many sigma trigger a shock.
    """

    mu: float = 3.0
    alpha: float = 5.0
    beta: float = 0.00115  # ln(2)/600 ≈ 10-min half-life
    max_history_seconds: float = 1800.0
    return_threshold_sigma: float = 4.0
    _shocks: Deque[Tuple[float, float]] = field(
        default_factory=deque, repr=False,
    )

    def record_shock(self, t: float, magnitude: float) -> None:
        """Record a shock at time t with given magnitude."""
        if magnitude == 0.0:
            return
        self._shocks.append((t, magnitude))

    def record_return(
        self, t: float, log_return: float, realized_vol: float,
    ) -> None:
        """Record a return; trigger shock if |return| > threshold * vol."""
        if realized_vol <= 0:
            return
        z_score = abs(log_return) / realized_vol
        if z_score >= self.return_threshold_sigma:
            # Magnitude proportional to how far beyond threshold
            magnitude = z_score / self.return_threshold_sigma
            self.record_shock(t, magnitude)

    def intensity(self, t: float) -> float:
        """Compute lambda(t) = mu + sum of decayed shocks."""
        self._prune(t)
        excitation = 0.0
        for t_i, m_i in self._shocks:
            dt = t - t_i
            if dt < 0:
                continue
            excitation += self.alpha * m_i * math.exp(-self.beta * dt)
        return self.mu + excitation

    def _prune(self, t: float) -> None:
        """Remove shocks older than max_history_seconds."""
        cutoff = t - self.max_history_seconds
        while self._shocks and self._shocks[0][0] < cutoff:
            self._shocks.popleft()
```

**config.py** — add after jump diffusion params:

```python
# Hawkes self-exciting jumps
hawkes_enabled: bool = True
hawkes_alpha: float = 5.0      # Excitation amplitude
hawkes_beta: float = 0.00115   # Decay rate (ln(2)/600 ≈ 10-min half-life)
hawkes_return_threshold_sigma: float = 4.0  # Sigma threshold to trigger shock
```

**engine.py** — in `__init__`, create HawkesIntensity:

```python
from arb_bot.crypto.hawkes import HawkesIntensity

# In __init__:
self._hawkes = HawkesIntensity(
    mu=self._settings.mc_jump_intensity,
    alpha=self._settings.hawkes_alpha,
    beta=self._settings.hawkes_beta,
    return_threshold_sigma=self._settings.hawkes_return_threshold_sigma,
)
```

Add `_update_hawkes_from_returns` method:

```python
def _update_hawkes_from_returns(self) -> None:
    """Check recent returns and feed large ones to Hawkes tracker."""
    now = time.monotonic()
    for binance_sym in _KALSHI_TO_BINANCE.values():
        returns = self._price_feed.get_returns(
            binance_sym, interval_seconds=60, window_minutes=5,
        )
        if len(returns) < 2:
            continue
        vol = float(np.std(returns)) if len(returns) >= 5 else 0.01
        # Check the most recent return
        latest_return = returns[-1]
        self._hawkes.record_return(now, latest_return, vol)
```

In `_compute_model_probabilities`, replace static `jump_intensity` with dynamic:

```python
# Before path generation:
if self._settings.hawkes_enabled:
    dynamic_intensity = self._hawkes.intensity(time.monotonic())
else:
    dynamic_intensity = self._settings.mc_jump_intensity

# In path generation call:
if self._settings.use_jump_diffusion:
    paths = self._price_model.generate_paths_jump_diffusion(
        current_price, vol, horizon, drift=drift,
        jump_intensity=dynamic_intensity,
        jump_mean=self._settings.mc_jump_mean,
        jump_vol=self._settings.mc_jump_vol,
    )
```

Call `_update_hawkes_from_returns()` at the start of `_run_cycle()` and `run_cycle_with_quotes()`.

### Step 4: Run tests

Run: `python3 -m pytest arb_bot/tests/test_crypto_hawkes.py -v`
Expected: PASS

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add arb_bot/crypto/hawkes.py arb_bot/crypto/config.py arb_bot/crypto/engine.py arb_bot/tests/test_crypto_hawkes.py
git commit -m "feat(crypto): Hawkes self-exciting jump intensity (Filimonov-Sornette)"
```

---

## Task 3: Fix Up/Down Reference Price Bug

The model uses the current Binance price as the reference for up/down contracts. But the contract settles based on whether the price is up/down from the **interval start price**. We need to infer or track the interval start price and use it as the reference.

**Files:**
- Modify: `arb_bot/crypto/market_scanner.py` — add `interval_start_time` to CryptoMarketMeta
- Modify: `arb_bot/crypto/price_feed.py` — add `get_price_at_time()` method
- Modify: `arb_bot/crypto/engine.py:514-525` — use interval start price as reference for up/down
- Modify: `arb_bot/crypto/paper_test.py` — pass `open_time` from Kalshi API into meta, re-enable up/down
- Create: `arb_bot/tests/test_crypto_updown_ref.py`

### Step 1: Write failing tests

```python
# arb_bot/tests/test_crypto_updown_ref.py
"""Tests for up/down reference price fix."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.market_scanner import (
    CryptoMarket, CryptoMarketMeta, CryptoMarketQuote, parse_ticker,
)
from arb_bot.crypto.price_feed import PriceFeed, PriceTick
from arb_bot.crypto.price_model import PriceModel


def _make_settings(**overrides) -> CryptoSettings:
    defaults = dict(
        enabled=True, paper_mode=True, bankroll=1000.0,
        mc_num_paths=500, min_edge_pct=0.05, min_edge_cents=0.02,
        max_model_uncertainty=0.50, kelly_fraction_cap=0.10,
        max_position_per_market=100.0, max_concurrent_positions=10,
        scan_interval_seconds=0.01, paper_slippage_cents=0.0,
        confidence_level=0.95, mc_vol_window_minutes=30,
        price_feed_symbols=["btcusdt"], symbols=["KXBTC"],
        min_minutes_to_expiry=1, max_minutes_to_expiry=60,
        allowed_directions=["above", "below", "up", "down"],
        model_uncertainty_multiplier=1.0,
    )
    defaults.update(overrides)
    return CryptoSettings(**defaults)


class TestIntervalStartPrice:
    """PriceFeed can look up the price at a past timestamp."""

    def test_get_price_at_time_exact(self):
        feed = PriceFeed(symbols=["btcusdt"])
        t0 = 1000.0
        feed.inject_tick(PriceTick("btcusdt", 69000.0, t0, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 70000.0, t0 + 60, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 71000.0, t0 + 120, 1.0))
        # Get price closest to t0 + 60
        price = feed.get_price_at_time("btcusdt", t0 + 60)
        assert price == pytest.approx(70000.0)

    def test_get_price_at_time_interpolates_nearest(self):
        """When no exact tick exists, return the closest tick."""
        feed = PriceFeed(symbols=["btcusdt"])
        t0 = 1000.0
        feed.inject_tick(PriceTick("btcusdt", 69000.0, t0, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 71000.0, t0 + 120, 1.0))
        # Query t0 + 50 — closer to first tick
        price = feed.get_price_at_time("btcusdt", t0 + 50)
        assert price == pytest.approx(69000.0)

    def test_get_price_at_time_none_if_no_data(self):
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_price_at_time("btcusdt", time.time()) is None


class TestMetaIntervalStart:
    """CryptoMarketMeta carries interval_start_time for up/down contracts."""

    def test_meta_has_interval_start_time(self):
        now = datetime.now(timezone.utc)
        meta = CryptoMarketMeta(
            underlying="BTC", interval="15min",
            expiry=now + timedelta(minutes=10),
            strike=None, direction="up", series_ticker="KXBTC15M",
            interval_start_time=now - timedelta(minutes=5),
        )
        assert meta.interval_start_time is not None
        assert meta.interval_start_time < meta.expiry


class TestUpDownUsesRefPrice:
    """Engine uses interval start price (not current price) for up/down."""

    def test_up_contract_uses_start_price(self):
        """If BTC started at 69000, is now 70000, model should compute
        P(terminal > 69000), not P(terminal > 70000)."""
        settings = _make_settings(mc_num_paths=2000, ofi_enabled=False)
        engine = CryptoEngine(settings)

        now_ts = time.time()
        interval_start_ts = now_ts - 300  # 5 min ago

        # Price was 69000 at interval start, now 70000
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 69000.0, interval_start_ts, 1.0)
        )
        for i in range(60):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 69000.0 + i * 17, interval_start_ts + i * 5, 1.0)
            )
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 70000.0, now_ts, 1.0)
        )

        # Create an "up" market with known interval start time
        now_dt = datetime.now(timezone.utc)
        start_dt = now_dt - timedelta(seconds=300)
        meta = CryptoMarketMeta(
            underlying="BTC", interval="15min",
            expiry=now_dt + timedelta(minutes=10),
            strike=None, direction="up", series_ticker="KXBTC15M",
            interval_start_time=start_dt,
        )
        market = CryptoMarket(ticker="KXBTC15M-TEST-U1", meta=meta)
        quote = CryptoMarketQuote(
            market=market,
            yes_buy_price=0.80, no_buy_price=0.20,
            yes_buy_size=100, no_buy_size=100,
            yes_bid_price=0.79, no_bid_price=0.19,
            time_to_expiry_minutes=10.0,
            implied_probability=0.80,
        )

        probs = engine._compute_model_probabilities([quote])
        assert market.ticker in probs
        prob = probs[market.ticker]
        # Since current price (70000) is already above start price (69000),
        # the model should predict high probability (>75%) of ending above 69000
        assert prob.probability > 0.75

    def test_down_contract_with_price_already_below(self):
        """If price dropped from 70000 to 68000, P(down) should be high."""
        settings = _make_settings(mc_num_paths=2000, ofi_enabled=False)
        engine = CryptoEngine(settings)

        now_ts = time.time()
        interval_start_ts = now_ts - 300

        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 70000.0, interval_start_ts, 1.0)
        )
        for i in range(60):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0 - i * 33, interval_start_ts + i * 5, 1.0)
            )
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 68000.0, now_ts, 1.0)
        )

        now_dt = datetime.now(timezone.utc)
        start_dt = now_dt - timedelta(seconds=300)
        meta = CryptoMarketMeta(
            underlying="BTC", interval="15min",
            expiry=now_dt + timedelta(minutes=10),
            strike=None, direction="down", series_ticker="KXBTC15M",
            interval_start_time=start_dt,
        )
        market = CryptoMarket(ticker="KXBTC15M-TEST-D1", meta=meta)
        quote = CryptoMarketQuote(
            market=market,
            yes_buy_price=0.80, no_buy_price=0.20,
            yes_buy_size=100, no_buy_size=100,
            yes_bid_price=0.79, no_bid_price=0.19,
            time_to_expiry_minutes=10.0,
            implied_probability=0.80,
        )

        probs = engine._compute_model_probabilities([quote])
        assert market.ticker in probs
        prob = probs[market.ticker]
        # Price already 2000 below start → P(down) should be high
        assert prob.probability > 0.75
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest arb_bot/tests/test_crypto_updown_ref.py -v`
Expected: FAIL — `interval_start_time` not a field, `get_price_at_time` doesn't exist

### Step 3: Implement

**market_scanner.py** — add `interval_start_time` field to `CryptoMarketMeta`:

```python
@dataclass(frozen=True)
class CryptoMarketMeta:
    underlying: str
    interval: str
    expiry: datetime
    strike: float | None
    direction: str
    series_ticker: str
    interval_index: int | None = None
    interval_start_time: datetime | None = None  # For up/down: start of the interval
```

**price_feed.py** — add `get_price_at_time()`:

```python
def get_price_at_time(
    self, symbol: str, target_timestamp: float,
) -> float | None:
    """Return the price closest to target_timestamp.

    Searches the tick history for the tick with the smallest
    absolute time difference from target_timestamp.
    """
    sym = symbol.lower()
    ticks = self._ticks.get(sym)
    if not ticks:
        return None

    best_tick = None
    best_diff = float("inf")
    for tick in ticks:
        diff = abs(tick.timestamp - target_timestamp)
        if diff < best_diff:
            best_diff = diff
            best_tick = tick

    return best_tick.price if best_tick is not None else None
```

**engine.py** — modify up/down handling in `_compute_model_probabilities` (lines 514-525):

```python
elif direction == "up":
    # Use interval start price as reference, not current price
    ref_price = current_price  # fallback
    if mq.market.meta.interval_start_time is not None:
        start_ts = mq.market.meta.interval_start_time.timestamp()
        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
        if looked_up is not None:
            ref_price = looked_up
    prob = self._price_model.probability_up(paths, ref_price)
elif direction == "down":
    ref_price = current_price  # fallback
    if mq.market.meta.interval_start_time is not None:
        start_ts = mq.market.meta.interval_start_time.timestamp()
        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
        if looked_up is not None:
            ref_price = looked_up
    up = self._price_model.probability_up(paths, ref_price)
    prob = ProbabilityEstimate(
        probability=1.0 - up.probability,
        ci_lower=1.0 - up.ci_upper,
        ci_upper=1.0 - up.ci_lower,
        uncertainty=up.uncertainty,
        num_paths=up.num_paths,
    )
```

**paper_test.py** — update `kalshi_raw_to_quote` to extract `open_time` from the Kalshi API response and compute `interval_start_time`:

```python
# In kalshi_raw_to_quote, after parsing close_time:
# For 15M/1H markets, the interval start is close_time minus the interval duration
interval_start = None
if interval == "15min":
    interval_start = close_time - timedelta(minutes=15)
elif interval == "1hour":
    interval_start = close_time - timedelta(hours=1)
# Also try to use open_time from API if available
open_str = raw.get("open_time", "")
if open_str:
    try:
        open_str = open_str.replace("Z", "+00:00")
        interval_start = datetime.fromisoformat(open_str)
    except (ValueError, TypeError):
        pass

# In meta construction:
meta = CryptoMarketMeta(
    ...,
    interval_start_time=interval_start,
)
```

Re-enable up/down in paper_test settings:
```python
allowed_directions=["above", "below", "up", "down"],
```

Update header print:
```python
print(f"  Directions:     above, below, up, down (ref-price fixed)")
```

### Step 4: Run tests

Run: `python3 -m pytest arb_bot/tests/test_crypto_updown_ref.py -v`
Expected: PASS

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: PASS (existing up/down tests use `interval_start_time=None` → falls back to current_price → same behavior)

### Step 5: Commit

```bash
git add arb_bot/crypto/market_scanner.py arb_bot/crypto/price_feed.py arb_bot/crypto/engine.py arb_bot/crypto/paper_test.py arb_bot/tests/test_crypto_updown_ref.py
git commit -m "fix(crypto): use interval start price for up/down contracts"
```

---

## Task 4: WebSocket AggTrades Streaming for OFI Data Quality

Replace REST polling of 100 aggTrades per cycle with a persistent WebSocket stream that feeds trades continuously. This eliminates the sparse OFI readings (OFI swinging to ±1.0 because only a few trades in the window).

**Files:**
- Modify: `arb_bot/crypto/price_feed.py` — add `connect_agg_trades_ws()` and stream handler
- Modify: `arb_bot/crypto/paper_test.py` — start WS stream at startup, remove per-cycle REST aggTrades polling
- Create: `arb_bot/tests/test_crypto_agg_trades_ws.py`

### Step 1: Write failing tests

```python
# arb_bot/tests/test_crypto_agg_trades_ws.py
"""Tests for aggTrades WebSocket streaming."""

from __future__ import annotations

import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arb_bot.crypto.price_feed import PriceFeed, PriceTick


class TestAggTradesHandler:
    """The _handle_agg_trade_message method parses WS messages."""

    def test_parses_buy_trade(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = {
            "e": "aggTrade", "s": "BTCUSDT",
            "p": "70000.50", "q": "0.5",
            "T": int(time.time() * 1000),
            "m": False,  # buyer is NOT maker → buy
        }
        feed._handle_agg_trade_message(msg)
        assert feed.get_current_price("btcusdt") == pytest.approx(70000.50)
        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol > 0

    def test_parses_sell_trade(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = {
            "e": "aggTrade", "s": "BTCUSDT",
            "p": "70000.50", "q": "0.5",
            "T": int(time.time() * 1000),
            "m": True,  # buyer IS maker → sell
        }
        feed._handle_agg_trade_message(msg)
        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert sell_vol > 0

    def test_ofi_from_stream_data(self):
        """Multiple trades should produce a meaningful OFI."""
        feed = PriceFeed(symbols=["btcusdt"])
        now_ms = int(time.time() * 1000)
        # 7 buys, 3 sells
        for i in range(7):
            feed._handle_agg_trade_message({
                "e": "aggTrade", "s": "BTCUSDT",
                "p": "70000.0", "q": "1.0",
                "T": now_ms + i * 100, "m": False,
            })
        for i in range(3):
            feed._handle_agg_trade_message({
                "e": "aggTrade", "s": "BTCUSDT",
                "p": "70000.0", "q": "1.0",
                "T": now_ms + 700 + i * 100, "m": True,
            })
        ofi = feed.get_ofi("btcusdt", window_seconds=60)
        # OFI = (7 - 3) / (7 + 3) = 0.4
        assert ofi == pytest.approx(0.4)

    def test_unknown_symbol_ignored(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = {
            "e": "aggTrade", "s": "XRPUSDT",
            "p": "0.50", "q": "100",
            "T": int(time.time() * 1000), "m": False,
        }
        feed._handle_agg_trade_message(msg)
        assert feed.get_current_price("xrpusdt") is None

    def test_malformed_message_handled(self):
        feed = PriceFeed(symbols=["btcusdt"])
        feed._handle_agg_trade_message({})  # no crash
        feed._handle_agg_trade_message({"e": "aggTrade"})  # partial


class TestAggTradesWSConfig:
    """Config for aggTrades streaming."""

    def test_has_agg_trades_ws_field(self):
        from arb_bot.crypto.config import CryptoSettings
        s = CryptoSettings()
        assert hasattr(s, 'agg_trades_ws_enabled')
        assert s.agg_trades_ws_enabled is True
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest arb_bot/tests/test_crypto_agg_trades_ws.py -v`
Expected: FAIL — `_handle_agg_trade_message` doesn't exist

### Step 3: Implement

**price_feed.py** — add `_handle_agg_trade_message` method:

```python
def _handle_agg_trade_message(self, msg: dict) -> None:
    """Parse a Binance aggTrade WebSocket message and inject as tick."""
    try:
        symbol = msg.get("s", "").lower()
        if symbol not in self._symbols_set:
            return
        price = float(msg["p"])
        qty = float(msg["q"])
        ts = float(msg["T"]) / 1000.0
        is_buyer_maker = msg.get("m")
    except (KeyError, ValueError, TypeError):
        return

    tick = PriceTick(
        symbol=symbol, price=price, timestamp=ts,
        volume=qty, is_buyer_maker=is_buyer_maker,
    )
    self._process_tick(tick)
```

Also refactor `inject_tick` and `_handle_message` to both call a shared `_process_tick`:

```python
def _process_tick(self, tick: PriceTick) -> None:
    """Common tick processing for both inject and WS paths."""
    sym = tick.symbol.lower()
    self._ticks[sym].append(tick)
    self._current_price[sym] = tick.price
    if tick.is_buyer_maker is not None:
        is_buy = not tick.is_buyer_maker
        self._buy_sells[sym].append((tick.timestamp, tick.volume, is_buy))
```

Add `_symbols_set` in `__init__` for O(1) lookup:

```python
self._symbols_set: set[str] = {s.lower() for s in symbols}
```

Add async method for connecting aggTrades WS:

```python
async def connect_agg_trades_ws(self) -> None:
    """Connect to Binance aggTrade WebSocket stream for all symbols."""
    import websockets
    streams = "/".join(f"{s}@aggTrade" for s in sorted(self._symbols_set))
    url = f"wss://stream.binance.us:9443/stream?streams={streams}"
    while True:
        try:
            async with websockets.connect(url) as ws:
                async for raw_msg in ws:
                    import json
                    data = json.loads(raw_msg)
                    payload = data.get("data", data)
                    self._handle_agg_trade_message(payload)
        except Exception:
            await asyncio.sleep(5)
```

**config.py** — add field:

```python
agg_trades_ws_enabled: bool = True
```

**paper_test.py** — start WS task at startup, remove per-cycle REST aggTrades loop:

In the main loop setup (after bootstrap), add:

```python
# Start aggTrades WebSocket for continuous OFI data
agg_trades_task = None
if settings.agg_trades_ws_enabled:
    agg_trades_task = asyncio.create_task(engine.price_feed.connect_agg_trades_ws())
```

Remove the per-cycle block:
```python
# DELETE THIS BLOCK:
# # Refresh aggTrades for live OFI updates
# for sym in binance_symbols:
#     trades = await fetch_binance_agg_trades(sym, 100, client)
#     ...
```

At the end of the test, cancel the task:
```python
if agg_trades_task:
    agg_trades_task.cancel()
    try:
        await agg_trades_task
    except asyncio.CancelledError:
        pass
```

### Step 4: Run tests

Run: `python3 -m pytest arb_bot/tests/test_crypto_agg_trades_ws.py -v`
Expected: PASS

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add arb_bot/crypto/price_feed.py arb_bot/crypto/config.py arb_bot/crypto/paper_test.py arb_bot/tests/test_crypto_agg_trades_ws.py
git commit -m "feat(crypto): WebSocket aggTrades streaming for continuous OFI"
```

---

## Task 5: Per-Cycle CSV Data Logger

Log OFI, volume rate, returns, Hawkes intensity, and model outputs to CSV every cycle for offline calibration of theta, beta, and model validation.

**Files:**
- Create: `arb_bot/crypto/cycle_logger.py`
- Modify: `arb_bot/crypto/engine.py` — call logger at end of each cycle
- Create: `arb_bot/tests/test_crypto_cycle_logger.py`

### Step 1: Write failing tests

```python
# arb_bot/tests/test_crypto_cycle_logger.py
"""Tests for per-cycle CSV data logger."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pytest

from arb_bot.crypto.cycle_logger import CycleLogger, CycleSnapshot


class TestCycleSnapshot:
    """CycleSnapshot data class captures per-cycle state."""

    def test_snapshot_fields(self):
        snap = CycleSnapshot(
            timestamp=time.time(),
            cycle=1,
            symbol="btcusdt",
            price=70000.0,
            ofi=0.35,
            volume_rate_short=150.0,
            volume_rate_long=100.0,
            activity_ratio=1.5,
            realized_vol=0.5,
            hawkes_intensity=5.2,
            num_edges=2,
            num_positions=1,
            session_pnl=-10.5,
            bankroll=490.0,
        )
        assert snap.ofi == 0.35
        assert snap.hawkes_intensity == 5.2


class TestCycleLogger:
    """CycleLogger writes snapshots to CSV."""

    def test_creates_csv_file(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        snap = CycleSnapshot(
            timestamp=1000.0, cycle=1, symbol="btcusdt",
            price=70000.0, ofi=0.3, volume_rate_short=100.0,
            volume_rate_long=80.0, activity_ratio=1.25,
            realized_vol=0.5, hawkes_intensity=3.0,
            num_edges=1, num_positions=0,
            session_pnl=0.0, bankroll=500.0,
        )
        logger.log(snap)
        logger.flush()
        assert path.exists()

    def test_csv_has_header_and_row(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        snap = CycleSnapshot(
            timestamp=1000.0, cycle=1, symbol="btcusdt",
            price=70000.0, ofi=0.3, volume_rate_short=100.0,
            volume_rate_long=80.0, activity_ratio=1.25,
            realized_vol=0.5, hawkes_intensity=3.0,
            num_edges=1, num_positions=0,
            session_pnl=0.0, bankroll=500.0,
        )
        logger.log(snap)
        logger.flush()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "btcusdt"
        assert float(rows[0]["ofi"]) == pytest.approx(0.3)
        assert float(rows[0]["hawkes_intensity"]) == pytest.approx(3.0)

    def test_multiple_snapshots_append(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        for i in range(5):
            logger.log(CycleSnapshot(
                timestamp=1000.0 + i, cycle=i + 1, symbol="btcusdt",
                price=70000.0 + i * 100, ofi=0.1 * i,
                volume_rate_short=100.0, volume_rate_long=80.0,
                activity_ratio=1.25, realized_vol=0.5,
                hawkes_intensity=3.0 + i, num_edges=0,
                num_positions=0, session_pnl=0.0, bankroll=500.0,
            ))
        logger.flush()

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_close_flushes(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.log(CycleSnapshot(
            timestamp=1000.0, cycle=1, symbol="btcusdt",
            price=70000.0, ofi=0.0, volume_rate_short=0.0,
            volume_rate_long=0.0, activity_ratio=1.0,
            realized_vol=0.5, hawkes_intensity=3.0,
            num_edges=0, num_positions=0,
            session_pnl=0.0, bankroll=500.0,
        ))
        logger.close()
        assert path.exists()
        with open(path) as f:
            assert len(list(csv.DictReader(f))) == 1
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest arb_bot/tests/test_crypto_cycle_logger.py -v`
Expected: FAIL — module doesn't exist

### Step 3: Implement

**arb_bot/crypto/cycle_logger.py:**

```python
"""Per-cycle CSV data logger for offline calibration.

Logs OFI, volume rates, Hawkes intensity, model outputs, and PnL
every cycle. Use for calibrating theta (power-law exponent),
beta (Hawkes decay), and validating model predictions.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from typing import Optional, TextIO


@dataclass
class CycleSnapshot:
    """State captured at the end of each engine cycle."""

    timestamp: float
    cycle: int
    symbol: str
    price: float
    ofi: float
    volume_rate_short: float
    volume_rate_long: float
    activity_ratio: float
    realized_vol: float
    hawkes_intensity: float
    num_edges: int
    num_positions: int
    session_pnl: float
    bankroll: float


_FIELDS = [f.name for f in fields(CycleSnapshot)]


class CycleLogger:
    """Append-only CSV logger for CycleSnapshot records."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        write_header = not os.path.exists(self._path) or os.path.getsize(self._path) == 0
        self._file = open(self._path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDS)
        if write_header:
            self._writer.writeheader()

    def log(self, snap: CycleSnapshot) -> None:
        """Append a snapshot row to the CSV."""
        self._ensure_open()
        assert self._writer is not None
        row = {f: getattr(snap, f) for f in _FIELDS}
        self._writer.writerow(row)

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
```

**engine.py** — add logging call at end of `_run_cycle` and `run_cycle_with_quotes`.

In `__init__`:
```python
from arb_bot.crypto.cycle_logger import CycleLogger, CycleSnapshot

self._cycle_logger: Optional[CycleLogger] = None
```

Add `set_cycle_logger` method:
```python
def set_cycle_logger(self, logger: CycleLogger) -> None:
    self._cycle_logger = logger
```

Add `_log_cycle` method:
```python
def _log_cycle(self, edges_count: int) -> None:
    if self._cycle_logger is None:
        return
    import time as _time
    now = _time.monotonic()
    for binance_sym in _KALSHI_TO_BINANCE.values():
        price = self._price_feed.get_current_price(binance_sym)
        if price is None:
            continue
        ofi = self._price_feed.get_ofi(binance_sym, self._settings.ofi_window_seconds)
        sr = self._price_feed.get_volume_flow_rate(
            binance_sym, self._settings.activity_scaling_short_window_seconds,
        )
        lr = self._price_feed.get_volume_flow_rate(
            binance_sym, self._settings.activity_scaling_long_window_seconds,
        )
        ar = sr / lr if lr > 0 and sr > 0 else 1.0
        returns = self._price_feed.get_returns(binance_sym, 60, 5)
        vol = float(np.std(returns)) if len(returns) >= 2 else 0.0
        hi = self._hawkes.intensity(now) if self._settings.hawkes_enabled else self._settings.mc_jump_intensity

        self._cycle_logger.log(CycleSnapshot(
            timestamp=_time.time(),
            cycle=self._cycle_count,
            symbol=binance_sym,
            price=price,
            ofi=ofi,
            volume_rate_short=sr,
            volume_rate_long=lr,
            activity_ratio=ar,
            realized_vol=vol,
            hawkes_intensity=hi,
            num_edges=edges_count,
            num_positions=len(self._positions),
            session_pnl=self._session_pnl,
            bankroll=self._bankroll,
        ))
    self._cycle_logger.flush()
```

Call `self._log_cycle(len(edges))` at the end of `_run_cycle()` and `run_cycle_with_quotes()`.

### Step 4: Run tests

Run: `python3 -m pytest arb_bot/tests/test_crypto_cycle_logger.py -v`
Expected: PASS

Run: `python3 -m pytest arb_bot/tests/test_crypto_engine.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add arb_bot/crypto/cycle_logger.py arb_bot/crypto/engine.py arb_bot/tests/test_crypto_cycle_logger.py
git commit -m "feat(crypto): per-cycle CSV data logger for offline calibration"
```

---

## Task 6: Paper Test Integration + Kalshi Execution Awareness

Update the paper test to use all new features: power-law OFI, Hawkes jumps, reference-price fix for up/down, aggTrades WS, and CSV logging. Also add Kalshi-specific execution awareness: model the thin order book by incorporating spread into edge calculation.

**Files:**
- Modify: `arb_bot/crypto/paper_test.py` — integrate all new features, add spread-aware edge display
- Modify: `arb_bot/crypto/edge_detector.py` — add `spread_adjusted_edge` field to CryptoEdge
- Create: `arb_bot/tests/test_crypto_spread_edge.py`

### Step 1: Write failing tests

```python
# arb_bot/tests/test_crypto_spread_edge.py
"""Tests for spread-adjusted edge calculation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arb_bot.crypto.edge_detector import CryptoEdge, EdgeDetector, compute_implied_probability
from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketMeta, CryptoMarketQuote
from arb_bot.crypto.price_model import ProbabilityEstimate


def _make_quote(
    yes_buy: float = 0.40, no_buy: float = 0.60,
    yes_bid: float = 0.38, no_bid: float = 0.58,
    tte: float = 10.0,
) -> CryptoMarketQuote:
    now = datetime.now(timezone.utc)
    meta = CryptoMarketMeta(
        underlying="BTC", interval="daily",
        expiry=now + timedelta(minutes=tte),
        strike=70000.0, direction="above",
        series_ticker="KXBTCD",
    )
    market = CryptoMarket(ticker="KXBTCD-TEST-T70000", meta=meta)
    implied = compute_implied_probability(
        (yes_bid + yes_buy) / 2, (no_bid + no_buy) / 2,
    )
    return CryptoMarketQuote(
        market=market,
        yes_buy_price=yes_buy, no_buy_price=no_buy,
        yes_buy_size=100, no_buy_size=100,
        yes_bid_price=yes_bid, no_bid_price=no_bid,
        time_to_expiry_minutes=tte, implied_probability=implied,
    )


class TestSpreadAdjustedEdge:
    """Edge detector computes spread-adjusted edge for thin books."""

    def test_edge_has_spread_cost(self):
        """CryptoEdge should carry the half-spread cost."""
        det = EdgeDetector(
            min_edge_pct=0.01, min_edge_cents=0.01,
            max_model_uncertainty=0.50,
        )
        quote = _make_quote(yes_buy=0.40, yes_bid=0.36)
        # YES spread = 0.40 - 0.36 = 0.04, half-spread = 0.02
        model_probs = {
            quote.market.ticker: ProbabilityEstimate(
                probability=0.55, ci_lower=0.50, ci_upper=0.60,
                uncertainty=0.05, num_paths=1000,
            ),
        }
        edges = det.detect_edges([quote], model_probs)
        assert len(edges) >= 1
        edge = edges[0]
        assert hasattr(edge, 'spread_cost')
        assert edge.spread_cost > 0

    def test_wide_spread_kills_edge(self):
        """A wide spread should reduce the effective edge."""
        det = EdgeDetector(
            min_edge_pct=0.05, min_edge_cents=0.02,
            max_model_uncertainty=0.50,
        )
        # Tight spread: yes_buy=0.40, yes_bid=0.39 → spread=0.01
        tight_quote = _make_quote(yes_buy=0.40, yes_bid=0.39, no_buy=0.60, no_bid=0.59)
        # Wide spread: yes_buy=0.40, yes_bid=0.30 → spread=0.10
        wide_quote = _make_quote(yes_buy=0.40, yes_bid=0.30, no_buy=0.60, no_bid=0.50)

        model_probs_tight = {
            tight_quote.market.ticker: ProbabilityEstimate(
                probability=0.48, ci_lower=0.43, ci_upper=0.53,
                uncertainty=0.05, num_paths=1000,
            ),
        }
        model_probs_wide = {
            wide_quote.market.ticker: ProbabilityEstimate(
                probability=0.48, ci_lower=0.43, ci_upper=0.53,
                uncertainty=0.05, num_paths=1000,
            ),
        }

        edges_tight = det.detect_edges([tight_quote], model_probs_tight)
        edges_wide = det.detect_edges([wide_quote], model_probs_wide)

        # The wide-spread market should have fewer or smaller edges
        if edges_tight and edges_wide:
            assert edges_wide[0].spread_cost > edges_tight[0].spread_cost
```

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest arb_bot/tests/test_crypto_spread_edge.py -v`
Expected: FAIL — `spread_cost` not on CryptoEdge

### Step 3: Implement

**edge_detector.py** — add `spread_cost` field to CryptoEdge:

```python
@dataclass(frozen=True)
class CryptoEdge:
    market: CryptoMarket
    model_prob: ProbabilityEstimate
    market_implied_prob: float
    edge: float
    edge_cents: float
    side: str
    recommended_price: float
    model_uncertainty: float
    time_to_expiry_minutes: float
    yes_buy_price: float
    no_buy_price: float
    blended_probability: float = 0.0
    spread_cost: float = 0.0  # Half-spread in dollar terms
```

In `detect_edges()`, compute spread_cost when building each edge:

```python
# After determining side:
if side == "yes":
    bid = mq.yes_bid_price if mq.yes_bid_price else mq.yes_buy_price
    half_spread = (mq.yes_buy_price - bid) / 2.0
else:
    bid = mq.no_bid_price if mq.no_bid_price else mq.no_buy_price
    half_spread = (mq.no_buy_price - bid) / 2.0

# Include in CryptoEdge construction:
spread_cost=max(0.0, half_spread),
```

**paper_test.py** — full integration update:

1. Set up CycleLogger:
```python
from arb_bot.crypto.cycle_logger import CycleLogger
log_path = f"arb_bot/output/paper_v3_{int(time.time())}.csv"
logger = CycleLogger(log_path)
engine.set_cycle_logger(logger)
```

2. Start aggTrades WS (already done in Task 4).

3. Update header to show "Paper Test v3" with all features.

4. Add spread display in edge output:
```python
f"spread={e.spread_cost*100:.1f}¢"
```

5. Close logger at end:
```python
logger.close()
print(f"  Cycle data logged to: {log_path}")
```

### Step 4: Run tests

Run: `python3 -m pytest arb_bot/tests/test_crypto_spread_edge.py -v`
Expected: PASS

Run: `python3 -m pytest arb_bot/tests/ -q --ignore=arb_bot/tests/test_engine_multi_venue.py`
Expected: All pass

### Step 5: Commit

```bash
git add arb_bot/crypto/edge_detector.py arb_bot/crypto/paper_test.py arb_bot/tests/test_crypto_spread_edge.py
git commit -m "feat(crypto): spread-aware edge + paper test v3 integration"
```

---

## Post-Implementation Checklist

After all 6 tasks:

1. Run full test suite: `python3 -m pytest arb_bot/tests/ -q --ignore=arb_bot/tests/test_engine_multi_venue.py`
2. Expected: All existing tests pass + ~40 new tests across 4 new test files
3. Run a short paper test (5 min) to validate integration
4. Commit any final cleanup
5. Push and create PR
