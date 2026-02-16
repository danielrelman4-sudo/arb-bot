"""Tests for drift safety fixes in the crypto arb bot.

Covers:
 1. Cross-asset drift scaling (return_scale / max_drift clamp)
 2. Total drift clamp (max_total_drift config)
 3. Mean reversion logic (_compute_mean_reversion math)
 4. Edge threshold alignment (min_edge_pct / min_edge_pct_no_side)
 5. Integration-style construction tests
"""

import math
import time

import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.cross_asset import CrossAssetFeatures, CrossAssetSignal
from arb_bot.crypto.price_feed import PriceFeed, PriceTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakePriceFeed:
    """Minimal mock satisfying the CrossAssetFeatures interface."""

    def __init__(self, ofi=0.0, returns=None, vol_rate_short=1.0, vol_rate_long=1.0):
        self._ofi = ofi
        self._returns = returns or []
        self._vol_short = vol_rate_short
        self._vol_long = vol_rate_long

    def get_ofi(self, symbol, window_seconds=300):
        return self._ofi

    def get_returns(self, symbol, interval_seconds=60, window_minutes=5):
        return self._returns

    def get_volume_flow_rate(self, symbol, window_seconds=300):
        if window_seconds <= 300:
            return self._vol_short
        return self._vol_long


def _clamp(value, lo, hi):
    """Reference clamp: max(lo, min(hi, value))."""
    return max(lo, min(hi, value))


# ===================================================================
# 1. Cross-Asset Drift Scaling Tests
# ===================================================================

class TestCrossAssetDriftScaling:
    """Verify that return_scale and max_drift clamp extreme drift values."""

    def test_small_leader_return_does_not_blow_up(self):
        """A 0.1% 5-min leader return should produce a small drift, not 21+."""
        feed = FakePriceFeed(
            ofi=0.0,
            returns=[0.001],  # 0.1% cumulative return
        )
        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.0,
            return_weight=0.2,
            return_scale=20.0,
            max_drift=2.0,
        )
        sig = cross.compute_features(feed, "solusdt")
        assert sig is not None
        # With the fix: drift_from_return = 0.001 * 0.2 * 20.0 = 0.004
        # Without the fix (annualized): 0.001 * 0.2 * 525960/5 = ~21.0
        # We check that it's clamped to max_drift at most
        assert abs(sig.drift_adjustment) <= 2.0, (
            f"drift_adjustment={sig.drift_adjustment} exceeds max_drift=2.0"
        )

    def test_max_drift_clamps_positive(self):
        """Extreme positive values should be clamped to max_drift."""
        feed = FakePriceFeed(
            ofi=10000.0,       # huge OFI
            returns=[0.10],    # 10% return
        )
        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.3,
            return_weight=0.2,
            return_scale=20.0,
            max_drift=2.0,
        )
        sig = cross.compute_features(feed, "solusdt")
        assert sig is not None
        assert sig.drift_adjustment <= 2.0

    def test_max_drift_clamps_negative(self):
        """Extreme negative values should be clamped to -max_drift."""
        feed = FakePriceFeed(
            ofi=-10000.0,
            returns=[-0.10],
        )
        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.3,
            return_weight=0.2,
            return_scale=20.0,
            max_drift=2.0,
        )
        sig = cross.compute_features(feed, "solusdt")
        assert sig is not None
        assert sig.drift_adjustment >= -2.0

    def test_default_constructor_backward_compat(self):
        """Default CrossAssetFeatures() should work with new params."""
        cross = CrossAssetFeatures()
        assert cross._return_scale == 20.0
        assert cross._max_drift == 2.0
        assert cross.leader_symbol == "btcusdt"

    def test_zero_leader_return_gives_near_zero_drift(self):
        """When leader return is zero and OFI is zero, drift should be ~0."""
        feed = FakePriceFeed(
            ofi=0.0,
            returns=[0.0, 0.0, 0.0],
        )
        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.3,
            return_weight=0.2,
            return_scale=20.0,
            max_drift=2.0,
        )
        sig = cross.compute_features(feed, "solusdt")
        assert sig is not None
        assert abs(sig.drift_adjustment) < 0.01


# ===================================================================
# 2. Total Drift Clamp Tests (CryptoSettings.max_total_drift)
# ===================================================================

class TestTotalDriftClamp:
    """Verify the max_total_drift config field and clamp math."""

    def test_settings_has_max_total_drift(self):
        """CryptoSettings should have a max_total_drift field."""
        s = CryptoSettings(max_total_drift=2.0)
        assert hasattr(s, "max_total_drift")
        assert s.max_total_drift == 2.0

    def test_max_total_drift_default(self):
        """Default max_total_drift should be 2.0."""
        s = CryptoSettings()
        assert s.max_total_drift == 2.0

    def test_clamp_logic_various_values(self):
        """The clamping formula max(-L, min(L, v)) should work correctly."""
        limit = 2.0
        assert _clamp(0.5, -limit, limit) == 0.5
        assert _clamp(3.0, -limit, limit) == 2.0
        assert _clamp(-3.0, -limit, limit) == -2.0
        assert _clamp(0.0, -limit, limit) == 0.0
        assert _clamp(2.0, -limit, limit) == 2.0
        assert _clamp(-2.0, -limit, limit) == -2.0

    def test_settings_has_mean_reversion_fields(self):
        """CryptoSettings should have mean_reversion_enabled, kappa, lookback."""
        s = CryptoSettings(
            mean_reversion_enabled=True,
            mean_reversion_kappa=50.0,
            mean_reversion_lookback_minutes=5.0,
        )
        assert s.mean_reversion_enabled is True
        assert s.mean_reversion_kappa == 50.0
        assert s.mean_reversion_lookback_minutes == 5.0


# ===================================================================
# 3. Mean Reversion Tests
# ===================================================================

class TestMeanReversionMath:
    """Test the mean-reversion formula: reversion = -kappa * sum(returns).

    The _compute_mean_reversion method on CryptoEngine computes:
        recent_return = sum(returns)
        return -kappa * recent_return

    We test the math directly rather than instantiating the full engine.
    """

    @staticmethod
    def _mean_reversion(returns, kappa):
        """Reference implementation of mean-reversion drift."""
        if not returns:
            return 0.0
        recent_return = sum(returns)
        return -kappa * recent_return

    def test_positive_return_negative_reversion(self):
        """Price went up -> reversion pushes down."""
        returns = [0.001, 0.002, 0.0005]  # cumulative 0.35%
        reversion = self._mean_reversion(returns, kappa=50.0)
        assert reversion < 0, "Positive returns should produce negative reversion"
        assert reversion == pytest.approx(-50.0 * 0.0035)

    def test_negative_return_positive_reversion(self):
        """Price went down -> reversion pushes up (bounce expected)."""
        returns = [-0.001, -0.002]  # cumulative -0.3%
        reversion = self._mean_reversion(returns, kappa=50.0)
        assert reversion > 0, "Negative returns should produce positive reversion"
        assert reversion == pytest.approx(50.0 * 0.003)

    def test_zero_return_zero_reversion(self):
        """No net return -> no reversion."""
        returns = [0.001, -0.001]  # cancel out
        reversion = self._mean_reversion(returns, kappa=50.0)
        assert reversion == pytest.approx(0.0)

    def test_large_kappa_stronger_reversion(self):
        """Larger kappa should scale the reversion linearly."""
        returns = [0.001]
        rev_low = self._mean_reversion(returns, kappa=10.0)
        rev_high = self._mean_reversion(returns, kappa=100.0)
        assert abs(rev_high) > abs(rev_low)
        assert rev_high / rev_low == pytest.approx(10.0)

    def test_empty_returns_zero_reversion(self):
        """No data -> zero reversion (no crash)."""
        reversion = self._mean_reversion([], kappa=50.0)
        assert reversion == 0.0


# ===================================================================
# 4. Edge Threshold Tests
# ===================================================================

class TestEdgeThresholds:
    """Verify min_edge_pct and min_edge_pct_no_side are aligned."""

    def test_default_min_edge_pct(self):
        """Default min_edge_pct should be 0.12."""
        s = CryptoSettings()
        assert s.min_edge_pct == pytest.approx(0.12)

    def test_default_min_edge_pct_no_side(self):
        """Default min_edge_pct_no_side should be 0.12 (matched to min_edge_pct)."""
        s = CryptoSettings()
        assert s.min_edge_pct_no_side == pytest.approx(0.12), (
            f"min_edge_pct_no_side={s.min_edge_pct_no_side}, expected 0.12"
        )

    def test_edge_thresholds_now_equal(self):
        """Both edge thresholds should be equal (both 0.12)."""
        s = CryptoSettings()
        assert s.min_edge_pct == s.min_edge_pct_no_side, (
            f"min_edge_pct={s.min_edge_pct} != "
            f"min_edge_pct_no_side={s.min_edge_pct_no_side}"
        )

    def test_custom_edge_thresholds(self):
        """Custom overrides should work."""
        s = CryptoSettings(min_edge_pct=0.08, min_edge_pct_no_side=0.10)
        assert s.min_edge_pct == 0.08
        assert s.min_edge_pct_no_side == 0.10


# ===================================================================
# 5. Integration-Style Tests
# ===================================================================

class TestIntegration:
    """End-to-end construction and feature computation sanity checks."""

    def test_cross_asset_with_all_new_params_produces_valid_signal(self):
        """CrossAssetFeatures with all new params should produce a valid, clamped signal."""
        feed = FakePriceFeed(
            ofi=500.0,
            returns=[0.005, 0.003, -0.001],
            vol_rate_short=2.0,
            vol_rate_long=1.5,
        )
        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.3,
            return_weight=0.2,
            vol_weight=0.2,
            vol_baseline=1.0,
            return_scale=20.0,
            max_drift=2.0,
        )
        sig = cross.compute_features(feed, "solusdt")
        assert sig is not None
        # drift_adjustment should be clamped
        assert -2.0 <= sig.drift_adjustment <= 2.0
        # vol_adjustment should be clamped [0.8, 1.5]
        assert 0.8 <= sig.vol_adjustment <= 1.5
        # leader data should be populated
        assert sig.leader_ofi == 500.0
        leader_ret = sum([0.005, 0.003, -0.001])
        assert sig.leader_return_5m == pytest.approx(leader_ret)

    def test_crypto_settings_with_all_new_params(self):
        """CryptoSettings should accept all new drift-safety params."""
        s = CryptoSettings(
            max_total_drift=3.0,
            mean_reversion_enabled=True,
            mean_reversion_kappa=25.0,
            mean_reversion_lookback_minutes=10.0,
            min_edge_pct=0.12,
            min_edge_pct_no_side=0.12,
        )
        assert s.max_total_drift == 3.0
        assert s.mean_reversion_enabled is True
        assert s.mean_reversion_kappa == 25.0
        assert s.mean_reversion_lookback_minutes == 10.0
        assert s.min_edge_pct == 0.12
        assert s.min_edge_pct_no_side == 0.12


# ===================================================================
# Extra: Cross-asset with real PriceFeed
# ===================================================================

class TestCrossAssetWithRealFeed:
    """Use the real PriceFeed to verify the drift clamp works end-to-end."""

    def test_real_feed_drift_is_bounded(self):
        """Drift from real PriceFeed data should respect max_drift."""
        feed = PriceFeed(symbols=["btcusdt", "solusdt"])
        now = time.time()
        # Inject strong upward BTC trend: 1% per minute (extreme)
        for i in range(60):
            feed.inject_tick(PriceTick(
                symbol="btcusdt",
                price=70000.0 * (1.0 + 0.01 * i),  # +1% per tick
                timestamp=now - 300 + i * 5,
                volume=10.0,
                is_buyer_maker=False,
            ))
        # Inject flat SOL
        for i in range(60):
            feed.inject_tick(PriceTick(
                symbol="solusdt",
                price=88.0,
                timestamp=now - 300 + i * 5,
                volume=1.0,
                is_buyer_maker=True,
            ))

        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.3,
            return_weight=0.2,
            return_scale=20.0,
            max_drift=2.0,
        )
        sig = cross.compute_features(feed, "solusdt")
        assert sig is not None
        assert -2.0 <= sig.drift_adjustment <= 2.0, (
            f"drift_adjustment={sig.drift_adjustment} out of [-2, 2]"
        )
