"""Tests for cross-asset feature construction (A5)."""
import time
import pytest
from arb_bot.crypto.price_feed import PriceFeed, PriceTick
from arb_bot.crypto.cross_asset import CrossAssetFeatures, CrossAssetSignal


def _make_dual_feed():
    """Create a PriceFeed with BTC and SOL data."""
    feed = PriceFeed(symbols=["btcusdt", "solusdt"])
    now = time.time()
    # BTC: trending up with strong buy OFI
    for i in range(100):
        feed.inject_tick(PriceTick(
            symbol="btcusdt",
            price=69000.0 + i * 5,
            timestamp=now - 300 + i * 3,
            volume=1.0,
            is_buyer_maker=False,  # buy pressure
        ))
    # SOL: flat
    for i in range(100):
        feed.inject_tick(PriceTick(
            symbol="solusdt",
            price=87.0,
            timestamp=now - 300 + i * 3,
            volume=0.5,
            is_buyer_maker=bool(i % 2 == 0),  # balanced
        ))
    return feed


class TestCrossAssetFeatures:
    def test_leader_excluded(self):
        """BTC pricing should not get cross-asset from itself."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "btcusdt")
        assert result is None

    def test_follower_gets_signal(self):
        """SOL should get cross-asset signal from BTC."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        assert isinstance(result, CrossAssetSignal)

    def test_leader_ofi_propagates(self):
        """When BTC has strong buy OFI, signal should reflect it."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        # BTC has all-buy ticks -> OFI should be positive
        assert result.leader_ofi > 0

    def test_drift_adjustment_positive_for_buy_pressure(self):
        """Positive leader OFI should produce positive drift adjustment."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt", ofi_weight=1.0, return_weight=0.0)
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        assert result.drift_adjustment > 0

    def test_vol_adjustment_in_bounds(self):
        """Vol adjustment should be clamped to [0.8, 1.5]."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        assert 0.8 <= result.vol_adjustment <= 1.5

    def test_cross_ofi_divergence_computed(self):
        """Should compute absolute difference between leader and target OFI."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        assert result.cross_ofi_divergence >= 0

    def test_zero_weights_give_zero_adjustment(self):
        """With all weights zero, adjustments should be neutral."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(
            leader_symbol="btcusdt",
            ofi_weight=0.0, return_weight=0.0, vol_weight=0.0,
        )
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        assert result.drift_adjustment == pytest.approx(0.0)
        assert result.vol_adjustment == pytest.approx(1.0)


class TestCrossAssetNoData:
    def test_empty_leader_returns_defaults(self):
        """When leader has no data, should return zeroed signal."""
        feed = PriceFeed(symbols=["btcusdt", "solusdt"])
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "solusdt")
        assert result is not None
        assert result.leader_ofi == 0.0
        assert result.leader_return_5m == 0.0
        assert result.vol_adjustment == pytest.approx(1.0)

    def test_unknown_target_returns_defaults(self):
        """When target has no data, should still compute leader features."""
        feed = _make_dual_feed()
        cross = CrossAssetFeatures(leader_symbol="btcusdt")
        result = cross.compute_features(feed, "ethusdt")
        assert result is not None
        assert result.leader_ofi > 0  # BTC data exists


class TestCrossAssetConfig:
    def test_custom_leader(self):
        """Should use configured leader symbol."""
        cross = CrossAssetFeatures(leader_symbol="ethusdt")
        assert cross.leader_symbol == "ethusdt"

    def test_custom_weights(self):
        """Custom weights should affect output."""
        feed = _make_dual_feed()
        cross_low = CrossAssetFeatures(leader_symbol="btcusdt", ofi_weight=0.1)
        cross_high = CrossAssetFeatures(leader_symbol="btcusdt", ofi_weight=0.9)
        r_low = cross_low.compute_features(feed, "solusdt")
        r_high = cross_high.compute_features(feed, "solusdt")
        assert r_low is not None and r_high is not None
        # Higher OFI weight -> larger drift when leader has strong OFI
        # (both should be positive since BTC has buy pressure)
        assert abs(r_high.drift_adjustment) >= abs(r_low.drift_adjustment) or \
               abs(r_high.drift_adjustment - r_low.drift_adjustment) < 0.01
