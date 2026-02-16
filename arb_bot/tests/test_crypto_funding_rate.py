"""Tests for funding rate ingestion (A2)."""
import time
import pytest
from arb_bot.crypto.funding_rate import FundingRateTracker, FundingSignal


class TestFundingRateTracker:
    def test_inject_and_get_rate(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("BTCUSDT", 0.0003, timestamp=1000.0)
        assert tracker.get_current_rate("BTCUSDT") == 0.0003

    def test_get_rate_case_insensitive(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("btcusdt", 0.0002)
        assert tracker.get_current_rate("BTCUSDT") is not None

    def test_get_rate_unknown_symbol(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        assert tracker.get_current_rate("XYZUSDT") is None

    def test_rolling_avg(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.0001, timestamp=now - 100)
        tracker.inject_rate("BTCUSDT", 0.0003, timestamp=now - 50)
        tracker.inject_rate("BTCUSDT", 0.0005, timestamp=now)
        avg = tracker.get_rolling_avg("BTCUSDT", window_hours=1.0)
        assert avg is not None
        assert abs(avg - 0.0003) < 0.0001

    def test_rolling_avg_no_data(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        assert tracker.get_rolling_avg("BTCUSDT") is None

    def test_rate_of_change(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.0001, timestamp=now - 100)
        tracker.inject_rate("BTCUSDT", 0.0005, timestamp=now)
        roc = tracker.get_rate_of_change("BTCUSDT", window_hours=1.0)
        assert roc is not None
        assert abs(roc - 0.0004) < 0.0001

    def test_rate_of_change_insufficient_data(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("BTCUSDT", 0.0001)
        assert tracker.get_rate_of_change("BTCUSDT") is None


class TestFundingSignal:
    def test_extreme_long_detected(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], extreme_threshold=0.0005)
        tracker.inject_rate("BTCUSDT", 0.001)  # 0.1% -- extreme
        signal = tracker.get_funding_signal("BTCUSDT")
        assert signal is not None
        assert signal.is_extreme_long
        assert not signal.is_extreme_short
        assert signal.drift_adjustment < 0  # Expect pullback

    def test_extreme_short_detected(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], extreme_threshold=0.0005)
        tracker.inject_rate("BTCUSDT", -0.001)  # -0.1% -- extreme short
        signal = tracker.get_funding_signal("BTCUSDT")
        assert signal is not None
        assert signal.is_extreme_short
        assert not signal.is_extreme_long
        assert signal.drift_adjustment > 0  # Expect squeeze

    def test_normal_funding_not_extreme(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], extreme_threshold=0.0005)
        tracker.inject_rate("BTCUSDT", 0.0001)  # Normal
        signal = tracker.get_funding_signal("BTCUSDT")
        assert signal is not None
        assert not signal.is_extreme_long
        assert not signal.is_extreme_short

    def test_zero_funding_no_drift(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("BTCUSDT", 0.0)
        signal = tracker.get_funding_signal("BTCUSDT")
        assert signal is not None
        assert signal.drift_adjustment == pytest.approx(0.0)

    def test_signal_none_when_no_data(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        assert tracker.get_funding_signal("BTCUSDT") is None

    def test_drift_magnitude_scales_with_rate(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], extreme_threshold=0.0005)
        tracker.inject_rate("BTCUSDT", 0.001)
        signal_small = tracker.get_funding_signal("BTCUSDT")
        tracker.inject_rate("BTCUSDT", 0.005)
        signal_large = tracker.get_funding_signal("BTCUSDT")
        assert abs(signal_large.drift_adjustment) > abs(signal_small.drift_adjustment)

    def test_signal_includes_rolling_avg(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.0002, timestamp=now - 100)
        tracker.inject_rate("BTCUSDT", 0.0004, timestamp=now)
        signal = tracker.get_funding_signal("BTCUSDT")
        assert signal is not None
        assert signal.rolling_avg_8h is not None

    def test_signal_includes_roc(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.0001, timestamp=now - 100)
        tracker.inject_rate("BTCUSDT", 0.0005, timestamp=now)
        signal = tracker.get_funding_signal("BTCUSDT")
        assert signal is not None
        assert signal.rate_of_change != 0.0


class TestFundingRateHistory:
    def test_history_capped_at_max(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], max_history=10)
        for i in range(20):
            tracker.inject_rate("BTCUSDT", 0.0001 * i, timestamp=float(i))
        assert len(tracker._history["BTCUSDT"]) == 10

    def test_multiple_symbols_independent(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT", "ETHUSDT"])
        tracker.inject_rate("BTCUSDT", 0.001)
        tracker.inject_rate("ETHUSDT", -0.001)
        btc = tracker.get_funding_signal("BTCUSDT")
        eth = tracker.get_funding_signal("ETHUSDT")
        assert btc.drift_adjustment < 0  # BTC positive -> negative drift
        assert eth.drift_adjustment > 0  # ETH negative -> positive drift

    def test_inject_unknown_symbol_creates_history(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("NEWUSDT", 0.0001)
        assert tracker.get_current_rate("NEWUSDT") == 0.0001
