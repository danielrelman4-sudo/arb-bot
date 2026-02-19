"""Tests for multi-timescale feature engineering (A3)."""
import time
import pytest
from arb_bot.crypto.price_feed import PriceFeed, PriceTick


def _make_feed_with_trades(num_trades=100, base_price=69000.0, interval=0.5):
    """Create a PriceFeed with injected trades."""
    feed = PriceFeed(symbols=["btcusdt"])
    now = time.time()
    for i in range(num_trades):
        tick = PriceTick(
            symbol="btcusdt",
            price=base_price + (i % 10) * 10,
            timestamp=now - (num_trades - i) * interval,
            volume=1.0 + (i % 5) * 0.5,
            is_buyer_maker=bool(i % 3 == 0),  # ~33% sell, ~67% buy
        )
        feed.inject_tick(tick)
    return feed


class TestMultiscaleOFI:
    def test_returns_dict_with_all_windows(self):
        feed = _make_feed_with_trades()
        result = feed.get_ofi_multiscale("btcusdt", windows=[30, 60, 120])
        assert isinstance(result, dict)
        assert set(result.keys()) == {30, 60, 120}

    def test_default_windows(self):
        feed = _make_feed_with_trades()
        result = feed.get_ofi_multiscale("btcusdt")
        assert set(result.keys()) == {30, 60, 120, 300}

    def test_ofi_values_in_range(self):
        feed = _make_feed_with_trades()
        result = feed.get_ofi_multiscale("btcusdt")
        for w, val in result.items():
            assert -1.0 <= val <= 1.0, f"OFI at {w}s window out of range: {val}"

    def test_empty_symbol_returns_zeros(self):
        feed = PriceFeed(symbols=["btcusdt"])
        result = feed.get_ofi_multiscale("btcusdt", windows=[30, 60])
        for val in result.values():
            assert val == 0.0

    def test_shorter_window_more_volatile(self):
        """Shorter windows should react more to recent trades."""
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Old trades: balanced
        for i in range(50):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 200 + i, volume=1.0,
                is_buyer_maker=bool(i % 2 == 0),
            ))
        # Recent trades: all buys
        for i in range(20):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 10 + i * 0.5, volume=1.0,
                is_buyer_maker=False,  # buy pressure
            ))
        result = feed.get_ofi_multiscale("btcusdt", windows=[15, 300])
        # Short window should show more buy pressure than long window
        assert result[15] >= result[300]


class TestAggressorRatio:
    def test_all_buys_returns_one(self):
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        for i in range(10):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 10 + i, volume=1.0,
                is_buyer_maker=False,  # buy-initiated
            ))
        assert feed.get_aggressor_ratio("btcusdt", window_seconds=60) == pytest.approx(1.0)

    def test_all_sells_returns_zero(self):
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        for i in range(10):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 10 + i, volume=1.0,
                is_buyer_maker=True,  # sell-initiated
            ))
        assert feed.get_aggressor_ratio("btcusdt", window_seconds=60) == pytest.approx(0.0)

    def test_balanced_returns_half(self):
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        for i in range(20):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 10 + i * 0.5, volume=1.0,
                is_buyer_maker=bool(i % 2 == 0),
            ))
        ratio = feed.get_aggressor_ratio("btcusdt", window_seconds=60)
        assert 0.4 <= ratio <= 0.6

    def test_no_data_returns_half(self):
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_aggressor_ratio("btcusdt") == 0.5


class TestVolumeAcceleration:
    def test_acceleration_when_recent_volume_high(self):
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Old period: low volume (1 trade/sec)
        for i in range(200):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 300 + i, volume=0.1,
                is_buyer_maker=False,
            ))
        # Recent: high volume (1 trade/0.1sec, 10x volume)
        for i in range(50):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 5 + i * 0.1, volume=1.0,
                is_buyer_maker=False,
            ))
        accel = feed.get_volume_acceleration("btcusdt", short_window=10, long_window=300)
        assert accel > 1.0  # Recent volume should exceed average

    def test_no_data_returns_one(self):
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_volume_acceleration("btcusdt") == 1.0

    def test_deceleration_when_recent_quiet(self):
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Old period: high volume
        for i in range(200):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 300 + i, volume=5.0,
                is_buyer_maker=False,
            ))
        # Recent: very low volume
        for i in range(2):
            feed.inject_tick(PriceTick(
                symbol="btcusdt", price=69000.0,
                timestamp=now - 2 + i, volume=0.01,
                is_buyer_maker=False,
            ))
        accel = feed.get_volume_acceleration("btcusdt", short_window=10, long_window=300)
        assert accel < 1.0  # Recent volume below average
