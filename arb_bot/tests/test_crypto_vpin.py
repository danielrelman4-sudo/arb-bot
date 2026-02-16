"""Tests for VPIN calculator (B1)."""
import pytest
import time

from arb_bot.crypto.vpin import VPINCalculator, VPINBucket
from arb_bot.crypto.price_feed import PriceFeed, PriceTick


class TestVPINBucketing:
    def test_trades_accumulate_in_bucket(self):
        calc = VPINCalculator(bucket_volume=10.0, num_buckets=50)
        calc.process_trade(100.0, 3.0, True, 1.0)
        calc.process_trade(100.0, 2.0, False, 2.0)
        # Not full yet (5 of 10)
        assert calc.num_completed_buckets == 0

    def test_bucket_completes_at_target_volume(self):
        calc = VPINCalculator(bucket_volume=10.0, num_buckets=50)
        calc.process_trade(100.0, 10.0, True, 1.0)
        assert calc.num_completed_buckets == 1

    def test_overflow_rolls_to_next_bucket(self):
        calc = VPINCalculator(bucket_volume=10.0, num_buckets=50)
        calc.process_trade(100.0, 25.0, True, 1.0)
        assert calc.num_completed_buckets == 2  # 10 + 10, with 5 remaining

    def test_multiple_small_trades_fill_bucket(self):
        calc = VPINCalculator(bucket_volume=5.0, num_buckets=50)
        for i in range(10):
            calc.process_trade(100.0, 1.0, True, float(i))
        assert calc.num_completed_buckets == 2  # 5 + 5

    def test_zero_bucket_volume_skips_processing(self):
        calc = VPINCalculator(bucket_volume=0.0, num_buckets=50)
        calc.process_trade(100.0, 10.0, True, 1.0)
        assert calc.num_completed_buckets == 0

    def test_buckets_capped_at_num_buckets(self):
        calc = VPINCalculator(bucket_volume=1.0, num_buckets=5)
        for i in range(20):
            calc.process_trade(100.0, 1.0, True, float(i))
        assert calc.num_completed_buckets == 5  # Capped at maxlen


class TestVPINComputation:
    def _fill_buckets(self, calc, num_buys, num_sells, vol_per_trade=1.0):
        """Helper: fill buckets with specified buy/sell pattern."""
        t = 0.0
        for _ in range(num_buys):
            calc.process_trade(100.0, vol_per_trade, True, t)
            t += 1.0
        for _ in range(num_sells):
            calc.process_trade(100.0, vol_per_trade, False, t)
            t += 1.0

    def test_all_buys_vpin_one(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        for i in range(20):
            calc.process_trade(100.0, 2.0, True, float(i))
        vpin = calc.get_vpin()
        assert vpin is not None
        assert vpin == pytest.approx(1.0)

    def test_all_sells_vpin_one(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        for i in range(20):
            calc.process_trade(100.0, 2.0, False, float(i))
        vpin = calc.get_vpin()
        assert vpin is not None
        assert vpin == pytest.approx(1.0)

    def test_balanced_vpin_near_zero(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        for i in range(20):
            calc.process_trade(100.0, 1.0, True, float(i * 2))
            calc.process_trade(100.0, 1.0, False, float(i * 2 + 1))
        vpin = calc.get_vpin()
        assert vpin is not None
        assert vpin < 0.2  # Near zero

    def test_vpin_none_with_few_buckets(self):
        calc = VPINCalculator(bucket_volume=100.0, num_buckets=50)
        calc.process_trade(100.0, 100.0, True, 1.0)
        assert calc.get_vpin() is None  # Only 1 bucket, need 5

    def test_vpin_in_zero_one_range(self):
        calc = VPINCalculator(bucket_volume=5.0, num_buckets=50)
        for i in range(100):
            is_buy = i % 3 != 0  # 67% buys
            calc.process_trade(100.0, 5.0, is_buy, float(i))
        vpin = calc.get_vpin()
        assert vpin is not None
        assert 0.0 <= vpin <= 1.0


class TestSignedVPIN:
    def test_all_buys_positive(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        for i in range(20):
            calc.process_trade(100.0, 2.0, True, float(i))
        sv = calc.get_signed_vpin()
        assert sv is not None
        assert sv > 0

    def test_all_sells_negative(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        for i in range(20):
            calc.process_trade(100.0, 2.0, False, float(i))
        sv = calc.get_signed_vpin()
        assert sv is not None
        assert sv < 0

    def test_balanced_near_zero(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        for i in range(20):
            calc.process_trade(100.0, 1.0, True, float(i * 2))
            calc.process_trade(100.0, 1.0, False, float(i * 2 + 1))
        sv = calc.get_signed_vpin()
        assert sv is not None
        assert abs(sv) < 0.2

    def test_signed_vpin_none_with_few_buckets(self):
        calc = VPINCalculator(bucket_volume=100.0, num_buckets=50)
        assert calc.get_signed_vpin() is None


class TestVPINTrend:
    def test_increasing_vpin_positive_trend(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        # Start balanced, then shift to all-buy
        # Use lookback that spans both regions
        for i in range(10):
            calc.process_trade(100.0, 1.0, True, float(i * 2))
            calc.process_trade(100.0, 1.0, False, float(i * 2 + 1))
        for i in range(10):
            calc.process_trade(100.0, 2.0, True, float(20 + i))
        # lookback=20 spans both balanced (VPIN~0) and all-buy (VPIN=1) buckets
        trend = calc.get_vpin_trend(lookback_buckets=20)
        assert trend is not None
        assert trend > 0  # VPIN increasing

    def test_trend_none_with_few_buckets(self):
        calc = VPINCalculator(bucket_volume=10.0, num_buckets=50)
        calc.process_trade(100.0, 10.0, True, 1.0)
        assert calc.get_vpin_trend() is None

    def test_flat_trend_near_zero(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50)
        # Consistent pattern -> flat trend
        for i in range(30):
            calc.process_trade(100.0, 2.0, True, float(i))
        trend = calc.get_vpin_trend(lookback_buckets=10)
        assert trend is not None
        assert abs(trend) < 0.1


class TestAutoCalibration:
    def test_auto_calibrate_sets_volume(self):
        calc = VPINCalculator(bucket_volume=0.0, num_buckets=50)
        result = calc.auto_calibrate_bucket_size(
            total_volume=1000.0, window_minutes=60.0, target_buckets_per_hour=50,
        )
        assert result == pytest.approx(20.0)  # 1000/hr / 50 = 20
        assert calc.bucket_volume == pytest.approx(20.0)

    def test_auto_calibrate_zero_volume_fallback(self):
        calc = VPINCalculator(bucket_volume=0.0)
        result = calc.auto_calibrate_bucket_size(total_volume=0.0, window_minutes=60.0)
        assert result == 1.0  # Fallback

    def test_auto_calibrate_enables_processing(self):
        calc = VPINCalculator(bucket_volume=0.0, num_buckets=50)
        # Before calibration, trades are skipped
        calc.process_trade(100.0, 10.0, True, 1.0)
        assert calc.num_completed_buckets == 0
        # After calibration
        calc.auto_calibrate_bucket_size(total_volume=100.0, window_minutes=60.0)
        calc.process_trade(100.0, calc.bucket_volume, True, 2.0)
        assert calc.num_completed_buckets == 1


class TestStalenessDetection:
    def test_is_stale_initially(self):
        calc = VPINCalculator(bucket_volume=10.0, num_buckets=50)
        assert calc.is_stale is True

    def test_not_stale_after_trade(self):
        calc = VPINCalculator(bucket_volume=10.0, num_buckets=50, staleness_seconds=120.0)
        calc.process_trade(100.0, 5.0, True, 1.0)
        assert calc.is_stale is False

    def test_vpin_none_when_stale(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, staleness_seconds=0.0)
        for i in range(20):
            calc.process_trade(100.0, 2.0, True, float(i))
        # staleness_seconds=0 means always stale
        assert calc.get_vpin() is None


class TestAdaptiveThresholds:
    def _fill_with_pattern(self, calc, num_buckets, buy_ratio=0.5):
        """Fill buckets with given buy ratio to produce predictable VPIN."""
        t = 0.0
        bv = calc.bucket_volume
        for _ in range(num_buckets):
            buy_vol = bv * buy_ratio
            sell_vol = bv * (1 - buy_ratio)
            if buy_vol > 0:
                calc.process_trade(100.0, buy_vol, True, t)
                t += 1.0
            if sell_vol > 0:
                calc.process_trade(100.0, sell_vol, False, t)
                t += 1.0

    def test_history_records_vpin_snapshots(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=10, adaptive_history_size=500)
        # Fill 20 buckets (only last 10 kept in ring buffer, but all recorded in history)
        for i in range(20):
            calc.process_trade(100.0, 2.0, True, float(i))
        # Should have 20 history entries (one per completed bucket, minus first 4 where <5 buckets)
        assert calc.vpin_history_size > 0
        assert calc.vpin_history_size <= 20

    def test_history_respects_maxlen(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=10, adaptive_history_size=10)
        for i in range(50):
            calc.process_trade(100.0, 2.0, True, float(i))
        assert calc.vpin_history_size <= 10

    def test_adaptive_threshold_none_with_insufficient_history(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=10, adaptive_history_size=500)
        # Only 3 buckets — not enough for VPIN, so no history
        for i in range(3):
            calc.process_trade(100.0, 2.0, True, float(i))
        result = calc.get_adaptive_threshold(percentile=75.0, min_history=30)
        assert result is None

    def test_adaptive_threshold_returns_value_with_enough_history(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        # Fill 60 buckets to generate sufficient history
        for i in range(60):
            calc.process_trade(100.0, 2.0, True, float(i))
        result = calc.get_adaptive_threshold(percentile=75.0, min_history=5)
        assert result is not None
        assert 0.50 <= result <= 0.95  # Within floor/ceiling

    def test_adaptive_threshold_respects_floor(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        # Balanced trades -> VPIN near 0 -> percentile will be low
        for i in range(60):
            calc.process_trade(100.0, 1.0, True, float(i * 2))
            calc.process_trade(100.0, 1.0, False, float(i * 2 + 1))
        result = calc.get_adaptive_threshold(
            percentile=50.0, floor=0.60, ceiling=0.95, min_history=5,
        )
        assert result is not None
        assert result >= 0.60  # Floor enforced even though VPIN is low

    def test_adaptive_threshold_respects_ceiling(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        # All buys -> VPIN=1.0 -> percentile will be 1.0
        for i in range(60):
            calc.process_trade(100.0, 2.0, True, float(i))
        result = calc.get_adaptive_threshold(
            percentile=99.0, floor=0.50, ceiling=0.90, min_history=5,
        )
        assert result is not None
        assert result <= 0.90  # Ceiling enforced

    def test_higher_percentile_gives_higher_threshold(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        # Mix of balanced and imbalanced trades
        for i in range(30):
            calc.process_trade(100.0, 1.0, True, float(i * 2))
            calc.process_trade(100.0, 1.0, False, float(i * 2 + 1))
        for i in range(30):
            calc.process_trade(100.0, 2.0, True, float(60 + i))
        p50 = calc.get_adaptive_threshold(
            percentile=50.0, floor=0.0, ceiling=1.0, min_history=5,
        )
        p90 = calc.get_adaptive_threshold(
            percentile=90.0, floor=0.0, ceiling=1.0, min_history=5,
        )
        assert p50 is not None and p90 is not None
        assert p90 >= p50

    def test_momentum_thresholds_halt_above_momentum(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        # Fill with all-buy trades so VPIN history is all ~1.0
        for i in range(60):
            calc.process_trade(100.0, 2.0, True, float(i))
        momentum_t, halt_t = calc.get_adaptive_momentum_thresholds(
            halt_percentile=90.0,
            momentum_percentile=75.0,
            halt_floor=0.50,
            halt_ceiling=0.98,
            momentum_floor=0.40,
            momentum_ceiling=0.95,
            min_history=5,
        )
        assert momentum_t is not None
        assert halt_t is not None
        # Halt must be above momentum (with 0.05 gap enforced)
        assert halt_t > momentum_t

    def test_momentum_thresholds_none_with_insufficient_history(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        for i in range(3):
            calc.process_trade(100.0, 2.0, True, float(i))
        m, h = calc.get_adaptive_momentum_thresholds(min_history=30)
        assert m is None
        assert h is None

    def test_momentum_gap_enforced(self):
        """When percentiles converge, gap enforcement pushes halt above momentum."""
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        # All-buy -> VPIN=1.0 everywhere -> 75th and 90th percentiles both = 1.0
        for i in range(60):
            calc.process_trade(100.0, 2.0, True, float(i))
        m, h = calc.get_adaptive_momentum_thresholds(
            halt_percentile=90.0,
            momentum_percentile=75.0,
            halt_floor=0.50,
            halt_ceiling=1.10,  # High ceiling so gap logic isn't ceiling-capped
            momentum_floor=0.50,
            momentum_ceiling=0.95,
            min_history=5,
        )
        assert m is not None and h is not None
        # When percentiles converge, gap enforcement adds 0.10
        # momentum = clamped to 0.95 (ceiling), halt = momentum + 0.10 = 1.05
        assert h >= m + 0.05

    def test_vpin_history_size_property(self):
        calc = VPINCalculator(bucket_volume=2.0, num_buckets=50, adaptive_history_size=500)
        assert calc.vpin_history_size == 0
        for i in range(10):
            calc.process_trade(100.0, 2.0, True, float(i))
        assert calc.vpin_history_size > 0


class TestVPINFeedIntegration:
    def test_price_feed_forwards_to_vpin(self):
        feed = PriceFeed(symbols=["btcusdt"])
        calc = VPINCalculator(bucket_volume=1.0, num_buckets=50)
        feed.register_vpin("btcusdt", calc)

        now = time.time()
        for i in range(10):
            feed.inject_tick(PriceTick(
                symbol="btcusdt",
                price=69000.0,
                timestamp=now + i,
                volume=1.0,
                is_buyer_maker=False,  # buy
            ))
        assert calc.num_completed_buckets == 10

    def test_vpin_only_receives_registered_symbol(self):
        feed = PriceFeed(symbols=["btcusdt", "ethusdt"])
        calc_btc = VPINCalculator(bucket_volume=1.0, num_buckets=50)
        feed.register_vpin("btcusdt", calc_btc)

        now = time.time()
        # ETH ticks should NOT go to BTC calculator
        feed.inject_tick(PriceTick(
            symbol="ethusdt", price=3500.0,
            timestamp=now, volume=1.0, is_buyer_maker=False,
        ))
        assert calc_btc.num_completed_buckets == 0

    def test_ticks_without_side_skipped(self):
        feed = PriceFeed(symbols=["btcusdt"])
        calc = VPINCalculator(bucket_volume=1.0, num_buckets=50)
        feed.register_vpin("btcusdt", calc)

        now = time.time()
        feed.inject_tick(PriceTick(
            symbol="btcusdt", price=69000.0,
            timestamp=now, volume=1.0,
            is_buyer_maker=None,  # Unknown side
        ))
        assert calc.num_completed_buckets == 0

    def test_get_vpin_calculator(self):
        feed = PriceFeed(symbols=["btcusdt"])
        calc = VPINCalculator(bucket_volume=1.0)
        feed.register_vpin("btcusdt", calc)
        assert feed.get_vpin_calculator("btcusdt") is calc
        assert feed.get_vpin_calculator("ethusdt") is None
