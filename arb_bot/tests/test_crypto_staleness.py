"""Tests for quote staleness detection (A1)."""
import pytest
from arb_bot.crypto.staleness_detector import StalenessDetector, StalenessResult


class TestStalenessResult:
    def test_fresh_quote_not_stale(self):
        det = StalenessDetector()
        det.record_quote_snapshot("KXBTCD-TEST", 0.65, 0.35, timestamp=100.0)
        result = det.compute_staleness(
            "KXBTCD-TEST",
            current_spot=69000.0,
            spot_at_lookback=69000.0,  # no movement
            current_yes_ask=0.65,
            current_no_ask=0.35,
            now=110.0,
        )
        assert not result.is_stale
        assert result.staleness_score == 0.0

    def test_stale_when_spot_moved_quote_unchanged(self):
        det = StalenessDetector(spot_move_threshold=0.003, quote_change_threshold=0.005)
        det.record_quote_snapshot("KXBTCD-TEST", 0.65, 0.35, timestamp=100.0)
        # Spot moved +0.5% but quote unchanged
        result = det.compute_staleness(
            "KXBTCD-TEST",
            current_spot=69345.0,
            spot_at_lookback=69000.0,  # +0.5%
            current_yes_ask=0.65,      # unchanged
            current_no_ask=0.35,
            now=120.0,  # 20 seconds later
        )
        assert result.is_stale
        assert result.staleness_score > 0.0
        assert abs(result.spot_delta_pct - 0.005) < 0.001

    def test_not_stale_when_quote_updated(self):
        det = StalenessDetector(spot_move_threshold=0.003)
        det.record_quote_snapshot("KXBTCD-TEST", 0.65, 0.35, timestamp=100.0)
        # Spot moved AND quote updated
        det.record_quote_snapshot("KXBTCD-TEST", 0.70, 0.30, timestamp=110.0)
        result = det.compute_staleness(
            "KXBTCD-TEST",
            current_spot=69345.0,
            spot_at_lookback=69000.0,
            current_yes_ask=0.70,
            current_no_ask=0.30,
            now=112.0,
        )
        assert not result.is_stale

    def test_not_stale_when_spot_barely_moved(self):
        det = StalenessDetector(spot_move_threshold=0.003)
        det.record_quote_snapshot("KXBTCD-TEST", 0.65, 0.35, timestamp=100.0)
        # Only 0.1% spot move -- below threshold
        result = det.compute_staleness(
            "KXBTCD-TEST",
            current_spot=69069.0,
            spot_at_lookback=69000.0,
            current_yes_ask=0.65,
            current_no_ask=0.35,
            now=120.0,
        )
        assert not result.is_stale


class TestStalenessScore:
    def test_score_increases_with_spot_movement(self):
        det = StalenessDetector(spot_move_threshold=0.003)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)
        det.record_quote_snapshot("T2", 0.50, 0.50, timestamp=100.0)

        r1 = det.compute_staleness("T1", 69207.0, 69000.0, 0.50, 0.50, now=130.0)  # 0.3%
        r2 = det.compute_staleness("T2", 69690.0, 69000.0, 0.50, 0.50, now=130.0)  # 1.0%
        assert r2.staleness_score > r1.staleness_score

    def test_score_increases_with_age(self):
        det = StalenessDetector(spot_move_threshold=0.003)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)

        r1 = det.compute_staleness("T1", 69345.0, 69000.0, 0.50, 0.50, now=110.0)  # 10s
        r2 = det.compute_staleness("T1", 69345.0, 69000.0, 0.50, 0.50, now=200.0)  # 100s
        assert r2.staleness_score > r1.staleness_score

    def test_score_zero_to_one_range(self):
        det = StalenessDetector(spot_move_threshold=0.003)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)
        result = det.compute_staleness("T1", 72000.0, 69000.0, 0.50, 0.50, now=300.0)
        assert 0.0 <= result.staleness_score <= 1.0


class TestStalenessEdgeCases:
    def test_no_snapshot_returns_not_stale(self):
        det = StalenessDetector()
        result = det.compute_staleness("UNKNOWN", 69000.0, 68500.0, 0.50, 0.50)
        assert not result.is_stale
        assert result.staleness_score == 0.0

    def test_no_lookback_price_returns_not_stale(self):
        det = StalenessDetector()
        det.record_quote_snapshot("T1", 0.50, 0.50)
        result = det.compute_staleness("T1", 69000.0, None, 0.50, 0.50)
        assert not result.is_stale

    def test_zero_spot_price_returns_not_stale(self):
        det = StalenessDetector()
        det.record_quote_snapshot("T1", 0.50, 0.50)
        result = det.compute_staleness("T1", 0.0, 69000.0, 0.50, 0.50)
        assert not result.is_stale

    def test_old_snapshot_discarded(self):
        det = StalenessDetector(max_age_seconds=60)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)
        result = det.compute_staleness("T1", 69345.0, 69000.0, 0.50, 0.50, now=200.0)
        # 100 seconds old, max_age=60 -> too old
        assert not result.is_stale

    def test_prune_old_snapshots(self):
        det = StalenessDetector(max_age_seconds=60)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)
        det.record_quote_snapshot("T2", 0.60, 0.40, timestamp=200.0)
        removed = det.prune_old_snapshots(now=300.0)
        assert removed == 1  # T1 should be removed (200 sec old > 2*60)

    def test_snapshot_timestamp_only_updates_on_change(self):
        det = StalenessDetector()
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=200.0)  # same quote
        # Timestamp should still be 100.0 (quote didn't change)
        assert det._snapshots["T1"][2] == 100.0

    def test_negative_spot_delta_detected(self):
        det = StalenessDetector(spot_move_threshold=0.003)
        det.record_quote_snapshot("T1", 0.50, 0.50, timestamp=100.0)
        # Spot dropped 0.5%
        result = det.compute_staleness("T1", 68655.0, 69000.0, 0.50, 0.50, now=120.0)
        assert result.is_stale
        assert result.spot_delta_pct < 0


class TestStalenessEdgeBonus:
    def test_edge_bonus_property(self):
        det = StalenessDetector(edge_bonus=0.05)
        assert det.edge_bonus == 0.05

    def test_edge_bonus_default(self):
        det = StalenessDetector()
        assert det.edge_bonus == 0.02
