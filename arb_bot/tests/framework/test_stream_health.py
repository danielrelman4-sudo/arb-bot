"""Tests for Phase 2D: Stream ingestion hardening."""

from __future__ import annotations

import pytest

from arb_bot.framework.stream_health import (
    BoundedQueueConfig,
    BoundedQuoteQueue,
    DropPolicy,
    QueueStats,
    StreamHealthConfig,
    StreamHealthMonitor,
    StreamHealthRegistry,
    StreamHealthStats,
)


# ---------------------------------------------------------------------------
# BoundedQueueConfig
# ---------------------------------------------------------------------------


class TestBoundedQueueConfig:
    def test_defaults(self) -> None:
        cfg = BoundedQueueConfig()
        assert cfg.max_size == 10000
        assert cfg.drop_policy == DropPolicy.DROP_OLDEST
        assert cfg.warn_threshold_pct == 80


# ---------------------------------------------------------------------------
# BoundedQuoteQueue — basic operations
# ---------------------------------------------------------------------------


class TestQueueBasic:
    def test_put_and_get(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(max_size=100))
        assert q.put("a") is True
        assert q.put("b") is True
        assert q.size() == 2
        assert q.get() == "a"
        assert q.get() == "b"
        assert q.get() is None

    def test_is_empty(self) -> None:
        q = BoundedQuoteQueue()
        assert q.is_empty() is True
        q.put("x")
        assert q.is_empty() is False

    def test_peek(self) -> None:
        q = BoundedQuoteQueue()
        assert q.peek() is None
        q.put("x")
        assert q.peek() == "x"
        assert q.size() == 1  # Not consumed.

    def test_clear(self) -> None:
        q = BoundedQuoteQueue()
        q.put("a")
        q.put("b")
        removed = q.clear()
        assert removed == 2
        assert q.is_empty() is True

    def test_utilization_pct(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(max_size=10))
        q.put("a")
        q.put("b")
        q.put("c")
        assert q.utilization_pct() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# BoundedQuoteQueue — drop policies
# ---------------------------------------------------------------------------


class TestDropPolicy:
    def test_drop_oldest(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=3,
            drop_policy=DropPolicy.DROP_OLDEST,
            warn_threshold_pct=0,  # Disable warnings.
        ))
        q.put("a")
        q.put("b")
        q.put("c")
        assert q.is_full() is True
        # Queue is full — putting drops oldest.
        assert q.put("d") is True
        assert q.size() == 3
        assert q.get() == "b"  # "a" was dropped.

    def test_drop_newest(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=3,
            drop_policy=DropPolicy.DROP_NEWEST,
            warn_threshold_pct=0,
        ))
        q.put("a")
        q.put("b")
        q.put("c")
        # Queue is full — new item is rejected.
        assert q.put("d") is False
        assert q.size() == 3
        assert q.get() == "a"  # Original items preserved.

    def test_drop_oldest_stats(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=2,
            drop_policy=DropPolicy.DROP_OLDEST,
            warn_threshold_pct=0,
        ))
        q.put("a")
        q.put("b")
        q.put("c")  # Drops "a".
        assert q.stats.total_enqueued == 3
        assert q.stats.total_dropped == 1

    def test_drop_newest_stats(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=2,
            drop_policy=DropPolicy.DROP_NEWEST,
            warn_threshold_pct=0,
        ))
        q.put("a")
        q.put("b")
        q.put("c")  # Rejected.
        assert q.stats.total_enqueued == 2
        assert q.stats.total_dropped == 1


# ---------------------------------------------------------------------------
# BoundedQuoteQueue — stats and warnings
# ---------------------------------------------------------------------------


class TestQueueStats:
    def test_stats_tracking(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=100,
            warn_threshold_pct=0,
        ))
        q.put("a")
        q.put("b")
        q.get()
        assert q.stats.total_enqueued == 2
        assert q.stats.total_dequeued == 1
        assert q.stats.total_dropped == 0
        assert q.stats.high_water_mark == 2

    def test_warn_threshold(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=10,
            warn_threshold_pct=50,
        ))
        for i in range(5):
            q.put(i)
        assert q.stats.warn_count == 1

    def test_warn_only_once(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=10,
            warn_threshold_pct=50,
        ))
        for i in range(10):
            q.put(i)
        assert q.stats.warn_count == 1

    def test_warn_resets_on_clear(self) -> None:
        q = BoundedQuoteQueue(BoundedQueueConfig(
            max_size=10,
            warn_threshold_pct=50,
        ))
        for i in range(6):
            q.put(i)
        assert q.stats.warn_count == 1
        q.clear()
        for i in range(6):
            q.put(i)
        assert q.stats.warn_count == 2  # Warned again after clear.

    def test_stats_reset(self) -> None:
        stats = QueueStats(total_enqueued=5, total_dropped=1)
        stats.reset()
        assert stats.total_enqueued == 0
        assert stats.total_dropped == 0


# ---------------------------------------------------------------------------
# StreamHealthConfig
# ---------------------------------------------------------------------------


class TestStreamHealthConfig:
    def test_defaults(self) -> None:
        cfg = StreamHealthConfig()
        assert cfg.heartbeat_timeout_seconds == 30.0
        assert cfg.max_reconnects_per_window == 5
        assert cfg.reconnect_window_seconds == 300.0
        assert cfg.max_gap_rate_per_minute == 10.0
        assert cfg.min_messages_per_minute == 0.0


# ---------------------------------------------------------------------------
# StreamHealthMonitor — heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_healthy_with_recent_message(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=30.0,
        ))
        mon.record_message(now=100.0)
        assert mon.is_healthy(now=110.0) is True

    def test_unhealthy_after_timeout(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=30.0,
        ))
        mon.record_message(now=100.0)
        assert mon.is_healthy(now=135.0) is False
        assert "heartbeat_timeout" in mon.stats.unhealthy_reason

    def test_healthy_before_first_message(self) -> None:
        """No messages yet — heartbeat check skipped (not unhealthy)."""
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=30.0,
        ))
        assert mon.is_healthy(now=100.0) is True


# ---------------------------------------------------------------------------
# StreamHealthMonitor — reconnect rate
# ---------------------------------------------------------------------------


class TestReconnectRate:
    def test_within_limit(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_reconnects_per_window=5,
            reconnect_window_seconds=300.0,
        ))
        for i in range(5):
            mon.record_reconnect(now=100.0 + i)
        assert mon.is_healthy(now=110.0) is True

    def test_exceeds_limit(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_reconnects_per_window=3,
            reconnect_window_seconds=300.0,
        ))
        for i in range(5):
            mon.record_reconnect(now=100.0 + i)
        assert mon.is_healthy(now=110.0) is False
        assert "reconnect_storm" in mon.stats.unhealthy_reason

    def test_old_reconnects_expire(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_reconnects_per_window=3,
            reconnect_window_seconds=60.0,
        ))
        for i in range(5):
            mon.record_reconnect(now=100.0 + i)
        # 70 seconds later — old reconnects expired.
        assert mon.is_healthy(now=175.0) is True


# ---------------------------------------------------------------------------
# StreamHealthMonitor — gap rate
# ---------------------------------------------------------------------------


class TestGapRate:
    def test_within_limit(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_gap_rate_per_minute=10.0,
        ))
        for i in range(8):
            mon.record_gap(now=100.0 + i)
        assert mon.is_healthy(now=110.0) is True

    def test_exceeds_limit(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_gap_rate_per_minute=5.0,
        ))
        for i in range(8):
            mon.record_gap(now=100.0 + i)
        assert mon.is_healthy(now=110.0) is False
        assert "gap_rate" in mon.stats.unhealthy_reason

    def test_old_gaps_expire(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_gap_rate_per_minute=5.0,
        ))
        for i in range(8):
            mon.record_gap(now=100.0 + i)
        # 65 seconds later — old gaps expired.
        assert mon.is_healthy(now=170.0) is True

    def test_disabled_when_zero(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            max_gap_rate_per_minute=0.0,
        ))
        for i in range(20):
            mon.record_gap(now=100.0 + i)
        assert mon.is_healthy(now=110.0) is True


# ---------------------------------------------------------------------------
# StreamHealthMonitor — message rate
# ---------------------------------------------------------------------------


class TestMessageRate:
    def test_disabled_by_default(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=0.0,  # Disable heartbeat for this test.
            min_messages_per_minute=0.0,     # Rate disabled by default.
        ))
        mon.record_message(now=100.0)
        assert mon.is_healthy(now=200.0) is True

    def test_sufficient_rate(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=0.0,  # Disable heartbeat.
            min_messages_per_minute=5.0,
        ))
        for i in range(10):
            mon.record_message(now=100.0 + i * 5)
        assert mon.is_healthy(now=145.0) is True

    def test_insufficient_rate(self) -> None:
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=0.0,
            min_messages_per_minute=10.0,
        ))
        mon.record_message(now=100.0)
        mon.record_message(now=110.0)
        assert mon.is_healthy(now=150.0) is False
        assert "low_message_rate" in mon.stats.unhealthy_reason


# ---------------------------------------------------------------------------
# StreamHealthMonitor — stats and reset
# ---------------------------------------------------------------------------


class TestMonitorStats:
    def test_total_messages(self) -> None:
        mon = StreamHealthMonitor("kalshi")
        mon.record_message(now=100.0)
        mon.record_message(now=110.0)
        assert mon.stats.total_messages == 2

    def test_total_reconnects(self) -> None:
        mon = StreamHealthMonitor("kalshi")
        mon.record_reconnect(now=100.0)
        assert mon.stats.total_reconnects == 1

    def test_total_gaps(self) -> None:
        mon = StreamHealthMonitor("kalshi")
        mon.record_gap(now=100.0)
        mon.record_gap(now=110.0)
        assert mon.stats.total_gaps == 2

    def test_reset(self) -> None:
        mon = StreamHealthMonitor("kalshi")
        mon.record_message(now=100.0)
        mon.record_reconnect(now=100.0)
        mon.record_gap(now=100.0)
        mon.reset()
        assert mon.stats.total_messages == 0
        assert mon.stats.total_reconnects == 0
        assert mon.stats.total_gaps == 0
        assert mon.stats.last_message_at is None

    def test_stats_defaults(self) -> None:
        stats = StreamHealthStats()
        assert stats.total_messages == 0
        assert stats.is_healthy is True
        assert stats.unhealthy_reason == ""


# ---------------------------------------------------------------------------
# StreamHealthRegistry
# ---------------------------------------------------------------------------


class TestStreamHealthRegistry:
    def test_get_or_create(self) -> None:
        registry = StreamHealthRegistry()
        mon = registry.get_or_create("kalshi")
        assert mon.venue == "kalshi"
        # Same venue returns same monitor.
        assert registry.get_or_create("kalshi") is mon

    def test_venues(self) -> None:
        registry = StreamHealthRegistry()
        registry.get_or_create("kalshi")
        registry.get_or_create("polymarket")
        assert sorted(registry.venues()) == ["kalshi", "polymarket"]

    def test_all_healthy(self) -> None:
        registry = StreamHealthRegistry(StreamHealthConfig(
            heartbeat_timeout_seconds=30.0,
        ))
        mon = registry.get_or_create("kalshi")
        mon.record_message(now=100.0)
        assert registry.all_healthy(now=110.0) is True

    def test_one_unhealthy(self) -> None:
        registry = StreamHealthRegistry(StreamHealthConfig(
            heartbeat_timeout_seconds=10.0,
        ))
        k = registry.get_or_create("kalshi")
        p = registry.get_or_create("polymarket")
        k.record_message(now=100.0)
        p.record_message(now=100.0)
        # Only kalshi goes stale.
        assert registry.all_healthy(now=105.0) is True
        assert registry.all_healthy(now=115.0) is False
        assert registry.unhealthy_venues(now=115.0) == ["kalshi", "polymarket"]

    def test_evaluate_all(self) -> None:
        registry = StreamHealthRegistry()
        registry.get_or_create("kalshi")
        registry.get_or_create("polymarket")
        results = registry.evaluate_all(now=100.0)
        assert "kalshi" in results
        assert "polymarket" in results

    def test_reset_all(self) -> None:
        registry = StreamHealthRegistry()
        mon = registry.get_or_create("kalshi")
        mon.record_message(now=100.0)
        mon.record_reconnect(now=100.0)
        registry.reset_all()
        assert mon.stats.total_messages == 0

    def test_custom_config_per_venue(self) -> None:
        registry = StreamHealthRegistry()
        custom_cfg = StreamHealthConfig(heartbeat_timeout_seconds=60.0)
        mon = registry.get_or_create("kalshi", config=custom_cfg)
        assert mon.config.heartbeat_timeout_seconds == 60.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_queue_and_health_combined(self) -> None:
        """Simulate ingestion with bounded queue and health monitoring."""
        queue = BoundedQuoteQueue(BoundedQueueConfig(max_size=5, warn_threshold_pct=0))
        monitor = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=30.0,
        ))

        # Simulate 10 messages — queue will overflow, health stays good.
        for i in range(10):
            now = 100.0 + i
            queue.put(f"msg_{i}")
            monitor.record_message(now=now)

        assert queue.size() == 5  # Bounded.
        assert queue.stats.total_dropped == 5
        assert monitor.is_healthy(now=110.0) is True

    def test_multiple_failure_modes(self) -> None:
        """Monitor can detect multiple issues simultaneously."""
        mon = StreamHealthMonitor("kalshi", StreamHealthConfig(
            heartbeat_timeout_seconds=10.0,
            max_reconnects_per_window=2,
            reconnect_window_seconds=60.0,
        ))
        mon.record_message(now=100.0)
        for i in range(5):
            mon.record_reconnect(now=100.0 + i)
        # 15 seconds later — heartbeat AND reconnect storm.
        assert mon.is_healthy(now=115.0) is False
        assert "heartbeat_timeout" in mon.stats.unhealthy_reason
        assert "reconnect_storm" in mon.stats.unhealthy_reason
