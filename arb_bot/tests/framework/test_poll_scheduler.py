"""Tests for Phase 5B: Tiered polling scheduler."""

from __future__ import annotations

import pytest

from arb_bot.framework.poll_scheduler import (
    DueItem,
    PollScheduler,
    PollSchedulerConfig,
    PollTier,
    SchedulerSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sched(**kw) -> PollScheduler:
    return PollScheduler(PollSchedulerConfig(**kw))


V = "kalshi"
M = "BTC-50K"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = PollSchedulerConfig()
        assert cfg.hot_interval == 0.5
        assert cfg.warm_interval == 5.0
        assert cfg.cold_interval == 60.0
        assert cfg.default_tier == PollTier.WARM
        assert cfg.hot_to_warm_seconds == 30.0
        assert cfg.warm_to_cold_seconds == 300.0
        assert cfg.max_hot_items == 50
        assert cfg.max_warm_items == 200

    def test_frozen(self) -> None:
        cfg = PollSchedulerConfig()
        with pytest.raises(AttributeError):
            cfg.hot_interval = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        s = _sched()
        s.register(V, M, now=100.0)
        state = s.get_state(V, M)
        assert state is not None
        assert state.venue == V
        assert state.market_id == M

    def test_default_tier(self) -> None:
        s = _sched(default_tier=PollTier.COLD)
        s.register(V, M, now=100.0)
        assert s.get_tier(V, M) == PollTier.COLD

    def test_explicit_tier(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        assert s.get_tier(V, M) == PollTier.HOT

    def test_no_duplicate_register(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        s.register(V, M, tier=PollTier.COLD, now=200.0)
        # Should keep original.
        assert s.get_tier(V, M) == PollTier.HOT

    def test_unregister(self) -> None:
        s = _sched()
        s.register(V, M, now=100.0)
        s.unregister(V, M)
        assert s.get_state(V, M) is None

    def test_unregister_nonexistent(self) -> None:
        s = _sched()
        s.unregister(V, M)  # No error.


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


class TestPromotion:
    def test_cold_to_warm(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.COLD, now=100.0)
        s.promote(V, M, reason="stream_update", now=110.0)
        assert s.get_tier(V, M) == PollTier.WARM

    def test_warm_to_hot(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.WARM, now=100.0)
        s.promote(V, M, reason="opportunity", now=110.0)
        assert s.get_tier(V, M) == PollTier.HOT

    def test_hot_stays_hot(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        s.promote(V, M, reason="position", now=110.0)
        assert s.get_tier(V, M) == PollTier.HOT

    def test_promote_to_hot_directly(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.COLD, now=100.0)
        s.promote_to_hot(V, M, reason="urgent", now=110.0)
        assert s.get_tier(V, M) == PollTier.HOT

    def test_promote_updates_activity_time(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.COLD, now=100.0)
        s.promote(V, M, reason="test", now=200.0)
        state = s.get_state(V, M)
        assert state is not None
        assert state.last_activity_time == 200.0

    def test_promote_sets_reason(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.COLD, now=100.0)
        s.promote(V, M, reason="stream_update", now=110.0)
        state = s.get_state(V, M)
        assert state is not None
        assert state.promotion_reason == "stream_update"

    def test_promote_nonexistent(self) -> None:
        s = _sched()
        s.promote(V, M, reason="test", now=100.0)  # No error.


# ---------------------------------------------------------------------------
# Manual demotion
# ---------------------------------------------------------------------------


class TestDemotion:
    def test_hot_to_warm(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        s.demote(V, M)
        assert s.get_tier(V, M) == PollTier.WARM

    def test_warm_to_cold(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.WARM, now=100.0)
        s.demote(V, M)
        assert s.get_tier(V, M) == PollTier.COLD

    def test_cold_stays_cold(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.COLD, now=100.0)
        s.demote(V, M)
        assert s.get_tier(V, M) == PollTier.COLD


# ---------------------------------------------------------------------------
# Auto-demotion
# ---------------------------------------------------------------------------


class TestAutoDemotion:
    def test_hot_demotes_to_warm_after_timeout(self) -> None:
        s = _sched(hot_to_warm_seconds=30.0)
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        s.due_items(now=131.0)  # 31s > 30s timeout.
        assert s.get_tier(V, M) == PollTier.WARM

    def test_hot_stays_if_within_timeout(self) -> None:
        s = _sched(hot_to_warm_seconds=30.0)
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        s.due_items(now=125.0)  # 25s < 30s timeout.
        assert s.get_tier(V, M) == PollTier.HOT

    def test_warm_demotes_to_cold_after_timeout(self) -> None:
        s = _sched(warm_to_cold_seconds=60.0)
        s.register(V, M, tier=PollTier.WARM, now=100.0)
        s.due_items(now=161.0)  # 61s > 60s timeout.
        assert s.get_tier(V, M) == PollTier.COLD

    def test_cold_never_demotes(self) -> None:
        s = _sched()
        s.register(V, M, tier=PollTier.COLD, now=100.0)
        s.due_items(now=99999.0)
        assert s.get_tier(V, M) == PollTier.COLD

    def test_activity_resets_demotion_timer(self) -> None:
        s = _sched(hot_to_warm_seconds=30.0)
        s.register(V, M, tier=PollTier.HOT, now=100.0)
        s.promote(V, M, reason="refresh", now=125.0)  # Reset timer.
        s.due_items(now=150.0)  # 25s since last activity < 30s.
        assert s.get_tier(V, M) == PollTier.HOT


# ---------------------------------------------------------------------------
# Due items
# ---------------------------------------------------------------------------


class TestDueItems:
    def test_never_polled_is_due(self) -> None:
        s = _sched()
        s.register(V, M, now=100.0)
        items = s.due_items(now=100.0)
        assert len(items) == 1
        assert items[0].venue == V
        assert items[0].market_id == M

    def test_not_due_within_interval(self) -> None:
        s = _sched(warm_interval=5.0)
        s.register(V, M, tier=PollTier.WARM, now=100.0)
        s.mark_polled(V, M, now=100.0)
        items = s.due_items(now=103.0)  # 3s < 5s interval.
        assert len(items) == 0

    def test_due_after_interval(self) -> None:
        s = _sched(warm_interval=5.0)
        s.register(V, M, tier=PollTier.WARM, now=100.0)
        s.mark_polled(V, M, now=100.0)
        items = s.due_items(now=106.0)  # 6s > 5s interval.
        assert len(items) == 1
        assert items[0].overdue_seconds == pytest.approx(1.0)

    def test_sorted_most_overdue_first(self) -> None:
        s = _sched(warm_interval=5.0)
        s.register("k", "A", tier=PollTier.WARM, now=100.0)
        s.register("k", "B", tier=PollTier.WARM, now=100.0)
        s.mark_polled("k", "A", now=100.0)
        s.mark_polled("k", "B", now=102.0)
        items = s.due_items(now=108.0)
        # A: 8s - 5s = 3s overdue. B: 6s - 5s = 1s overdue.
        assert len(items) == 2
        assert items[0].market_id == "A"
        assert items[1].market_id == "B"

    def test_hot_tier_shorter_interval(self) -> None:
        s = _sched(hot_interval=0.5, warm_interval=5.0)
        s.register("k", "A", tier=PollTier.HOT, now=100.0)
        s.register("k", "B", tier=PollTier.WARM, now=100.0)
        s.mark_polled("k", "A", now=100.0)
        s.mark_polled("k", "B", now=100.0)
        items = s.due_items(now=101.0)
        # Hot A: 1s > 0.5s → due. Warm B: 1s < 5s → not due.
        assert len(items) == 1
        assert items[0].market_id == "A"


# ---------------------------------------------------------------------------
# Mark polled
# ---------------------------------------------------------------------------


class TestMarkPolled:
    def test_updates_poll_time(self) -> None:
        s = _sched()
        s.register(V, M, now=100.0)
        s.mark_polled(V, M, now=105.0)
        state = s.get_state(V, M)
        assert state is not None
        assert state.last_poll_time == 105.0

    def test_increments_count(self) -> None:
        s = _sched()
        s.register(V, M, now=100.0)
        s.mark_polled(V, M, now=105.0)
        s.mark_polled(V, M, now=110.0)
        state = s.get_state(V, M)
        assert state is not None
        assert state.poll_count == 2


# ---------------------------------------------------------------------------
# Tier limits
# ---------------------------------------------------------------------------


class TestTierLimits:
    def test_max_hot_enforced(self) -> None:
        s = _sched(max_hot_items=3, hot_to_warm_seconds=9999.0)
        for i in range(5):
            s.register("k", f"M{i}", tier=PollTier.HOT, now=float(i))
        s.due_items(now=10.0)  # Triggers enforcement.
        counts = s.tier_counts()
        assert counts[PollTier.HOT] <= 3
        # Oldest activity should be demoted.
        assert s.get_tier("k", "M0") == PollTier.WARM
        assert s.get_tier("k", "M1") == PollTier.WARM
        assert s.get_tier("k", "M4") == PollTier.HOT

    def test_max_warm_enforced(self) -> None:
        s = _sched(max_warm_items=2, warm_to_cold_seconds=9999.0)
        for i in range(4):
            s.register("k", f"M{i}", tier=PollTier.WARM, now=float(i))
        s.due_items(now=10.0)
        counts = s.tier_counts()
        assert counts[PollTier.WARM] <= 2
        assert s.get_tier("k", "M0") == PollTier.COLD
        assert s.get_tier("k", "M1") == PollTier.COLD
        assert s.get_tier("k", "M3") == PollTier.WARM


# ---------------------------------------------------------------------------
# Tier counts
# ---------------------------------------------------------------------------


class TestTierCounts:
    def test_empty(self) -> None:
        s = _sched()
        counts = s.tier_counts()
        assert counts[PollTier.HOT] == 0
        assert counts[PollTier.WARM] == 0
        assert counts[PollTier.COLD] == 0

    def test_mixed(self) -> None:
        s = _sched()
        s.register("k", "A", tier=PollTier.HOT, now=100.0)
        s.register("k", "B", tier=PollTier.WARM, now=100.0)
        s.register("k", "C", tier=PollTier.COLD, now=100.0)
        s.register("k", "D", tier=PollTier.HOT, now=100.0)
        counts = s.tier_counts()
        assert counts[PollTier.HOT] == 2
        assert counts[PollTier.WARM] == 1
        assert counts[PollTier.COLD] == 1


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty(self) -> None:
        s = _sched()
        snap = s.snapshot()
        assert snap.total_items == 0
        assert snap.hot_count == 0
        assert snap.total_polls == 0

    def test_populated(self) -> None:
        s = _sched()
        s.register("k", "A", tier=PollTier.HOT, now=100.0)
        s.register("k", "B", tier=PollTier.WARM, now=100.0)
        s.mark_polled("k", "A", now=101.0)
        s.mark_polled("k", "A", now=102.0)
        s.mark_polled("k", "B", now=101.0)
        snap = s.snapshot()
        assert snap.total_items == 2
        assert snap.hot_count == 1
        assert snap.warm_count == 1
        assert snap.total_polls == 3


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        s = _sched()
        s.register(V, M, now=100.0)
        s.clear()
        assert s.get_state(V, M) is None
        assert s.snapshot().total_items == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = PollSchedulerConfig(hot_interval=0.25)
        s = PollScheduler(cfg)
        assert s.config.hot_interval == 0.25


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_lifecycle(self) -> None:
        """Market goes cold → warm → hot → demoted back."""
        s = _sched(
            hot_interval=0.5,
            warm_interval=5.0,
            cold_interval=30.0,
            hot_to_warm_seconds=10.0,
            warm_to_cold_seconds=20.0,
        )

        # Register cold.
        s.register(V, M, tier=PollTier.COLD, now=0.0)
        assert s.get_tier(V, M) == PollTier.COLD

        # Promote twice to hot.
        s.promote(V, M, reason="stream_update", now=5.0)
        assert s.get_tier(V, M) == PollTier.WARM
        s.promote(V, M, reason="opportunity", now=6.0)
        assert s.get_tier(V, M) == PollTier.HOT

        # Due items at hot rate.
        s.mark_polled(V, M, now=6.0)
        items = s.due_items(now=7.0)  # 1s > 0.5s hot interval.
        assert len(items) == 1

        # Inactivity → auto-demote to warm.
        items = s.due_items(now=17.0)  # 11s > 10s hot_to_warm.
        assert s.get_tier(V, M) == PollTier.WARM

        # More inactivity → auto-demote to cold.
        items = s.due_items(now=27.0)  # 21s > 20s warm_to_cold.
        assert s.get_tier(V, M) == PollTier.COLD

    def test_multiple_markets_mixed_tiers(self) -> None:
        s = _sched(hot_interval=1.0, warm_interval=5.0, cold_interval=30.0)
        s.register("k", "A", tier=PollTier.HOT, now=100.0)
        s.register("k", "B", tier=PollTier.WARM, now=100.0)
        s.register("k", "C", tier=PollTier.COLD, now=100.0)

        s.mark_polled("k", "A", now=100.0)
        s.mark_polled("k", "B", now=100.0)
        s.mark_polled("k", "C", now=100.0)

        # At t=102: hot A due (2s > 1s), warm B not (2s < 5s), cold C not (2s < 30s).
        items = s.due_items(now=102.0)
        assert len(items) == 1
        assert items[0].market_id == "A"

        # At t=106: hot A due, warm B due, cold C not.
        s.mark_polled("k", "A", now=102.0)
        items = s.due_items(now=106.0)
        markets_due = {i.market_id for i in items}
        assert "A" in markets_due
        assert "B" in markets_due
        assert "C" not in markets_due
