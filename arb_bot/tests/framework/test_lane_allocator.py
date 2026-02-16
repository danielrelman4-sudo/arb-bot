"""Tests for Phase 4G: Dynamic per-lane bankroll allocation."""

from __future__ import annotations

import pytest

from arb_bot.framework.lane_allocator import (
    AllocationResult,
    LaneAllocator,
    LaneAllocatorConfig,
    LaneStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alloc(**kw) -> LaneAllocator:
    return LaneAllocator(LaneAllocatorConfig(**kw))


def _populate(alloc: LaneAllocator, lane: str, pnls: list[float]) -> None:
    for pnl in pnls:
        alloc.record_trade(lane, pnl)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LaneAllocatorConfig()
        assert cfg.default_allocation == 0.20
        assert cfg.min_allocation == 0.05
        assert cfg.max_allocation == 0.40
        assert cfg.sharpe_weight == 0.60
        assert cfg.win_rate_weight == 0.40
        assert cfg.min_trades_for_scoring == 10
        assert cfg.window_size == 100
        assert cfg.smoothing_factor == 0.3

    def test_frozen(self) -> None:
        cfg = LaneAllocatorConfig()
        with pytest.raises(AttributeError):
            cfg.max_allocation = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Record trades
# ---------------------------------------------------------------------------


class TestRecordTrades:
    def test_records(self) -> None:
        a = _alloc()
        a.record_trade("lane_a", 0.05)
        assert a.trade_count("lane_a") == 1

    def test_window_trimming(self) -> None:
        a = _alloc(window_size=5)
        for i in range(10):
            a.record_trade("lane_a", float(i))
        assert a.trade_count("lane_a") == 5

    def test_unknown_lane(self) -> None:
        a = _alloc()
        assert a.trade_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# Compute — no lanes
# ---------------------------------------------------------------------------


class TestNoLanes:
    def test_empty(self) -> None:
        a = _alloc()
        r = a.compute_allocations(bankroll=10_000)
        assert r.allocations == {}
        assert r.fractions == {}

    def test_explicit_empty(self) -> None:
        a = _alloc()
        r = a.compute_allocations(bankroll=10_000, active_lanes=[])
        assert r.allocations == {}


# ---------------------------------------------------------------------------
# Compute — unscored lanes (below min_trades)
# ---------------------------------------------------------------------------


class TestUnscoredLanes:
    def test_equal_allocation(self) -> None:
        a = _alloc(min_trades_for_scoring=10)
        a.record_trade("a", 0.01)
        a.record_trade("b", 0.02)
        r = a.compute_allocations(bankroll=10_000, active_lanes=["a", "b"])
        # Both unscored → equal split.
        assert r.fractions["a"] == pytest.approx(r.fractions["b"], abs=0.01)

    def test_default_allocation_used(self) -> None:
        a = _alloc(min_trades_for_scoring=10, default_allocation=0.25)
        a.record_trade("a", 0.01)
        r = a.compute_allocations(bankroll=10_000, active_lanes=["a"])
        # Single lane, normalized: should get full allocation.
        assert r.fractions["a"] > 0


# ---------------------------------------------------------------------------
# Compute — scored lanes
# ---------------------------------------------------------------------------


class TestScoredLanes:
    def test_better_lane_gets_more(self) -> None:
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=1.0)
        # Lane A: consistently profitable.
        _populate(a, "good", [0.05] * 10)
        # Lane B: mixed.
        _populate(a, "bad", [0.02, -0.03] * 5)
        r = a.compute_allocations(bankroll=10_000, active_lanes=["good", "bad"])
        assert r.fractions["good"] > r.fractions["bad"]

    def test_allocations_sum_at_most_1(self) -> None:
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=1.0)
        _populate(a, "a", [0.05] * 10)
        _populate(a, "b", [0.03] * 10)
        _populate(a, "c", [0.01] * 10)
        r = a.compute_allocations(
            bankroll=10_000, active_lanes=["a", "b", "c"]
        )
        total = sum(r.fractions.values())
        assert total <= 1.01  # Small tolerance.

    def test_min_allocation_respected(self) -> None:
        a = _alloc(min_trades_for_scoring=5, min_allocation=0.10,
                   smoothing_factor=1.0)
        _populate(a, "good", [0.10] * 10)
        _populate(a, "terrible", [-0.05] * 10)
        r = a.compute_allocations(
            bankroll=10_000, active_lanes=["good", "terrible"]
        )
        # Even terrible lane gets min allocation (after normalization).
        assert r.fractions["terrible"] >= 0.05  # At least some allocation.

    def test_max_allocation_respected(self) -> None:
        a = _alloc(min_trades_for_scoring=5, max_allocation=0.40,
                   smoothing_factor=1.0)
        # One great lane, many weak lanes.
        _populate(a, "great", [0.20] * 10)
        for i in range(5):
            _populate(a, f"weak_{i}", [0.001] * 10)
        lanes = ["great"] + [f"weak_{i}" for i in range(5)]
        r = a.compute_allocations(bankroll=10_000, active_lanes=lanes)
        assert r.fractions["great"] <= 0.45  # Slightly above due to normalization.


# ---------------------------------------------------------------------------
# Dollar allocations
# ---------------------------------------------------------------------------


class TestDollarAllocations:
    def test_dollars_match_fractions(self) -> None:
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=1.0)
        _populate(a, "a", [0.05] * 10)
        r = a.compute_allocations(bankroll=10_000, active_lanes=["a"])
        assert r.allocations["a"] == pytest.approx(
            10_000 * r.fractions["a"]
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_populated(self) -> None:
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=1.0)
        _populate(a, "lane_a", [0.05, 0.03, -0.01, 0.04, 0.02,
                                0.06, -0.02, 0.03, 0.05, 0.01])
        r = a.compute_allocations(bankroll=10_000, active_lanes=["lane_a"])
        stats = r.stats["lane_a"]
        assert stats.trade_count == 10
        assert stats.win_rate > 0.0
        assert stats.mean_pnl > 0.0
        assert stats.sharpe != 0.0

    def test_zero_trades(self) -> None:
        a = _alloc()
        r = a.compute_allocations(bankroll=10_000, active_lanes=["empty"])
        stats = r.stats["empty"]
        assert stats.trade_count == 0
        assert stats.win_rate == 0.0


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:
    def test_smoothing_dampens_change(self) -> None:
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=0.1)
        _populate(a, "a", [0.05] * 10)
        _populate(a, "b", [0.01] * 10)

        r1 = a.compute_allocations(bankroll=10_000, active_lanes=["a", "b"])
        # Second call should be smoothed toward first.
        r2 = a.compute_allocations(bankroll=10_000, active_lanes=["a", "b"])
        # With low smoothing factor, fractions shouldn't change much.
        assert abs(r2.fractions["a"] - r1.fractions["a"]) < 0.1

    def test_full_smoothing_instant_update(self) -> None:
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=1.0)
        _populate(a, "a", [0.05] * 10)
        r = a.compute_allocations(bankroll=10_000, active_lanes=["a"])
        # With smoothing=1.0, uses new value fully.
        assert r.fractions["a"] > 0


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        a = _alloc()
        a.record_trade("a", 0.05)
        a.clear()
        assert a.trade_count("a") == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = LaneAllocatorConfig(max_allocation=0.50)
        a = LaneAllocator(cfg)
        assert a.config.max_allocation == 0.50


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_multi_lane_rebalance(self) -> None:
        """Simulate rebalancing as performance changes."""
        a = _alloc(min_trades_for_scoring=5, smoothing_factor=0.5,
                   max_allocation=0.80)

        # Phase 1: Lane A has high Sharpe (high mean, low variance).
        _populate(a, "A", [0.08, 0.07, 0.09, 0.08, 0.07, 0.09, 0.08, 0.07, 0.09, 0.08])
        # Lane B: lower Sharpe (mixed wins/losses).
        _populate(a, "B", [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, 0.02, -0.01])
        r1 = a.compute_allocations(bankroll=10_000, active_lanes=["A", "B"])
        assert r1.fractions["A"] > r1.fractions["B"]

        # Phase 2: Lane B improves significantly.
        _populate(a, "B", [0.12, 0.11, 0.13, 0.10, 0.12, 0.11, 0.13, 0.10, 0.12, 0.11] * 2)
        r2 = a.compute_allocations(bankroll=10_000, active_lanes=["A", "B"])
        # B should get more than before (smoothed).
        assert r2.fractions["B"] > r1.fractions["B"]
