"""Tests for Phase 5I: Parallel lane execution architecture."""

from __future__ import annotations

import pytest

from arb_bot.framework.lane_executor import (
    LaneExecutor,
    LaneExecutorConfig,
    LaneResult,
    MergedResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec(**kw) -> LaneExecutor:
    return LaneExecutor(LaneExecutorConfig(**kw))


def _ok_worker(items=None):
    def worker():
        return items if items is not None else [1, 2, 3]
    return worker


def _fail_worker():
    def worker():
        raise RuntimeError("boom")
    return worker


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LaneExecutorConfig()
        assert cfg.max_concurrent_lanes == 5
        assert cfg.default_timeout == 10.0
        assert cfg.total_budget == 30.0
        assert cfg.merge_strategy == "append"

    def test_frozen(self) -> None:
        cfg = LaneExecutorConfig()
        with pytest.raises(AttributeError):
            cfg.max_concurrent_lanes = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        ex = _exec()
        ex.register_lane("cross", _ok_worker())
        assert "cross" in ex.registered_lanes()

    def test_unregister(self) -> None:
        ex = _exec()
        ex.register_lane("cross", _ok_worker())
        ex.unregister_lane("cross")
        assert "cross" not in ex.registered_lanes()

    def test_custom_timeout(self) -> None:
        ex = _exec()
        ex.register_lane("cross", _ok_worker(), timeout=5.0)
        stats = ex.get_lane_stats("cross")
        assert stats is not None


# ---------------------------------------------------------------------------
# Execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    def test_single_lane(self) -> None:
        ex = _exec()
        ex.register_lane("cross", _ok_worker([10, 20]))
        result = ex.execute_all()
        assert result.successful_lanes == 1
        assert result.failed_lanes == 0
        assert result.all_items == (10, 20)

    def test_multiple_lanes(self) -> None:
        ex = _exec()
        ex.register_lane("a", _ok_worker([1]))
        ex.register_lane("b", _ok_worker([2]))
        result = ex.execute_all()
        assert result.successful_lanes == 2
        assert set(result.all_items) == {1, 2}

    def test_empty_results(self) -> None:
        ex = _exec()
        ex.register_lane("a", _ok_worker([]))
        result = ex.execute_all()
        assert result.successful_lanes == 1
        assert result.all_items == ()


# ---------------------------------------------------------------------------
# Execute — failure
# ---------------------------------------------------------------------------


class TestExecuteFailure:
    def test_lane_failure(self) -> None:
        ex = _exec()
        ex.register_lane("bad", _fail_worker())
        result = ex.execute_all()
        assert result.failed_lanes == 1
        assert result.successful_lanes == 0
        assert result.lane_results["bad"].error == "boom"

    def test_partial_failure(self) -> None:
        ex = _exec()
        ex.register_lane("good", _ok_worker([1]))
        ex.register_lane("bad", _fail_worker())
        result = ex.execute_all()
        assert result.successful_lanes == 1
        assert result.failed_lanes == 1
        assert result.all_items == (1,)


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------


class TestPriority:
    def test_priority_merge_order(self) -> None:
        ex = _exec(merge_strategy="priority")
        ex.register_lane("low", _ok_worker(["low"]), priority=1)
        ex.register_lane("high", _ok_worker(["high"]), priority=10)
        result = ex.execute_all()
        # High priority items should come first.
        assert result.all_items[0] == "high"

    def test_default_append_order(self) -> None:
        ex = _exec(merge_strategy="append")
        ex.register_lane("a", _ok_worker(["a_item"]))
        ex.register_lane("b", _ok_worker(["b_item"]))
        result = ex.execute_all()
        # Append order = registration order.
        assert result.all_items == ("a_item", "b_item")


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------


class TestConcurrencyLimit:
    def test_max_concurrent(self) -> None:
        ex = _exec(max_concurrent_lanes=2)
        counts = [0]

        def counting_worker():
            counts[0] += 1
            return [counts[0]]

        ex.register_lane("a", counting_worker, priority=3)
        ex.register_lane("b", counting_worker, priority=2)
        ex.register_lane("c", counting_worker, priority=1)
        result = ex.execute_all()
        # Only top 2 by priority should execute.
        assert len(result.lane_results) == 2
        assert "a" in result.lane_results
        assert "b" in result.lane_results
        assert "c" not in result.lane_results


# ---------------------------------------------------------------------------
# Lane stats
# ---------------------------------------------------------------------------


class TestLaneStats:
    def test_stats_after_execution(self) -> None:
        ex = _exec()
        ex.register_lane("a", _ok_worker())
        ex.execute_all()
        stats = ex.get_lane_stats("a")
        assert stats is not None
        assert stats["total_runs"] == 1
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 0
        assert stats["avg_elapsed"] > 0

    def test_failure_stats(self) -> None:
        ex = _exec()
        ex.register_lane("bad", _fail_worker())
        ex.execute_all()
        stats = ex.get_lane_stats("bad")
        assert stats is not None
        assert stats["total_failures"] == 1

    def test_nonexistent_lane(self) -> None:
        ex = _exec()
        assert ex.get_lane_stats("nope") is None


# ---------------------------------------------------------------------------
# No lanes
# ---------------------------------------------------------------------------


class TestNoLanes:
    def test_empty_execute(self) -> None:
        ex = _exec()
        result = ex.execute_all()
        assert result.successful_lanes == 0
        assert result.all_items == ()


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        ex = _exec()
        ex.register_lane("a", _ok_worker())
        ex.clear()
        assert ex.registered_lanes() == []


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = LaneExecutorConfig(max_concurrent_lanes=3)
        ex = LaneExecutor(cfg)
        assert ex.config.max_concurrent_lanes == 3


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_mixed_lanes(self) -> None:
        """Multiple lanes with different outcomes."""
        ex = _exec(max_concurrent_lanes=5)
        ex.register_lane("cross", _ok_worker(["cross_opp_1"]), priority=10)
        ex.register_lane("intra", _ok_worker(["intra_opp_1", "intra_opp_2"]), priority=5)
        ex.register_lane("parity", _fail_worker(), priority=3)

        result = ex.execute_all()
        assert result.successful_lanes == 2
        assert result.failed_lanes == 1
        assert len(result.all_items) == 3
        assert "cross_opp_1" in result.all_items

        # Stats accumulated.
        cross_stats = ex.get_lane_stats("cross")
        assert cross_stats is not None
        assert cross_stats["total_successes"] == 1

        parity_stats = ex.get_lane_stats("parity")
        assert parity_stats is not None
        assert parity_stats["total_failures"] == 1
