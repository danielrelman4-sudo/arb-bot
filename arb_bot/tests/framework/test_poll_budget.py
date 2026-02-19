"""Tests for Phase 5C: Poll budget allocator + fairness."""

from __future__ import annotations

import pytest

from arb_bot.framework.poll_budget import (
    BudgetAllocation,
    LaneBudgetState,
    PollBudgetAllocator,
    PollBudgetConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alloc(**kw) -> PollBudgetAllocator:
    return PollBudgetAllocator(PollBudgetConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = PollBudgetConfig()
        assert cfg.default_weight == 1.0
        assert cfg.min_allocation == 2
        assert cfg.max_allocation_fraction == 0.50
        assert cfg.burst_multiplier == 1.5
        assert cfg.burst_decay == 0.5
        assert cfg.starvation_boost == 2.0

    def test_frozen(self) -> None:
        cfg = PollBudgetConfig()
        with pytest.raises(AttributeError):
            cfg.min_allocation = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        a = _alloc()
        a.register_lane("cross_venue", weight=2.0)
        assert "cross_venue" in a.registered_lanes()

    def test_default_weight(self) -> None:
        a = _alloc(default_weight=3.0)
        a.register_lane("lane_a")
        state = a.get_state("lane_a")
        assert state is not None
        assert state.weight == 3.0

    def test_update_weight(self) -> None:
        a = _alloc()
        a.register_lane("lane_a", weight=1.0)
        a.register_lane("lane_a", weight=5.0)
        state = a.get_state("lane_a")
        assert state is not None
        assert state.weight == 5.0

    def test_unregister(self) -> None:
        a = _alloc()
        a.register_lane("lane_a")
        a.unregister_lane("lane_a")
        assert a.get_state("lane_a") is None

    def test_unregister_nonexistent(self) -> None:
        a = _alloc()
        a.unregister_lane("nope")  # No error.


# ---------------------------------------------------------------------------
# Basic allocation
# ---------------------------------------------------------------------------


class TestBasicAllocation:
    def test_empty(self) -> None:
        a = _alloc()
        result = a.allocate(100)
        assert result.allocations == {}
        assert result.total_allocated == 0

    def test_single_lane(self) -> None:
        a = _alloc(min_allocation=2, max_allocation_fraction=0.50)
        a.register_lane("lane_a", weight=1.0)
        result = a.allocate(100)
        alloc = result.allocations["lane_a"]
        # Single lane gets up to max_fraction.
        assert alloc >= 2
        assert alloc <= 50

    def test_equal_weights(self) -> None:
        a = _alloc(min_allocation=1)
        a.register_lane("lane_a", weight=1.0)
        a.register_lane("lane_b", weight=1.0)
        result = a.allocate(100)
        # Should be roughly equal.
        assert abs(result.allocations["lane_a"] - result.allocations["lane_b"]) <= 2

    def test_weighted_allocation(self) -> None:
        a = _alloc(min_allocation=1, max_allocation_fraction=1.0)
        a.register_lane("lane_a", weight=3.0)
        a.register_lane("lane_b", weight=1.0)
        result = a.allocate(100)
        # lane_a should get ~75%, lane_b ~25%.
        assert result.allocations["lane_a"] > result.allocations["lane_b"]
        assert result.allocations["lane_a"] >= 60

    def test_total_within_budget(self) -> None:
        a = _alloc(min_allocation=1, max_allocation_fraction=1.0)
        a.register_lane("a", weight=1.0)
        a.register_lane("b", weight=1.0)
        a.register_lane("c", weight=1.0)
        result = a.allocate(30)
        assert result.total_allocated <= 30


# ---------------------------------------------------------------------------
# Minimum allocation
# ---------------------------------------------------------------------------


class TestMinAllocation:
    def test_min_guaranteed(self) -> None:
        a = _alloc(min_allocation=5, max_allocation_fraction=1.0)
        a.register_lane("lane_a", weight=0.01)  # Very small weight.
        a.register_lane("lane_b", weight=100.0)  # Very large weight.
        result = a.allocate(200)
        assert result.allocations["lane_a"] >= 5

    def test_min_when_budget_tight(self) -> None:
        a = _alloc(min_allocation=3, max_allocation_fraction=1.0)
        a.register_lane("a", weight=1.0)
        a.register_lane("b", weight=1.0)
        a.register_lane("c", weight=1.0)
        result = a.allocate(10)
        # Each should get at least 3, but total 9 < 10.
        for lane in ["a", "b", "c"]:
            assert result.allocations[lane] >= 3


# ---------------------------------------------------------------------------
# Max allocation fraction
# ---------------------------------------------------------------------------


class TestMaxAllocation:
    def test_max_cap(self) -> None:
        a = _alloc(min_allocation=1, max_allocation_fraction=0.40)
        a.register_lane("lane_a", weight=100.0)
        a.register_lane("lane_b", weight=1.0)
        result = a.allocate(100)
        assert result.allocations["lane_a"] <= 40


# ---------------------------------------------------------------------------
# Starvation prevention
# ---------------------------------------------------------------------------


class TestStarvation:
    def test_starvation_boost(self) -> None:
        a = _alloc(
            min_allocation=1,
            max_allocation_fraction=1.0,
            starvation_boost=3.0,
        )
        a.register_lane("a", weight=1.0)
        a.register_lane("b", weight=1.0)

        # First cycle: normal allocation.
        result1 = a.allocate(100)

        # Lane B uses zero polls.
        a.record_usage("a", result1.allocations["a"])
        a.record_usage("b", 0)

        # Second cycle: B should get starvation boost.
        result2 = a.allocate(100)
        assert "b" in result2.starvation_boosted
        # B should get more than A now.
        assert result2.allocations["b"] > result2.allocations["a"]

    def test_starved_cycles_counter(self) -> None:
        a = _alloc(starvation_boost=2.0)
        a.register_lane("a", weight=1.0)

        a.allocate(100)
        a.record_usage("a", 0)
        a.allocate(100)

        state = a.get_state("a")
        assert state is not None
        assert state.starved_cycles >= 1


# ---------------------------------------------------------------------------
# Burst credit
# ---------------------------------------------------------------------------


class TestBurst:
    def test_burst_credit_accumulates(self) -> None:
        a = _alloc(
            min_allocation=1,
            max_allocation_fraction=1.0,
            burst_multiplier=2.0,
            burst_decay=1.0,
        )
        a.register_lane("a", weight=1.0)
        a.register_lane("b", weight=1.0)

        # First cycle.
        result1 = a.allocate(100)
        # Lane A uses only half its allocation.
        a.record_usage("a", result1.allocations["a"] // 2)
        a.record_usage("b", result1.allocations["b"])

        # Second cycle: A should get burst bonus.
        result2 = a.allocate(100)
        assert "a" in result2.burst_boosted

    def test_burst_decays(self) -> None:
        a = _alloc(burst_decay=0.5, burst_multiplier=2.0)
        a.register_lane("a", weight=1.0)

        # Cycle 1: allocate, use nothing → builds burst credit next cycle.
        a.allocate(100)
        a.record_usage("a", 0)

        # Cycle 2: allocate computes burst credit from unused cycle 1.
        result2 = a.allocate(100)
        state = a.get_state("a")
        assert state is not None
        credit_after_buildup = state.burst_credit
        assert credit_after_buildup > 0

        # Use FULL allocation this cycle → no new unused.
        a.record_usage("a", result2.allocations["a"])

        # Cycle 3: burst credit should decay (no new unused to add).
        a.allocate(100)
        assert state.burst_credit < credit_after_buildup


# ---------------------------------------------------------------------------
# Record usage
# ---------------------------------------------------------------------------


class TestRecordUsage:
    def test_updates_state(self) -> None:
        a = _alloc()
        a.register_lane("a", weight=1.0)
        a.allocate(100)
        a.record_usage("a", 42)
        state = a.get_state("a")
        assert state is not None
        assert state.last_used == 42
        assert state.total_used == 42

    def test_cumulative_usage(self) -> None:
        a = _alloc()
        a.register_lane("a", weight=1.0)
        a.record_usage("a", 10)
        a.record_usage("a", 20)
        state = a.get_state("a")
        assert state is not None
        assert state.total_used == 30

    def test_nonexistent_lane(self) -> None:
        a = _alloc()
        a.record_usage("nope", 10)  # No error.


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        a = _alloc()
        a.register_lane("a")
        a.clear()
        assert a.registered_lanes() == []


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = PollBudgetConfig(min_allocation=10)
        a = PollBudgetAllocator(cfg)
        assert a.config.min_allocation == 10


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_multi_cycle_fairness(self) -> None:
        """Over multiple cycles, all lanes get meaningful allocation."""
        a = _alloc(
            min_allocation=2,
            max_allocation_fraction=0.60,
            starvation_boost=2.0,
            burst_multiplier=1.5,
            burst_decay=0.5,
        )
        a.register_lane("cross", weight=3.0)
        a.register_lane("intra", weight=1.0)
        a.register_lane("parity", weight=1.0)

        total_by_lane: dict[str, int] = {"cross": 0, "intra": 0, "parity": 0}

        for cycle in range(10):
            result = a.allocate(50)
            for lane, alloc in result.allocations.items():
                total_by_lane[lane] += alloc
                # Simulate using 80% of allocation.
                a.record_usage(lane, int(alloc * 0.8))

        # All lanes should have received significant allocations.
        for lane in total_by_lane:
            assert total_by_lane[lane] >= 20  # At least 2 * 10 cycles.

        # Cross should get the most (highest weight).
        assert total_by_lane["cross"] > total_by_lane["intra"]
        assert total_by_lane["cross"] > total_by_lane["parity"]

    def test_starvation_recovery(self) -> None:
        """A starved lane recovers in the next cycle."""
        a = _alloc(
            min_allocation=1,
            max_allocation_fraction=1.0,
            starvation_boost=3.0,
        )
        a.register_lane("a", weight=1.0)
        a.register_lane("b", weight=1.0)

        # Cycle 1: normal.
        r1 = a.allocate(100)
        a.record_usage("a", r1.allocations["a"])
        a.record_usage("b", 0)  # B starved.

        # Cycle 2: B should be boosted.
        r2 = a.allocate(100)
        assert r2.allocations["b"] > r2.allocations["a"]
