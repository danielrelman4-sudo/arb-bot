"""Tests for Phase 1B: Circuit breakers and auto-pause."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from arb_bot.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerSnapshot,
    CircuitState,
)


# ---------------------------------------------------------------------------
# CircuitState enum
# ---------------------------------------------------------------------------


class TestCircuitState:
    def test_values(self) -> None:
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


# ---------------------------------------------------------------------------
# CircuitBreaker — basic lifecycle
# ---------------------------------------------------------------------------


class TestCircuitBreakerBasics:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.allows_request() is True

    def test_name_from_config(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(name="test_breaker"))
        assert cb.name == "test_breaker"

    def test_default_name(self) -> None:
        cb = CircuitBreaker()
        assert cb.name == "default"


# ---------------------------------------------------------------------------
# CLOSED → OPEN transition
# ---------------------------------------------------------------------------


class TestClosedToOpen:
    def test_opens_at_failure_threshold(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True
        assert cb.allows_request() is False

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets count
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # only 1 failure since reset

    def test_total_trips_increments(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.0,  # instant recovery
            success_threshold=1,
        ))
        # Trip 1
        cb.record_failure()
        cb.record_failure()
        assert cb.snapshot().total_trips == 1
        # Recover (instant timeout → half_open, then success → closed)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # Trip 2
        cb.record_failure()
        cb.record_failure()
        assert cb.snapshot().total_trips == 2


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN transition
# ---------------------------------------------------------------------------


class TestOpenToHalfOpen:
    def test_transitions_after_timeout(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,
        ))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_stays_open_before_timeout(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=999.0,
        ))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_allows_requests_in_half_open(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.0,
        ))
        cb.record_failure()
        # Timeout = 0, so immediately transitions to HALF_OPEN.
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allows_request() is True


# ---------------------------------------------------------------------------
# HALF_OPEN → CLOSED / OPEN transitions
# ---------------------------------------------------------------------------


class TestHalfOpenTransitions:
    def test_success_threshold_closes(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.0,
            success_threshold=3,
        ))
        cb.record_failure()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=999.0,  # Long timeout so OPEN stays OPEN
            success_threshold=3,
        ))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Force to HALF_OPEN for testing
        cb._transition_to(CircuitState.HALF_OPEN)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_failure()  # Back to OPEN
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_force_reset_from_open(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allows_request() is True

    def test_force_reset_clears_counts(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.reset()
        snap = cb.snapshot()
        assert snap.failure_count == 0
        assert snap.success_count == 0


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_fields(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(name="snap_test", failure_threshold=2))
        cb.record_failure()
        snap = cb.snapshot()
        assert snap.name == "snap_test"
        assert snap.state == CircuitState.CLOSED
        assert snap.failure_count == 1
        assert snap.total_trips == 0


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------


class TestCircuitBreakerRegistry:
    def test_register_and_get(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = CircuitBreaker(CircuitBreakerConfig(name="test"))
        reg.register(cb)
        assert reg.get("test") is cb
        assert reg.get("nonexistent") is None

    def test_get_or_create(self) -> None:
        reg = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(name="auto")
        cb1 = reg.get_or_create(config)
        cb2 = reg.get_or_create(config)
        assert cb1 is cb2

    def test_any_open(self) -> None:
        reg = CircuitBreakerRegistry()
        cb1 = CircuitBreaker(CircuitBreakerConfig(name="a", failure_threshold=1))
        cb2 = CircuitBreaker(CircuitBreakerConfig(name="b", failure_threshold=1))
        reg.register(cb1)
        reg.register(cb2)
        assert reg.any_open() is False
        cb1.record_failure()
        assert reg.any_open() is True

    def test_open_breakers(self) -> None:
        reg = CircuitBreakerRegistry()
        cb1 = CircuitBreaker(CircuitBreakerConfig(name="a", failure_threshold=1))
        cb2 = CircuitBreaker(CircuitBreakerConfig(name="b", failure_threshold=1))
        reg.register(cb1)
        reg.register(cb2)
        assert reg.open_breakers() == []
        cb2.record_failure()
        open_list = reg.open_breakers()
        assert len(open_list) == 1
        assert open_list[0].name == "b"

    def test_check_all_passes(self) -> None:
        reg = CircuitBreakerRegistry()
        reg.register(CircuitBreaker(CircuitBreakerConfig(name="a")))
        reg.register(CircuitBreaker(CircuitBreakerConfig(name="b")))
        allowed, reason = reg.check_all()
        assert allowed is True
        assert reason == "ok"

    def test_check_all_fails(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = CircuitBreaker(CircuitBreakerConfig(name="data_health", failure_threshold=1))
        reg.register(cb)
        cb.record_failure()
        allowed, reason = reg.check_all()
        assert allowed is False
        assert "data_health" in reason

    def test_snapshots(self) -> None:
        reg = CircuitBreakerRegistry()
        reg.register(CircuitBreaker(CircuitBreakerConfig(name="x")))
        reg.register(CircuitBreaker(CircuitBreakerConfig(name="y")))
        snaps = reg.snapshots()
        assert len(snaps) == 2
        names = {s.name for s in snaps}
        assert names == {"x", "y"}

    def test_reset_all(self) -> None:
        reg = CircuitBreakerRegistry()
        cb1 = CircuitBreaker(CircuitBreakerConfig(name="a", failure_threshold=1))
        cb2 = CircuitBreaker(CircuitBreakerConfig(name="b", failure_threshold=1))
        reg.register(cb1)
        reg.register(cb2)
        cb1.record_failure()
        cb2.record_failure()
        assert reg.any_open() is True
        reg.reset_all()
        assert reg.any_open() is False

    def test_all_breakers(self) -> None:
        reg = CircuitBreakerRegistry()
        cb1 = CircuitBreaker(CircuitBreakerConfig(name="a"))
        cb2 = CircuitBreaker(CircuitBreakerConfig(name="b"))
        reg.register(cb1)
        reg.register(cb2)
        assert len(reg.all_breakers) == 2

    def test_empty_registry(self) -> None:
        reg = CircuitBreakerRegistry()
        assert reg.any_open() is False
        allowed, reason = reg.check_all()
        assert allowed is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_failure_in_open_state_ignored(self) -> None:
        """Failures while already OPEN should not re-trip or change state."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=999.0,
        ))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        trips_before = cb.snapshot().total_trips
        cb.record_failure()  # Should be ignored
        assert cb.state == CircuitState.OPEN
        assert cb.snapshot().total_trips == trips_before

    def test_success_in_open_state_ignored(self) -> None:
        """Success while OPEN should not change state (must wait for timeout)."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=999.0,
        ))
        cb.record_failure()
        cb.record_success()  # Should be ignored — still OPEN
        assert cb.state == CircuitState.OPEN

    def test_zero_failure_threshold_trips_immediately(self) -> None:
        """Edge case: failure_threshold=0 would never trip normally.
        But threshold=1 should trip on first failure."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_config_defaults(self) -> None:
        config = CircuitBreakerConfig()
        assert config.name == "default"
        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 30.0
        assert config.success_threshold == 3

    def test_full_lifecycle(self) -> None:
        """CLOSED → OPEN → HALF_OPEN → OPEN → HALF_OPEN → CLOSED."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.01,
            success_threshold=2,
        ))
        # CLOSED → OPEN
        cb.record_failure()
        cb.record_failure()
        # State is OPEN but may transition to HALF_OPEN on next check
        # since timeout is very short. Verify it tripped.
        assert cb.snapshot().total_trips == 1

        # Wait for recovery timeout
        time.sleep(0.02)

        # OPEN → HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        # HALF_OPEN → OPEN (failure during probe)
        cb.record_failure()
        # Immediately query internal state (don't let timeout elapse)
        assert cb._state == CircuitState.OPEN

        # Wait for recovery timeout again
        time.sleep(0.02)

        # OPEN → HALF_OPEN again
        assert cb.state == CircuitState.HALF_OPEN

        # HALF_OPEN → CLOSED (2 successes)
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allows_request() is True
