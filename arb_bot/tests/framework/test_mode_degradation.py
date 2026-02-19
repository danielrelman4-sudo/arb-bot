"""Tests for Phase 1D: Automatic mode degradation with hysteresis."""

from __future__ import annotations

import time

import pytest

from arb_bot.framework.mode_degradation import (
    DataMode,
    DegradationConfig,
    DegradationSnapshot,
    ModeDegradationManager,
)


# ---------------------------------------------------------------------------
# DataMode enum
# ---------------------------------------------------------------------------


class TestDataMode:
    def test_ordering(self) -> None:
        assert DataMode.STREAM > DataMode.HYBRID > DataMode.POLL_ONLY

    def test_values(self) -> None:
        assert DataMode.STREAM == 3
        assert DataMode.HYBRID == 2
        assert DataMode.POLL_ONLY == 1


# ---------------------------------------------------------------------------
# ModeDegradationManager — initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_in_stream(self) -> None:
        mgr = ModeDegradationManager()
        assert mgr.current_mode == DataMode.STREAM
        assert mgr.is_degraded is False
        assert mgr.is_poll_only is False

    def test_custom_initial_mode(self) -> None:
        mgr = ModeDegradationManager(initial_mode=DataMode.POLL_ONLY)
        assert mgr.current_mode == DataMode.POLL_ONLY
        assert mgr.is_degraded is True
        assert mgr.is_poll_only is True


# ---------------------------------------------------------------------------
# Degradation: STREAM → HYBRID → POLL_ONLY
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_degrades_after_threshold(self) -> None:
        config = DegradationConfig(degrade_after_failures=3)
        mgr = ModeDegradationManager(config=config)

        mgr.record_stream_failure()
        mgr.record_stream_failure()
        assert mgr.current_mode == DataMode.STREAM
        mgr.record_stream_failure()  # 3rd failure triggers degrade
        assert mgr.current_mode == DataMode.HYBRID

    def test_degrades_to_poll_only(self) -> None:
        config = DegradationConfig(degrade_after_failures=1)
        mgr = ModeDegradationManager(config=config)

        mgr.record_stream_failure()  # STREAM → HYBRID
        assert mgr.current_mode == DataMode.HYBRID
        mgr.record_stream_failure()  # HYBRID → POLL_ONLY
        assert mgr.current_mode == DataMode.POLL_ONLY

    def test_stays_at_poll_only(self) -> None:
        config = DegradationConfig(degrade_after_failures=1)
        mgr = ModeDegradationManager(config=config)

        mgr.record_stream_failure()
        mgr.record_stream_failure()
        assert mgr.current_mode == DataMode.POLL_ONLY
        mgr.record_stream_failure()  # Can't go lower
        assert mgr.current_mode == DataMode.POLL_ONLY

    def test_success_resets_failure_count(self) -> None:
        config = DegradationConfig(degrade_after_failures=3)
        mgr = ModeDegradationManager(config=config)

        mgr.record_stream_failure()
        mgr.record_stream_failure()
        mgr.record_stream_success()  # Resets counter
        mgr.record_stream_failure()
        mgr.record_stream_failure()
        assert mgr.current_mode == DataMode.STREAM  # Still < 3 consecutive

    def test_degradation_count_tracked(self) -> None:
        config = DegradationConfig(degrade_after_failures=1)
        mgr = ModeDegradationManager(config=config)

        mgr.record_stream_failure()
        mgr.record_stream_failure()
        snap = mgr.snapshot()
        assert snap.degradation_count == 2


# ---------------------------------------------------------------------------
# Upgrade: POLL_ONLY → HYBRID → STREAM (with hysteresis)
# ---------------------------------------------------------------------------


class TestUpgrade:
    def test_upgrades_after_success_threshold(self) -> None:
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=3,
            min_time_in_degraded_seconds=0.0,  # No time requirement
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()  # STREAM → HYBRID

        mgr.record_stream_success()
        mgr.record_stream_success()
        assert mgr.current_mode == DataMode.HYBRID
        mgr.record_stream_success()  # 3rd success → upgrade
        assert mgr.current_mode == DataMode.STREAM

    def test_upgrade_requires_min_time(self) -> None:
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=1,
            min_time_in_degraded_seconds=999.0,  # Long min time
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()  # STREAM → HYBRID
        mgr.record_stream_success()  # Success but too soon
        assert mgr.current_mode == DataMode.HYBRID  # Hysteresis blocks upgrade

    def test_upgrade_after_min_time_elapsed(self) -> None:
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=1,
            min_time_in_degraded_seconds=0.01,
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()
        time.sleep(0.02)
        mgr.record_stream_success()
        assert mgr.current_mode == DataMode.STREAM

    def test_failure_during_upgrade_resets(self) -> None:
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=5,
            min_time_in_degraded_seconds=0.0,
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()  # STREAM → HYBRID

        mgr.record_stream_success()
        mgr.record_stream_success()
        mgr.record_stream_success()
        mgr.record_stream_failure()  # Resets success count + degrades to POLL_ONLY
        assert mgr.current_mode == DataMode.POLL_ONLY

    def test_stepwise_upgrade(self) -> None:
        """POLL_ONLY → HYBRID → STREAM requires two upgrade cycles."""
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=2,
            min_time_in_degraded_seconds=0.0,
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()
        mgr.record_stream_failure()
        assert mgr.current_mode == DataMode.POLL_ONLY

        # First upgrade: POLL_ONLY → HYBRID
        mgr.record_stream_success()
        mgr.record_stream_success()
        assert mgr.current_mode == DataMode.HYBRID

        # Second upgrade: HYBRID → STREAM
        mgr.record_stream_success()
        mgr.record_stream_success()
        assert mgr.current_mode == DataMode.STREAM

    def test_upgrade_count_tracked(self) -> None:
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=1,
            min_time_in_degraded_seconds=0.0,
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()
        mgr.record_stream_success()  # Upgrade
        snap = mgr.snapshot()
        assert snap.upgrade_count == 1

    def test_no_upgrade_above_stream(self) -> None:
        config = DegradationConfig(
            upgrade_after_successes=1,
            min_time_in_degraded_seconds=0.0,
        )
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_success()
        assert mgr.current_mode == DataMode.STREAM  # Already at top


# ---------------------------------------------------------------------------
# Force mode
# ---------------------------------------------------------------------------


class TestForceMode:
    def test_force_to_poll_only(self) -> None:
        mgr = ModeDegradationManager()
        mgr.force_mode(DataMode.POLL_ONLY)
        assert mgr.current_mode == DataMode.POLL_ONLY
        assert mgr.target_mode == DataMode.POLL_ONLY

    def test_force_resets_counters(self) -> None:
        config = DegradationConfig(degrade_after_failures=3)
        mgr = ModeDegradationManager(config=config)
        mgr.record_stream_failure()
        mgr.record_stream_failure()
        mgr.force_mode(DataMode.STREAM)
        snap = mgr.snapshot()
        assert snap.consecutive_failures == 0
        assert snap.consecutive_successes == 0

    def test_force_same_mode_noop(self) -> None:
        mgr = ModeDegradationManager()
        mgr.force_mode(DataMode.STREAM)
        assert mgr.current_mode == DataMode.STREAM


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_fields(self) -> None:
        mgr = ModeDegradationManager()
        snap = mgr.snapshot()
        assert snap.current_mode == DataMode.STREAM
        assert snap.target_mode == DataMode.STREAM
        assert snap.consecutive_failures == 0
        assert snap.consecutive_successes == 0
        assert snap.degradation_count == 0
        assert snap.upgrade_count == 0


# ---------------------------------------------------------------------------
# DegradationConfig
# ---------------------------------------------------------------------------


class TestDegradationConfig:
    def test_defaults(self) -> None:
        config = DegradationConfig()
        assert config.degrade_after_failures == 3
        assert config.upgrade_after_successes == 10
        assert config.min_time_in_degraded_seconds == 60.0


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_degrade_and_recover(self) -> None:
        config = DegradationConfig(
            degrade_after_failures=2,
            upgrade_after_successes=3,
            min_time_in_degraded_seconds=0.0,
        )
        mgr = ModeDegradationManager(config=config)

        # Degrade STREAM → HYBRID → POLL_ONLY
        mgr.record_stream_failure()
        mgr.record_stream_failure()
        assert mgr.current_mode == DataMode.HYBRID
        mgr.record_stream_failure()
        mgr.record_stream_failure()
        assert mgr.current_mode == DataMode.POLL_ONLY

        # Recover POLL_ONLY → HYBRID → STREAM
        for _ in range(3):
            mgr.record_stream_success()
        assert mgr.current_mode == DataMode.HYBRID
        for _ in range(3):
            mgr.record_stream_success()
        assert mgr.current_mode == DataMode.STREAM

        snap = mgr.snapshot()
        assert snap.degradation_count == 2
        assert snap.upgrade_count == 2

    def test_flap_prevention(self) -> None:
        """With hysteresis, rapid failure/success doesn't cause flapping."""
        config = DegradationConfig(
            degrade_after_failures=1,
            upgrade_after_successes=1,
            min_time_in_degraded_seconds=999.0,  # Long hysteresis
        )
        mgr = ModeDegradationManager(config=config)

        mgr.record_stream_failure()  # STREAM → HYBRID
        assert mgr.current_mode == DataMode.HYBRID

        # Success alone won't upgrade because of min_time hysteresis
        mgr.record_stream_success()
        assert mgr.current_mode == DataMode.HYBRID  # Still degraded
