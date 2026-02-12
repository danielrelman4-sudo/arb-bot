"""Tests for Phase 4J: Lane readiness watchdog enforcement."""

from __future__ import annotations

import pytest

from arb_bot.lane_watchdog import (
    LaneCheckResult,
    LaneWatchdog,
    LaneWatchdogConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wd(**kw) -> LaneWatchdog:
    return LaneWatchdog(LaneWatchdogConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LaneWatchdogConfig()
        assert cfg.grace_period_seconds == 60.0
        assert cfg.max_consecutive_failures == 10
        assert cfg.check_interval_seconds == 5.0

    def test_frozen(self) -> None:
        cfg = LaneWatchdogConfig()
        with pytest.raises(AttributeError):
            cfg.grace_period_seconds = 30.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register checks
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_check(self) -> None:
        wd = _wd()
        wd.register_check("lane_a", "data_fresh", lambda: True)
        assert "lane_a" in wd.registered_lanes()

    def test_multiple_checks(self) -> None:
        wd = _wd()
        wd.register_check("lane_a", "data_fresh", lambda: True)
        wd.register_check("lane_a", "calibrated", lambda: True)
        assert "lane_a" in wd.registered_lanes()

    def test_multiple_lanes(self) -> None:
        wd = _wd()
        wd.register_check("a", "check", lambda: True)
        wd.register_check("b", "check", lambda: True)
        assert set(wd.registered_lanes()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Check — ready
# ---------------------------------------------------------------------------


class TestCheckReady:
    def test_all_pass(self) -> None:
        wd = _wd()
        wd.register_check("lane_a", "data_fresh", lambda: True)
        wd.register_check("lane_a", "calibrated", lambda: True)
        r = wd.check_lane("lane_a", now=100.0)
        assert r.ready is True
        assert r.failed_checks == ()
        assert set(r.passed_checks) == {"data_fresh", "calibrated"}

    def test_unregistered_lane(self) -> None:
        wd = _wd()
        r = wd.check_lane("unknown", now=100.0)
        assert r.ready is False
        assert "no_checks_registered" in r.failed_checks


# ---------------------------------------------------------------------------
# Check — not ready
# ---------------------------------------------------------------------------


class TestCheckNotReady:
    def test_one_fails(self) -> None:
        wd = _wd()
        wd.register_check("lane_a", "data_fresh", lambda: True)
        wd.register_check("lane_a", "calibrated", lambda: False)
        r = wd.check_lane("lane_a", now=100.0)
        assert r.ready is False
        assert "calibrated" in r.failed_checks
        assert "data_fresh" in r.passed_checks

    def test_all_fail(self) -> None:
        wd = _wd()
        wd.register_check("lane_a", "a", lambda: False)
        wd.register_check("lane_a", "b", lambda: False)
        r = wd.check_lane("lane_a", now=100.0)
        assert r.ready is False
        assert set(r.failed_checks) == {"a", "b"}

    def test_exception_counts_as_failure(self) -> None:
        wd = _wd()

        def bad_check() -> bool:
            raise RuntimeError("oops")

        wd.register_check("lane_a", "bad", bad_check)
        r = wd.check_lane("lane_a", now=100.0)
        assert r.ready is False
        assert "bad" in r.failed_checks


# ---------------------------------------------------------------------------
# Grace period
# ---------------------------------------------------------------------------


class TestGracePeriod:
    def test_enters_grace_after_failure(self) -> None:
        wd = _wd(grace_period_seconds=60.0)
        failing = [False]
        wd.register_check("lane_a", "check", lambda: failing[0])

        # Fail at t=100.
        wd.check_lane("lane_a", now=100.0)

        # Now passes, but within grace period.
        failing[0] = True
        r = wd.check_lane("lane_a", now=130.0)
        assert r.ready is False
        assert r.in_grace_period is True

    def test_exits_grace_after_period(self) -> None:
        wd = _wd(grace_period_seconds=60.0)
        failing = [False]
        wd.register_check("lane_a", "check", lambda: failing[0])

        wd.check_lane("lane_a", now=100.0)  # Fail.
        failing[0] = True
        r = wd.check_lane("lane_a", now=200.0)  # 100s after fail > 60s grace.
        assert r.ready is True
        assert r.in_grace_period is False

    def test_zero_grace_period(self) -> None:
        wd = _wd(grace_period_seconds=0.0)
        failing = [False]
        wd.register_check("lane_a", "check", lambda: failing[0])

        wd.check_lane("lane_a", now=100.0)  # Fail.
        failing[0] = True
        r = wd.check_lane("lane_a", now=100.0)  # Same time but passes.
        assert r.ready is True


# ---------------------------------------------------------------------------
# Consecutive failures and lockout
# ---------------------------------------------------------------------------


class TestLockout:
    def test_lockout_after_max_failures(self) -> None:
        wd = _wd(max_consecutive_failures=3)
        wd.register_check("lane_a", "check", lambda: False)

        for i in range(3):
            wd.check_lane("lane_a", now=float(i))

        assert wd.is_locked_out("lane_a") is True
        r = wd.check_lane("lane_a", now=100.0)
        assert r.locked_out is True
        assert r.ready is False

    def test_consecutive_counter_resets_after_grace(self) -> None:
        wd = _wd(max_consecutive_failures=10, grace_period_seconds=5.0)
        failing = [False]
        wd.register_check("lane_a", "check", lambda: failing[0])

        # Fail twice.
        wd.check_lane("lane_a", now=100.0)
        wd.check_lane("lane_a", now=101.0)

        # Now passes, wait out grace.
        failing[0] = True
        r = wd.check_lane("lane_a", now=200.0)
        assert r.ready is True
        assert r.consecutive_failures == 0

    def test_reset_clears_lockout(self) -> None:
        wd = _wd(max_consecutive_failures=2)
        wd.register_check("lane_a", "check", lambda: False)
        wd.check_lane("lane_a", now=1.0)
        wd.check_lane("lane_a", now=2.0)
        assert wd.is_locked_out("lane_a") is True

        wd.reset_lane("lane_a")
        assert wd.is_locked_out("lane_a") is False

    def test_not_locked_unknown_lane(self) -> None:
        wd = _wd()
        assert wd.is_locked_out("unknown") is False


# ---------------------------------------------------------------------------
# Dynamic checks
# ---------------------------------------------------------------------------


class TestDynamicChecks:
    def test_check_function_called_each_time(self) -> None:
        wd = _wd(grace_period_seconds=0.0)
        counter = [0]

        def counting_check() -> bool:
            counter[0] += 1
            return True

        wd.register_check("lane_a", "counter", counting_check)
        wd.check_lane("lane_a", now=1.0)
        wd.check_lane("lane_a", now=2.0)
        assert counter[0] == 2

    def test_mutable_state_check(self) -> None:
        wd = _wd(grace_period_seconds=0.0)
        state = {"ready": False}
        wd.register_check("lane_a", "dynamic", lambda: state["ready"])

        r = wd.check_lane("lane_a", now=1.0)
        assert r.ready is False

        state["ready"] = True
        r = wd.check_lane("lane_a", now=2.0)
        assert r.ready is True


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        wd = _wd()
        wd.register_check("a", "check", lambda: True)
        wd.clear()
        assert wd.registered_lanes() == []


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = LaneWatchdogConfig(grace_period_seconds=30.0)
        wd = LaneWatchdog(cfg)
        assert wd.config.grace_period_seconds == 30.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_lifecycle(self) -> None:
        """Simulate lane going through not-ready → ready → failure → recovery."""
        wd = _wd(grace_period_seconds=10.0, max_consecutive_failures=5)

        data_age = [0.0]
        cal_count = [0]

        wd.register_check("cross_venue", "data_fresh",
                          lambda: data_age[0] < 30.0)
        wd.register_check("cross_venue", "calibrated",
                          lambda: cal_count[0] >= 20)

        # Not ready: no calibration.
        r = wd.check_lane("cross_venue", now=0.0)
        assert r.ready is False

        # Calibration builds up.
        cal_count[0] = 25
        data_age[0] = 5.0
        r = wd.check_lane("cross_venue", now=10.0)
        assert r.ready is True

        # Data goes stale.
        data_age[0] = 50.0
        r = wd.check_lane("cross_venue", now=20.0)
        assert r.ready is False
        assert "data_fresh" in r.failed_checks

        # Data recovers — in grace period.
        data_age[0] = 2.0
        r = wd.check_lane("cross_venue", now=25.0)
        assert r.ready is False
        assert r.in_grace_period is True

        # Grace period expires.
        r = wd.check_lane("cross_venue", now=35.0)
        assert r.ready is True
