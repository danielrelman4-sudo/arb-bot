"""Tests for Phase 5H: Initialization speed + startup progress."""

from __future__ import annotations

import pytest

from arb_bot.startup_progress import (
    PhaseStatus,
    StartupReport,
    StartupTracker,
    StartupTrackerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker(**kw) -> StartupTracker:
    return StartupTracker(StartupTrackerConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = StartupTrackerConfig()
        assert cfg.default_timeout == 60.0
        assert cfg.fail_fast is True
        assert cfg.max_total_startup_time == 300.0

    def test_frozen(self) -> None:
        cfg = StartupTrackerConfig()
        with pytest.raises(AttributeError):
            cfg.fail_fast = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Define phase
# ---------------------------------------------------------------------------


class TestDefinePhase:
    def test_define(self) -> None:
        t = _tracker()
        t.define_phase("discovery", timeout=30.0, total=100)
        state = t.get_phase("discovery")
        assert state is not None
        assert state.name == "discovery"
        assert state.timeout == 30.0
        assert state.total == 100
        assert state.status == PhaseStatus.PENDING

    def test_no_duplicate(self) -> None:
        t = _tracker()
        t.define_phase("discovery", timeout=30.0)
        t.define_phase("discovery", timeout=60.0)
        state = t.get_phase("discovery")
        assert state is not None
        assert state.timeout == 30.0  # Kept original.

    def test_default_timeout(self) -> None:
        t = _tracker(default_timeout=45.0)
        t.define_phase("bootstrap")
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.timeout == 45.0


# ---------------------------------------------------------------------------
# Begin / complete phase
# ---------------------------------------------------------------------------


class TestBeginComplete:
    def test_begin(self) -> None:
        t = _tracker()
        t.define_phase("discovery")
        result = t.begin_phase("discovery", now=100.0)
        assert result is True
        state = t.get_phase("discovery")
        assert state is not None
        assert state.status == PhaseStatus.RUNNING
        assert state.start_time == 100.0

    def test_complete(self) -> None:
        t = _tracker()
        t.define_phase("discovery")
        t.begin_phase("discovery", now=100.0)
        t.complete_phase("discovery", now=105.0)
        state = t.get_phase("discovery")
        assert state is not None
        assert state.status == PhaseStatus.COMPLETED
        assert state.end_time == 105.0

    def test_auto_define_on_begin(self) -> None:
        t = _tracker()
        t.begin_phase("discovery", now=100.0)
        state = t.get_phase("discovery")
        assert state is not None
        assert state.status == PhaseStatus.RUNNING


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


class TestProgress:
    def test_update_progress(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap", total=200)
        t.begin_phase("bootstrap", now=100.0)
        t.update_progress("bootstrap", current=50)
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.current == 50
        assert state.progress_fraction == pytest.approx(0.25)

    def test_update_total(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap")
        t.begin_phase("bootstrap", now=100.0)
        t.update_progress("bootstrap", current=30, total=100)
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.total == 100

    def test_message(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap")
        t.begin_phase("bootstrap", now=100.0)
        t.update_progress("bootstrap", message="Loading markets...")
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.message == "Loading markets..."

    def test_complete_sets_full_progress(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap", total=100)
        t.begin_phase("bootstrap", now=100.0)
        t.update_progress("bootstrap", current=50)
        t.complete_phase("bootstrap", now=110.0)
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.current == 100
        assert state.progress_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


class TestFailure:
    def test_fail_phase(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap")
        t.begin_phase("bootstrap", now=100.0)
        t.fail_phase("bootstrap", error="Connection refused", now=105.0)
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.status == PhaseStatus.FAILED
        assert state.error == "Connection refused"

    def test_fail_fast_aborts(self) -> None:
        t = _tracker(fail_fast=True)
        t.define_phase("a")
        t.define_phase("b")
        t.begin_phase("a", now=100.0)
        t.fail_phase("a", now=105.0)
        assert t.is_aborted() is True
        # Next phase should be rejected.
        assert t.begin_phase("b", now=110.0) is False

    def test_no_fail_fast(self) -> None:
        t = _tracker(fail_fast=False)
        t.define_phase("a")
        t.define_phase("b")
        t.begin_phase("a", now=100.0)
        t.fail_phase("a", now=105.0)
        assert t.is_aborted() is False
        assert t.begin_phase("b", now=110.0) is True


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_check_timeout(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap", timeout=10.0)
        t.begin_phase("bootstrap", now=100.0)
        assert t.check_timeout("bootstrap", now=105.0) is False
        assert t.check_timeout("bootstrap", now=111.0) is True
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.status == PhaseStatus.TIMED_OUT

    def test_timeout_phase_directly(self) -> None:
        t = _tracker()
        t.define_phase("bootstrap")
        t.begin_phase("bootstrap", now=100.0)
        t.timeout_phase("bootstrap", now=200.0)
        state = t.get_phase("bootstrap")
        assert state is not None
        assert state.status == PhaseStatus.TIMED_OUT

    def test_timeout_aborts_with_fail_fast(self) -> None:
        t = _tracker(fail_fast=True)
        t.define_phase("a", timeout=5.0)
        t.begin_phase("a", now=100.0)
        t.check_timeout("a", now=106.0)
        assert t.is_aborted() is True

    def test_check_timeout_not_running(self) -> None:
        t = _tracker()
        t.define_phase("a")
        assert t.check_timeout("a", now=100.0) is False  # PENDING, not running.


# ---------------------------------------------------------------------------
# Skip
# ---------------------------------------------------------------------------


class TestSkip:
    def test_skip_phase(self) -> None:
        t = _tracker()
        t.define_phase("optional")
        t.skip_phase("optional")
        state = t.get_phase("optional")
        assert state is not None
        assert state.status == PhaseStatus.SKIPPED


# ---------------------------------------------------------------------------
# Total startup timeout
# ---------------------------------------------------------------------------


class TestTotalTimeout:
    def test_total_timeout_aborts(self) -> None:
        t = _tracker(max_total_startup_time=60.0)
        t.define_phase("a")
        t.define_phase("b")
        t.begin_phase("a", now=100.0)
        t.complete_phase("a", now=150.0)
        # 70s into startup > 60s max.
        assert t.begin_phase("b", now=170.0) is False
        assert t.is_aborted() is True


# ---------------------------------------------------------------------------
# PhaseState properties
# ---------------------------------------------------------------------------


class TestPhaseStateProps:
    def test_elapsed(self) -> None:
        t = _tracker()
        t.define_phase("a")
        t.begin_phase("a", now=100.0)
        t.complete_phase("a", now=110.0)
        state = t.get_phase("a")
        assert state is not None
        assert state.elapsed == pytest.approx(10.0)

    def test_elapsed_pending(self) -> None:
        t = _tracker()
        t.define_phase("a")
        state = t.get_phase("a")
        assert state is not None
        assert state.elapsed == 0.0

    def test_progress_fraction_zero_total(self) -> None:
        t = _tracker()
        t.define_phase("a", total=0)
        state = t.get_phase("a")
        assert state is not None
        assert state.progress_fraction == 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_all_completed(self) -> None:
        t = _tracker()
        t.define_phase("a")
        t.define_phase("b")
        t.begin_phase("a", now=100.0)
        t.complete_phase("a", now=105.0)
        t.begin_phase("b", now=105.0)
        t.complete_phase("b", now=110.0)

        report = t.report(now=110.0)
        assert report.completed_count == 2
        assert report.failed_count == 0
        assert report.overall_status == PhaseStatus.COMPLETED
        assert report.is_ready is True
        assert report.total_elapsed == pytest.approx(10.0)

    def test_with_failure(self) -> None:
        t = _tracker(fail_fast=False)
        t.define_phase("a")
        t.define_phase("b")
        t.begin_phase("a", now=100.0)
        t.fail_phase("a", now=105.0)
        t.begin_phase("b", now=105.0)
        t.complete_phase("b", now=110.0)

        report = t.report(now=110.0)
        assert report.failed_count == 1
        assert report.overall_status == PhaseStatus.FAILED
        assert report.is_ready is False
        assert "a" in report.failed_phases

    def test_still_running(self) -> None:
        t = _tracker()
        t.define_phase("a")
        t.begin_phase("a", now=100.0)
        report = t.report(now=105.0)
        assert report.pending_count == 1  # Running counts as pending.
        assert report.overall_status == PhaseStatus.RUNNING
        assert report.is_ready is False

    def test_aborted_not_ready(self) -> None:
        t = _tracker(fail_fast=True)
        t.define_phase("a")
        t.begin_phase("a", now=100.0)
        t.fail_phase("a", now=105.0)
        t.define_phase("b")  # Still pending but aborted.
        report = t.report(now=105.0)
        assert report.is_ready is False


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        t = _tracker()
        t.define_phase("a")
        t.begin_phase("a", now=100.0)
        t.clear()
        assert t.get_phase("a") is None
        assert t.is_aborted() is False


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = StartupTrackerConfig(default_timeout=45.0)
        t = StartupTracker(cfg)
        assert t.config.default_timeout == 45.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_startup(self) -> None:
        """Simulate complete startup sequence."""
        t = _tracker(
            default_timeout=30.0,
            fail_fast=True,
            max_total_startup_time=120.0,
        )

        # Define phases.
        t.define_phase("discovery", total=2)
        t.define_phase("bootstrap", total=100)
        t.define_phase("warmup", total=0)

        # Discovery — use non-zero start to avoid sentinel confusion.
        t.begin_phase("discovery", now=100.0)
        t.update_progress("discovery", current=1, message="Kalshi done")
        t.update_progress("discovery", current=2, message="Polymarket done")
        t.complete_phase("discovery", now=105.0)

        # Bootstrap.
        t.begin_phase("bootstrap", now=105.0)
        for i in range(0, 100, 10):
            t.update_progress("bootstrap", current=i)
        t.complete_phase("bootstrap", now=130.0)

        # Warmup.
        t.begin_phase("warmup", now=130.0)
        t.complete_phase("warmup", now=135.0)

        report = t.report(now=135.0)
        assert report.is_ready is True
        assert report.completed_count == 3
        assert report.total_elapsed == pytest.approx(35.0)
        assert report.overall_status == PhaseStatus.COMPLETED
