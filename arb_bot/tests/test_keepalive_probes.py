"""Tests for Phase 5E: Cross/parity keepalive probe set."""

from __future__ import annotations

import pytest

from arb_bot.keepalive_probes import (
    DueProbe,
    KeepaliveProbeConfig,
    KeepaliveProbeSet,
    ProbeReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probes(**kw) -> KeepaliveProbeSet:
    return KeepaliveProbeSet(KeepaliveProbeConfig(**kw))


def _ok() -> bool:
    return True


def _fail() -> bool:
    return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = KeepaliveProbeConfig()
        assert cfg.probe_interval_seconds == 30.0
        assert cfg.failure_threshold == 3
        assert cfg.coverage_floor == 0.70
        assert cfg.max_probes_per_cycle == 20
        assert cfg.backoff_factor == 1.5
        assert cfg.max_probe_interval == 300.0
        assert cfg.recovery_probes == 2

    def test_frozen(self) -> None:
        cfg = KeepaliveProbeConfig()
        with pytest.raises(AttributeError):
            cfg.failure_threshold = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        ps = _probes()
        ps.register_probe("pair_1", _ok)
        assert ps.probe_count() == 1

    def test_no_duplicate(self) -> None:
        ps = _probes()
        ps.register_probe("pair_1", _ok)
        ps.register_probe("pair_1", _fail)
        assert ps.probe_count() == 1
        # Should keep original check_fn.
        state = ps.get_state("pair_1")
        assert state is not None
        assert state.check_fn is _ok

    def test_custom_interval(self) -> None:
        ps = _probes()
        ps.register_probe("pair_1", _ok, interval=10.0)
        state = ps.get_state("pair_1")
        assert state is not None
        assert state.interval == 10.0

    def test_unregister(self) -> None:
        ps = _probes()
        ps.register_probe("pair_1", _ok)
        ps.unregister_probe("pair_1")
        assert ps.probe_count() == 0

    def test_unregister_nonexistent(self) -> None:
        ps = _probes()
        ps.unregister_probe("nope")  # No error.


# ---------------------------------------------------------------------------
# Due probes
# ---------------------------------------------------------------------------


class TestDueProbes:
    def test_never_probed_is_due(self) -> None:
        ps = _probes(probe_interval_seconds=10.0)
        ps.register_probe("pair_1", _ok)
        due = ps.due_probes(now=100.0)
        assert len(due) == 1
        assert due[0].probe_id == "pair_1"

    def test_not_due_within_interval(self) -> None:
        ps = _probes(probe_interval_seconds=10.0)
        ps.register_probe("pair_1", _ok)
        ps.record_result("pair_1", success=True, now=100.0)
        due = ps.due_probes(now=105.0)  # 5s < 10s.
        assert len(due) == 0

    def test_due_after_interval(self) -> None:
        ps = _probes(probe_interval_seconds=10.0)
        ps.register_probe("pair_1", _ok)
        ps.record_result("pair_1", success=True, now=100.0)
        due = ps.due_probes(now=111.0)  # 11s > 10s.
        assert len(due) == 1

    def test_sorted_most_overdue_first(self) -> None:
        ps = _probes(probe_interval_seconds=10.0)
        ps.register_probe("A", _ok)
        ps.register_probe("B", _ok)
        ps.record_result("A", success=True, now=100.0)
        ps.record_result("B", success=True, now=105.0)
        due = ps.due_probes(now=115.0)
        # A: 15s overdue (5s), B: 10s overdue (0s).
        assert due[0].probe_id == "A"

    def test_max_per_cycle(self) -> None:
        ps = _probes(max_probes_per_cycle=2)
        for i in range(5):
            ps.register_probe(f"p{i}", _ok)
        due = ps.due_probes(now=100.0)
        assert len(due) <= 2


# ---------------------------------------------------------------------------
# Record result — success
# ---------------------------------------------------------------------------


class TestRecordSuccess:
    def test_updates_last_probe_time(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=True, now=100.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.last_probe_time == 100.0

    def test_resets_consecutive_failures(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=False, now=100.0)
        ps.record_result("p1", success=True, now=101.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.consecutive_failures == 0

    def test_increments_consecutive_successes(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=True, now=100.0)
        ps.record_result("p1", success=True, now=101.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.consecutive_successes == 2


# ---------------------------------------------------------------------------
# Record result — failure
# ---------------------------------------------------------------------------


class TestRecordFailure:
    def test_increments_failures(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=False, now=100.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.consecutive_failures == 1
        assert state.total_failures == 1

    def test_resets_consecutive_successes(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=True, now=100.0)
        ps.record_result("p1", success=False, now=101.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.consecutive_successes == 0

    def test_backoff_on_failure(self) -> None:
        ps = _probes(probe_interval_seconds=10.0, backoff_factor=2.0)
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=False, now=100.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.interval == pytest.approx(20.0)

    def test_backoff_capped(self) -> None:
        ps = _probes(
            probe_interval_seconds=10.0,
            backoff_factor=100.0,
            max_probe_interval=60.0,
        )
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=False, now=100.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.interval <= 60.0


# ---------------------------------------------------------------------------
# Dead / alive
# ---------------------------------------------------------------------------


class TestDeadAlive:
    def test_alive_by_default(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        state = ps.get_state("p1")
        assert state is not None
        assert state.alive is True

    def test_dies_after_threshold(self) -> None:
        ps = _probes(failure_threshold=3)
        ps.register_probe("p1", _ok)
        for i in range(3):
            ps.record_result("p1", success=False, now=float(i))
        state = ps.get_state("p1")
        assert state is not None
        assert state.alive is False

    def test_not_dead_before_threshold(self) -> None:
        ps = _probes(failure_threshold=3)
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=False, now=100.0)
        ps.record_result("p1", success=False, now=101.0)
        state = ps.get_state("p1")
        assert state is not None
        assert state.alive is True

    def test_recovers_after_successes(self) -> None:
        ps = _probes(failure_threshold=2, recovery_probes=2)
        ps.register_probe("p1", _ok)
        # Kill it.
        ps.record_result("p1", success=False, now=100.0)
        ps.record_result("p1", success=False, now=101.0)
        assert ps.get_state("p1").alive is False  # type: ignore

        # One success — not enough.
        ps.record_result("p1", success=True, now=102.0)
        assert ps.get_state("p1").alive is False  # type: ignore

        # Two successes — recovers.
        ps.record_result("p1", success=True, now=103.0)
        assert ps.get_state("p1").alive is True  # type: ignore

    def test_recovery_resets_interval(self) -> None:
        ps = _probes(
            probe_interval_seconds=10.0,
            failure_threshold=2,
            recovery_probes=1,
            backoff_factor=2.0,
        )
        ps.register_probe("p1", _ok)
        ps.record_result("p1", success=False, now=100.0)
        ps.record_result("p1", success=False, now=101.0)
        # Interval backed off.
        state = ps.get_state("p1")
        assert state is not None
        assert state.interval > 10.0

        # Recover.
        ps.record_result("p1", success=True, now=200.0)
        assert state.interval == 10.0  # Reset to default.


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_empty(self) -> None:
        ps = _probes()
        report = ps.report()
        assert report.total_probes == 0
        assert report.coverage == 1.0
        assert report.floor_breached is False

    def test_all_alive(self) -> None:
        ps = _probes()
        for i in range(5):
            ps.register_probe(f"p{i}", _ok)
        report = ps.report()
        assert report.alive_count == 5
        assert report.dead_count == 0
        assert report.coverage == pytest.approx(1.0)

    def test_some_dead(self) -> None:
        ps = _probes(failure_threshold=1, coverage_floor=0.70)
        for i in range(10):
            ps.register_probe(f"p{i}", _ok)
        # Kill 4 → 6/10 = 0.60 < 0.70 floor.
        for i in range(4):
            ps.record_result(f"p{i}", success=False, now=100.0)

        report = ps.report()
        assert report.alive_count == 6
        assert report.dead_count == 4
        assert report.coverage == pytest.approx(0.60)
        assert report.floor_breached is True
        assert len(report.dead_probes) == 4

    def test_above_floor(self) -> None:
        ps = _probes(failure_threshold=1, coverage_floor=0.50)
        for i in range(10):
            ps.register_probe(f"p{i}", _ok)
        # Kill 4 → 6/10 = 0.60 > 0.50.
        for i in range(4):
            ps.record_result(f"p{i}", success=False, now=100.0)

        report = ps.report()
        assert report.floor_breached is False


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        ps = _probes()
        ps.register_probe("p1", _ok)
        ps.clear()
        assert ps.probe_count() == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = KeepaliveProbeConfig(probe_interval_seconds=15.0)
        ps = KeepaliveProbeSet(cfg)
        assert ps.config.probe_interval_seconds == 15.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_probe_lifecycle(self) -> None:
        """Probe starts alive, fails, dies, recovers."""
        ps = _probes(
            probe_interval_seconds=10.0,
            failure_threshold=3,
            recovery_probes=2,
            backoff_factor=1.5,
            coverage_floor=0.50,
        )
        ps.register_probe("cross_1", _ok)
        ps.register_probe("cross_2", _ok)

        # Both alive.
        report = ps.report()
        assert report.coverage == pytest.approx(1.0)
        assert report.floor_breached is False

        # cross_1 fails 3 times → dead.
        for i in range(3):
            ps.record_result("cross_1", success=False, now=float(100 + i))

        report = ps.report()
        assert report.alive_count == 1
        assert report.dead_count == 1
        assert report.coverage == pytest.approx(0.50)
        assert report.floor_breached is False  # 0.50 == 0.50.

        # Interval should have backed off.
        state = ps.get_state("cross_1")
        assert state is not None
        assert state.interval > 10.0

        # Recover cross_1 with 2 successes.
        ps.record_result("cross_1", success=True, now=200.0)
        ps.record_result("cross_1", success=True, now=201.0)

        report = ps.report()
        assert report.alive_count == 2
        assert report.coverage == pytest.approx(1.0)
