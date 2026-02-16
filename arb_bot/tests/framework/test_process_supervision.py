"""Tests for Phase 1F: Process supervision policy."""

from __future__ import annotations

import time

import pytest

from arb_bot.framework.process_supervision import (
    CrashLoopConfig,
    CrashLoopDetector,
    GracefulShutdown,
)


# ---------------------------------------------------------------------------
# CrashLoopConfig
# ---------------------------------------------------------------------------


class TestCrashLoopConfig:
    def test_defaults(self) -> None:
        config = CrashLoopConfig()
        assert config.window_seconds == 300.0
        assert config.max_crashes == 3
        assert config.cooldown_seconds == 300.0


# ---------------------------------------------------------------------------
# CrashLoopDetector
# ---------------------------------------------------------------------------


class TestCrashLoopDetector:
    def test_no_crashes_no_loop(self) -> None:
        d = CrashLoopDetector()
        assert d.in_crash_loop is False
        assert d.crash_count == 0

    def test_below_threshold_no_loop(self) -> None:
        config = CrashLoopConfig(max_crashes=3, window_seconds=300.0)
        d = CrashLoopDetector(config)
        d.record_crash()
        d.record_crash()
        assert d.in_crash_loop is False

    def test_at_threshold_triggers_loop(self) -> None:
        config = CrashLoopConfig(max_crashes=3, window_seconds=300.0)
        d = CrashLoopDetector(config)
        d.record_crash()
        d.record_crash()
        result = d.record_crash()
        assert result is True
        assert d.in_crash_loop is True

    def test_record_crash_returns_bool(self) -> None:
        config = CrashLoopConfig(max_crashes=2)
        d = CrashLoopDetector(config)
        assert d.record_crash() is False
        assert d.record_crash() is True

    def test_crash_count(self) -> None:
        config = CrashLoopConfig(max_crashes=10, window_seconds=300.0)
        d = CrashLoopDetector(config)
        d.record_crash()
        d.record_crash()
        d.record_crash()
        assert d.crash_count == 3

    def test_old_crashes_pruned(self) -> None:
        config = CrashLoopConfig(max_crashes=3, window_seconds=0.01)
        d = CrashLoopDetector(config)
        d.record_crash()
        d.record_crash()
        time.sleep(0.02)
        # Old crashes should be pruned
        assert d.crash_count == 0

    def test_cooldown_clears_loop(self) -> None:
        config = CrashLoopConfig(max_crashes=1, cooldown_seconds=0.01)
        d = CrashLoopDetector(config)
        d.record_crash()
        assert d.in_crash_loop is True
        time.sleep(0.02)
        assert d.in_crash_loop is False

    def test_seconds_until_retry(self) -> None:
        config = CrashLoopConfig(max_crashes=1, cooldown_seconds=999.0)
        d = CrashLoopDetector(config)
        assert d.seconds_until_retry == 0.0
        d.record_crash()
        assert d.seconds_until_retry > 0.0

    def test_seconds_until_retry_not_in_loop(self) -> None:
        d = CrashLoopDetector()
        assert d.seconds_until_retry == 0.0

    def test_reset(self) -> None:
        config = CrashLoopConfig(max_crashes=1)
        d = CrashLoopDetector(config)
        d.record_crash()
        assert d.in_crash_loop is True
        d.reset()
        assert d.in_crash_loop is False
        assert d.crash_count == 0

    def test_successful_start_counted(self) -> None:
        config = CrashLoopConfig(max_crashes=10, window_seconds=300.0)
        d = CrashLoopDetector(config)
        d.record_successful_start()
        assert d.crash_count == 1  # Start is recorded in window


# ---------------------------------------------------------------------------
# GracefulShutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    def test_not_requested_initially(self) -> None:
        gs = GracefulShutdown()
        assert gs.shutdown_requested is False

    def test_request_shutdown(self) -> None:
        gs = GracefulShutdown()
        gs.request_shutdown("test")
        assert gs.shutdown_requested is True

    def test_double_request_idempotent(self) -> None:
        gs = GracefulShutdown()
        gs.request_shutdown("first")
        gs.request_shutdown("second")  # Should not log again
        assert gs.shutdown_requested is True

    def test_reset(self) -> None:
        gs = GracefulShutdown()
        gs.request_shutdown("test")
        gs.reset()
        assert gs.shutdown_requested is False

    def test_register_and_run_callbacks(self) -> None:
        gs = GracefulShutdown()
        results: list[str] = []
        gs.register_callback("step1", lambda: results.append("a"))
        gs.register_callback("step2", lambda: results.append("b"))
        callback_results = gs.run_callbacks()
        assert results == ["a", "b"]
        assert all(success for _, success in callback_results)

    def test_callback_failure_handled(self) -> None:
        gs = GracefulShutdown()
        gs.register_callback("good", lambda: None)
        gs.register_callback("bad", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        gs.register_callback("also_good", lambda: None)
        callback_results = gs.run_callbacks()
        assert callback_results[0] == ("good", True)
        assert callback_results[1] == ("bad", False)
        assert callback_results[2] == ("also_good", True)

    def test_no_callbacks(self) -> None:
        gs = GracefulShutdown()
        results = gs.run_callbacks()
        assert results == []


# ---------------------------------------------------------------------------
# GracefulShutdown — signal handlers
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    def test_install_signal_handlers(self) -> None:
        gs = GracefulShutdown()
        gs.install_signal_handlers()
        assert gs._signals_installed is True

    def test_double_install_safe(self) -> None:
        gs = GracefulShutdown()
        gs.install_signal_handlers()
        gs.install_signal_handlers()  # Should not raise
        assert gs._signals_installed is True
