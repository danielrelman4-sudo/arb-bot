"""Process supervision: crash-loop detection and graceful shutdown (Phase 1F).

Provides crash-loop detection to prevent a failing bot from restarting
endlessly, and a graceful shutdown coordinator that ensures in-flight
orders are handled safely before exit.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrashLoopConfig:
    """Configuration for crash-loop detection.

    Parameters
    ----------
    window_seconds:
        Time window for counting crashes. Default 300s (5 min).
    max_crashes:
        Max crashes within the window before declaring crash loop.
        Default 3.
    cooldown_seconds:
        How long to pause before allowing restart after crash loop
        is detected. Default 300s (5 min).
    """

    window_seconds: float = 300.0
    max_crashes: int = 3
    cooldown_seconds: float = 300.0


class CrashLoopDetector:
    """Detects rapid restart patterns that indicate a crash loop.

    Records crash timestamps and declares a crash loop if too many
    crashes occur within the configured window. When a crash loop is
    detected, the process should wait for cooldown_seconds before
    retrying, or alert the operator.
    """

    def __init__(self, config: CrashLoopConfig | None = None) -> None:
        self._config = config or CrashLoopConfig()
        self._crash_timestamps: list[float] = []
        self._in_crash_loop = False
        self._crash_loop_detected_at: float = 0.0

    def record_crash(self) -> bool:
        """Record a crash event. Returns True if now in crash loop."""
        now = time.monotonic()
        self._crash_timestamps.append(now)
        self._prune_old_crashes(now)

        if len(self._crash_timestamps) >= self._config.max_crashes:
            if not self._in_crash_loop:
                self._in_crash_loop = True
                self._crash_loop_detected_at = now
                LOGGER.error(
                    "CRASH LOOP DETECTED: %d crashes in %.0fs (limit: %d in %.0fs)",
                    len(self._crash_timestamps),
                    self._config.window_seconds,
                    self._config.max_crashes,
                    self._config.window_seconds,
                )
            return True
        return False

    def record_successful_start(self) -> None:
        """Record that the bot started successfully (not a crash restart)."""
        now = time.monotonic()
        self._crash_timestamps.append(now)
        self._prune_old_crashes(now)

    @property
    def in_crash_loop(self) -> bool:
        if not self._in_crash_loop:
            return False
        # Check if cooldown has elapsed
        elapsed = time.monotonic() - self._crash_loop_detected_at
        if elapsed >= self._config.cooldown_seconds:
            self._in_crash_loop = False
            self._crash_timestamps.clear()
            LOGGER.info("Crash loop cooldown expired, allowing restart")
            return False
        return True

    @property
    def crash_count(self) -> int:
        self._prune_old_crashes(time.monotonic())
        return len(self._crash_timestamps)

    @property
    def seconds_until_retry(self) -> float:
        """Seconds remaining in crash loop cooldown. 0 if not in crash loop."""
        if not self._in_crash_loop:
            return 0.0
        elapsed = time.monotonic() - self._crash_loop_detected_at
        remaining = self._config.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def reset(self) -> None:
        """Manual reset — clears crash history and crash loop state."""
        self._crash_timestamps.clear()
        self._in_crash_loop = False
        self._crash_loop_detected_at = 0.0

    def _prune_old_crashes(self, now: float) -> None:
        cutoff = now - self._config.window_seconds
        self._crash_timestamps = [ts for ts in self._crash_timestamps if ts > cutoff]


class GracefulShutdown:
    """Coordinates graceful shutdown of the trading engine.

    Registers signal handlers for SIGINT and SIGTERM. When a shutdown
    signal is received, sets a flag that the engine loop should check.
    The engine should then:
    1. Stop accepting new trades
    2. Cancel any pending unfilled orders
    3. Flush state to persistent store
    4. Exit cleanly

    Supports registering cleanup callbacks that run in order on shutdown.
    """

    def __init__(self) -> None:
        self._shutdown_requested = False
        self._shutdown_ts: float = 0.0
        self._callbacks: list[tuple[str, Callable[[], None]]] = []
        self._signals_installed = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def request_shutdown(self, reason: str = "manual") -> None:
        """Request a graceful shutdown."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._shutdown_ts = time.monotonic()
            LOGGER.info("Graceful shutdown requested: %s", reason)

    def register_callback(self, name: str, callback: Callable[[], None]) -> None:
        """Register a cleanup callback to run on shutdown."""
        self._callbacks.append((name, callback))

    def run_callbacks(self) -> list[tuple[str, bool]]:
        """Run all registered callbacks. Returns list of (name, success)."""
        results: list[tuple[str, bool]] = []
        for name, callback in self._callbacks:
            try:
                callback()
                results.append((name, True))
                LOGGER.info("Shutdown callback '%s' completed", name)
            except Exception as exc:
                results.append((name, False))
                LOGGER.error("Shutdown callback '%s' failed: %s", name, exc)
        return results

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown.

        Safe to call multiple times — only installs once.
        """
        if self._signals_installed:
            return

        def _handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            self.request_shutdown(reason=f"signal {sig_name}")

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        self._signals_installed = True
        LOGGER.info("Signal handlers installed for graceful shutdown")

    def reset(self) -> None:
        """Reset shutdown state (for testing)."""
        self._shutdown_requested = False
        self._shutdown_ts = 0.0
