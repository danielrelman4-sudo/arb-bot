"""Automatic mode degradation with hysteresis (Phase 1D).

Manages data-ingestion mode transitions between STREAM, HYBRID, and
POLL_ONLY. Degradation happens quickly on failures; recovery requires
sustained health before upgrading (hysteresis prevents flapping).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum

LOGGER = logging.getLogger(__name__)


class DataMode(IntEnum):
    """Data ingestion modes ordered from best to worst."""

    STREAM = 3       # Full WebSocket streaming
    HYBRID = 2       # Stream + poll fallback
    POLL_ONLY = 1    # REST polling only


@dataclass(frozen=True)
class DegradationConfig:
    """Configuration for mode degradation behavior.

    Parameters
    ----------
    degrade_after_failures:
        Consecutive stream failures before degrading one level.
    upgrade_after_successes:
        Consecutive healthy checks before upgrading one level (hysteresis).
    min_time_in_degraded_seconds:
        Minimum time to stay in a degraded mode before allowing upgrade.
    """

    degrade_after_failures: int = 3
    upgrade_after_successes: int = 10
    min_time_in_degraded_seconds: float = 60.0


@dataclass
class DegradationSnapshot:
    """Read-only snapshot of degradation state."""

    current_mode: DataMode
    target_mode: DataMode
    consecutive_failures: int
    consecutive_successes: int
    last_transition_ts: float
    degradation_count: int
    upgrade_count: int


class ModeDegradationManager:
    """Manages mode transitions with hysteresis.

    The manager tracks stream health and transitions between modes:

    STREAM → HYBRID → POLL_ONLY  (degradation, fast)
    POLL_ONLY → HYBRID → STREAM  (upgrade, slow with hysteresis)

    Degradation is aggressive: a few consecutive failures trigger immediate
    downgrade. Upgrade is conservative: many consecutive successes are
    required, plus a minimum time in the degraded mode.
    """

    def __init__(
        self,
        config: DegradationConfig | None = None,
        initial_mode: DataMode = DataMode.STREAM,
    ) -> None:
        self._config = config or DegradationConfig()
        self._current_mode = initial_mode
        self._target_mode = initial_mode
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_transition_ts = time.monotonic()
        self._degradation_count = 0
        self._upgrade_count = 0

    @property
    def current_mode(self) -> DataMode:
        return self._current_mode

    @property
    def target_mode(self) -> DataMode:
        return self._target_mode

    @property
    def is_degraded(self) -> bool:
        return self._current_mode < DataMode.STREAM

    @property
    def is_poll_only(self) -> bool:
        return self._current_mode == DataMode.POLL_ONLY

    def record_stream_failure(self) -> DataMode:
        """Record a stream health failure. Returns the new mode."""
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        if self._consecutive_failures >= self._config.degrade_after_failures:
            self._degrade()

        return self._current_mode

    def record_stream_success(self) -> DataMode:
        """Record a stream health success. Returns the new mode."""
        self._consecutive_successes += 1
        self._consecutive_failures = 0

        if self._consecutive_successes >= self._config.upgrade_after_successes:
            self._try_upgrade()

        return self._current_mode

    def force_mode(self, mode: DataMode) -> None:
        """Force a specific mode (operator override)."""
        old = self._current_mode
        self._current_mode = mode
        self._target_mode = mode
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_transition_ts = time.monotonic()
        if mode != old:
            LOGGER.info("Mode force-set: %s → %s", old.name, mode.name)

    def snapshot(self) -> DegradationSnapshot:
        return DegradationSnapshot(
            current_mode=self._current_mode,
            target_mode=self._target_mode,
            consecutive_failures=self._consecutive_failures,
            consecutive_successes=self._consecutive_successes,
            last_transition_ts=self._last_transition_ts,
            degradation_count=self._degradation_count,
            upgrade_count=self._upgrade_count,
        )

    def _degrade(self) -> None:
        if self._current_mode == DataMode.POLL_ONLY:
            return  # Already at lowest mode

        old = self._current_mode
        if old == DataMode.STREAM:
            self._current_mode = DataMode.HYBRID
        elif old == DataMode.HYBRID:
            self._current_mode = DataMode.POLL_ONLY

        self._consecutive_failures = 0
        self._last_transition_ts = time.monotonic()
        self._degradation_count += 1
        LOGGER.warning(
            "Mode degraded: %s → %s (degradation #%d)",
            old.name,
            self._current_mode.name,
            self._degradation_count,
        )

    def _try_upgrade(self) -> None:
        if self._current_mode == DataMode.STREAM:
            return  # Already at best mode

        # Hysteresis: require minimum time in degraded mode
        elapsed = time.monotonic() - self._last_transition_ts
        if elapsed < self._config.min_time_in_degraded_seconds:
            return

        old = self._current_mode
        if old == DataMode.POLL_ONLY:
            self._current_mode = DataMode.HYBRID
        elif old == DataMode.HYBRID:
            self._current_mode = DataMode.STREAM

        self._consecutive_successes = 0
        self._last_transition_ts = time.monotonic()
        self._upgrade_count += 1
        LOGGER.info(
            "Mode upgraded: %s → %s (upgrade #%d)",
            old.name,
            self._current_mode.name,
            self._upgrade_count,
        )
