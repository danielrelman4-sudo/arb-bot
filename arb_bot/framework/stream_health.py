"""Stream ingestion hardening (Phase 2D).

Provides bounded per-venue queues with configurable backpressure/drop
policy, and stream health watchdogs that track heartbeat lag, reconnect
rate, and message gap rate.

The BoundedQuoteQueue wraps an asyncio-style interface but caps memory
usage. The StreamHealthMonitor tracks per-venue health metrics and
exposes readiness checks for the engine.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drop policy
# ---------------------------------------------------------------------------


class DropPolicy(Enum):
    """What to do when the queue is full."""

    DROP_OLDEST = "drop_oldest"   # Drop the oldest item to make room.
    DROP_NEWEST = "drop_newest"   # Reject the new item.


# ---------------------------------------------------------------------------
# Bounded queue config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundedQueueConfig:
    """Configuration for a bounded quote queue.

    Parameters
    ----------
    max_size:
        Maximum number of items in the queue. Default 10000.
    drop_policy:
        What to do when the queue is full. Default DROP_OLDEST.
    warn_threshold_pct:
        Log a warning when the queue reaches this percent of capacity.
        Default 80 (i.e., warn at 80% full). 0 = disabled.
    """

    max_size: int = 10000
    drop_policy: DropPolicy = DropPolicy.DROP_OLDEST
    warn_threshold_pct: int = 80


@dataclass
class QueueStats:
    """Statistics for a bounded queue."""

    total_enqueued: int = 0
    total_dropped: int = 0
    total_dequeued: int = 0
    high_water_mark: int = 0
    warn_count: int = 0

    def reset(self) -> None:
        self.total_enqueued = 0
        self.total_dropped = 0
        self.total_dequeued = 0
        self.high_water_mark = 0
        self.warn_count = 0


class BoundedQuoteQueue:
    """Bounded FIFO queue with configurable drop policy.

    This is a synchronous queue backed by a deque, suitable for use
    in the engine's stream ingestion path. For async integration,
    wrap with an asyncio.Queue or use put/get directly.
    """

    def __init__(self, config: BoundedQueueConfig | None = None) -> None:
        self._config = config or BoundedQueueConfig()
        self._items: deque[Any] = deque()
        self._stats = QueueStats()
        self._warned = False

    @property
    def config(self) -> BoundedQueueConfig:
        return self._config

    @property
    def stats(self) -> QueueStats:
        return self._stats

    def put(self, item: Any) -> bool:
        """Add an item to the queue. Returns True if accepted, False if dropped."""
        cfg = self._config

        if len(self._items) >= cfg.max_size:
            if cfg.drop_policy == DropPolicy.DROP_OLDEST:
                self._items.popleft()
                self._stats.total_dropped += 1
                self._items.append(item)
                self._stats.total_enqueued += 1
                return True
            else:
                # DROP_NEWEST: reject the new item.
                self._stats.total_dropped += 1
                return False

        self._items.append(item)
        self._stats.total_enqueued += 1

        current_size = len(self._items)
        if current_size > self._stats.high_water_mark:
            self._stats.high_water_mark = current_size

        # Warn once when crossing the threshold.
        if cfg.warn_threshold_pct > 0:
            threshold = int(cfg.max_size * cfg.warn_threshold_pct / 100)
            if current_size >= threshold and not self._warned:
                self._warned = True
                self._stats.warn_count += 1
                LOGGER.warning(
                    "Queue at %d/%d (%d%% of capacity)",
                    current_size,
                    cfg.max_size,
                    int(current_size * 100 / cfg.max_size),
                )

        return True

    def get(self) -> Any | None:
        """Remove and return the oldest item, or None if empty."""
        if not self._items:
            return None
        self._stats.total_dequeued += 1
        return self._items.popleft()

    def peek(self) -> Any | None:
        """Return the oldest item without removing it, or None if empty."""
        if not self._items:
            return None
        return self._items[0]

    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def is_full(self) -> bool:
        return len(self._items) >= self._config.max_size

    def clear(self) -> int:
        """Remove all items. Returns the number removed."""
        count = len(self._items)
        self._items.clear()
        self._warned = False
        return count

    def utilization_pct(self) -> float:
        """Current queue utilization as a percentage."""
        if self._config.max_size == 0:
            return 100.0
        return len(self._items) * 100.0 / self._config.max_size


# ---------------------------------------------------------------------------
# Stream health watchdog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamHealthConfig:
    """Configuration for stream health monitoring.

    Parameters
    ----------
    heartbeat_timeout_seconds:
        If no message received for this long, mark stream unhealthy.
        Default 30.0.
    max_reconnects_per_window:
        Maximum reconnects allowed in the window before marking unhealthy.
        Default 5.
    reconnect_window_seconds:
        Sliding window for reconnect counting. Default 300.0 (5 min).
    max_gap_rate_per_minute:
        Maximum message gaps (missing sequence numbers) per minute
        before marking unhealthy. Default 10.0. 0 = disabled.
    min_messages_per_minute:
        Minimum message rate to consider stream healthy.
        Default 0.0 (disabled).
    """

    heartbeat_timeout_seconds: float = 30.0
    max_reconnects_per_window: int = 5
    reconnect_window_seconds: float = 300.0
    max_gap_rate_per_minute: float = 10.0
    min_messages_per_minute: float = 0.0


@dataclass
class StreamHealthStats:
    """Health statistics for a single venue's stream."""

    venue: str = ""
    total_messages: int = 0
    total_reconnects: int = 0
    total_gaps: int = 0
    last_message_at: Optional[float] = None
    last_reconnect_at: Optional[float] = None
    is_healthy: bool = True
    unhealthy_reason: str = ""

    def reset(self) -> None:
        self.total_messages = 0
        self.total_reconnects = 0
        self.total_gaps = 0
        self.last_message_at = None
        self.last_reconnect_at = None
        self.is_healthy = True
        self.unhealthy_reason = ""


class StreamHealthMonitor:
    """Tracks stream health for a single venue.

    Records message arrivals, reconnections, and gaps, then evaluates
    health based on configurable thresholds.
    """

    def __init__(
        self,
        venue: str,
        config: StreamHealthConfig | None = None,
    ) -> None:
        self._venue = venue
        self._config = config or StreamHealthConfig()
        self._stats = StreamHealthStats(venue=venue)
        self._reconnect_times: deque[float] = deque()
        self._gap_times: deque[float] = deque()
        self._message_times: deque[float] = deque()

    @property
    def venue(self) -> str:
        return self._venue

    @property
    def config(self) -> StreamHealthConfig:
        return self._config

    @property
    def stats(self) -> StreamHealthStats:
        return self._stats

    def record_message(self, now: float | None = None) -> None:
        """Record receipt of a stream message."""
        if now is None:
            now = time.monotonic()
        self._stats.total_messages += 1
        self._stats.last_message_at = now
        self._message_times.append(now)
        # Keep last 2 minutes of message timestamps.
        cutoff = now - 120.0
        while self._message_times and self._message_times[0] < cutoff:
            self._message_times.popleft()

    def record_reconnect(self, now: float | None = None) -> None:
        """Record a stream reconnection event."""
        if now is None:
            now = time.monotonic()
        self._stats.total_reconnects += 1
        self._stats.last_reconnect_at = now
        self._reconnect_times.append(now)

    def record_gap(self, now: float | None = None) -> None:
        """Record a detected message gap (missing sequence)."""
        if now is None:
            now = time.monotonic()
        self._stats.total_gaps += 1
        self._gap_times.append(now)

    def evaluate(self, now: float | None = None) -> StreamHealthStats:
        """Evaluate current stream health and update stats.

        Returns the updated stats object.
        """
        if now is None:
            now = time.monotonic()
        cfg = self._config
        reasons: list[str] = []

        # 1. Heartbeat check.
        if cfg.heartbeat_timeout_seconds > 0 and self._stats.last_message_at is not None:
            idle = now - self._stats.last_message_at
            if idle > cfg.heartbeat_timeout_seconds:
                reasons.append(
                    f"heartbeat_timeout(idle={idle:.1f}s, max={cfg.heartbeat_timeout_seconds:.1f}s)"
                )

        # 2. Reconnect rate.
        if cfg.max_reconnects_per_window > 0:
            cutoff = now - cfg.reconnect_window_seconds
            while self._reconnect_times and self._reconnect_times[0] < cutoff:
                self._reconnect_times.popleft()
            if len(self._reconnect_times) > cfg.max_reconnects_per_window:
                reasons.append(
                    f"reconnect_storm({len(self._reconnect_times)} in "
                    f"{cfg.reconnect_window_seconds:.0f}s, max={cfg.max_reconnects_per_window})"
                )

        # 3. Gap rate.
        if cfg.max_gap_rate_per_minute > 0:
            cutoff = now - 60.0
            while self._gap_times and self._gap_times[0] < cutoff:
                self._gap_times.popleft()
            gap_rate = len(self._gap_times)
            if gap_rate > cfg.max_gap_rate_per_minute:
                reasons.append(
                    f"gap_rate({gap_rate}/min, max={cfg.max_gap_rate_per_minute:.0f}/min)"
                )

        # 4. Message rate.
        if cfg.min_messages_per_minute > 0 and self._stats.last_message_at is not None:
            cutoff = now - 60.0
            while self._message_times and self._message_times[0] < cutoff:
                self._message_times.popleft()
            msg_rate = len(self._message_times)
            if msg_rate < cfg.min_messages_per_minute:
                reasons.append(
                    f"low_message_rate({msg_rate}/min, min={cfg.min_messages_per_minute:.0f}/min)"
                )

        if reasons:
            self._stats.is_healthy = False
            self._stats.unhealthy_reason = "; ".join(reasons)
            LOGGER.warning(
                "Stream %s unhealthy: %s",
                self._venue,
                self._stats.unhealthy_reason,
            )
        else:
            self._stats.is_healthy = True
            self._stats.unhealthy_reason = ""

        return self._stats

    def is_healthy(self, now: float | None = None) -> bool:
        """Convenience: evaluate and return health status."""
        self.evaluate(now=now)
        return self._stats.is_healthy

    def reset(self) -> None:
        """Reset all state."""
        self._stats.reset()
        self._reconnect_times.clear()
        self._gap_times.clear()
        self._message_times.clear()


# ---------------------------------------------------------------------------
# Multi-venue health registry
# ---------------------------------------------------------------------------


class StreamHealthRegistry:
    """Registry of per-venue stream health monitors."""

    def __init__(self, config: StreamHealthConfig | None = None) -> None:
        self._default_config = config or StreamHealthConfig()
        self._monitors: Dict[str, StreamHealthMonitor] = {}

    def get_or_create(
        self,
        venue: str,
        config: StreamHealthConfig | None = None,
    ) -> StreamHealthMonitor:
        """Get or create a monitor for the given venue."""
        if venue not in self._monitors:
            self._monitors[venue] = StreamHealthMonitor(
                venue, config or self._default_config
            )
        return self._monitors[venue]

    def evaluate_all(self, now: float | None = None) -> dict[str, StreamHealthStats]:
        """Evaluate health for all registered venues."""
        return {
            venue: monitor.evaluate(now=now)
            for venue, monitor in self._monitors.items()
        }

    def all_healthy(self, now: float | None = None) -> bool:
        """Check if all venues are healthy."""
        return all(
            monitor.is_healthy(now=now)
            for monitor in self._monitors.values()
        )

    def unhealthy_venues(self, now: float | None = None) -> list[str]:
        """Return list of unhealthy venue names."""
        return [
            venue
            for venue, monitor in self._monitors.items()
            if not monitor.is_healthy(now=now)
        ]

    def venues(self) -> list[str]:
        return list(self._monitors.keys())

    def reset_all(self) -> None:
        for monitor in self._monitors.values():
            monitor.reset()
