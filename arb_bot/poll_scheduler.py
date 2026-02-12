"""Tiered polling scheduler (Phase 5B).

Markets are placed in hot/warm/cold tiers with different polling
intervals. Promotion happens on stream updates, position changes,
or opportunity detection. Demotion is automatic after inactivity.

Usage::

    sched = PollScheduler(config)
    sched.register("kalshi", "BTC-50K")
    sched.promote("kalshi", "BTC-50K", reason="stream_update")
    items = sched.due_items(now)
    for item in items:
        # poll item.venue / item.market_id
        sched.mark_polled(item.venue, item.market_id, now)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class PollTier(str, Enum):
    """Polling tier — determines poll frequency."""

    HOT = "hot"  # Sub-second to 1s.
    WARM = "warm"  # A few seconds.
    COLD = "cold"  # Tens of seconds to minutes.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollSchedulerConfig:
    """Configuration for tiered polling scheduler.

    Parameters
    ----------
    hot_interval:
        Polling interval for hot tier (seconds). Default 0.5.
    warm_interval:
        Polling interval for warm tier (seconds). Default 5.0.
    cold_interval:
        Polling interval for cold tier (seconds). Default 60.0.
    default_tier:
        Initial tier for new markets. Default WARM.
    hot_to_warm_seconds:
        Inactivity timeout before demoting hot→warm. Default 30.
    warm_to_cold_seconds:
        Inactivity timeout before demoting warm→cold. Default 300.
    max_hot_items:
        Maximum number of items in hot tier. Default 50.
    max_warm_items:
        Maximum number of items in warm tier. Default 200.
    """

    hot_interval: float = 0.5
    warm_interval: float = 5.0
    cold_interval: float = 60.0
    default_tier: PollTier = PollTier.WARM
    hot_to_warm_seconds: float = 30.0
    warm_to_cold_seconds: float = 300.0
    max_hot_items: int = 50
    max_warm_items: int = 200


# ---------------------------------------------------------------------------
# Poll item state
# ---------------------------------------------------------------------------


@dataclass
class PollItemState:
    """State for a single polled market."""

    venue: str
    market_id: str
    tier: PollTier
    last_activity_time: float
    last_poll_time: float
    poll_count: int = 0
    promotion_reason: str = ""


# ---------------------------------------------------------------------------
# Due item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DueItem:
    """An item that is due for polling."""

    venue: str
    market_id: str
    tier: PollTier
    overdue_seconds: float


# ---------------------------------------------------------------------------
# Scheduler snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Snapshot of scheduler state for monitoring."""

    total_items: int
    hot_count: int
    warm_count: int
    cold_count: int
    total_polls: int


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class PollScheduler:
    """Tiered polling scheduler for market data.

    Markets are assigned to hot/warm/cold tiers with different
    polling intervals. Tiers are promoted on activity events
    and auto-demoted after inactivity.
    """

    def __init__(self, config: PollSchedulerConfig | None = None) -> None:
        self._config = config or PollSchedulerConfig()
        self._items: Dict[str, PollItemState] = {}

    @property
    def config(self) -> PollSchedulerConfig:
        return self._config

    def _key(self, venue: str, market_id: str) -> str:
        return f"{venue}:{market_id}"

    def _tier_interval(self, tier: PollTier) -> float:
        if tier == PollTier.HOT:
            return self._config.hot_interval
        elif tier == PollTier.WARM:
            return self._config.warm_interval
        else:
            return self._config.cold_interval

    def register(
        self,
        venue: str,
        market_id: str,
        tier: PollTier | None = None,
        now: float | None = None,
    ) -> None:
        """Register a market for polling.

        Parameters
        ----------
        venue:
            Venue identifier.
        market_id:
            Market identifier.
        tier:
            Initial tier. If None, uses default_tier from config.
        now:
            Current timestamp. If None, uses time.monotonic().
        """
        if now is None:
            now = time.monotonic()
        key = self._key(venue, market_id)
        if key not in self._items:
            self._items[key] = PollItemState(
                venue=venue,
                market_id=market_id,
                tier=tier or self._config.default_tier,
                last_activity_time=now,
                last_poll_time=0.0,  # Never polled.
            )

    def unregister(self, venue: str, market_id: str) -> None:
        """Remove a market from polling."""
        key = self._key(venue, market_id)
        self._items.pop(key, None)

    def promote(
        self,
        venue: str,
        market_id: str,
        reason: str = "",
        now: float | None = None,
    ) -> None:
        """Promote a market to a hotter tier.

        Parameters
        ----------
        venue:
            Venue identifier.
        market_id:
            Market identifier.
        reason:
            Reason for promotion (e.g., "stream_update", "position",
            "opportunity").
        now:
            Current timestamp. If None, uses time.monotonic().
        """
        if now is None:
            now = time.monotonic()
        key = self._key(venue, market_id)
        state = self._items.get(key)
        if state is None:
            return

        state.last_activity_time = now
        state.promotion_reason = reason

        if state.tier == PollTier.COLD:
            state.tier = PollTier.WARM
        elif state.tier == PollTier.WARM:
            state.tier = PollTier.HOT

    def promote_to_hot(
        self,
        venue: str,
        market_id: str,
        reason: str = "",
        now: float | None = None,
    ) -> None:
        """Promote a market directly to hot tier."""
        if now is None:
            now = time.monotonic()
        key = self._key(venue, market_id)
        state = self._items.get(key)
        if state is None:
            return

        state.last_activity_time = now
        state.promotion_reason = reason
        state.tier = PollTier.HOT

    def demote(self, venue: str, market_id: str) -> None:
        """Manually demote a market one tier."""
        key = self._key(venue, market_id)
        state = self._items.get(key)
        if state is None:
            return

        if state.tier == PollTier.HOT:
            state.tier = PollTier.WARM
        elif state.tier == PollTier.WARM:
            state.tier = PollTier.COLD

    def _auto_demote(self, now: float) -> None:
        """Auto-demote items that have been inactive too long."""
        cfg = self._config
        for state in self._items.values():
            elapsed = now - state.last_activity_time
            if state.tier == PollTier.HOT:
                if elapsed >= cfg.hot_to_warm_seconds:
                    state.tier = PollTier.WARM
            elif state.tier == PollTier.WARM:
                if elapsed >= cfg.warm_to_cold_seconds:
                    state.tier = PollTier.COLD

    def _enforce_tier_limits(self) -> None:
        """Demote excess items from hot/warm tiers (oldest activity first)."""
        cfg = self._config

        # Enforce hot tier limit.
        hot_items = [s for s in self._items.values() if s.tier == PollTier.HOT]
        if len(hot_items) > cfg.max_hot_items:
            hot_items.sort(key=lambda s: s.last_activity_time)
            for item in hot_items[: len(hot_items) - cfg.max_hot_items]:
                item.tier = PollTier.WARM

        # Enforce warm tier limit.
        warm_items = [s for s in self._items.values() if s.tier == PollTier.WARM]
        if len(warm_items) > cfg.max_warm_items:
            warm_items.sort(key=lambda s: s.last_activity_time)
            for item in warm_items[: len(warm_items) - cfg.max_warm_items]:
                item.tier = PollTier.COLD

    def due_items(self, now: float | None = None) -> List[DueItem]:
        """Get items that are due for polling.

        Returns items sorted by overdue time (most overdue first).
        Also runs auto-demotion and tier limit enforcement.
        """
        if now is None:
            now = time.monotonic()

        self._auto_demote(now)
        self._enforce_tier_limits()

        due: List[DueItem] = []
        for state in self._items.values():
            interval = self._tier_interval(state.tier)
            if state.last_poll_time == 0.0:
                # Never polled — always due.
                overdue = interval
            else:
                elapsed = now - state.last_poll_time
                if elapsed >= interval:
                    overdue = elapsed - interval
                else:
                    continue
            due.append(DueItem(
                venue=state.venue,
                market_id=state.market_id,
                tier=state.tier,
                overdue_seconds=overdue,
            ))

        due.sort(key=lambda d: d.overdue_seconds, reverse=True)
        return due

    def mark_polled(
        self,
        venue: str,
        market_id: str,
        now: float | None = None,
    ) -> None:
        """Mark an item as just polled."""
        if now is None:
            now = time.monotonic()
        key = self._key(venue, market_id)
        state = self._items.get(key)
        if state is not None:
            state.last_poll_time = now
            state.poll_count += 1

    def get_state(self, venue: str, market_id: str) -> PollItemState | None:
        """Get state for a specific item."""
        key = self._key(venue, market_id)
        return self._items.get(key)

    def get_tier(self, venue: str, market_id: str) -> PollTier | None:
        """Get current tier for an item."""
        state = self.get_state(venue, market_id)
        return state.tier if state else None

    def tier_counts(self) -> Dict[PollTier, int]:
        """Count items in each tier."""
        counts: Dict[PollTier, int] = {t: 0 for t in PollTier}
        for state in self._items.values():
            counts[state.tier] += 1
        return counts

    def snapshot(self) -> SchedulerSnapshot:
        """Get a snapshot of scheduler state."""
        counts = self.tier_counts()
        total_polls = sum(s.poll_count for s in self._items.values())
        return SchedulerSnapshot(
            total_items=len(self._items),
            hot_count=counts[PollTier.HOT],
            warm_count=counts[PollTier.WARM],
            cold_count=counts[PollTier.COLD],
            total_polls=total_polls,
        )

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()
