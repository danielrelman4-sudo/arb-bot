"""Deterministic dedupe + idempotency keys (Phase 2E).

Provides stable opportunity IDs derived from the canonical leg set,
a cross-cycle dedupe registry that prevents re-executing the same
opportunity within a configurable cooldown window, and idempotency
key generation for safe order resubmission.

The dedupe key is deterministic: same legs → same key, regardless of
leg ordering or metadata differences. The cooldown prevents rapid
re-execution while allowing the same opportunity to be traded again
after the window expires.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dedupe key generation
# ---------------------------------------------------------------------------


def opportunity_dedupe_key(
    kind: str,
    execution_style: str,
    legs: Sequence[tuple[str, str, str]],
) -> str:
    """Generate a deterministic dedupe key from opportunity legs.

    Parameters
    ----------
    kind:
        Opportunity kind (e.g., "cross_venue", "intra_venue").
    execution_style:
        Execution style (e.g., "taker", "maker_estimate").
    legs:
        Sequence of (venue, market_id, side) tuples.

    Returns
    -------
    Hex SHA-256 hash (first 16 chars) of the canonical representation.
    """
    # Sort legs to ensure deterministic ordering.
    sorted_legs = sorted(legs)
    parts = [kind, execution_style]
    for venue, market_id, side in sorted_legs:
        parts.extend([venue, market_id, side])
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def idempotency_key(
    dedupe_key: str,
    timestamp_bucket_seconds: float = 60.0,
    now: float | None = None,
) -> str:
    """Generate an idempotency key for order submission.

    Combines the dedupe key with a time bucket so the same opportunity
    in the same time window gets the same idempotency key (safe for
    retries), but a different bucket means a new submission.

    Parameters
    ----------
    dedupe_key:
        The opportunity dedupe key.
    timestamp_bucket_seconds:
        Time bucket width. Default 60 seconds.
    now:
        Current time as Unix timestamp. Defaults to time.time().

    Returns
    -------
    Hex SHA-256 hash (first 24 chars).
    """
    if now is None:
        now = time.time()
    bucket = int(now / timestamp_bucket_seconds)
    raw = f"{dedupe_key}:{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Dedupe registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupeConfig:
    """Configuration for the dedupe registry.

    Parameters
    ----------
    cooldown_seconds:
        How long after submission before the same opportunity can be
        re-executed. Default 300 (5 minutes).
    max_entries:
        Maximum number of entries to track. Oldest entries are evicted
        when this is exceeded. Default 10000.
    enabled:
        Master enable/disable. Default True.
    """

    cooldown_seconds: float = 300.0
    max_entries: int = 10000
    enabled: bool = True


@dataclass
class DedupeEntry:
    """A single dedupe registry entry."""

    dedupe_key: str
    kind: str
    execution_style: str
    submitted_at: float
    intent_id: str = ""
    idempotency_key: str = ""


@dataclass
class DedupeStats:
    """Statistics for the dedupe registry."""

    total_checked: int = 0
    total_blocked: int = 0
    total_registered: int = 0
    total_evicted: int = 0

    def reset(self) -> None:
        self.total_checked = 0
        self.total_blocked = 0
        self.total_registered = 0
        self.total_evicted = 0


class DedupeRegistry:
    """Cross-cycle dedupe registry.

    Tracks recently submitted opportunities and blocks re-execution
    within the cooldown window. Supports manual clearing for testing
    and operational reset.
    """

    def __init__(self, config: DedupeConfig | None = None) -> None:
        self._config = config or DedupeConfig()
        self._entries: Dict[str, DedupeEntry] = {}
        self._insertion_order: list[str] = []
        self._stats = DedupeStats()

    @property
    def config(self) -> DedupeConfig:
        return self._config

    @property
    def stats(self) -> DedupeStats:
        return self._stats

    def is_duplicate(
        self,
        dedupe_key: str,
        now: float | None = None,
    ) -> bool:
        """Check if this opportunity is a duplicate (within cooldown).

        Parameters
        ----------
        dedupe_key:
            The opportunity dedupe key.
        now:
            Current time as Unix timestamp. Defaults to time.time().

        Returns
        -------
        True if the key is in the registry and within cooldown.
        """
        if not self._config.enabled:
            return False

        if now is None:
            now = time.time()

        self._stats.total_checked += 1

        entry = self._entries.get(dedupe_key)
        if entry is None:
            return False

        elapsed = now - entry.submitted_at
        if elapsed < self._config.cooldown_seconds:
            self._stats.total_blocked += 1
            LOGGER.debug(
                "Dedupe blocked %s (%.1fs remaining in cooldown)",
                dedupe_key,
                self._config.cooldown_seconds - elapsed,
            )
            return True

        # Cooldown expired — remove stale entry.
        self._remove(dedupe_key)
        return False

    def register(
        self,
        dedupe_key: str,
        kind: str = "",
        execution_style: str = "",
        intent_id: str = "",
        idempotency_key: str = "",
        now: float | None = None,
    ) -> None:
        """Register an opportunity as submitted.

        Parameters
        ----------
        dedupe_key:
            The opportunity dedupe key.
        kind:
            Opportunity kind (for logging/debugging).
        execution_style:
            Execution style (for logging/debugging).
        intent_id:
            Order intent ID (for cross-reference).
        idempotency_key:
            Idempotency key used for this submission.
        now:
            Current time as Unix timestamp.
        """
        if now is None:
            now = time.time()

        self._stats.total_registered += 1

        entry = DedupeEntry(
            dedupe_key=dedupe_key,
            kind=kind,
            execution_style=execution_style,
            submitted_at=now,
            intent_id=intent_id,
            idempotency_key=idempotency_key,
        )
        self._entries[dedupe_key] = entry
        self._insertion_order.append(dedupe_key)

        # Evict oldest if over capacity.
        self._evict_if_needed()

    def remove(self, dedupe_key: str) -> bool:
        """Remove a specific entry. Returns True if found."""
        return self._remove(dedupe_key)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._insertion_order.clear()

    def size(self) -> int:
        return len(self._entries)

    def get_entry(self, dedupe_key: str) -> DedupeEntry | None:
        return self._entries.get(dedupe_key)

    def purge_expired(self, now: float | None = None) -> int:
        """Remove all expired entries. Returns count removed."""
        if now is None:
            now = time.time()

        expired = [
            key for key, entry in self._entries.items()
            if (now - entry.submitted_at) >= self._config.cooldown_seconds
        ]
        for key in expired:
            self._remove(key)
        return len(expired)

    def _remove(self, key: str) -> bool:
        if key in self._entries:
            del self._entries[key]
            # Don't bother removing from insertion_order — it's
            # only used for eviction and we check existence.
            return True
        return False

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._config.max_entries:
            # Find oldest still-present key.
            while self._insertion_order:
                oldest = self._insertion_order.pop(0)
                if oldest in self._entries:
                    del self._entries[oldest]
                    self._stats.total_evicted += 1
                    break
