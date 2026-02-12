"""Persistent warm-cache snapshots (Phase 5G).

Persists quote cache and coverage state to disk, reloads at startup
to reduce time-to-readiness after restarts.

Usage::

    cache = WarmCache(config)
    cache.store("kalshi:BTC-50K", {"bid": 0.55, "ask": 0.57}, ttl=60.0)
    cache.save_snapshot()
    # After restart:
    cache.load_snapshot()
    entry = cache.get("kalshi:BTC-50K")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmCacheConfig:
    """Configuration for warm cache.

    Parameters
    ----------
    snapshot_path:
        Path to snapshot file. Default "warm_cache.json".
    default_ttl:
        Default time-to-live for cache entries (seconds).
        Default 300.
    max_entries:
        Maximum cache entries. Default 10000.
    max_snapshot_age:
        Maximum age of snapshot to load (seconds). Older
        snapshots are discarded. Default 3600.
    auto_save_interval:
        Seconds between auto-saves. 0 = manual only. Default 60.
    """

    snapshot_path: str = "warm_cache.json"
    default_ttl: float = 300.0
    max_entries: int = 10000
    max_snapshot_age: float = 3600.0
    auto_save_interval: float = 60.0


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cache entry."""

    key: str
    data: Any
    stored_at: float
    ttl: float
    hit_count: int = 0

    def is_expired(self, now: float) -> bool:
        return now - self.stored_at > self.ttl


# ---------------------------------------------------------------------------
# Cache snapshot metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotMeta:
    """Metadata about a loaded snapshot."""

    path: str
    saved_at: float
    entry_count: int
    loaded_count: int
    expired_count: int
    age_seconds: float


# ---------------------------------------------------------------------------
# Warm cache
# ---------------------------------------------------------------------------


class WarmCache:
    """Persistent warm cache for quote data and coverage state.

    Stores key-value entries with TTL, persists to disk as JSON,
    and reloads on startup. Expired entries are pruned on load.
    """

    def __init__(self, config: WarmCacheConfig | None = None) -> None:
        self._config = config or WarmCacheConfig()
        self._entries: Dict[str, CacheEntry] = {}
        self._last_save_time: float = 0.0

    @property
    def config(self) -> WarmCacheConfig:
        return self._config

    def store(
        self,
        key: str,
        data: Any,
        ttl: float | None = None,
        now: float | None = None,
    ) -> None:
        """Store a value in the cache.

        Parameters
        ----------
        key:
            Cache key.
        data:
            JSON-serializable data.
        ttl:
            Time-to-live in seconds. If None, uses default_ttl.
        now:
            Current timestamp.
        """
        if now is None:
            now = time.time()
        if ttl is None:
            ttl = self._config.default_ttl

        self._entries[key] = CacheEntry(
            key=key, data=data, stored_at=now, ttl=ttl,
        )

        # Evict if over capacity.
        if len(self._entries) > self._config.max_entries:
            self._evict(now)

    def get(
        self,
        key: str,
        now: float | None = None,
    ) -> Any | None:
        """Get a value from the cache.

        Returns None if not found or expired.
        """
        if now is None:
            now = time.time()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_expired(now):
            del self._entries[key]
            return None
        entry.hit_count += 1
        return entry.data

    def has(self, key: str, now: float | None = None) -> bool:
        """Check if a key exists and is not expired."""
        if now is None:
            now = time.time()
        entry = self._entries.get(key)
        if entry is None:
            return False
        return not entry.is_expired(now)

    def remove(self, key: str) -> bool:
        """Remove a key from the cache."""
        return self._entries.pop(key, None) is not None

    def _evict(self, now: float) -> None:
        """Evict expired and oldest entries to stay under max_entries."""
        # First remove expired.
        expired = [k for k, v in self._entries.items() if v.is_expired(now)]
        for k in expired:
            del self._entries[k]

        # If still over, remove oldest (lowest stored_at).
        if len(self._entries) > self._config.max_entries:
            sorted_keys = sorted(
                self._entries.keys(),
                key=lambda k: self._entries[k].stored_at,
            )
            excess = len(self._entries) - self._config.max_entries
            for k in sorted_keys[:excess]:
                del self._entries[k]

    def save_snapshot(self, now: float | None = None) -> str:
        """Save cache to disk.

        Returns the path where snapshot was saved.
        """
        if now is None:
            now = time.time()

        snapshot = {
            "saved_at": now,
            "entry_count": len(self._entries),
            "entries": {},
        }

        for key, entry in self._entries.items():
            if not entry.is_expired(now):
                snapshot["entries"][key] = {
                    "data": entry.data,
                    "stored_at": entry.stored_at,
                    "ttl": entry.ttl,
                    "hit_count": entry.hit_count,
                }

        path = self._config.snapshot_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snapshot, f)

        self._last_save_time = now
        return path

    def load_snapshot(self, now: float | None = None) -> SnapshotMeta | None:
        """Load cache from disk.

        Returns metadata about the load, or None if no snapshot
        or snapshot is too old.
        """
        if now is None:
            now = time.time()

        path = self._config.snapshot_path
        if not os.path.exists(path):
            return None

        try:
            with open(path) as f:
                snapshot = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        saved_at = snapshot.get("saved_at", 0.0)
        age = now - saved_at

        if age > self._config.max_snapshot_age:
            return SnapshotMeta(
                path=path,
                saved_at=saved_at,
                entry_count=snapshot.get("entry_count", 0),
                loaded_count=0,
                expired_count=0,
                age_seconds=age,
            )

        entries = snapshot.get("entries", {})
        loaded = 0
        expired = 0

        for key, entry_data in entries.items():
            stored_at = entry_data.get("stored_at", 0.0)
            ttl = entry_data.get("ttl", self._config.default_ttl)
            data = entry_data.get("data")

            entry = CacheEntry(
                key=key, data=data,
                stored_at=stored_at, ttl=ttl,
                hit_count=entry_data.get("hit_count", 0),
            )

            if entry.is_expired(now):
                expired += 1
            else:
                self._entries[key] = entry
                loaded += 1

        return SnapshotMeta(
            path=path,
            saved_at=saved_at,
            entry_count=len(entries),
            loaded_count=loaded,
            expired_count=expired,
            age_seconds=age,
        )

    def should_auto_save(self, now: float | None = None) -> bool:
        """Check if an auto-save is due."""
        if self._config.auto_save_interval <= 0:
            return False
        if now is None:
            now = time.time()
        return (now - self._last_save_time) >= self._config.auto_save_interval

    def entry_count(self) -> int:
        """Number of entries in cache."""
        return len(self._entries)

    def keys(self) -> List[str]:
        """All cache keys."""
        return list(self._entries.keys())

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
