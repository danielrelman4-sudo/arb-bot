"""Tests for Phase 5G: Persistent warm-cache snapshots."""

from __future__ import annotations

import json
import os

import pytest

from arb_bot.framework.warm_cache import (
    CacheEntry,
    SnapshotMeta,
    WarmCache,
    WarmCacheConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache(tmp_path, **kw) -> WarmCache:
    path = str(tmp_path / "warm_cache.json")
    return WarmCache(WarmCacheConfig(snapshot_path=path, **kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = WarmCacheConfig()
        assert cfg.default_ttl == 300.0
        assert cfg.max_entries == 10000
        assert cfg.max_snapshot_age == 3600.0
        assert cfg.auto_save_interval == 60.0

    def test_frozen(self) -> None:
        cfg = WarmCacheConfig()
        with pytest.raises(AttributeError):
            cfg.default_ttl = 100.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Store / get
# ---------------------------------------------------------------------------


class TestStoreGet:
    def test_store_and_get(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        result = c.get("k:BTC", now=100.0)
        assert result == {"bid": 0.55}

    def test_get_nonexistent(self, tmp_path) -> None:
        c = _cache(tmp_path)
        assert c.get("nope", now=100.0) is None

    def test_overwrite(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        c.store("k:BTC", {"bid": 0.60}, now=101.0)
        assert c.get("k:BTC", now=101.0) == {"bid": 0.60}

    def test_hit_count(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        c.get("k:BTC", now=100.0)
        c.get("k:BTC", now=101.0)
        entry = c._entries["k:BTC"]
        assert entry.hit_count == 2


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_expired_entry_returns_none(self, tmp_path) -> None:
        c = _cache(tmp_path, default_ttl=10.0)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        assert c.get("k:BTC", now=111.0) is None

    def test_not_expired(self, tmp_path) -> None:
        c = _cache(tmp_path, default_ttl=10.0)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        assert c.get("k:BTC", now=109.0) is not None

    def test_custom_ttl(self, tmp_path) -> None:
        c = _cache(tmp_path, default_ttl=300.0)
        c.store("k:BTC", {"bid": 0.55}, ttl=5.0, now=100.0)
        assert c.get("k:BTC", now=106.0) is None

    def test_has_returns_false_for_expired(self, tmp_path) -> None:
        c = _cache(tmp_path, default_ttl=10.0)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        assert c.has("k:BTC", now=100.0) is True
        assert c.has("k:BTC", now=111.0) is False

    def test_has_nonexistent(self, tmp_path) -> None:
        c = _cache(tmp_path)
        assert c.has("nope", now=100.0) is False


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        assert c.remove("k:BTC") is True
        assert c.get("k:BTC", now=100.0) is None

    def test_remove_nonexistent(self, tmp_path) -> None:
        c = _cache(tmp_path)
        assert c.remove("nope") is False


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_evicts_when_over_capacity(self, tmp_path) -> None:
        c = _cache(tmp_path, max_entries=3)
        for i in range(5):
            c.store(f"key_{i}", i, now=100.0 + i)
        assert c.entry_count() <= 3

    def test_evicts_oldest_first(self, tmp_path) -> None:
        c = _cache(tmp_path, max_entries=3, default_ttl=1000.0)
        c.store("old", 0, now=100.0)
        c.store("mid", 1, now=200.0)
        c.store("new1", 2, now=300.0)
        c.store("new2", 3, now=400.0)  # Triggers eviction.
        assert c.has("old", now=400.0) is False
        assert c.has("new2", now=400.0) is True


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_save_creates_file(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("k:BTC", {"bid": 0.55}, now=100.0)
        path = c.save_snapshot(now=100.0)
        assert os.path.exists(path)

    def test_save_and_load(self, tmp_path) -> None:
        c1 = _cache(tmp_path, default_ttl=300.0)
        c1.store("k:BTC", {"bid": 0.55}, now=100.0)
        c1.store("k:ETH", {"bid": 0.30}, now=100.0)
        c1.save_snapshot(now=100.0)

        c2 = _cache(tmp_path, default_ttl=300.0)
        meta = c2.load_snapshot(now=150.0)
        assert meta is not None
        assert meta.loaded_count == 2
        assert meta.expired_count == 0
        assert c2.get("k:BTC", now=150.0) == {"bid": 0.55}

    def test_expired_entries_not_loaded(self, tmp_path) -> None:
        c1 = _cache(tmp_path, default_ttl=30.0)
        c1.store("k:BTC", {"bid": 0.55}, now=100.0)
        c1.save_snapshot(now=100.0)

        c2 = _cache(tmp_path, default_ttl=30.0)
        meta = c2.load_snapshot(now=200.0)  # 100s > 30s TTL.
        assert meta is not None
        assert meta.loaded_count == 0
        assert meta.expired_count == 1

    def test_snapshot_too_old(self, tmp_path) -> None:
        c1 = _cache(tmp_path, max_snapshot_age=60.0, default_ttl=1000.0)
        c1.store("k:BTC", {"bid": 0.55}, now=100.0)
        c1.save_snapshot(now=100.0)

        c2 = _cache(tmp_path, max_snapshot_age=60.0)
        meta = c2.load_snapshot(now=200.0)  # 100s > 60s max age.
        assert meta is not None
        assert meta.loaded_count == 0

    def test_no_snapshot_file(self, tmp_path) -> None:
        c = _cache(tmp_path)
        meta = c.load_snapshot(now=100.0)
        assert meta is None

    def test_corrupt_snapshot(self, tmp_path) -> None:
        path = str(tmp_path / "warm_cache.json")
        with open(path, "w") as f:
            f.write("not json!")
        c = WarmCache(WarmCacheConfig(snapshot_path=path))
        meta = c.load_snapshot(now=100.0)
        assert meta is None

    def test_expired_entries_pruned_from_save(self, tmp_path) -> None:
        c = _cache(tmp_path, default_ttl=10.0)
        c.store("fresh", 1, now=100.0)
        c.store("stale", 2, now=80.0)  # Will be expired at save time.
        c.save_snapshot(now=100.0)

        with open(c.config.snapshot_path) as f:
            data = json.load(f)
        assert "fresh" in data["entries"]
        assert "stale" not in data["entries"]


# ---------------------------------------------------------------------------
# Auto-save
# ---------------------------------------------------------------------------


class TestAutoSave:
    def test_should_auto_save(self, tmp_path) -> None:
        c = _cache(tmp_path, auto_save_interval=30.0)
        assert c.should_auto_save(now=100.0) is True  # Never saved.

    def test_not_due_after_save(self, tmp_path) -> None:
        c = _cache(tmp_path, auto_save_interval=30.0)
        c.save_snapshot(now=100.0)
        assert c.should_auto_save(now=120.0) is False  # 20s < 30s.

    def test_due_after_interval(self, tmp_path) -> None:
        c = _cache(tmp_path, auto_save_interval=30.0)
        c.save_snapshot(now=100.0)
        assert c.should_auto_save(now=131.0) is True  # 31s > 30s.

    def test_disabled(self, tmp_path) -> None:
        c = _cache(tmp_path, auto_save_interval=0.0)
        assert c.should_auto_save(now=100.0) is False


# ---------------------------------------------------------------------------
# Entry count / keys
# ---------------------------------------------------------------------------


class TestMisc:
    def test_entry_count(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("a", 1, now=100.0)
        c.store("b", 2, now=100.0)
        assert c.entry_count() == 2

    def test_keys(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("a", 1, now=100.0)
        c.store("b", 2, now=100.0)
        assert set(c.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self, tmp_path) -> None:
        c = _cache(tmp_path)
        c.store("a", 1, now=100.0)
        c.clear()
        assert c.entry_count() == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self, tmp_path) -> None:
        path = str(tmp_path / "cache.json")
        cfg = WarmCacheConfig(snapshot_path=path, default_ttl=120.0)
        c = WarmCache(cfg)
        assert c.config.default_ttl == 120.0


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_is_expired(self) -> None:
        e = CacheEntry(key="k", data=1, stored_at=100.0, ttl=10.0)
        assert e.is_expired(now=109.0) is False
        assert e.is_expired(now=111.0) is True


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self, tmp_path) -> None:
        """Store → save → new instance → load → verify."""
        c1 = _cache(tmp_path, default_ttl=300.0, max_snapshot_age=600.0)

        # Populate cache.
        for i in range(20):
            c1.store(f"market_{i}", {"bid": 0.5 + i * 0.01}, now=1000.0)

        # Save.
        c1.save_snapshot(now=1000.0)

        # Load in new instance.
        c2 = _cache(tmp_path, default_ttl=300.0, max_snapshot_age=600.0)
        meta = c2.load_snapshot(now=1100.0)
        assert meta is not None
        assert meta.loaded_count == 20

        # Verify data integrity.
        for i in range(20):
            data = c2.get(f"market_{i}", now=1100.0)
            assert data is not None
            assert data["bid"] == pytest.approx(0.5 + i * 0.01)

        # After TTL, entries expire.
        expired_data = c2.get("market_0", now=1400.0)
        assert expired_data is None
