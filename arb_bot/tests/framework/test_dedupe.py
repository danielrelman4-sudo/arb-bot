"""Tests for Phase 2E: Deterministic dedupe + idempotency keys."""

from __future__ import annotations

import pytest

from arb_bot.framework.dedupe import (
    DedupeConfig,
    DedupeEntry,
    DedupeRegistry,
    DedupeStats,
    idempotency_key,
    opportunity_dedupe_key,
)


# ---------------------------------------------------------------------------
# Dedupe key generation
# ---------------------------------------------------------------------------


class TestDedupeKeyGeneration:
    def test_deterministic(self) -> None:
        """Same inputs produce same key."""
        legs = [("kalshi", "M1", "yes"), ("polymarket", "P1", "no")]
        k1 = opportunity_dedupe_key("cross_venue", "taker", legs)
        k2 = opportunity_dedupe_key("cross_venue", "taker", legs)
        assert k1 == k2

    def test_order_invariant(self) -> None:
        """Leg order doesn't matter."""
        legs_a = [("kalshi", "M1", "yes"), ("polymarket", "P1", "no")]
        legs_b = [("polymarket", "P1", "no"), ("kalshi", "M1", "yes")]
        assert opportunity_dedupe_key("cross_venue", "taker", legs_a) == \
               opportunity_dedupe_key("cross_venue", "taker", legs_b)

    def test_different_kind_different_key(self) -> None:
        legs = [("kalshi", "M1", "yes")]
        k1 = opportunity_dedupe_key("intra_venue", "taker", legs)
        k2 = opportunity_dedupe_key("cross_venue", "taker", legs)
        assert k1 != k2

    def test_different_style_different_key(self) -> None:
        legs = [("kalshi", "M1", "yes")]
        k1 = opportunity_dedupe_key("intra_venue", "taker", legs)
        k2 = opportunity_dedupe_key("intra_venue", "maker_estimate", legs)
        assert k1 != k2

    def test_different_legs_different_key(self) -> None:
        legs_a = [("kalshi", "M1", "yes")]
        legs_b = [("kalshi", "M2", "yes")]
        k1 = opportunity_dedupe_key("intra_venue", "taker", legs_a)
        k2 = opportunity_dedupe_key("intra_venue", "taker", legs_b)
        assert k1 != k2

    def test_key_length(self) -> None:
        legs = [("kalshi", "M1", "yes")]
        key = opportunity_dedupe_key("intra_venue", "taker", legs)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Idempotency key generation
# ---------------------------------------------------------------------------


class TestIdempotencyKey:
    def test_deterministic_same_bucket(self) -> None:
        """Same dedupe key in same time bucket → same idempotency key."""
        # Both in bucket 16 (960-1019).
        k1 = idempotency_key("abc123", timestamp_bucket_seconds=60.0, now=960.0)
        k2 = idempotency_key("abc123", timestamp_bucket_seconds=60.0, now=990.0)
        assert k1 == k2

    def test_different_bucket_different_key(self) -> None:
        """Different time bucket → different idempotency key."""
        k1 = idempotency_key("abc123", timestamp_bucket_seconds=60.0, now=1000.0)
        k2 = idempotency_key("abc123", timestamp_bucket_seconds=60.0, now=1061.0)
        assert k1 != k2

    def test_different_dedupe_key_different_idem(self) -> None:
        k1 = idempotency_key("abc", timestamp_bucket_seconds=60.0, now=1000.0)
        k2 = idempotency_key("xyz", timestamp_bucket_seconds=60.0, now=1000.0)
        assert k1 != k2

    def test_key_length(self) -> None:
        key = idempotency_key("test", now=1000.0)
        assert len(key) == 24


# ---------------------------------------------------------------------------
# DedupeConfig
# ---------------------------------------------------------------------------


class TestDedupeConfig:
    def test_defaults(self) -> None:
        cfg = DedupeConfig()
        assert cfg.cooldown_seconds == 300.0
        assert cfg.max_entries == 10000
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# DedupeEntry
# ---------------------------------------------------------------------------


class TestDedupeEntry:
    def test_fields(self) -> None:
        entry = DedupeEntry(
            dedupe_key="abc",
            kind="cross_venue",
            execution_style="taker",
            submitted_at=1000.0,
            intent_id="intent_1",
            idempotency_key="idem_1",
        )
        assert entry.dedupe_key == "abc"
        assert entry.kind == "cross_venue"
        assert entry.intent_id == "intent_1"


# ---------------------------------------------------------------------------
# DedupeStats
# ---------------------------------------------------------------------------


class TestDedupeStats:
    def test_defaults(self) -> None:
        stats = DedupeStats()
        assert stats.total_checked == 0
        assert stats.total_blocked == 0
        assert stats.total_registered == 0
        assert stats.total_evicted == 0

    def test_reset(self) -> None:
        stats = DedupeStats(total_checked=5, total_blocked=2)
        stats.reset()
        assert stats.total_checked == 0
        assert stats.total_blocked == 0


# ---------------------------------------------------------------------------
# DedupeRegistry — basic
# ---------------------------------------------------------------------------


class TestRegistryBasic:
    def test_not_duplicate_when_empty(self) -> None:
        reg = DedupeRegistry()
        assert reg.is_duplicate("abc", now=1000.0) is False

    def test_register_then_duplicate(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        assert reg.is_duplicate("abc", now=1010.0) is True

    def test_cooldown_expired(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        assert reg.is_duplicate("abc", now=1070.0) is False

    def test_different_keys_independent(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        assert reg.is_duplicate("xyz", now=1010.0) is False

    def test_size(self) -> None:
        reg = DedupeRegistry()
        reg.register("a", now=1000.0)
        reg.register("b", now=1000.0)
        assert reg.size() == 2


# ---------------------------------------------------------------------------
# DedupeRegistry — disabled
# ---------------------------------------------------------------------------


class TestRegistryDisabled:
    def test_disabled_never_blocks(self) -> None:
        reg = DedupeRegistry(DedupeConfig(enabled=False, cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        assert reg.is_duplicate("abc", now=1010.0) is False


# ---------------------------------------------------------------------------
# DedupeRegistry — removal
# ---------------------------------------------------------------------------


class TestRegistryRemoval:
    def test_remove_existing(self) -> None:
        reg = DedupeRegistry()
        reg.register("abc", now=1000.0)
        assert reg.remove("abc") is True
        assert reg.size() == 0

    def test_remove_nonexistent(self) -> None:
        reg = DedupeRegistry()
        assert reg.remove("abc") is False

    def test_clear(self) -> None:
        reg = DedupeRegistry()
        reg.register("a", now=1000.0)
        reg.register("b", now=1000.0)
        reg.clear()
        assert reg.size() == 0


# ---------------------------------------------------------------------------
# DedupeRegistry — eviction
# ---------------------------------------------------------------------------


class TestRegistryEviction:
    def test_evicts_oldest(self) -> None:
        reg = DedupeRegistry(DedupeConfig(max_entries=3))
        reg.register("a", now=1000.0)
        reg.register("b", now=1001.0)
        reg.register("c", now=1002.0)
        reg.register("d", now=1003.0)  # Should evict "a".
        assert reg.size() == 3
        assert reg.get_entry("a") is None
        assert reg.get_entry("d") is not None

    def test_eviction_stats(self) -> None:
        reg = DedupeRegistry(DedupeConfig(max_entries=2))
        reg.register("a", now=1000.0)
        reg.register("b", now=1001.0)
        reg.register("c", now=1002.0)
        assert reg.stats.total_evicted == 1


# ---------------------------------------------------------------------------
# DedupeRegistry — purge expired
# ---------------------------------------------------------------------------


class TestRegistryPurge:
    def test_purge_expired(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("a", now=1000.0)
        reg.register("b", now=1050.0)
        removed = reg.purge_expired(now=1065.0)
        assert removed == 1  # "a" expired, "b" still valid.
        assert reg.size() == 1
        assert reg.get_entry("b") is not None

    def test_purge_none_expired(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("a", now=1000.0)
        removed = reg.purge_expired(now=1010.0)
        assert removed == 0

    def test_purge_all_expired(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("a", now=1000.0)
        reg.register("b", now=1000.0)
        removed = reg.purge_expired(now=1070.0)
        assert removed == 2
        assert reg.size() == 0


# ---------------------------------------------------------------------------
# DedupeRegistry — get_entry
# ---------------------------------------------------------------------------


class TestGetEntry:
    def test_existing(self) -> None:
        reg = DedupeRegistry()
        reg.register("abc", kind="cross_venue", intent_id="i1", now=1000.0)
        entry = reg.get_entry("abc")
        assert entry is not None
        assert entry.kind == "cross_venue"
        assert entry.intent_id == "i1"

    def test_nonexistent(self) -> None:
        reg = DedupeRegistry()
        assert reg.get_entry("abc") is None


# ---------------------------------------------------------------------------
# DedupeRegistry — stats tracking
# ---------------------------------------------------------------------------


class TestRegistryStats:
    def test_check_increments(self) -> None:
        reg = DedupeRegistry()
        reg.is_duplicate("abc", now=1000.0)
        assert reg.stats.total_checked == 1

    def test_blocked_increments(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        reg.is_duplicate("abc", now=1010.0)
        assert reg.stats.total_blocked == 1

    def test_registered_increments(self) -> None:
        reg = DedupeRegistry()
        reg.register("abc", now=1000.0)
        assert reg.stats.total_registered == 1


# ---------------------------------------------------------------------------
# DedupeRegistry — re-register after cooldown
# ---------------------------------------------------------------------------


class TestReRegister:
    def test_reregister_after_cooldown(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        # After cooldown, not a duplicate.
        assert reg.is_duplicate("abc", now=1070.0) is False
        # Re-register.
        reg.register("abc", now=1070.0)
        # Now it's a duplicate again.
        assert reg.is_duplicate("abc", now=1080.0) is True

    def test_overwrite_updates_timestamp(self) -> None:
        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register("abc", now=1000.0)
        reg.register("abc", now=1050.0)  # Overwrite.
        # Cooldown from 1050, so at 1070 it's still blocking.
        assert reg.is_duplicate("abc", now=1070.0) is True
        # But at 1115 it's expired.
        assert reg.is_duplicate("abc", now=1115.0) is False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self) -> None:
        """Generate key → check dedupe → register → check again."""
        legs = [("kalshi", "M1", "yes"), ("polymarket", "P1", "no")]
        key = opportunity_dedupe_key("cross_venue", "taker", legs)
        idem = idempotency_key(key, timestamp_bucket_seconds=60.0, now=1000.0)

        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=120.0))

        # First check — not duplicate.
        assert reg.is_duplicate(key, now=1000.0) is False

        # Register submission.
        reg.register(
            key,
            kind="cross_venue",
            execution_style="taker",
            intent_id="intent_abc",
            idempotency_key=idem,
            now=1000.0,
        )

        # Second check — duplicate.
        assert reg.is_duplicate(key, now=1010.0) is True

        # After cooldown — not duplicate.
        assert reg.is_duplicate(key, now=1130.0) is False

    def test_order_invariant_workflow(self) -> None:
        """Same legs in different order → same dedupe key."""
        legs_a = [("kalshi", "M1", "yes"), ("polymarket", "P1", "no")]
        legs_b = [("polymarket", "P1", "no"), ("kalshi", "M1", "yes")]
        key_a = opportunity_dedupe_key("cross_venue", "taker", legs_a)
        key_b = opportunity_dedupe_key("cross_venue", "taker", legs_b)

        reg = DedupeRegistry(DedupeConfig(cooldown_seconds=60.0))
        reg.register(key_a, now=1000.0)
        # Same opportunity detected with different leg order → blocked.
        assert reg.is_duplicate(key_b, now=1010.0) is True
