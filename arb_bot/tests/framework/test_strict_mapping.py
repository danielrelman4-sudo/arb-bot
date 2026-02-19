"""Tests for strict_mapping module (Phase 6A)."""

from __future__ import annotations

import pytest

from arb_bot.framework.strict_mapping import (
    AuditEntry,
    MappingEntry,
    MappingReport,
    MappingResolution,
    MappingTier,
    StrictMapper,
    StrictMapperConfig,
)


def _mapper(**kw: object) -> StrictMapper:
    return StrictMapper(StrictMapperConfig(**kw))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = StrictMapperConfig()
        assert cfg.strict_confidence_threshold == 0.90
        assert cfg.fallback_confidence_threshold == 0.70
        assert cfg.min_signals_strict == 2
        assert cfg.min_signals_fallback == 1
        assert cfg.fallback_enabled is True
        assert cfg.max_audit_entries == 10000
        assert cfg.stale_mapping_seconds == 3600.0

    def test_frozen(self) -> None:
        cfg = StrictMapperConfig()
        with pytest.raises(AttributeError):
            cfg.strict_confidence_threshold = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registration and tier classification
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_strict_tier(self) -> None:
        m = _mapper()
        tier = m.register_mapping("BTC-50K", "kalshi", "poly", confidence=0.95, signals=3, now=100.0)
        assert tier == MappingTier.STRICT

    def test_fallback_tier(self) -> None:
        m = _mapper()
        tier = m.register_mapping("BTC-50K", "kalshi", "poly", confidence=0.75, signals=1, now=100.0)
        assert tier == MappingTier.FALLBACK

    def test_rejected_low_confidence(self) -> None:
        m = _mapper()
        tier = m.register_mapping("BTC-50K", "kalshi", "poly", confidence=0.50, signals=1, now=100.0)
        assert tier == MappingTier.REJECTED

    def test_high_confidence_low_signals_fallback(self) -> None:
        """High confidence but only 1 signal → fallback (needs 2 for strict)."""
        m = _mapper()
        tier = m.register_mapping("BTC-50K", "kalshi", "poly", confidence=0.95, signals=1, now=100.0)
        assert tier == MappingTier.FALLBACK

    def test_fallback_disabled_unverified(self) -> None:
        """With fallback disabled, moderate confidence → unverified."""
        m = _mapper(fallback_enabled=False)
        tier = m.register_mapping("BTC-50K", "kalshi", "poly", confidence=0.80, signals=1, now=100.0)
        assert tier == MappingTier.UNVERIFIED

    def test_mapping_count(self) -> None:
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.95, signals=2, now=1.0)
        m.register_mapping("b", "k", "p", confidence=0.80, signals=1, now=1.0)
        assert m.mapping_count() == 2

    def test_overwrite_mapping(self) -> None:
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.50, signals=1, now=1.0)
        m.register_mapping("a", "k", "p", confidence=0.95, signals=3, now=2.0)
        assert m.mapping_count() == 1
        entry = m.get_mapping("a")
        assert entry is not None
        assert entry.confidence == 0.95

    def test_confidence_clamped_above_1(self) -> None:
        """Confidence > 1.0 is clamped to 1.0."""
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=1.5, signals=2, now=1.0)
        entry = m.get_mapping("a")
        assert entry is not None
        assert entry.confidence == 1.0

    def test_confidence_clamped_below_0(self) -> None:
        """Negative confidence is clamped to 0.0 → REJECTED tier."""
        m = _mapper()
        tier = m.register_mapping("a", "k", "p", confidence=-0.5, signals=2, now=1.0)
        assert tier == MappingTier.REJECTED
        entry = m.get_mapping("a")
        assert entry is not None
        assert entry.confidence == 0.0

    def test_update_confidence_clamped(self) -> None:
        """update_confidence clamps values too."""
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.95, signals=2, now=1.0)
        m.update_confidence("a", confidence=2.0, now=2.0)
        entry = m.get_mapping("a")
        assert entry is not None
        assert entry.confidence == 1.0


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_strict(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        res = m.resolve("BTC", now=100.0)
        assert res.accepted is True
        assert res.tier == MappingTier.STRICT
        assert res.confidence == 0.95

    def test_resolve_fallback(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.75, signals=1, now=100.0)
        res = m.resolve("BTC", now=100.0)
        assert res.accepted is True
        assert res.tier == MappingTier.FALLBACK

    def test_resolve_not_found(self) -> None:
        m = _mapper()
        res = m.resolve("MISSING", now=100.0)
        assert res.accepted is False
        assert res.rejection_reason == "not_found"

    def test_resolve_rejected_tier(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.50, signals=1, now=100.0)
        res = m.resolve("BTC", now=100.0)
        assert res.accepted is False
        assert res.rejection_reason == "below_threshold"

    def test_resolve_stale(self) -> None:
        m = _mapper(stale_mapping_seconds=60.0)
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        res = m.resolve("BTC", now=200.0)  # 100s > 60s stale threshold.
        assert res.accepted is False
        assert res.rejection_reason == "stale"

    def test_resolve_not_stale_within_window(self) -> None:
        m = _mapper(stale_mapping_seconds=60.0)
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        res = m.resolve("BTC", now=150.0)  # 50s < 60s.
        assert res.accepted is True

    def test_resolve_unverified(self) -> None:
        m = _mapper(fallback_enabled=False)
        m.register_mapping("BTC", "k", "p", confidence=0.80, signals=1, now=100.0)
        res = m.resolve("BTC", now=100.0)
        assert res.accepted is False
        assert res.rejection_reason == "unverified"

    def test_resolve_fallback_disabled(self) -> None:
        """Register with fallback enabled, then resolve with fallback disabled."""
        m = _mapper(fallback_enabled=True)
        m.register_mapping("BTC", "k", "p", confidence=0.75, signals=1, now=100.0)
        # Now create a mapper with fallback disabled and transfer mapping.
        m2 = _mapper(fallback_enabled=False)
        m2.register_mapping("BTC", "k", "p", confidence=0.75, signals=1, now=100.0)
        # The classify will set it to UNVERIFIED when fallback is disabled.
        res = m2.resolve("BTC", now=100.0)
        assert res.accepted is False

    def test_resolve_increments_use_count(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        m.resolve("BTC", now=100.0)
        m.resolve("BTC", now=101.0)
        entry = m.get_mapping("BTC")
        assert entry is not None
        assert entry.use_count == 2

    def test_resolve_rejected_increments_rejection_count(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.50, signals=1, now=100.0)
        m.resolve("BTC", now=100.0)
        m.resolve("BTC", now=101.0)
        entry = m.get_mapping("BTC")
        assert entry is not None
        assert entry.rejection_count == 2

    def test_resolve_returns_venues(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "kalshi", "poly", confidence=0.95, signals=2, now=100.0)
        res = m.resolve("BTC", now=100.0)
        assert res.venue_a == "kalshi"
        assert res.venue_b == "poly"


# ---------------------------------------------------------------------------
# Confidence updates
# ---------------------------------------------------------------------------


class TestUpdateConfidence:
    def test_promote_fallback_to_strict(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.75, signals=1, now=100.0)
        new_tier = m.update_confidence("BTC", confidence=0.95, signals=3, now=200.0)
        assert new_tier == MappingTier.STRICT

    def test_demote_strict_to_fallback(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=3, now=100.0)
        new_tier = m.update_confidence("BTC", confidence=0.75, signals=1, now=200.0)
        assert new_tier == MappingTier.FALLBACK

    def test_update_nonexistent(self) -> None:
        m = _mapper()
        result = m.update_confidence("MISSING", confidence=0.95, now=100.0)
        assert result is None

    def test_update_refreshes_verified_time(self) -> None:
        m = _mapper(stale_mapping_seconds=60.0)
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        m.update_confidence("BTC", confidence=0.95, now=150.0)
        # Now resolve at 200.0 — only 50s since last verification (< 60s stale).
        res = m.resolve("BTC", now=200.0)
        assert res.accepted is True


# ---------------------------------------------------------------------------
# Mappings by tier
# ---------------------------------------------------------------------------


class TestMappingsByTier:
    def test_filter_by_tier(self) -> None:
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.95, signals=2, now=1.0)
        m.register_mapping("b", "k", "p", confidence=0.75, signals=1, now=1.0)
        m.register_mapping("c", "k", "p", confidence=0.50, signals=1, now=1.0)
        m.register_mapping("d", "k", "p", confidence=0.92, signals=3, now=1.0)

        strict = m.mappings_by_tier(MappingTier.STRICT)
        assert set(strict) == {"a", "d"}
        fallback = m.mappings_by_tier(MappingTier.FALLBACK)
        assert fallback == ["b"]
        rejected = m.mappings_by_tier(MappingTier.REJECTED)
        assert rejected == ["c"]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_audit_on_register(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        log = m.audit_log()
        assert len(log) == 1
        assert log[0].action == "registered"
        assert log[0].market_id == "BTC"

    def test_audit_on_resolve(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=2, now=100.0)
        m.resolve("BTC", now=100.0)
        log = m.audit_log()
        assert len(log) == 2
        assert log[0].action == "resolved"  # Newest first.
        assert log[1].action == "registered"

    def test_audit_on_rejection(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.50, signals=1, now=100.0)
        m.resolve("BTC", now=100.0)
        log = m.audit_log()
        assert any(e.action == "rejected" for e in log)

    def test_audit_on_promotion(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.75, signals=1, now=100.0)
        m.update_confidence("BTC", confidence=0.95, signals=3, now=200.0)
        log = m.audit_log()
        assert any(e.action == "promoted" for e in log)

    def test_audit_on_demotion(self) -> None:
        m = _mapper()
        m.register_mapping("BTC", "k", "p", confidence=0.95, signals=3, now=100.0)
        m.update_confidence("BTC", confidence=0.50, signals=1, now=200.0)
        log = m.audit_log()
        assert any(e.action == "demoted" for e in log)

    def test_audit_max_entries(self) -> None:
        m = _mapper(max_audit_entries=10)
        for i in range(20):
            m.register_mapping(f"m{i}", "k", "p", confidence=0.95, signals=2, now=float(i))
        # Should have pruned — never more than max.
        assert len(m.audit_log(limit=100)) <= 10

    def test_audit_limit(self) -> None:
        m = _mapper()
        for i in range(10):
            m.register_mapping(f"m{i}", "k", "p", confidence=0.95, signals=2, now=float(i))
        log = m.audit_log(limit=3)
        assert len(log) == 3


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_empty_report(self) -> None:
        m = _mapper()
        r = m.report(now=100.0)
        assert r.total_mappings == 0
        assert r.avg_confidence == 0.0

    def test_report_counts(self) -> None:
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.95, signals=2, now=100.0)  # strict
        m.register_mapping("b", "k", "p", confidence=0.75, signals=1, now=100.0)  # fallback
        m.register_mapping("c", "k", "p", confidence=0.50, signals=1, now=100.0)  # rejected
        r = m.report(now=100.0)
        assert r.total_mappings == 3
        assert r.strict_count == 1
        assert r.fallback_count == 1
        assert r.rejected_count == 1

    def test_report_stale(self) -> None:
        m = _mapper(stale_mapping_seconds=60.0)
        m.register_mapping("a", "k", "p", confidence=0.95, signals=2, now=100.0)
        r = m.report(now=200.0)  # 100s > 60s stale.
        assert r.stale_count == 1

    def test_report_avg_confidence(self) -> None:
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.90, signals=2, now=1.0)
        m.register_mapping("b", "k", "p", confidence=0.80, signals=2, now=1.0)
        r = m.report(now=1.0)
        assert abs(r.avg_confidence - 0.85) < 0.001


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets(self) -> None:
        m = _mapper()
        m.register_mapping("a", "k", "p", confidence=0.95, signals=2, now=1.0)
        m.resolve("a", now=1.0)
        m.clear()
        assert m.mapping_count() == 0
        assert len(m.audit_log()) == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Register, resolve, update, re-resolve, report."""
        m = _mapper(stale_mapping_seconds=120.0)

        # Register with low confidence.
        tier = m.register_mapping("BTC", "kalshi", "poly", confidence=0.72, signals=1, now=100.0)
        assert tier == MappingTier.FALLBACK

        # Resolve as fallback — accepted.
        res = m.resolve("BTC", now=100.0)
        assert res.accepted is True
        assert res.tier == MappingTier.FALLBACK

        # Improve confidence.
        new_tier = m.update_confidence("BTC", confidence=0.96, signals=3, now=150.0)
        assert new_tier == MappingTier.STRICT

        # Resolve again — now strict.
        res = m.resolve("BTC", now=160.0)
        assert res.accepted is True
        assert res.tier == MappingTier.STRICT

        # Report.
        r = m.report(now=160.0)
        assert r.strict_count == 1
        assert r.total_mappings == 1

        # Audit trail.
        log = m.audit_log()
        actions = [e.action for e in log]
        assert "registered" in actions
        assert "promoted" in actions
        assert "resolved" in actions
