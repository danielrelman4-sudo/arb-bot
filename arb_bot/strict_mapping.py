"""Confidence-scored strict mapping mode (Phase 6A).

Primary strict mapping with gated confidence fallback, multi-signal
agreement requirements, and separate audit logging for mapping decisions.

Usage::

    mapper = StrictMapper(config)
    mapper.register_mapping("BTC-50K", "kalshi", "poly", confidence=0.95, signals=3)
    result = mapper.resolve("BTC-50K")
    if result.accepted:
        # Use result.mapping
    else:
        # result.rejection_reason explains why

    # Audit trail
    audit = mapper.audit_log()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mapping quality tier
# ---------------------------------------------------------------------------


class MappingTier(str, Enum):
    """Quality tier for a mapping."""

    STRICT = "strict"  # High confidence, multi-signal agreement.
    FALLBACK = "fallback"  # Lower confidence, gated by extra checks.
    UNVERIFIED = "unverified"  # No confidence scoring yet.
    REJECTED = "rejected"  # Below threshold.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrictMapperConfig:
    """Configuration for strict mapping mode.

    Parameters
    ----------
    strict_confidence_threshold:
        Minimum confidence for strict tier. Default 0.90.
    fallback_confidence_threshold:
        Minimum confidence for fallback tier. Default 0.70.
    min_signals_strict:
        Minimum agreeing signals for strict. Default 2.
    min_signals_fallback:
        Minimum agreeing signals for fallback. Default 1.
    fallback_enabled:
        Whether to allow fallback-tier mappings. Default True.
    max_audit_entries:
        Maximum audit log entries. Default 10000.
    stale_mapping_seconds:
        Seconds after which a mapping is considered stale. Default 3600.
    """

    strict_confidence_threshold: float = 0.90
    fallback_confidence_threshold: float = 0.70
    min_signals_strict: int = 2
    min_signals_fallback: int = 1
    fallback_enabled: bool = True
    max_audit_entries: int = 10000
    stale_mapping_seconds: float = 3600.0


# ---------------------------------------------------------------------------
# Mapping entry
# ---------------------------------------------------------------------------


@dataclass
class MappingEntry:
    """A cross-venue mapping with confidence metadata."""

    market_id: str
    venue_a: str
    venue_b: str
    confidence: float
    signal_count: int
    tier: MappingTier
    created_at: float
    last_verified_at: float
    use_count: int = 0
    rejection_count: int = 0
    signals: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingResolution:
    """Result of resolving a mapping."""

    market_id: str
    accepted: bool
    tier: MappingTier
    confidence: float
    signal_count: int
    rejection_reason: str = ""
    venue_a: str = ""
    venue_b: str = ""


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """An audit log entry for a mapping decision."""

    timestamp: float
    market_id: str
    action: str  # "resolved", "rejected", "promoted", "demoted", "registered"
    tier: MappingTier
    confidence: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Mapper report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingReport:
    """Summary report of all mappings."""

    total_mappings: int
    strict_count: int
    fallback_count: int
    unverified_count: int
    rejected_count: int
    stale_count: int
    avg_confidence: float


# ---------------------------------------------------------------------------
# Strict mapper
# ---------------------------------------------------------------------------


class StrictMapper:
    """Confidence-scored mapping with strict/fallback tiers.

    Primary strict mappings require high confidence and multi-signal
    agreement. Fallback mappings have lower requirements but are gated
    and separately audited.
    """

    def __init__(self, config: StrictMapperConfig | None = None) -> None:
        self._config = config or StrictMapperConfig()
        self._mappings: Dict[str, MappingEntry] = {}
        self._audit: List[AuditEntry] = []

    @property
    def config(self) -> StrictMapperConfig:
        return self._config

    def _classify(self, confidence: float, signal_count: int) -> MappingTier:
        """Classify a mapping into a tier based on confidence/signals."""
        cfg = self._config
        if (
            confidence >= cfg.strict_confidence_threshold
            and signal_count >= cfg.min_signals_strict
        ):
            return MappingTier.STRICT
        if (
            cfg.fallback_enabled
            and confidence >= cfg.fallback_confidence_threshold
            and signal_count >= cfg.min_signals_fallback
        ):
            return MappingTier.FALLBACK
        if confidence < cfg.fallback_confidence_threshold:
            return MappingTier.REJECTED
        return MappingTier.UNVERIFIED

    def _add_audit(
        self,
        market_id: str,
        action: str,
        tier: MappingTier,
        confidence: float,
        detail: str = "",
        now: float | None = None,
    ) -> None:
        """Add an audit entry."""
        if now is None:
            now = time.time()
        if len(self._audit) >= self._config.max_audit_entries:
            # Drop oldest 10%.
            drop = max(1, self._config.max_audit_entries // 10)
            self._audit = self._audit[drop:]
        self._audit.append(AuditEntry(
            timestamp=now,
            market_id=market_id,
            action=action,
            tier=tier,
            confidence=confidence,
            detail=detail,
        ))

    def register_mapping(
        self,
        market_id: str,
        venue_a: str,
        venue_b: str,
        confidence: float,
        signals: int = 1,
        signal_names: Tuple[str, ...] = (),
        now: float | None = None,
    ) -> MappingTier:
        """Register or update a cross-venue mapping.

        Returns the assigned tier. Confidence is clamped to [0.0, 1.0].
        """
        if now is None:
            now = time.time()

        # Clamp confidence to valid range.
        confidence = max(0.0, min(1.0, confidence))

        tier = self._classify(confidence, signals)
        self._mappings[market_id] = MappingEntry(
            market_id=market_id,
            venue_a=venue_a,
            venue_b=venue_b,
            confidence=confidence,
            signal_count=signals,
            tier=tier,
            created_at=now,
            last_verified_at=now,
            signals=signal_names,
        )

        self._add_audit(market_id, "registered", tier, confidence, now=now)
        return tier

    def update_confidence(
        self,
        market_id: str,
        confidence: float,
        signals: int | None = None,
        now: float | None = None,
    ) -> MappingTier | None:
        """Update confidence for an existing mapping.

        Returns new tier, or None if mapping not found.
        Confidence is clamped to [0.0, 1.0].
        """
        if now is None:
            now = time.time()
        entry = self._mappings.get(market_id)
        if entry is None:
            return None

        # Clamp confidence to valid range.
        confidence = max(0.0, min(1.0, confidence))

        old_tier = entry.tier
        entry.confidence = confidence
        if signals is not None:
            entry.signal_count = signals
        entry.last_verified_at = now

        new_tier = self._classify(entry.confidence, entry.signal_count)
        entry.tier = new_tier

        if new_tier != old_tier:
            action = "promoted" if _tier_rank(new_tier) > _tier_rank(old_tier) else "demoted"
            self._add_audit(
                market_id, action, new_tier, confidence,
                detail=f"{old_tier.value}->{new_tier.value}", now=now,
            )

        return new_tier

    def resolve(
        self,
        market_id: str,
        now: float | None = None,
    ) -> MappingResolution:
        """Resolve a mapping for trading use.

        Checks tier, staleness, and confidence gates.
        """
        if now is None:
            now = time.time()

        entry = self._mappings.get(market_id)
        if entry is None:
            return MappingResolution(
                market_id=market_id,
                accepted=False,
                tier=MappingTier.REJECTED,
                confidence=0.0,
                signal_count=0,
                rejection_reason="not_found",
            )

        # Check staleness.
        age = now - entry.last_verified_at
        if age > self._config.stale_mapping_seconds:
            entry.rejection_count += 1
            self._add_audit(
                market_id, "rejected", entry.tier, entry.confidence,
                detail="stale", now=now,
            )
            return MappingResolution(
                market_id=market_id,
                accepted=False,
                tier=entry.tier,
                confidence=entry.confidence,
                signal_count=entry.signal_count,
                rejection_reason="stale",
                venue_a=entry.venue_a,
                venue_b=entry.venue_b,
            )

        # Check tier.
        if entry.tier == MappingTier.REJECTED:
            entry.rejection_count += 1
            self._add_audit(
                market_id, "rejected", entry.tier, entry.confidence,
                detail="below_threshold", now=now,
            )
            return MappingResolution(
                market_id=market_id,
                accepted=False,
                tier=entry.tier,
                confidence=entry.confidence,
                signal_count=entry.signal_count,
                rejection_reason="below_threshold",
                venue_a=entry.venue_a,
                venue_b=entry.venue_b,
            )

        if entry.tier == MappingTier.UNVERIFIED:
            entry.rejection_count += 1
            self._add_audit(
                market_id, "rejected", entry.tier, entry.confidence,
                detail="unverified", now=now,
            )
            return MappingResolution(
                market_id=market_id,
                accepted=False,
                tier=entry.tier,
                confidence=entry.confidence,
                signal_count=entry.signal_count,
                rejection_reason="unverified",
                venue_a=entry.venue_a,
                venue_b=entry.venue_b,
            )

        if entry.tier == MappingTier.FALLBACK and not self._config.fallback_enabled:
            entry.rejection_count += 1
            self._add_audit(
                market_id, "rejected", entry.tier, entry.confidence,
                detail="fallback_disabled", now=now,
            )
            return MappingResolution(
                market_id=market_id,
                accepted=False,
                tier=entry.tier,
                confidence=entry.confidence,
                signal_count=entry.signal_count,
                rejection_reason="fallback_disabled",
                venue_a=entry.venue_a,
                venue_b=entry.venue_b,
            )

        # Accepted.
        entry.use_count += 1
        self._add_audit(
            market_id, "resolved", entry.tier, entry.confidence, now=now,
        )
        return MappingResolution(
            market_id=market_id,
            accepted=True,
            tier=entry.tier,
            confidence=entry.confidence,
            signal_count=entry.signal_count,
            venue_a=entry.venue_a,
            venue_b=entry.venue_b,
        )

    def get_mapping(self, market_id: str) -> MappingEntry | None:
        """Get a mapping entry."""
        return self._mappings.get(market_id)

    def mappings_by_tier(self, tier: MappingTier) -> List[str]:
        """List market IDs in a given tier."""
        return [
            mid for mid, entry in self._mappings.items()
            if entry.tier == tier
        ]

    def audit_log(self, limit: int = 100) -> List[AuditEntry]:
        """Get recent audit entries (newest first)."""
        return list(reversed(self._audit[-limit:]))

    def report(self, now: float | None = None) -> MappingReport:
        """Generate a mapping quality report."""
        if now is None:
            now = time.time()
        cfg = self._config

        strict = 0
        fallback = 0
        unverified = 0
        rejected = 0
        stale = 0
        total_conf = 0.0

        for entry in self._mappings.values():
            total_conf += entry.confidence
            if entry.tier == MappingTier.STRICT:
                strict += 1
            elif entry.tier == MappingTier.FALLBACK:
                fallback += 1
            elif entry.tier == MappingTier.UNVERIFIED:
                unverified += 1
            elif entry.tier == MappingTier.REJECTED:
                rejected += 1

            age = now - entry.last_verified_at
            if age > cfg.stale_mapping_seconds:
                stale += 1

        total = len(self._mappings)
        avg = total_conf / total if total > 0 else 0.0

        return MappingReport(
            total_mappings=total,
            strict_count=strict,
            fallback_count=fallback,
            unverified_count=unverified,
            rejected_count=rejected,
            stale_count=stale,
            avg_confidence=avg,
        )

    def mapping_count(self) -> int:
        """Total registered mappings."""
        return len(self._mappings)

    def clear(self) -> None:
        """Clear all mappings and audit log."""
        self._mappings.clear()
        self._audit.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tier_rank(tier: MappingTier) -> int:
    """Rank tiers for promotion/demotion comparison."""
    return {
        MappingTier.REJECTED: 0,
        MappingTier.UNVERIFIED: 1,
        MappingTier.FALLBACK: 2,
        MappingTier.STRICT: 3,
    }.get(tier, 0)
