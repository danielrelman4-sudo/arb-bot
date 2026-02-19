"""Cross/parity coverage watchdog (Phase 5D).

Tracks coverage health for cross-venue pairs and parity rules.
Raises alerts when coverage drops below thresholds and identifies
specific items needing repair.

Usage::

    wd = CoverageWatchdog(config)
    wd.register_cross_pair("kalshi:BTC-50K", "poly:BTC-50K")
    wd.update_coverage("kalshi:BTC-50K", "poly:BTC-50K", covered=True)
    report = wd.report()
    if report.cross_coverage < report.cross_threshold:
        for pair in report.uncovered_cross_pairs:
            # attempt repair
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageWatchdogConfig:
    """Configuration for coverage watchdog.

    Parameters
    ----------
    cross_coverage_threshold:
        Minimum fraction of cross pairs that must be covered.
        Default 0.80.
    parity_coverage_threshold:
        Minimum fraction of parity rules that must be covered.
        Default 0.80.
    alert_cooldown_seconds:
        Minimum time between alerts for the same item. Default 60.
    repair_batch_size:
        Maximum number of items to flag for repair per check.
        Default 10.
    stale_timeout_seconds:
        Coverage data older than this is considered stale/uncovered.
        Default 300.
    """

    cross_coverage_threshold: float = 0.80
    parity_coverage_threshold: float = 0.80
    alert_cooldown_seconds: float = 60.0
    repair_batch_size: int = 10
    stale_timeout_seconds: float = 300.0


# ---------------------------------------------------------------------------
# Coverage item state
# ---------------------------------------------------------------------------


@dataclass
class CoverageItemState:
    """State for a single coverage item (cross pair or parity rule)."""

    item_id: str
    covered: bool = False
    last_update_time: float = 0.0
    last_alert_time: float = 0.0
    consecutive_uncovered: int = 0
    total_checks: int = 0
    total_covered: int = 0


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    """Coverage health report."""

    cross_total: int
    cross_covered: int
    cross_coverage: float
    cross_threshold: float
    cross_alert: bool
    uncovered_cross_pairs: Tuple[str, ...]
    parity_total: int
    parity_covered: int
    parity_coverage: float
    parity_threshold: float
    parity_alert: bool
    uncovered_parity_rules: Tuple[str, ...]
    repair_candidates: Tuple[str, ...]
    any_alert: bool


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class CoverageWatchdog:
    """Monitors cross-venue pair and parity rule coverage health.

    Tracks which pairs/rules are actively covered, raises alerts
    on coverage drops, and identifies items for targeted repair.
    """

    def __init__(self, config: CoverageWatchdogConfig | None = None) -> None:
        self._config = config or CoverageWatchdogConfig()
        self._cross_pairs: Dict[str, CoverageItemState] = {}
        self._parity_rules: Dict[str, CoverageItemState] = {}

    @property
    def config(self) -> CoverageWatchdogConfig:
        return self._config

    @staticmethod
    def _pair_key(venue_a: str, venue_b: str) -> str:
        """Create a canonical key for a venue pair."""
        return f"{venue_a}|{venue_b}"

    def register_cross_pair(self, venue_a: str, venue_b: str) -> None:
        """Register a cross-venue pair to monitor."""
        key = self._pair_key(venue_a, venue_b)
        if key not in self._cross_pairs:
            self._cross_pairs[key] = CoverageItemState(item_id=key)

    def register_parity_rule(self, rule_id: str) -> None:
        """Register a parity rule to monitor."""
        if rule_id not in self._parity_rules:
            self._parity_rules[rule_id] = CoverageItemState(item_id=rule_id)

    def update_cross_coverage(
        self,
        venue_a: str,
        venue_b: str,
        covered: bool,
        now: float | None = None,
    ) -> None:
        """Update coverage status for a cross-venue pair."""
        if now is None:
            now = time.monotonic()
        key = self._pair_key(venue_a, venue_b)
        state = self._cross_pairs.get(key)
        if state is None:
            return
        state.covered = covered
        state.last_update_time = now
        state.total_checks += 1
        if covered:
            state.total_covered += 1
            state.consecutive_uncovered = 0
        else:
            state.consecutive_uncovered += 1

    def update_parity_coverage(
        self,
        rule_id: str,
        covered: bool,
        now: float | None = None,
    ) -> None:
        """Update coverage status for a parity rule."""
        if now is None:
            now = time.monotonic()
        state = self._parity_rules.get(rule_id)
        if state is None:
            return
        state.covered = covered
        state.last_update_time = now
        state.total_checks += 1
        if covered:
            state.total_covered += 1
            state.consecutive_uncovered = 0
        else:
            state.consecutive_uncovered += 1

    def _is_effectively_covered(
        self, state: CoverageItemState, now: float
    ) -> bool:
        """Check if an item is effectively covered (not stale)."""
        if not state.covered:
            return False
        if state.last_update_time == 0.0:
            return False
        elapsed = now - state.last_update_time
        if elapsed > self._config.stale_timeout_seconds:
            return False
        return True

    def _compute_coverage(
        self,
        items: Dict[str, CoverageItemState],
        now: float,
    ) -> Tuple[int, int, List[str]]:
        """Compute coverage stats for a set of items."""
        total = len(items)
        covered = 0
        uncovered: List[str] = []
        for key, state in items.items():
            if self._is_effectively_covered(state, now):
                covered += 1
            else:
                uncovered.append(key)
        return total, covered, uncovered

    def _select_repair_candidates(
        self,
        uncovered_cross: List[str],
        uncovered_parity: List[str],
        now: float,
    ) -> List[str]:
        """Select items for repair, prioritizing by consecutive_uncovered."""
        candidates: List[Tuple[int, str]] = []

        for key in uncovered_cross:
            state = self._cross_pairs[key]
            if now - state.last_alert_time >= self._config.alert_cooldown_seconds:
                candidates.append((state.consecutive_uncovered, key))
                state.last_alert_time = now

        for key in uncovered_parity:
            state = self._parity_rules[key]
            if now - state.last_alert_time >= self._config.alert_cooldown_seconds:
                candidates.append((state.consecutive_uncovered, key))
                state.last_alert_time = now

        # Sort by consecutive_uncovered (highest first).
        candidates.sort(reverse=True, key=lambda x: x[0])
        return [c[1] for c in candidates[: self._config.repair_batch_size]]

    def report(self, now: float | None = None) -> CoverageReport:
        """Generate coverage health report."""
        if now is None:
            now = time.monotonic()
        cfg = self._config

        cross_total, cross_covered, uncovered_cross = self._compute_coverage(
            self._cross_pairs, now
        )
        parity_total, parity_covered, uncovered_parity = self._compute_coverage(
            self._parity_rules, now
        )

        cross_coverage = cross_covered / cross_total if cross_total > 0 else 1.0
        parity_coverage = parity_covered / parity_total if parity_total > 0 else 1.0

        cross_alert = cross_coverage < cfg.cross_coverage_threshold
        parity_alert = parity_coverage < cfg.parity_coverage_threshold

        repair_candidates = self._select_repair_candidates(
            uncovered_cross, uncovered_parity, now
        )

        return CoverageReport(
            cross_total=cross_total,
            cross_covered=cross_covered,
            cross_coverage=cross_coverage,
            cross_threshold=cfg.cross_coverage_threshold,
            cross_alert=cross_alert,
            uncovered_cross_pairs=tuple(uncovered_cross),
            parity_total=parity_total,
            parity_covered=parity_covered,
            parity_coverage=parity_coverage,
            parity_threshold=cfg.parity_coverage_threshold,
            parity_alert=parity_alert,
            uncovered_parity_rules=tuple(uncovered_parity),
            repair_candidates=tuple(repair_candidates),
            any_alert=cross_alert or parity_alert,
        )

    def get_cross_state(self, venue_a: str, venue_b: str) -> CoverageItemState | None:
        """Get state for a cross pair."""
        key = self._pair_key(venue_a, venue_b)
        return self._cross_pairs.get(key)

    def get_parity_state(self, rule_id: str) -> CoverageItemState | None:
        """Get state for a parity rule."""
        return self._parity_rules.get(rule_id)

    def cross_pair_count(self) -> int:
        """Number of registered cross pairs."""
        return len(self._cross_pairs)

    def parity_rule_count(self) -> int:
        """Number of registered parity rules."""
        return len(self._parity_rules)

    def clear(self) -> None:
        """Clear all registered items."""
        self._cross_pairs.clear()
        self._parity_rules.clear()
