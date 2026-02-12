"""Event-tree basket coverage (Phase 6B).

Robust rule generation from real parent/child metadata, payout
math validation, diagnostics, and basket coverage tracking.

Usage::

    tree = EventTree(config)
    tree.add_event("election", children=["dem_win", "rep_win", "other"])
    tree.set_payout("dem_win", 1.0)
    tree.set_price("dem_win", 0.55)
    tree.set_price("rep_win", 0.40)
    tree.set_price("other", 0.06)
    report = tree.validate("election")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Validation status
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    """Result of event tree validation."""

    VALID = "valid"
    SUM_VIOLATION = "sum_violation"
    MISSING_PRICES = "missing_prices"
    NO_CHILDREN = "no_children"
    NOT_FOUND = "not_found"
    STALE_PRICES = "stale_prices"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventTreeConfig:
    """Configuration for event tree basket coverage.

    Parameters
    ----------
    sum_tolerance:
        Maximum deviation of child prices from expected sum.
        Default 0.05 (5 cents).
    expected_sum:
        Expected sum of child prices (usually 1.0 for a
        complete partition). Default 1.0.
    min_children:
        Minimum children for a valid event. Default 2.
    price_stale_seconds:
        Seconds after which a price is considered stale.
        Default 300.
    max_events:
        Maximum tracked events. Default 5000.
    """

    sum_tolerance: float = 0.05
    expected_sum: float = 1.0
    min_children: int = 2
    price_stale_seconds: float = 300.0
    max_events: int = 5000


# ---------------------------------------------------------------------------
# Child outcome
# ---------------------------------------------------------------------------


@dataclass
class ChildOutcome:
    """A child outcome in an event tree."""

    outcome_id: str
    payout: float  # Usually 1.0 for binary, can vary.
    price: float = 0.0
    price_updated_at: float = 0.0
    venue: str = ""


# ---------------------------------------------------------------------------
# Event node
# ---------------------------------------------------------------------------


@dataclass
class EventNode:
    """A parent event with child outcomes."""

    event_id: str
    children: Dict[str, ChildOutcome] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeValidation:
    """Result of validating an event tree."""

    event_id: str
    status: ValidationStatus
    child_count: int
    price_sum: float
    expected_sum: float
    deviation: float
    missing_prices: Tuple[str, ...]
    stale_prices: Tuple[str, ...]
    detail: str = ""


# ---------------------------------------------------------------------------
# Basket opportunity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasketOpportunity:
    """An arbitrage opportunity from a basket mispricing."""

    event_id: str
    price_sum: float
    expected_sum: float
    edge: float  # price_sum - expected_sum (positive = overpriced basket).
    child_prices: Dict[str, float]


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeCoverageReport:
    """Coverage report for all event trees."""

    total_events: int
    valid_count: int
    violation_count: int
    missing_count: int
    stale_count: int
    no_children_count: int
    opportunities: Tuple[BasketOpportunity, ...]


# ---------------------------------------------------------------------------
# Event tree
# ---------------------------------------------------------------------------


class EventTree:
    """Manages event trees and basket coverage.

    Tracks parent/child event hierarchies, validates payout math
    (child prices should sum to expected value), and identifies
    basket mispricing opportunities.
    """

    def __init__(self, config: EventTreeConfig | None = None) -> None:
        self._config = config or EventTreeConfig()
        self._events: Dict[str, EventNode] = {}

    @property
    def config(self) -> EventTreeConfig:
        return self._config

    def add_event(
        self,
        event_id: str,
        children: List[str],
        payouts: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        """Add a parent event with child outcomes.

        Parameters
        ----------
        event_id:
            Unique event identifier.
        children:
            List of child outcome IDs.
        payouts:
            Per-child payout map. Default 1.0 for each.
        metadata:
            Optional event metadata.
        """
        if now is None:
            now = time.time()
        payouts = payouts or {}
        node = EventNode(
            event_id=event_id,
            created_at=now,
            metadata=dict(metadata or {}),
        )
        for child_id in children:
            node.children[child_id] = ChildOutcome(
                outcome_id=child_id,
                payout=payouts.get(child_id, 1.0),
            )
        self._events[event_id] = node

    def set_price(
        self,
        event_id: str,
        outcome_id: str,
        price: float,
        venue: str = "",
        now: float | None = None,
    ) -> bool:
        """Set the price for a child outcome.

        Returns False if event/outcome not found.
        """
        if now is None:
            now = time.time()
        node = self._events.get(event_id)
        if node is None:
            return False
        child = node.children.get(outcome_id)
        if child is None:
            return False
        child.price = price
        child.price_updated_at = now
        child.venue = venue
        return True

    def set_payout(
        self,
        event_id: str,
        outcome_id: str,
        payout: float,
    ) -> bool:
        """Set the payout for a child outcome.

        Returns False if event/outcome not found.
        """
        node = self._events.get(event_id)
        if node is None:
            return False
        child = node.children.get(outcome_id)
        if child is None:
            return False
        child.payout = payout
        return True

    def validate(
        self,
        event_id: str,
        now: float | None = None,
    ) -> TreeValidation:
        """Validate an event tree's payout math.

        Checks that child prices sum to the expected value within
        tolerance, and that no prices are missing or stale.
        """
        if now is None:
            now = time.time()
        cfg = self._config

        node = self._events.get(event_id)
        if node is None:
            return TreeValidation(
                event_id=event_id,
                status=ValidationStatus.NOT_FOUND,
                child_count=0,
                price_sum=0.0,
                expected_sum=cfg.expected_sum,
                deviation=0.0,
                missing_prices=(),
                stale_prices=(),
            )

        if len(node.children) < cfg.min_children:
            return TreeValidation(
                event_id=event_id,
                status=ValidationStatus.NO_CHILDREN,
                child_count=len(node.children),
                price_sum=0.0,
                expected_sum=cfg.expected_sum,
                deviation=0.0,
                missing_prices=(),
                stale_prices=(),
                detail=f"need {cfg.min_children}, have {len(node.children)}",
            )

        # Check for missing prices.
        missing = [
            cid for cid, c in node.children.items()
            if c.price == 0.0 and c.price_updated_at == 0.0
        ]
        if missing:
            return TreeValidation(
                event_id=event_id,
                status=ValidationStatus.MISSING_PRICES,
                child_count=len(node.children),
                price_sum=0.0,
                expected_sum=cfg.expected_sum,
                deviation=0.0,
                missing_prices=tuple(missing),
                stale_prices=(),
            )

        # Check for stale prices.
        stale = [
            cid for cid, c in node.children.items()
            if c.price_updated_at > 0 and (now - c.price_updated_at) > cfg.price_stale_seconds
        ]
        if stale:
            return TreeValidation(
                event_id=event_id,
                status=ValidationStatus.STALE_PRICES,
                child_count=len(node.children),
                price_sum=sum(c.price for c in node.children.values()),
                expected_sum=cfg.expected_sum,
                deviation=0.0,
                missing_prices=(),
                stale_prices=tuple(stale),
            )

        # Sum validation.
        price_sum = sum(c.price for c in node.children.values())
        deviation = abs(price_sum - cfg.expected_sum)

        if deviation > cfg.sum_tolerance:
            return TreeValidation(
                event_id=event_id,
                status=ValidationStatus.SUM_VIOLATION,
                child_count=len(node.children),
                price_sum=price_sum,
                expected_sum=cfg.expected_sum,
                deviation=deviation,
                missing_prices=(),
                stale_prices=(),
                detail=f"sum={price_sum:.4f}, expected={cfg.expected_sum:.4f}",
            )

        return TreeValidation(
            event_id=event_id,
            status=ValidationStatus.VALID,
            child_count=len(node.children),
            price_sum=price_sum,
            expected_sum=cfg.expected_sum,
            deviation=deviation,
            missing_prices=(),
            stale_prices=(),
        )

    def find_opportunities(
        self,
        min_edge: float = 0.01,
        now: float | None = None,
    ) -> List[BasketOpportunity]:
        """Find basket mispricing opportunities across all events.

        An opportunity exists when the sum of child prices deviates
        from the expected sum by more than min_edge.
        """
        if now is None:
            now = time.time()

        opportunities: List[BasketOpportunity] = []
        for event_id, node in self._events.items():
            validation = self.validate(event_id, now=now)
            if validation.status != ValidationStatus.VALID and validation.status != ValidationStatus.SUM_VIOLATION:
                continue

            price_sum = validation.price_sum
            edge = price_sum - self._config.expected_sum
            if abs(edge) >= min_edge:
                child_prices = {
                    cid: c.price for cid, c in node.children.items()
                }
                opportunities.append(BasketOpportunity(
                    event_id=event_id,
                    price_sum=price_sum,
                    expected_sum=self._config.expected_sum,
                    edge=edge,
                    child_prices=child_prices,
                ))

        # Sort by absolute edge descending.
        opportunities.sort(key=lambda o: abs(o.edge), reverse=True)
        return opportunities

    def coverage_report(self, now: float | None = None) -> TreeCoverageReport:
        """Generate a coverage report for all event trees."""
        if now is None:
            now = time.time()

        valid = 0
        violations = 0
        missing = 0
        stale = 0
        no_children = 0

        for event_id in self._events:
            v = self.validate(event_id, now=now)
            if v.status == ValidationStatus.VALID:
                valid += 1
            elif v.status == ValidationStatus.SUM_VIOLATION:
                violations += 1
            elif v.status == ValidationStatus.MISSING_PRICES:
                missing += 1
            elif v.status == ValidationStatus.STALE_PRICES:
                stale += 1
            elif v.status == ValidationStatus.NO_CHILDREN:
                no_children += 1

        opps = self.find_opportunities(now=now)

        return TreeCoverageReport(
            total_events=len(self._events),
            valid_count=valid,
            violation_count=violations,
            missing_count=missing,
            stale_count=stale,
            no_children_count=no_children,
            opportunities=tuple(opps),
        )

    def get_event(self, event_id: str) -> EventNode | None:
        """Get an event node."""
        return self._events.get(event_id)

    def event_ids(self) -> List[str]:
        """List all event IDs."""
        return list(self._events.keys())

    def event_count(self) -> int:
        """Total registered events."""
        return len(self._events)

    def remove_event(self, event_id: str) -> None:
        """Remove an event."""
        self._events.pop(event_id, None)

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
