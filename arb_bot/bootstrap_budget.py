"""Dual-venue bootstrap budget enforcement (Phase 5F).

Reserves startup API budget for mapped overlap symbols, preventing
one venue from consuming all available calls during initialization
and starving the other.

Usage::

    bb = BootstrapBudget(config)
    bb.register_venue("kalshi", budget=200)
    bb.register_venue("polymarket", budget=200)
    bb.reserve_overlap(["BTC-50K", "ETH-3K"], per_symbol=5)
    if bb.can_spend("kalshi", category="discovery"):
        bb.spend("kalshi", category="discovery", count=1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapBudgetConfig:
    """Configuration for bootstrap budget enforcement.

    Parameters
    ----------
    overlap_reserve_fraction:
        Fraction of total budget reserved for overlap symbols.
        Default 0.30 (30%).
    discovery_fraction:
        Fraction of total budget for initial discovery/listing.
        Default 0.20.
    min_per_venue:
        Minimum budget per venue regardless of fractions.
        Default 20.
    enforce_balance:
        If True, prevents any single venue from using more than
        balance_max_fraction of the total combined budget.
        Default True.
    balance_max_fraction:
        Maximum fraction of combined budget any venue can use.
        Default 0.65.
    """

    overlap_reserve_fraction: float = 0.30
    discovery_fraction: float = 0.20
    min_per_venue: int = 20
    enforce_balance: bool = True
    balance_max_fraction: float = 0.65


# ---------------------------------------------------------------------------
# Venue budget state
# ---------------------------------------------------------------------------


@dataclass
class VenueBudgetState:
    """Budget state for a single venue."""

    venue: str
    total_budget: int
    overlap_reserve: int = 0
    discovery_reserve: int = 0
    spent_overlap: int = 0
    spent_discovery: int = 0
    spent_general: int = 0

    @property
    def total_spent(self) -> int:
        return self.spent_overlap + self.spent_discovery + self.spent_general

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.total_spent)

    @property
    def overlap_remaining(self) -> int:
        return max(0, self.overlap_reserve - self.spent_overlap)

    @property
    def discovery_remaining(self) -> int:
        return max(0, self.discovery_reserve - self.spent_discovery)


# ---------------------------------------------------------------------------
# Budget report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapBudgetReport:
    """Report on bootstrap budget status."""

    venues: Dict[str, VenueBudgetState]
    total_budget: int
    total_spent: int
    total_remaining: int
    overlap_symbols_count: int
    any_venue_starved: bool
    starved_venues: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Bootstrap budget
# ---------------------------------------------------------------------------


class BootstrapBudget:
    """Enforces balanced API budget allocation during startup.

    Reserves portions of each venue's budget for overlap symbol
    loading and discovery, preventing one venue from consuming
    all calls and starving the other.
    """

    def __init__(self, config: BootstrapBudgetConfig | None = None) -> None:
        self._config = config or BootstrapBudgetConfig()
        self._venues: Dict[str, VenueBudgetState] = {}
        self._overlap_symbols: Set[str] = set()

    @property
    def config(self) -> BootstrapBudgetConfig:
        return self._config

    def register_venue(self, venue: str, budget: int) -> None:
        """Register a venue with its startup budget.

        Parameters
        ----------
        venue:
            Venue identifier.
        budget:
            Total API calls available for bootstrap.
        """
        cfg = self._config
        budget = max(budget, cfg.min_per_venue)
        overlap_reserve = int(budget * cfg.overlap_reserve_fraction)
        discovery_reserve = int(budget * cfg.discovery_fraction)

        self._venues[venue] = VenueBudgetState(
            venue=venue,
            total_budget=budget,
            overlap_reserve=overlap_reserve,
            discovery_reserve=discovery_reserve,
        )

    def reserve_overlap(
        self,
        symbols: List[str],
        per_symbol: int = 5,
    ) -> None:
        """Register overlap symbols and adjust reserves.

        Parameters
        ----------
        symbols:
            List of overlap symbol identifiers.
        per_symbol:
            API calls to reserve per symbol per venue.
        """
        self._overlap_symbols.update(symbols)
        needed = len(self._overlap_symbols) * per_symbol
        for state in self._venues.values():
            state.overlap_reserve = max(state.overlap_reserve, needed)
            # Don't exceed total budget.
            state.overlap_reserve = min(
                state.overlap_reserve,
                state.total_budget - state.discovery_reserve,
            )

    def can_spend(
        self,
        venue: str,
        category: str = "general",
        count: int = 1,
    ) -> bool:
        """Check if a venue can spend API calls.

        Parameters
        ----------
        venue:
            Venue identifier.
        category:
            Spending category: "overlap", "discovery", or "general".
        count:
            Number of calls to check.
        """
        state = self._venues.get(venue)
        if state is None:
            return False

        if state.remaining < count:
            return False

        if category == "overlap":
            if state.overlap_remaining < count:
                return False
        elif category == "discovery":
            if state.discovery_remaining < count:
                return False
        else:
            # General spending can't eat into reserves.
            general_available = (
                state.remaining
                - state.overlap_remaining
                - state.discovery_remaining
            )
            if general_available < count:
                return False

        # Balance check.
        if self._config.enforce_balance and len(self._venues) > 1:
            total_budget = sum(v.total_budget for v in self._venues.values())
            max_allowed = int(total_budget * self._config.balance_max_fraction)
            if state.total_spent + count > max_allowed:
                return False

        return True

    def spend(
        self,
        venue: str,
        category: str = "general",
        count: int = 1,
    ) -> bool:
        """Record API call spending.

        Returns True if the spend was allowed, False if denied.
        """
        if not self.can_spend(venue, category, count):
            return False

        state = self._venues[venue]
        if category == "overlap":
            state.spent_overlap += count
        elif category == "discovery":
            state.spent_discovery += count
        else:
            state.spent_general += count
        return True

    def get_state(self, venue: str) -> VenueBudgetState | None:
        """Get budget state for a venue."""
        return self._venues.get(venue)

    def report(self) -> BootstrapBudgetReport:
        """Generate budget status report."""
        total_budget = sum(v.total_budget for v in self._venues.values())
        total_spent = sum(v.total_spent for v in self._venues.values())
        total_remaining = sum(v.remaining for v in self._venues.values())

        starved: List[str] = []
        for venue, state in self._venues.items():
            if state.remaining == 0 and state.total_spent < state.total_budget * 0.5:
                starved.append(venue)

        return BootstrapBudgetReport(
            venues=dict(self._venues),
            total_budget=total_budget,
            total_spent=total_spent,
            total_remaining=total_remaining,
            overlap_symbols_count=len(self._overlap_symbols),
            any_venue_starved=len(starved) > 0,
            starved_venues=tuple(starved),
        )

    def registered_venues(self) -> List[str]:
        """List registered venues."""
        return list(self._venues.keys())

    def clear(self) -> None:
        """Clear all venues and state."""
        self._venues.clear()
        self._overlap_symbols.clear()
