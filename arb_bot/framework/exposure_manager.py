"""Exposure and concentration controls (Phase 4H).

Tracks and limits portfolio-wide exposure and concentration risk.
Prevents over-concentration in any single venue, market category,
or correlated cluster.

Usage::

    exp = ExposureManager(config)
    exp.add_position("kalshi", "politics", capital=500.0)
    check = exp.check_new_position("kalshi", "politics", capital=200.0)
    if check.allowed:
        exp.add_position("kalshi", "politics", capital=200.0)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureConfig:
    """Configuration for exposure and concentration limits.

    Parameters
    ----------
    max_total_exposure:
        Maximum total capital at risk across all positions.
        Default 5000.0.
    max_venue_exposure:
        Maximum capital at risk on any single venue.
        Default 3000.0.
    max_category_exposure:
        Maximum capital at risk in any single market category.
        Default 2000.0.
    max_venue_fraction:
        Maximum fraction of total exposure on any venue.
        Default 0.60.
    max_category_fraction:
        Maximum fraction of total exposure in any category.
        Default 0.40.
    max_single_market:
        Maximum capital in a single market. Default 1000.0.
    max_open_positions:
        Maximum number of open positions. Default 20.
    """

    max_total_exposure: float = 5000.0
    max_venue_exposure: float = 3000.0
    max_category_exposure: float = 2000.0
    max_venue_fraction: float = 0.60
    max_category_fraction: float = 0.40
    max_single_market: float = 1000.0
    max_open_positions: int = 20


# ---------------------------------------------------------------------------
# Position record
# ---------------------------------------------------------------------------


@dataclass
class PositionRecord:
    """A tracked open position."""

    venue: str
    category: str
    market_id: str
    capital: float


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureCheckResult:
    """Result of exposure check for a new position."""

    allowed: bool
    max_additional_capital: float
    breached_limits: Tuple[str, ...]
    current_total_exposure: float
    current_venue_exposure: float
    current_category_exposure: float
    current_market_exposure: float
    position_count: int


# ---------------------------------------------------------------------------
# Exposure snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureSnapshot:
    """Current exposure state."""

    total_exposure: float
    venue_exposures: Dict[str, float]
    category_exposures: Dict[str, float]
    market_exposures: Dict[str, float]
    position_count: int
    utilization: float  # total / max_total


# ---------------------------------------------------------------------------
# Exposure manager
# ---------------------------------------------------------------------------


class ExposureManager:
    """Manages portfolio exposure and concentration limits.

    Tracks open positions and checks new positions against
    configurable limits for total, per-venue, per-category,
    and per-market exposure.
    """

    def __init__(self, config: ExposureConfig | None = None) -> None:
        self._config = config or ExposureConfig()
        self._positions: List[PositionRecord] = []

    @property
    def config(self) -> ExposureConfig:
        return self._config

    def add_position(
        self,
        venue: str,
        category: str,
        capital: float,
        market_id: str = "",
    ) -> None:
        """Add an open position to tracking."""
        self._positions.append(PositionRecord(
            venue=venue, category=category,
            market_id=market_id, capital=capital,
        ))

    def remove_position(self, market_id: str) -> None:
        """Remove a position by market_id."""
        self._positions = [
            p for p in self._positions if p.market_id != market_id
        ]

    def check_new_position(
        self,
        venue: str,
        category: str,
        capital: float,
        market_id: str = "",
    ) -> ExposureCheckResult:
        """Check if a new position is within limits."""
        cfg = self._config
        breaches: List[str] = []

        total = self._total_exposure()
        venue_exp = self._venue_exposure(venue)
        cat_exp = self._category_exposure(category)
        mkt_exp = self._market_exposure(market_id)
        count = len(self._positions)

        # Check position count.
        if count >= cfg.max_open_positions:
            breaches.append("max_open_positions")

        # Check total exposure.
        if total + capital > cfg.max_total_exposure:
            breaches.append("max_total_exposure")

        # Check venue exposure.
        if venue_exp + capital > cfg.max_venue_exposure:
            breaches.append("max_venue_exposure")

        # Check venue fraction (only meaningful when there are other positions).
        new_total = total + capital
        if total > 0 and new_total > 0 and (venue_exp + capital) / new_total > cfg.max_venue_fraction:
            breaches.append("max_venue_fraction")

        # Check category exposure.
        if cat_exp + capital > cfg.max_category_exposure:
            breaches.append("max_category_exposure")

        # Check category fraction (only meaningful when there are other positions).
        if total > 0 and new_total > 0 and (cat_exp + capital) / new_total > cfg.max_category_fraction:
            breaches.append("max_category_fraction")

        # Check single market.
        if market_id and mkt_exp + capital > cfg.max_single_market:
            breaches.append("max_single_market")

        # Compute max additional capital.
        caps = [
            cfg.max_total_exposure - total,
            cfg.max_venue_exposure - venue_exp,
            cfg.max_category_exposure - cat_exp,
        ]
        if market_id:
            caps.append(cfg.max_single_market - mkt_exp)
        max_add = max(0.0, min(caps))

        return ExposureCheckResult(
            allowed=len(breaches) == 0,
            max_additional_capital=max_add,
            breached_limits=tuple(breaches),
            current_total_exposure=total,
            current_venue_exposure=venue_exp,
            current_category_exposure=cat_exp,
            current_market_exposure=mkt_exp,
            position_count=count,
        )

    def snapshot(self) -> ExposureSnapshot:
        """Get current exposure snapshot."""
        cfg = self._config
        total = self._total_exposure()
        venue_exps: Dict[str, float] = defaultdict(float)
        cat_exps: Dict[str, float] = defaultdict(float)
        mkt_exps: Dict[str, float] = defaultdict(float)

        for p in self._positions:
            venue_exps[p.venue] += p.capital
            cat_exps[p.category] += p.capital
            if p.market_id:
                mkt_exps[p.market_id] += p.capital

        utilization = total / cfg.max_total_exposure if cfg.max_total_exposure > 0 else 0.0

        return ExposureSnapshot(
            total_exposure=total,
            venue_exposures=dict(venue_exps),
            category_exposures=dict(cat_exps),
            market_exposures=dict(mkt_exps),
            position_count=len(self._positions),
            utilization=utilization,
        )

    def clear(self) -> None:
        """Clear all positions."""
        self._positions.clear()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _total_exposure(self) -> float:
        return sum(p.capital for p in self._positions)

    def _venue_exposure(self, venue: str) -> float:
        return sum(p.capital for p in self._positions if p.venue == venue)

    def _category_exposure(self, category: str) -> float:
        return sum(p.capital for p in self._positions if p.category == category)

    def _market_exposure(self, market_id: str) -> float:
        if not market_id:
            return 0.0
        return sum(p.capital for p in self._positions if p.market_id == market_id)
