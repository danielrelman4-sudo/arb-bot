"""Liquidity impact curve integration (Phase 4B).

Models the relationship between order size and market impact.
Learns empirical slippage curves per venue/market and automatically
sizes down when marginal impact erodes edge.

Usage::

    model = LiquidityImpactModel(config)
    model.record_fill("kalshi", contracts=10, expected_price=0.55, actual_price=0.56)
    impact = model.estimate_impact("kalshi", contracts=20, book_depth=100)
    capped = model.max_contracts_for_edge("kalshi", edge=0.03, cost=0.55, book_depth=100)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidityImpactConfig:
    """Configuration for liquidity impact model.

    Parameters
    ----------
    default_impact_coefficient:
        Default price impact per unit of depth fraction consumed.
        Impact = coeff * (contracts / book_depth) ^ exponent.
        Default 0.10.
    default_impact_exponent:
        Power law exponent for impact curve. 1.0 = linear,
        >1.0 = convex (large orders disproportionately worse).
        Default 1.5.
    max_depth_fraction:
        Maximum fraction of book depth to consume in one order.
        Orders beyond this are capped. Default 0.5.
    min_book_depth:
        Minimum book depth required to trade. Below this,
        impact is considered infinite. Default 5.
    learning_rate:
        Exponential smoothing factor for updating learned
        coefficients from realized fills. Default 0.1.
    window_size:
        Number of recent fills to keep for learning. Default 200.
    """

    default_impact_coefficient: float = 0.10
    default_impact_exponent: float = 1.5
    max_depth_fraction: float = 0.5
    min_book_depth: int = 5
    learning_rate: float = 0.1
    window_size: int = 200


# ---------------------------------------------------------------------------
# Impact estimate result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactEstimate:
    """Estimated market impact for a given order size."""

    venue: str
    contracts: int
    book_depth: int
    depth_fraction: float
    estimated_impact: float
    marginal_impact: float
    coefficient: float
    exponent: float
    blocked: bool
    block_reason: str


# ---------------------------------------------------------------------------
# Fill record for learning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactFillRecord:
    """A realized fill used to calibrate impact curves."""

    venue: str
    contracts: int
    book_depth: int
    expected_price: float
    actual_price: float
    slippage: float  # actual - expected (positive = adverse)


# ---------------------------------------------------------------------------
# Venue-specific learned parameters
# ---------------------------------------------------------------------------


@dataclass
class VenueImpactParams:
    """Learned impact parameters for a venue."""

    coefficient: float
    exponent: float
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Liquidity impact model
# ---------------------------------------------------------------------------


class LiquidityImpactModel:
    """Market impact model with empirical learning.

    Estimates price impact as a function of order size relative
    to book depth. Learns from realized fills to calibrate
    per-venue impact curves.
    """

    def __init__(self, config: LiquidityImpactConfig | None = None) -> None:
        self._config = config or LiquidityImpactConfig()
        self._venue_params: Dict[str, VenueImpactParams] = {}
        self._fill_history: Dict[str, List[ImpactFillRecord]] = {}

    @property
    def config(self) -> LiquidityImpactConfig:
        return self._config

    def get_venue_params(self, venue: str) -> VenueImpactParams:
        """Get current impact parameters for a venue."""
        if venue not in self._venue_params:
            return VenueImpactParams(
                coefficient=self._config.default_impact_coefficient,
                exponent=self._config.default_impact_exponent,
            )
        return self._venue_params[venue]

    def estimate_impact(
        self,
        venue: str,
        contracts: int,
        book_depth: int,
    ) -> ImpactEstimate:
        """Estimate market impact for an order.

        Parameters
        ----------
        venue:
            Venue identifier.
        contracts:
            Number of contracts to trade.
        book_depth:
            Available depth at the target price level.
        """
        cfg = self._config

        # Block on insufficient depth.
        if book_depth < cfg.min_book_depth:
            return ImpactEstimate(
                venue=venue, contracts=contracts, book_depth=book_depth,
                depth_fraction=1.0, estimated_impact=float("inf"),
                marginal_impact=float("inf"),
                coefficient=0.0, exponent=0.0,
                blocked=True, block_reason="insufficient_depth",
            )

        depth_fraction = contracts / book_depth

        # Block on excessive depth consumption.
        if depth_fraction > cfg.max_depth_fraction:
            return ImpactEstimate(
                venue=venue, contracts=contracts, book_depth=book_depth,
                depth_fraction=depth_fraction,
                estimated_impact=float("inf"),
                marginal_impact=float("inf"),
                coefficient=0.0, exponent=0.0,
                blocked=True, block_reason="exceeds_max_depth_fraction",
            )

        params = self.get_venue_params(venue)

        # Impact = coeff * (contracts / book_depth) ^ exponent.
        impact = params.coefficient * (depth_fraction ** params.exponent)

        # Marginal impact: derivative at current size.
        # d/dC [coeff * (C/D)^exp] = coeff * exp * C^(exp-1) / D^exp
        if contracts > 0:
            marginal = (
                params.coefficient * params.exponent
                * (depth_fraction ** (params.exponent - 1.0))
                / book_depth
            )
        else:
            marginal = 0.0

        return ImpactEstimate(
            venue=venue,
            contracts=contracts,
            book_depth=book_depth,
            depth_fraction=depth_fraction,
            estimated_impact=impact,
            marginal_impact=marginal,
            coefficient=params.coefficient,
            exponent=params.exponent,
            blocked=False,
            block_reason="",
        )

    def max_contracts_for_edge(
        self,
        venue: str,
        edge: float,
        cost: float,
        book_depth: int,
    ) -> int:
        """Find maximum contracts where impact doesn't erode edge.

        Returns the largest order size where estimated impact < edge/cost.
        """
        if edge <= 0 or cost <= 0 or book_depth < self._config.min_book_depth:
            return 0

        # Impact threshold: impact as fraction of price.
        impact_threshold = edge / cost
        params = self.get_venue_params(venue)
        max_depth = int(book_depth * self._config.max_depth_fraction)

        # Solve: coeff * (C/D)^exp = threshold
        # → (C/D)^exp = threshold / coeff
        # → C/D = (threshold / coeff) ^ (1/exp)
        # → C = D * (threshold / coeff) ^ (1/exp)
        if params.coefficient <= 0:
            return max_depth

        ratio = impact_threshold / params.coefficient
        if ratio <= 0:
            return 0

        depth_frac = ratio ** (1.0 / params.exponent)
        max_from_impact = int(book_depth * depth_frac)

        return max(0, min(max_from_impact, max_depth))

    def record_fill(
        self,
        venue: str,
        contracts: int,
        book_depth: int,
        expected_price: float,
        actual_price: float,
    ) -> None:
        """Record a realized fill for impact curve calibration.

        Parameters
        ----------
        venue:
            Venue identifier.
        contracts:
            Number of contracts filled.
        book_depth:
            Book depth at time of order.
        expected_price:
            Expected fill price.
        actual_price:
            Actual fill price.
        """
        slippage = actual_price - expected_price
        record = ImpactFillRecord(
            venue=venue,
            contracts=contracts,
            book_depth=book_depth,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage=slippage,
        )

        if venue not in self._fill_history:
            self._fill_history[venue] = []
        history = self._fill_history[venue]
        history.append(record)
        if len(history) > self._config.window_size:
            self._fill_history[venue] = history[-self._config.window_size:]

        self._update_params(venue, record)

    def _update_params(self, venue: str, record: ImpactFillRecord) -> None:
        """Update venue parameters from a new fill observation."""
        if record.book_depth < self._config.min_book_depth:
            return
        if record.contracts <= 0:
            return

        depth_fraction = record.contracts / record.book_depth
        if depth_fraction <= 0:
            return

        # Observed impact as fraction of price.
        if record.expected_price <= 0:
            return
        observed_impact = abs(record.slippage) / record.expected_price

        # Current params.
        params = self.get_venue_params(venue)
        exponent = params.exponent

        # What coefficient would explain this observation?
        # observed = coeff * depth_fraction ^ exponent
        # coeff_obs = observed / depth_fraction ^ exponent
        denom = depth_fraction ** exponent
        if denom <= 0:
            return
        observed_coeff = observed_impact / denom

        # Exponential smoothing.
        lr = self._config.learning_rate
        new_coeff = params.coefficient * (1.0 - lr) + observed_coeff * lr

        if venue not in self._venue_params:
            self._venue_params[venue] = VenueImpactParams(
                coefficient=new_coeff,
                exponent=exponent,
                sample_count=1,
            )
        else:
            self._venue_params[venue].coefficient = new_coeff
            self._venue_params[venue].sample_count += 1

    def fill_count(self, venue: str) -> int:
        """Number of recorded fills for a venue."""
        return len(self._fill_history.get(venue, []))

    def clear(self) -> None:
        """Clear all learned parameters and history."""
        self._venue_params.clear()
        self._fill_history.clear()
