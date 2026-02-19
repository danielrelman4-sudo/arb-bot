"""Execution realism upgrades (Phase 3A).

Enhances the basic fill model with:
- Independent leg fill simulation with graduated partial fill levels
- Queue position decay over time (latency-aware)
- Market impact / liquidity-aware slippage per leg
- Per-leg timing model for sequential execution scenarios

This module is used by the sizing/decision pipeline to produce
more realistic expected outcomes, and by the paper simulation to
generate more representative backtesting results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionModelConfig:
    """Configuration for the execution realism model.

    Parameters
    ----------
    queue_decay_half_life_seconds:
        Queue position decays exponentially with this half-life.
        Models the probability of being displaced at the front of the
        queue as time passes. Default 5.0 seconds.
    latency_seconds:
        Expected order-to-exchange latency in seconds. Used to compute
        initial queue position offset. Default 0.2 (200ms).
    market_impact_factor:
        How much each contract moves the price. Applied as a fraction
        of the spread per contract. Default 0.01.
    max_market_impact:
        Cap on total market impact as a fraction of spread.
        Default 0.5 (50% of spread max).
    min_fill_fraction:
        Minimum partial fill fraction to consider (below this, treat
        as zero fill). Default 0.1.
    fill_fraction_steps:
        Number of discrete fill levels for graduated partial fill
        simulation. Default 5 (0%, 20%, 40%, 60%, 80%, 100%).
    sequential_leg_delay_seconds:
        Expected delay between sequential leg submissions. Degrades
        later legs' fill probability. Default 1.0.
    enable_queue_decay:
        Enable time-based queue position decay. Default True.
    enable_market_impact:
        Enable market impact modeling. Default True.
    """

    queue_decay_half_life_seconds: float = 5.0
    latency_seconds: float = 0.2
    market_impact_factor: float = 0.01
    max_market_impact: float = 0.5
    min_fill_fraction: float = 0.1
    fill_fraction_steps: int = 5
    sequential_leg_delay_seconds: float = 1.0
    enable_queue_decay: bool = True
    enable_market_impact: bool = True


# ---------------------------------------------------------------------------
# Per-leg result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegFillEstimate:
    """Per-leg fill simulation result."""

    venue: str
    market_id: str
    side: str
    fill_probability: float
    expected_fill_fraction: float
    queue_position_score: float
    market_impact: float
    expected_slippage: float
    time_offset_seconds: float


# ---------------------------------------------------------------------------
# Execution estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionEstimate:
    """Full execution simulation result with graduated fill levels."""

    legs: tuple[LegFillEstimate, ...]
    all_fill_probability: float
    expected_fill_fraction: float
    expected_slippage_per_contract: float
    expected_market_impact_per_contract: float
    graduated_fill_distribution: tuple[tuple[float, float], ...]
    # List of (fill_fraction, probability) pairs.

    @property
    def partial_fill_probability(self) -> float:
        return 1.0 - self.all_fill_probability

    @property
    def expected_total_cost_adjustment(self) -> float:
        """Total expected cost adjustment from slippage + market impact."""
        return self.expected_slippage_per_contract + self.expected_market_impact_per_contract


# ---------------------------------------------------------------------------
# Execution model
# ---------------------------------------------------------------------------


class ExecutionModel:
    """Enhanced execution simulator with queue decay and market impact.

    Usage::

        model = ExecutionModel(config)
        estimate = model.simulate(legs, contracts, staleness_seconds)
    """

    def __init__(self, config: ExecutionModelConfig | None = None) -> None:
        self._config = config or ExecutionModelConfig()

    @property
    def config(self) -> ExecutionModelConfig:
        return self._config

    def simulate(
        self,
        legs: list[LegInput],
        contracts: int,
        staleness_seconds: float = 0.0,
        sequential: bool = False,
    ) -> ExecutionEstimate:
        """Simulate execution for a set of legs.

        Parameters
        ----------
        legs:
            List of leg inputs with venue, market, price, size info.
        contracts:
            Number of contracts to execute.
        staleness_seconds:
            How old the quote is at decision time.
        sequential:
            If True, legs are executed in order with delays between.
        """
        cfg = self._config

        leg_estimates: list[LegFillEstimate] = []
        total_slippage = 0.0
        total_impact = 0.0

        for i, leg in enumerate(legs):
            # Time offset: latency + staleness + sequential delay.
            time_offset = staleness_seconds + cfg.latency_seconds
            if sequential and i > 0:
                time_offset += cfg.sequential_leg_delay_seconds * i

            # Queue position score: decays over time.
            queue_score = self._queue_position_score(
                available_size=leg.available_size,
                contracts=contracts,
                time_offset=time_offset,
            )

            # Market impact: price displacement from our order.
            impact = self._market_impact(
                contracts=contracts,
                available_size=leg.available_size,
                spread=leg.spread,
            )

            # Expected slippage for this leg.
            slippage = self._expected_slippage(
                spread=leg.spread,
                queue_score=queue_score,
                impact=impact,
            )

            # Fill probability: queue score * depth coverage.
            depth_coverage = min(1.0, leg.available_size / max(1, contracts))
            fill_prob = max(0.0, min(1.0, queue_score * depth_coverage))

            # Expected fill fraction (how much of our order gets filled).
            fill_fraction = self._expected_fill_fraction(
                fill_probability=fill_prob,
                available_size=leg.available_size,
                contracts=contracts,
            )

            leg_estimates.append(LegFillEstimate(
                venue=leg.venue,
                market_id=leg.market_id,
                side=leg.side,
                fill_probability=fill_prob,
                expected_fill_fraction=fill_fraction,
                queue_position_score=queue_score,
                market_impact=impact,
                expected_slippage=slippage,
                time_offset_seconds=time_offset,
            ))

            total_slippage += slippage
            total_impact += impact

        # All-fill probability: product of leg fill probabilities.
        all_fill_prob = 1.0
        for est in leg_estimates:
            all_fill_prob *= est.fill_probability

        # Expected fill fraction: minimum across legs.
        expected_fill = min(
            (est.expected_fill_fraction for est in leg_estimates),
            default=0.0,
        )

        # Graduated fill distribution.
        graduated = self._graduated_fill_distribution(leg_estimates)

        avg_slippage = total_slippage / max(1, len(legs))
        avg_impact = total_impact / max(1, len(legs))

        return ExecutionEstimate(
            legs=tuple(leg_estimates),
            all_fill_probability=all_fill_prob,
            expected_fill_fraction=expected_fill,
            expected_slippage_per_contract=avg_slippage,
            expected_market_impact_per_contract=avg_impact,
            graduated_fill_distribution=tuple(graduated),
        )

    def _queue_position_score(
        self,
        available_size: float,
        contracts: int,
        time_offset: float,
    ) -> float:
        """Score representing our queue position quality (0-1).

        Decays over time as other orders arrive ahead of us.
        """
        cfg = self._config

        # Base score from available liquidity.
        base = min(1.0, available_size / max(1, contracts))

        if not cfg.enable_queue_decay or cfg.queue_decay_half_life_seconds <= 0:
            return base

        # Exponential decay: half-life means 50% of queue advantage lost.
        decay = math.exp(
            -time_offset * math.log(2) / max(0.01, cfg.queue_decay_half_life_seconds)
        )
        return base * decay

    def _market_impact(
        self,
        contracts: int,
        available_size: float,
        spread: float,
    ) -> float:
        """Expected price impact from our order."""
        cfg = self._config
        if not cfg.enable_market_impact or spread <= 0:
            return 0.0

        # Impact scales linearly with contracts relative to available.
        size_ratio = contracts / max(1.0, available_size)
        raw_impact = spread * size_ratio * cfg.market_impact_factor
        max_impact = spread * cfg.max_market_impact
        return min(raw_impact, max_impact)

    def _expected_slippage(
        self,
        spread: float,
        queue_score: float,
        impact: float,
    ) -> float:
        """Expected slippage for a single leg."""
        # Slippage is the spread we pay when we don't get queue priority,
        # plus the market impact of our own order.
        queue_miss_slippage = spread * (1.0 - queue_score)
        return queue_miss_slippage + impact

    def _expected_fill_fraction(
        self,
        fill_probability: float,
        available_size: float,
        contracts: int,
    ) -> float:
        """Expected fraction of our order that gets filled."""
        if contracts <= 0:
            return 0.0

        # Available size caps how much can fill.
        size_cap = min(1.0, available_size / contracts)
        raw_fraction = fill_probability * size_cap

        if raw_fraction < self._config.min_fill_fraction:
            return 0.0
        return raw_fraction

    def _graduated_fill_distribution(
        self,
        leg_estimates: list[LegFillEstimate],
    ) -> list[tuple[float, float]]:
        """Compute discrete fill level probabilities.

        Returns a list of (fill_fraction, probability) pairs.
        E.g., [(0.0, 0.1), (0.2, 0.05), (0.4, 0.1), ..., (1.0, 0.5)]
        """
        steps = max(2, self._config.fill_fraction_steps + 1)
        fractions = [i / (steps - 1) for i in range(steps)]

        if not leg_estimates:
            return [(0.0, 1.0)]

        # Simple model: distribute probability across fill levels
        # based on the minimum leg fill probability.
        min_fill_prob = min(e.fill_probability for e in leg_estimates)

        distribution: list[tuple[float, float]] = []
        remaining_prob = 1.0

        for i, frac in enumerate(fractions):
            if i == len(fractions) - 1:
                # Last level (full fill) gets all_fill_probability.
                prob = remaining_prob
            elif frac == 0.0:
                # Zero fill: probability of total failure.
                prob = max(0.0, (1.0 - min_fill_prob) ** len(leg_estimates))
                prob = min(prob, remaining_prob)
            else:
                # Intermediate levels: uniform distribution of remaining.
                intermediate_count = steps - 2  # Exclude 0.0 and 1.0.
                if intermediate_count > 0:
                    partial_total = remaining_prob - min_fill_prob ** len(leg_estimates)
                    prob = max(0.0, partial_total / intermediate_count)
                    prob = min(prob, remaining_prob)
                else:
                    prob = 0.0

            distribution.append((frac, prob))
            remaining_prob -= prob

        return distribution


# ---------------------------------------------------------------------------
# Leg input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegInput:
    """Input data for a single leg in execution simulation."""

    venue: str
    market_id: str
    side: str
    buy_price: float
    available_size: float
    spread: float = 0.0
    bid_price: float | None = None


# ---------------------------------------------------------------------------
# Rust dispatch for ExecutionModel.simulate().
# When arb_engine_rs is installed and ARB_USE_RUST_EXECUTION_MODEL=1 (or
# ARB_USE_RUST_ALL=1), replaces simulate() with Rust implementation.
# Set env var to "0" for instant rollback.
# ---------------------------------------------------------------------------


def _try_rust_dispatch() -> bool:
    """Attempt to replace ExecutionModel.simulate with Rust implementation."""
    import json
    import os

    if os.environ.get("ARB_USE_RUST_EXECUTION_MODEL", "") != "1" and \
       os.environ.get("ARB_USE_RUST_ALL", "") != "1":
        return False

    try:
        import arb_engine_rs  # type: ignore[import-untyped]
    except ImportError:
        return False

    _py_simulate = ExecutionModel.simulate

    def _rs_simulate(
        self: ExecutionModel,
        legs: list,
        contracts: int,
        staleness_seconds: float = 0.0,
        sequential: bool = False,
    ) -> ExecutionEstimate:
        cfg = self._config
        legs_json = json.dumps([
            {
                "venue": l.venue, "market_id": l.market_id,
                "side": l.side, "buy_price": l.buy_price,
                "available_size": l.available_size, "spread": l.spread,
            }
            for l in legs
        ])
        config_json = json.dumps({
            "queue_decay_half_life_seconds": cfg.queue_decay_half_life_seconds,
            "latency_seconds": cfg.latency_seconds,
            "market_impact_factor": cfg.market_impact_factor,
            "max_market_impact": cfg.max_market_impact,
            "min_fill_fraction": cfg.min_fill_fraction,
            "fill_fraction_steps": cfg.fill_fraction_steps,
            "sequential_leg_delay_seconds": cfg.sequential_leg_delay_seconds,
            "enable_queue_decay": cfg.enable_queue_decay,
            "enable_market_impact": cfg.enable_market_impact,
        })
        rs_json = arb_engine_rs.simulate_execution(
            legs_json, contracts, staleness_seconds, sequential, config_json,
        )
        d = json.loads(rs_json)
        leg_estimates = tuple(
            LegFillEstimate(
                venue=le["venue"], market_id=le["market_id"],
                side=le["side"], fill_probability=le["fill_probability"],
                expected_fill_fraction=le["expected_fill_fraction"],
                queue_position_score=le["queue_position_score"],
                market_impact=le["market_impact"],
                expected_slippage=le["expected_slippage"],
                time_offset_seconds=le["time_offset_seconds"],
            )
            for le in d["legs"]
        )
        return ExecutionEstimate(
            legs=leg_estimates,
            all_fill_probability=d["all_fill_probability"],
            expected_fill_fraction=d["expected_fill_fraction"],
            expected_slippage_per_contract=d["expected_slippage_per_contract"],
            expected_market_impact_per_contract=d["expected_market_impact_per_contract"],
            graduated_fill_distribution=tuple(
                tuple(pair) for pair in d["graduated_fill_distribution"]
            ),
        )

    ExecutionModel.simulate = _rs_simulate  # type: ignore[assignment]

    return True


_RUST_ACTIVE = _try_rust_dispatch()
