"""Monte Carlo execution simulation for paper runs (B2).

Replaces deterministic EV booking with randomized per-trade outcome simulation:
- Per-leg fill/no-fill using individual fill probabilities
- Legging risk: leg 1 fills but leg 2 doesn't → exposed position with loss
- Adverse selection: filled orders may have worse outcomes
- Slippage: actual fill price differs from quoted price
- Edge decay from execution latency

Produces realistic PnL distributions with wins AND losses.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

from arb_bot.models import ArbitrageOpportunity, OpportunityKind, Side, TradePlan

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonteCarloSettings:
    """Configuration for Monte Carlo paper simulation."""

    enabled: bool = True

    # --- Legging risk ---
    # When one leg fills and another doesn't, what fraction of capital is lost.
    # In real trading this is the cost of unwinding the filled leg at market.
    legging_loss_fraction: float = 0.03  # 3% of leg capital lost on unwind

    # --- Adverse selection ---
    # Probability that a filled trade experiences adverse selection
    # (market moves against us after fill).
    adverse_selection_probability: float = 0.15
    # When adverse selection occurs, what fraction of edge is lost (0-1).
    adverse_selection_edge_loss: float = 0.5

    # --- Slippage ---
    # Standard deviation of slippage in cents per contract.
    slippage_std_cents: float = 0.5
    # Maximum slippage in cents per contract.
    slippage_max_cents: float = 2.0

    # --- Edge decay from latency ---
    # Expected latency in seconds between detection and execution.
    expected_latency_seconds: float = 1.5
    # Edge decay half-life in seconds.
    edge_decay_half_life_seconds: float = 30.0

    # --- Outcome simulation ---
    # For settled positions: probability the arb resolves as expected.
    # Most arbs are near-certain (e.g., parity constraints), but some
    # may not resolve cleanly.
    resolution_success_rate: float = 0.95

    # Random seed for reproducibility (None = non-deterministic).
    seed: int | None = None


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of a Monte Carlo simulation for a single trade."""

    # Per-leg fill outcomes.
    leg_fills: tuple[bool, ...]
    # Whether ALL legs filled (trade is fully hedged).
    all_legs_filled: bool
    # Number of legs that filled.
    filled_leg_count: int

    # Simulated fill contracts (may differ from plan due to partial fills).
    simulated_filled_contracts: int

    # Simulated P&L components.
    gross_edge_pnl: float  # Edge profit if all goes well
    slippage_cost: float  # Cost of slippage on fills
    adverse_selection_cost: float  # Cost of adverse selection
    legging_loss: float  # Loss from incomplete hedges
    edge_decay_cost: float  # Loss from latency-induced edge decay
    resolution_pnl: float  # Adjustment for resolution risk

    # Final simulated P&L.
    simulated_pnl: float

    # Whether adverse selection occurred.
    adverse_selection_hit: bool
    # Simulated execution latency in ms.
    simulated_latency_ms: float


class MonteCarloSimulator:
    """Simulates realistic trade outcomes for paper runs."""

    def __init__(self, settings: MonteCarloSettings | None = None) -> None:
        self._settings = settings or MonteCarloSettings()
        self._rng = random.Random(self._settings.seed)

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def simulate_trade(
        self,
        opportunity: ArbitrageOpportunity,
        plan: TradePlan,
        leg_fill_probabilities: tuple[float, ...] | None = None,
        all_fill_probability: float = 1.0,
        expected_realized_profit: float = 0.0,
    ) -> MonteCarloResult:
        """Simulate a single trade outcome using Monte Carlo.

        Parameters
        ----------
        opportunity:
            The detected arbitrage opportunity.
        plan:
            The trade plan with contracts and pricing.
        leg_fill_probabilities:
            Per-leg fill probabilities from the fill model. If None,
            uses ``all_fill_probability`` for all legs uniformly.
        all_fill_probability:
            Combined fill probability (used if per-leg not available).
        expected_realized_profit:
            The EV-based profit estimate (used as base for simulation).
        """
        n_legs = len(opportunity.legs)

        # 1. Determine per-leg fill probabilities.
        if leg_fill_probabilities and len(leg_fill_probabilities) == n_legs:
            probs = list(leg_fill_probabilities)
        else:
            # Distribute combined probability evenly across legs.
            per_leg = all_fill_probability ** (1.0 / max(1, n_legs))
            probs = [per_leg] * n_legs

        # 2. Simulate per-leg fills.
        leg_fills = tuple(self._rng.random() < p for p in probs)
        filled_count = sum(leg_fills)
        all_filled = filled_count == n_legs

        # 3. Simulate execution latency.
        latency_seconds = max(
            0.0,
            self._rng.gauss(
                self._settings.expected_latency_seconds,
                self._settings.expected_latency_seconds * 0.5,
            ),
        )
        simulated_latency_ms = latency_seconds * 1000.0

        # 4. Calculate edge decay from latency.
        half_life = max(0.1, self._settings.edge_decay_half_life_seconds)
        decay_factor = math.exp(-latency_seconds / half_life)
        edge_after_decay = plan.edge_per_contract * decay_factor
        edge_decay_cost_per_contract = plan.edge_per_contract - edge_after_decay

        # 5. Determine filled contracts.
        if all_filled:
            simulated_contracts = plan.contracts
        elif filled_count == 0:
            simulated_contracts = 0
        else:
            # Partial fill: use fill ratio.
            simulated_contracts = max(1, int(plan.contracts * (filled_count / n_legs)))

        # 6. Compute P&L components.
        gross_edge_pnl = 0.0
        slippage_cost = 0.0
        adverse_selection_cost = 0.0
        legging_loss = 0.0
        edge_decay_cost = 0.0
        resolution_pnl = 0.0
        adverse_selection_hit = False

        if all_filled:
            # All legs filled — trade is fully hedged.
            gross_edge_pnl = edge_after_decay * simulated_contracts

            # Slippage simulation.
            slippage_per = min(
                self._settings.slippage_max_cents / 100.0,
                max(0.0, self._rng.gauss(0.0, self._settings.slippage_std_cents / 100.0)),
            )
            slippage_cost = slippage_per * simulated_contracts

            # Adverse selection simulation.
            if self._rng.random() < self._settings.adverse_selection_probability:
                adverse_selection_hit = True
                adverse_selection_cost = (
                    abs(gross_edge_pnl) * self._settings.adverse_selection_edge_loss
                )

            # Edge decay cost.
            edge_decay_cost = edge_decay_cost_per_contract * simulated_contracts

            # Resolution risk: does the arb resolve as expected?
            if self._rng.random() > self._settings.resolution_success_rate:
                # Arb doesn't resolve cleanly — lose some or all of edge.
                resolution_loss = gross_edge_pnl * self._rng.uniform(0.5, 1.0)
                resolution_pnl = -resolution_loss

        elif filled_count > 0:
            # Legging risk: some legs filled, some didn't.
            # The filled legs are now unhedged exposures.
            # Estimate loss as fraction of capital committed to filled legs.
            committed_total = sum(plan.capital_required_by_venue.values())
            filled_fraction = filled_count / max(1, n_legs)
            legging_loss = (
                committed_total
                * filled_fraction
                * self._settings.legging_loss_fraction
            )
            # Reset contracts — no successful trade.
            simulated_contracts = 0

        # else: no legs filled — no trade, no PnL.

        simulated_pnl = (
            gross_edge_pnl
            - slippage_cost
            - adverse_selection_cost
            - legging_loss
            - edge_decay_cost
            + resolution_pnl
        )

        return MonteCarloResult(
            leg_fills=leg_fills,
            all_legs_filled=all_filled,
            filled_leg_count=filled_count,
            simulated_filled_contracts=simulated_contracts,
            gross_edge_pnl=gross_edge_pnl,
            slippage_cost=slippage_cost,
            adverse_selection_cost=adverse_selection_cost,
            legging_loss=legging_loss,
            edge_decay_cost=edge_decay_cost,
            resolution_pnl=resolution_pnl,
            simulated_pnl=simulated_pnl,
            adverse_selection_hit=adverse_selection_hit,
            simulated_latency_ms=simulated_latency_ms,
        )

    def simulate_settlement(
        self,
        expected_realized_profit: float,
        filled_contracts: int,
        opportunity: ArbitrageOpportunity,
    ) -> float:
        """Simulate the settlement outcome for a paper position.

        Called when a paper position reaches its lifetime and settles.
        Instead of deterministically returning ``expected_realized_profit``,
        this applies resolution risk and adverse selection to produce a
        realistic settlement P&L.

        Parameters
        ----------
        expected_realized_profit:
            The EV-based profit estimate.
        filled_contracts:
            Number of contracts that were filled.
        opportunity:
            The arbitrage opportunity.

        Returns
        -------
        Simulated realized profit (may be negative).
        """
        if filled_contracts <= 0 or expected_realized_profit == 0.0:
            return 0.0

        # Resolution risk: does the arb resolve as expected?
        if self._rng.random() > self._settings.resolution_success_rate:
            loss_fraction = self._rng.uniform(0.3, 1.0)
            return -abs(expected_realized_profit) * loss_fraction

        # Adverse selection at settlement.
        if self._rng.random() < self._settings.adverse_selection_probability:
            reduction = self._settings.adverse_selection_edge_loss
            return expected_realized_profit * (1.0 - reduction)

        # Normal resolution: small random variation around expected.
        noise = self._rng.gauss(0.0, 0.05)  # ±5% noise
        return expected_realized_profit * (1.0 + noise)
