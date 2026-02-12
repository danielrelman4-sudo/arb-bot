from __future__ import annotations

import math

from arb_bot.config import SizingSettings, StrategySettings
from arb_bot.models import ArbitrageOpportunity, TradeLegPlan, TradePlan


class PositionSizer:
    def __init__(self, sizing: SizingSettings, strategy: StrategySettings) -> None:
        self._sizing = sizing
        self._strategy = strategy

    def build_trade_plan(
        self,
        opportunity: ArbitrageOpportunity,
        available_cash_by_venue: dict[str, float],
        min_expected_profit_override: float | None = None,
        max_dollars_override: float | None = None,
        max_liquidity_fraction_override: float | None = None,
    ) -> TradePlan | None:
        cost_per_contract = opportunity.total_cost_per_contract
        if cost_per_contract <= 0:
            return None

        per_venue_cost = opportunity.capital_per_contract_by_venue

        venue_contract_caps: list[int] = []
        for venue, unit_cost in per_venue_cost.items():
            if unit_cost <= 0:
                continue
            available_cash = available_cash_by_venue.get(venue, 0.0)
            cap = math.floor((available_cash * self._sizing.max_bankroll_fraction_per_trade) / unit_cost)
            venue_contract_caps.append(cap)

        if not venue_contract_caps:
            return None

        max_from_cash = min(venue_contract_caps)
        max_trade_dollars = self._sizing.max_dollars_per_trade if max_dollars_override is None else max_dollars_override
        max_from_trade_cap = math.floor(max(0.0, max_trade_dollars) / cost_per_contract)

        liquidity_fraction = (
            self._sizing.max_liquidity_fraction_per_trade
            if max_liquidity_fraction_override is None
            else max_liquidity_fraction_override
        )
        liquidity_fraction = max(0.0, min(1.0, liquidity_fraction))
        max_from_liquidity = math.floor(
            max(0.0, min(leg.buy_size * liquidity_fraction for leg in opportunity.legs))
        )

        max_contracts = min(
            max_from_cash,
            max_from_trade_cap,
            max_from_liquidity,
            self._sizing.max_contracts_per_trade,
        )

        return self.resize_trade_plan(
            opportunity=opportunity,
            existing_plan=None,
            target_contracts=max_contracts,
            min_expected_profit_override=min_expected_profit_override,
        )

    def resize_trade_plan(
        self,
        opportunity: ArbitrageOpportunity,
        existing_plan: TradePlan | None,
        target_contracts: int,
        min_expected_profit_override: float | None = None,
    ) -> TradePlan | None:
        if target_contracts <= 0:
            return None

        if existing_plan is not None:
            contracts = min(target_contracts, existing_plan.contracts)
        else:
            contracts = target_contracts

        if contracts <= 0:
            return None

        expected_profit = contracts * opportunity.net_edge_per_contract
        min_expected_profit = (
            self._strategy.min_expected_profit_usd
            if min_expected_profit_override is None
            else min_expected_profit_override
        )
        if expected_profit < min_expected_profit:
            return None

        per_venue_cost = opportunity.capital_per_contract_by_venue
        legs = tuple(
            TradeLegPlan(
                venue=leg.venue,
                market_id=leg.market_id,
                side=leg.side,
                contracts=contracts,
                limit_price=leg.buy_price,
                metadata=leg.metadata,
            )
            for leg in opportunity.legs
        )

        capital_required_by_venue = {
            venue: unit_cost * contracts for venue, unit_cost in per_venue_cost.items()
        }

        return TradePlan(
            kind=opportunity.kind,
            execution_style=opportunity.execution_style,
            legs=legs,
            contracts=contracts,
            capital_required=contracts * opportunity.total_cost_per_contract,
            capital_required_by_venue=capital_required_by_venue,
            expected_profit=contracts * opportunity.net_edge_per_contract,
            edge_per_contract=opportunity.net_edge_per_contract,
            metadata={
                **opportunity.metadata,
                "match_key": opportunity.match_key,
                "match_score": opportunity.match_score,
            },
        )

    @staticmethod
    def execution_aware_kelly_fraction(
        edge_per_contract: float,
        cost_per_contract: float,
        fill_probability: float,
        failure_loss_per_contract: float | None = None,
    ) -> float:
        """Modified Kelly fraction that downweights sizing by fill uncertainty.

        Generalized binary Kelly with execution uncertainty:
        - b = upside gain fraction (edge / cost)
        - a = downside loss fraction on a failed execution
        - p = full-fill probability, q = 1 - p
        - f_raw = max(0, (b*p - a*q)/(a*b)), then damped by sqrt(p).
        """
        if edge_per_contract <= 0.0 or cost_per_contract <= 0.0:
            return 0.0

        p = max(0.0, min(1.0, float(fill_probability)))
        q = 1.0 - p
        b = edge_per_contract / cost_per_contract
        if b <= 0.0:
            return 0.0

        if failure_loss_per_contract is None:
            failure_loss_per_contract = cost_per_contract
        failure_loss_per_contract = max(0.0, min(cost_per_contract, failure_loss_per_contract))
        a = failure_loss_per_contract / cost_per_contract
        if a <= 0.0:
            raw = 1.0
        else:
            raw = max(0.0, (b * p - a * q) / (a * b))
        adjusted = raw * math.sqrt(p)
        return max(0.0, min(1.0, adjusted))


# ---------------------------------------------------------------------------
# Rust dispatch for execution_aware_kelly_fraction.
# When arb_engine_rs is installed and ARB_USE_RUST_SIZING=1 (or
# ARB_USE_RUST_ALL=1), replaces the static method with the Rust
# implementation. Set env var to "0" for instant rollback.
# ---------------------------------------------------------------------------


def _try_rust_dispatch() -> bool:
    """Attempt to replace static method with Rust implementation."""
    import os

    if os.environ.get("ARB_USE_RUST_SIZING", "") != "1" and \
       os.environ.get("ARB_USE_RUST_ALL", "") != "1":
        return False

    try:
        import arb_engine_rs  # type: ignore[import-untyped]
    except ImportError:
        return False

    _py_kelly = PositionSizer.execution_aware_kelly_fraction

    @staticmethod  # type: ignore[misc]
    def _rs_kelly(
        edge_per_contract: float,
        cost_per_contract: float,
        fill_probability: float,
        failure_loss_per_contract: float | None = None,
    ) -> float:
        return arb_engine_rs.execution_aware_kelly_fraction(
            edge_per_contract, cost_per_contract,
            fill_probability, failure_loss_per_contract,
        )

    PositionSizer.execution_aware_kelly_fraction = _rs_kelly  # type: ignore[assignment]

    return True


_RUST_ACTIVE = _try_rust_dispatch()
