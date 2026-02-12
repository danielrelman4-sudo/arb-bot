from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from arb_bot.config import FillModelSettings
from arb_bot.models import ArbitrageOpportunity, Side, TradePlan


@dataclass(frozen=True)
class FillEstimate:
    leg_fill_probabilities: tuple[float, ...]
    all_fill_probability: float
    partial_fill_probability: float
    expected_slippage_per_contract: float
    fill_quality_score: float
    adverse_selection_flag: bool
    expected_realized_edge_per_contract: float
    expected_realized_profit: float


class FillModel:
    def __init__(self, settings: FillModelSettings) -> None:
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def estimate(
        self,
        opportunity: ArbitrageOpportunity,
        plan: TradePlan,
        now: datetime | None = None,
    ) -> FillEstimate:
        if plan.contracts <= 0:
            return FillEstimate(
                leg_fill_probabilities=tuple(),
                all_fill_probability=0.0,
                partial_fill_probability=1.0,
                expected_slippage_per_contract=0.0,
                fill_quality_score=0.0,
                adverse_selection_flag=False,
                expected_realized_edge_per_contract=-self._settings.partial_fill_penalty_per_contract,
                expected_realized_profit=0.0,
            )

        ts = now or datetime.now(timezone.utc)
        staleness_seconds = max(0.0, (ts - opportunity.observed_at).total_seconds())
        half_life = max(0.1, self._settings.stale_quote_half_life_seconds)
        staleness_factor = math.exp(-staleness_seconds / half_life)

        leg_probs: list[float] = []
        slippage_terms: list[float] = []
        fill_quality_terms: list[float] = []

        for leg in opportunity.legs:
            available = max(0.0, float(leg.buy_size))
            denom = max(1.0, plan.contracts * max(1.0, self._settings.queue_depth_factor))
            depth_ratio = min(1.0, available / denom)

            spread = _side_spread(leg.side, leg.metadata, leg.buy_price)
            spread_penalty = min(0.9, max(0.0, spread) * self._settings.spread_penalty_weight)

            source_penalty = 0.0
            source_key = "yes_buy_source" if leg.side is Side.YES else "no_buy_source"
            source = str(leg.metadata.get(source_key) or "").strip().lower()
            if source == "opposite_bid_transform":
                source_penalty = max(0.0, self._settings.transform_source_penalty)

            probability = depth_ratio * staleness_factor * (1.0 - spread_penalty) * (1.0 - source_penalty)
            probability = max(0.0, min(1.0, probability))
            leg_probs.append(probability)

            slippage_terms.append(max(0.0, spread) * (1.0 - probability))
            fill_quality_terms.append(_fill_quality_score(leg.side, leg.metadata, leg.buy_price, spread))

        all_fill_probability = 1.0
        for probability in leg_probs:
            all_fill_probability *= probability

        partial_fill_probability = 1.0 - all_fill_probability
        expected_slippage_per_contract = sum(slippage_terms) / max(1, len(slippage_terms))
        fill_quality_score = sum(fill_quality_terms) / max(1, len(fill_quality_terms))
        adverse_selection_flag = fill_quality_score < 0.0
        expected_realized_edge_per_contract = (
            plan.edge_per_contract * all_fill_probability
            - self._settings.partial_fill_penalty_per_contract * partial_fill_probability
            - expected_slippage_per_contract
        )
        expected_realized_profit = expected_realized_edge_per_contract * plan.contracts

        return FillEstimate(
            leg_fill_probabilities=tuple(leg_probs),
            all_fill_probability=all_fill_probability,
            partial_fill_probability=partial_fill_probability,
            expected_slippage_per_contract=expected_slippage_per_contract,
            fill_quality_score=fill_quality_score,
            adverse_selection_flag=adverse_selection_flag,
            expected_realized_edge_per_contract=expected_realized_edge_per_contract,
            expected_realized_profit=expected_realized_profit,
        )


def _side_spread(side: Side, metadata: dict[str, Any], leg_buy_price: float) -> float:
    key = "yes_spread" if side is Side.YES else "no_spread"
    spread = _to_float(metadata.get(key))
    if spread is not None:
        return max(0.0, spread)

    bid_key = "yes_bid_price" if side is Side.YES else "no_bid_price"
    bid_price = _to_float(metadata.get(bid_key))
    if bid_price is None:
        return 0.0
    return max(0.0, leg_buy_price - bid_price)


def _fill_quality_score(
    side: Side,
    metadata: dict[str, Any],
    leg_buy_price: float,
    spread: float,
) -> float:
    if spread <= 0:
        return 0.0

    midpoint = _side_midpoint(side, metadata, leg_buy_price, spread)
    if midpoint is None:
        return 0.0

    # For buys: positive score means paying below midpoint; negative means adverse selection.
    score = (midpoint - leg_buy_price) / spread
    return max(-2.0, min(2.0, score))


def _side_midpoint(
    side: Side,
    metadata: dict[str, Any],
    leg_buy_price: float,
    spread: float,
) -> float | None:
    bid_key = "yes_bid_price" if side is Side.YES else "no_bid_price"
    bid = _to_float(metadata.get(bid_key))
    if bid is not None:
        return (bid + leg_buy_price) / 2.0

    # Fallback when bid is unavailable but spread diagnostic exists.
    if spread > 0:
        return leg_buy_price - (spread / 2.0)
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
