from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_EPSILON = 1e-9


@dataclass(frozen=True)
class QuoteDecomposition:
    implied_probability: float
    edge_per_side: float


@dataclass(frozen=True)
class EffectiveBuy:
    price: float
    size: float
    source: Literal["direct_ask", "opposite_bid_transform"]


@dataclass(frozen=True)
class QuoteDiagnostics:
    ask_implied_probability: float
    ask_edge_per_side: float
    bid_implied_probability: float | None
    bid_edge_per_side: float | None
    midpoint_consistency_gap: float | None
    yes_spread: float | None
    no_spread: float | None
    spread_asymmetry: float | None


def zero_spread_yes_buy_from_no_bid(no_bid_price: float | None) -> float | None:
    if no_bid_price is None:
        return None
    return _clamp_price(1.0 - no_bid_price)


def zero_spread_no_buy_from_yes_bid(yes_bid_price: float | None) -> float | None:
    if yes_bid_price is None:
        return None
    return _clamp_price(1.0 - yes_bid_price)


def decompose_binary_quote(yes_price: float, no_price: float) -> QuoteDecomposition:
    yes = _validate_price(yes_price)
    no = _validate_price(no_price)
    implied_probability = 0.5 * (yes + (1.0 - no))
    edge_per_side = 0.5 * (yes + no - 1.0)
    return QuoteDecomposition(
        implied_probability=implied_probability,
        edge_per_side=edge_per_side,
    )


def reconstruct_binary_quote(
    implied_probability: float,
    edge_per_side: float,
) -> tuple[float, float]:
    p = _validate_price(implied_probability)
    edge = float(edge_per_side)
    yes = _clamp_price(p + edge)
    no = _clamp_price((1.0 - p) + edge)
    return yes, no


def choose_effective_buy_price(
    side: Literal["yes", "no"],
    direct_ask_price: float | None,
    direct_ask_size: float,
    opposite_bid_price: float | None,
    opposite_bid_size: float,
) -> EffectiveBuy | None:
    candidates: list[EffectiveBuy] = []

    normalized_ask = _normalize_price_or_none(direct_ask_price)
    normalized_ask_size = _normalize_size(direct_ask_size)
    if normalized_ask is not None and normalized_ask_size > 0:
        candidates.append(
            EffectiveBuy(
                price=normalized_ask,
                size=normalized_ask_size,
                source="direct_ask",
            )
        )

    transformed_price = (
        zero_spread_yes_buy_from_no_bid(opposite_bid_price)
        if side == "yes"
        else zero_spread_no_buy_from_yes_bid(opposite_bid_price)
    )
    normalized_transformed_size = _normalize_size(opposite_bid_size)
    if transformed_price is not None and normalized_transformed_size > 0:
        candidates.append(
            EffectiveBuy(
                price=transformed_price,
                size=normalized_transformed_size,
                source="opposite_bid_transform",
            )
        )

    if not candidates:
        return None

    # Favor the lowest executable price. On ties, favor deeper size.
    return min(candidates, key=lambda item: (item.price, -item.size))


def build_quote_diagnostics(
    yes_buy_price: float,
    no_buy_price: float,
    yes_bid_price: float | None,
    no_bid_price: float | None,
) -> QuoteDiagnostics:
    ask = decompose_binary_quote(yes_buy_price, no_buy_price)

    yes_bid = _normalize_price_or_none(yes_bid_price)
    no_bid = _normalize_price_or_none(no_bid_price)
    if yes_bid is None or no_bid is None:
        return QuoteDiagnostics(
            ask_implied_probability=ask.implied_probability,
            ask_edge_per_side=ask.edge_per_side,
            bid_implied_probability=None,
            bid_edge_per_side=None,
            midpoint_consistency_gap=None,
            yes_spread=None,
            no_spread=None,
            spread_asymmetry=None,
        )

    bid_implied_probability = 0.5 * (yes_bid + (1.0 - no_bid))
    bid_edge_per_side = 0.5 * (1.0 - yes_bid - no_bid)
    yes_spread = yes_buy_price - yes_bid
    no_spread = no_buy_price - no_bid

    return QuoteDiagnostics(
        ask_implied_probability=ask.implied_probability,
        ask_edge_per_side=ask.edge_per_side,
        bid_implied_probability=bid_implied_probability,
        bid_edge_per_side=bid_edge_per_side,
        midpoint_consistency_gap=abs(ask.implied_probability - bid_implied_probability),
        yes_spread=yes_spread,
        no_spread=no_spread,
        spread_asymmetry=yes_spread - no_spread,
    )


def _validate_price(value: float) -> float:
    numeric = float(value)
    if numeric < -_EPSILON or numeric > 1.0 + _EPSILON:
        raise ValueError(f"price out of range [0, 1]: {value}")
    return min(1.0, max(0.0, numeric))


def _normalize_price_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric < -_EPSILON or numeric > 1.0 + _EPSILON:
        return None
    return min(1.0, max(0.0, numeric))


def _clamp_price(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalize_size(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, numeric)
