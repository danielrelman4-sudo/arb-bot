"""Edge detection: compare model probabilities against market prices.

Given Monte Carlo probability estimates and live Kalshi quotes,
identifies mispriced contracts and emits trading signals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketQuote
from arb_bot.crypto.price_model import ProbabilityEstimate

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CryptoEdge:
    """A detected edge between model and market."""

    market: CryptoMarket
    model_prob: ProbabilityEstimate
    market_implied_prob: float
    edge: float                    # model_prob.probability - market_implied_prob
    edge_cents: float              # edge * $1.00 payout = dollar edge
    side: str                      # "yes" or "no"
    recommended_price: float       # Price to bid/offer
    model_uncertainty: float       # Wilson CI half-width
    time_to_expiry_minutes: float
    yes_buy_price: float
    no_buy_price: float
    blended_probability: float = 0.0  # After market blending


def compute_implied_probability(yes_price: float, no_price: float) -> float:
    """Convert Kalshi binary prices to implied probability.

    Uses the midpoint method: implied_prob = 0.5 * (yes + (1 - no))
    This is the same formula as ``decompose_binary_quote`` in binary_math.py.
    """
    return 0.5 * (yes_price + (1.0 - no_price))


def blend_probabilities(
    model_prob: float,
    market_prob: float,
    model_uncertainty: float,
    base_model_weight: float = 0.7,
    uncertainty_scaling: float = 5.0,
) -> float:
    """Blend model and market probabilities based on model confidence.

    When model uncertainty is low (tight CI), weight model more heavily.
    When uncertainty is high (wide CI), defer to market price.

    Parameters
    ----------
    model_prob: MC model probability estimate
    market_prob: Market-implied probability from bid/ask
    model_uncertainty: Wilson CI half-width
    base_model_weight: Starting weight for model (0-1)
    uncertainty_scaling: How fast to reduce model weight as uncertainty grows
    """
    # Reduce model weight as uncertainty increases
    # At uncertainty=0, weight = base_model_weight
    # At uncertainty=0.10, weight ~ base_model_weight * 0.6
    # At uncertainty=0.20, weight ~ base_model_weight * 0.37
    decay = math.exp(-uncertainty_scaling * model_uncertainty)
    model_weight = base_model_weight * decay
    market_weight = 1.0 - model_weight

    blended = model_weight * model_prob + market_weight * market_prob
    return max(0.0, min(1.0, blended))


class EdgeDetector:
    """Finds mispriced crypto markets by comparing model vs market.

    Parameters
    ----------
    min_edge_pct:
        Minimum absolute edge as a fraction (e.g. 0.05 = 5%).
    min_edge_pct_daily:
        Higher minimum edge for daily (above/below, tte > 30 min) markets.
    min_edge_cents:
        Minimum edge in dollar terms (e.g. 0.02 = 2¢).
    max_model_uncertainty:
        Max Wilson CI half-width to trade.
    use_blending:
        Whether to blend model probability with market implied probability.
    """

    def __init__(
        self,
        min_edge_pct: float = 0.05,
        min_edge_pct_daily: float = 0.06,
        min_edge_cents: float = 0.02,
        max_model_uncertainty: float = 0.15,
        use_blending: bool = True,
    ) -> None:
        self._min_edge_pct = min_edge_pct
        self._min_edge_pct_daily = min_edge_pct_daily
        self._min_edge_cents = min_edge_cents
        self._max_uncertainty = max_model_uncertainty
        self._use_blending = use_blending

    def detect_edges(
        self,
        market_quotes: list[CryptoMarketQuote],
        model_probs: dict[str, ProbabilityEstimate],
    ) -> list[CryptoEdge]:
        """Compare model probabilities vs market implied probabilities.

        Parameters
        ----------
        market_quotes:
            Live Kalshi quotes with parsed metadata.
        model_probs:
            Map of market ticker -> model probability estimate.

        Returns
        -------
        list[CryptoEdge]
            Edges that pass all filters, sorted by absolute edge descending.
        """
        edges: list[CryptoEdge] = []

        for mq in market_quotes:
            ticker = mq.market.ticker
            model = model_probs.get(ticker)
            if model is None:
                continue

            # Market implied probability
            market_prob = mq.implied_probability

            # Blend model probability with market probability if enabled
            if self._use_blending:
                blended = blend_probabilities(
                    model_prob=model.probability,
                    market_prob=market_prob,
                    model_uncertainty=model.uncertainty,
                )
            else:
                blended = model.probability

            # Raw edge: blended probability vs market implied
            raw_edge = blended - market_prob

            # Determine side and effective edge measured against execution price
            # (the actual cost of buying), not the midpoint.
            if raw_edge > 0:
                # Buy YES -- edge is blended_prob minus the actual cost (yes_buy_price)
                side = "yes"
                effective_edge = blended - mq.yes_buy_price
                recommended_price = mq.yes_buy_price
            elif raw_edge < 0:
                # Buy NO -- edge is (1-blended_prob) minus the actual cost (no_buy_price)
                side = "no"
                effective_edge = (1.0 - blended) - mq.no_buy_price
                recommended_price = mq.no_buy_price
            else:
                continue  # No edge

            edge_cents = effective_edge  # $1 payout, so edge_pct = edge_cents

            # -- Apply filters -----------------------------------------
            # Use higher threshold for daily (above/below) markets
            is_daily = (
                mq.market.meta.direction in ("above", "below")
                and mq.time_to_expiry_minutes > 30
            )
            min_edge = self._min_edge_pct_daily if is_daily else self._min_edge_pct
            if effective_edge < min_edge:
                continue
            if edge_cents < self._min_edge_cents:
                continue
            if model.uncertainty > self._max_uncertainty:
                LOGGER.debug(
                    "Edge on %s rejected: uncertainty %.3f > %.3f",
                    ticker, model.uncertainty, self._max_uncertainty,
                )
                continue

            edges.append(CryptoEdge(
                market=mq.market,
                model_prob=model,
                market_implied_prob=market_prob,
                edge=raw_edge,
                edge_cents=edge_cents,
                side=side,
                recommended_price=recommended_price,
                model_uncertainty=model.uncertainty,
                time_to_expiry_minutes=mq.time_to_expiry_minutes,
                yes_buy_price=mq.yes_buy_price,
                no_buy_price=mq.no_buy_price,
                blended_probability=blended,
            ))

        # Sort by largest absolute edge first
        edges.sort(key=lambda e: abs(e.edge), reverse=True)
        return edges
