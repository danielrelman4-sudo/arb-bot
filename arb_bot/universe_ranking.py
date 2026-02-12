from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from arb_bot.config import UniverseRankingSettings
from arb_bot.models import BinaryQuote


def rank_quotes(
    quotes: list[BinaryQuote],
    settings: UniverseRankingSettings,
    now: datetime | None = None,
) -> list[BinaryQuote]:
    if not settings.enabled:
        return list(quotes)

    ts = now or datetime.now(timezone.utc)
    return sorted(
        quotes,
        key=lambda quote: quote_rank_score(quote, settings, ts),
        reverse=True,
    )


def quote_rank_score(
    quote: BinaryQuote,
    settings: UniverseRankingSettings,
    now: datetime,
) -> float:
    volume = _to_float(quote.metadata.get("volume_24h") or quote.metadata.get("volume") or 0.0)
    liquidity = _to_float(
        quote.metadata.get("liquidity")
        or quote.metadata.get("liquidity_dollars")
        or quote.metadata.get("open_interest")
        or 0.0
    )
    spread = _coalesce_spread(quote)

    staleness_seconds = max(0.0, (now - quote.observed_at).total_seconds())
    staleness_minutes = staleness_seconds / 60.0

    score = 0.0
    score += settings.volume_weight * math.log1p(max(0.0, volume))
    score += settings.liquidity_weight * math.log1p(max(0.0, liquidity))
    score -= settings.spread_weight * max(0.0, spread)
    score -= settings.staleness_weight * staleness_minutes
    return score


def _coalesce_spread(quote: BinaryQuote) -> float:
    yes_spread = _to_float(quote.metadata.get("yes_spread"))
    no_spread = _to_float(quote.metadata.get("no_spread"))
    if yes_spread is not None and no_spread is not None:
        return 0.5 * (max(0.0, yes_spread) + max(0.0, no_spread))

    if quote.yes_bid_price is not None and quote.no_bid_price is not None:
        yes = max(0.0, quote.yes_buy_price - quote.yes_bid_price)
        no = max(0.0, quote.no_buy_price - quote.no_bid_price)
        return 0.5 * (yes + no)

    # Fallback: cost deviation from perfect no-arb.
    return max(0.0, quote.yes_buy_price + quote.no_buy_price - 1.0)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
