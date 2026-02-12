"""Pre-trade quote and mapping risk gates (Phase 2A).

Provides configurable gates that validate quote freshness, price sanity,
and mapping confidence before a TradePlan is accepted for execution.
These run after opportunity detection and sizing but before risk precheck.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from arb_bot.models import BinaryQuote, TradePlan

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuoteGateConfig:
    """Configuration for pre-trade quote gates.

    Parameters
    ----------
    max_quote_age_seconds:
        Reject if any leg's quote is older than this. 0 = disabled.
    min_mapping_confidence:
        For cross-venue plans, reject if mapping confidence (match_score)
        is below this threshold. 0 = disabled.
    max_bid_ask_spread:
        Reject if any leg's bid-ask spread exceeds this. 0 = disabled.
    require_both_sides_quoted:
        If True, reject if a quote is missing bid-side pricing.
    max_price_vs_complement_deviation:
        Reject if |yes_price + no_price - 1.0| exceeds this.
        Catches micro-price artifacts. 0 = disabled.
    min_size_contracts:
        Reject if any leg's available size is below this minimum. 0 = disabled.
    """

    max_quote_age_seconds: float = 0.0
    min_mapping_confidence: float = 0.0
    max_bid_ask_spread: float = 0.0
    require_both_sides_quoted: bool = False
    max_price_vs_complement_deviation: float = 0.0
    min_size_contracts: float = 0.0


@dataclass(frozen=True)
class QuoteGateResult:
    """Result of quote gate checks for a single plan."""

    passed: bool
    reason: str
    checks_run: int = 0
    checks_failed: int = 0


class QuoteGateChecker:
    """Runs pre-trade quote gates against a TradePlan.

    Call ``check()`` with a plan and a dict of current quotes (keyed by
    ``(venue, market_id)``). Returns a QuoteGateResult indicating whether
    the plan should proceed.
    """

    def __init__(self, config: QuoteGateConfig | None = None) -> None:
        self._config = config or QuoteGateConfig()

    @property
    def config(self) -> QuoteGateConfig:
        return self._config

    def check(
        self,
        plan: TradePlan,
        quotes: Dict[tuple[str, str], BinaryQuote],
        now: datetime | None = None,
    ) -> QuoteGateResult:
        """Run all enabled gates against the plan.

        Parameters
        ----------
        plan:
            The trade plan to validate.
        quotes:
            Dict of current quotes keyed by (venue, market_id).
        now:
            Current time (for staleness checks). Defaults to utcnow.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        failures: list[str] = []
        checks_run = 0

        # Collect quotes for each leg.
        leg_quotes: list[tuple[Any, BinaryQuote | None]] = []
        for leg in plan.legs:
            key = (leg.venue, leg.market_id)
            leg_quotes.append((leg, quotes.get(key)))

        # Gate 1: Quote availability
        checks_run += 1
        missing = [(leg.venue, leg.market_id) for leg, q in leg_quotes if q is None]
        if missing:
            ids = ", ".join(f"{v}/{m}" for v, m in missing)
            failures.append(f"missing quotes: {ids}")

        # Gate 2: Quote freshness
        if self._config.max_quote_age_seconds > 0:
            checks_run += 1
            for leg, quote in leg_quotes:
                if quote is None:
                    continue
                age = (now - quote.observed_at).total_seconds()
                if age > self._config.max_quote_age_seconds:
                    failures.append(
                        f"stale quote {leg.venue}/{leg.market_id} "
                        f"(age={age:.1f}s, max={self._config.max_quote_age_seconds:.1f}s)"
                    )

        # Gate 3: Mapping confidence (cross-venue only)
        if self._config.min_mapping_confidence > 0:
            checks_run += 1
            match_score = plan.metadata.get("match_score")
            if match_score is not None:
                if match_score < self._config.min_mapping_confidence:
                    failures.append(
                        f"low mapping confidence "
                        f"(score={match_score:.3f}, min={self._config.min_mapping_confidence:.3f})"
                    )

        # Gate 4: Bid-ask spread
        if self._config.max_bid_ask_spread > 0:
            checks_run += 1
            for leg, quote in leg_quotes:
                if quote is None:
                    continue
                spread = _bid_ask_spread(quote, leg.side.value)
                if spread is not None and spread > self._config.max_bid_ask_spread:
                    failures.append(
                        f"wide spread {leg.venue}/{leg.market_id}/{leg.side.value} "
                        f"(spread={spread:.4f}, max={self._config.max_bid_ask_spread:.4f})"
                    )

        # Gate 5: Both sides quoted
        if self._config.require_both_sides_quoted:
            checks_run += 1
            for leg, quote in leg_quotes:
                if quote is None:
                    continue
                if leg.side.value == "yes" and quote.yes_bid_price is None:
                    failures.append(
                        f"no bid for {leg.venue}/{leg.market_id}/yes"
                    )
                elif leg.side.value == "no" and quote.no_bid_price is None:
                    failures.append(
                        f"no bid for {leg.venue}/{leg.market_id}/no"
                    )

        # Gate 6: Price vs complement deviation
        if self._config.max_price_vs_complement_deviation > 0:
            checks_run += 1
            for leg, quote in leg_quotes:
                if quote is None:
                    continue
                deviation = abs(quote.yes_buy_price + quote.no_buy_price - 1.0)
                if deviation > self._config.max_price_vs_complement_deviation:
                    failures.append(
                        f"price complement deviation {leg.venue}/{leg.market_id} "
                        f"(yes={quote.yes_buy_price:.4f}, no={quote.no_buy_price:.4f}, "
                        f"dev={deviation:.4f}, max={self._config.max_price_vs_complement_deviation:.4f})"
                    )

        # Gate 7: Minimum size
        if self._config.min_size_contracts > 0:
            checks_run += 1
            for leg, quote in leg_quotes:
                if quote is None:
                    continue
                size = quote.yes_buy_size if leg.side.value == "yes" else quote.no_buy_size
                if size < self._config.min_size_contracts:
                    failures.append(
                        f"thin book {leg.venue}/{leg.market_id}/{leg.side.value} "
                        f"(size={size:.1f}, min={self._config.min_size_contracts:.1f})"
                    )

        if failures:
            reason = "; ".join(failures)
            LOGGER.debug("Quote gate rejected plan %s: %s", plan.market_key, reason)
            return QuoteGateResult(
                passed=False,
                reason=reason,
                checks_run=checks_run,
                checks_failed=len(failures),
            )

        return QuoteGateResult(
            passed=True,
            reason="ok",
            checks_run=checks_run,
            checks_failed=0,
        )


def _bid_ask_spread(quote: BinaryQuote, side: str) -> float | None:
    """Compute bid-ask spread for the given side. Returns None if no bid."""
    if side == "yes":
        bid = quote.yes_bid_price
        ask = quote.yes_buy_price
    else:
        bid = quote.no_bid_price
        ask = quote.no_buy_price
    if bid is None:
        return None
    return ask - bid
