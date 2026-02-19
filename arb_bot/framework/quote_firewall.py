"""Quote sanity firewall hardening (Phase 2B).

A second-pass filter that catches subtle quote anomalies that pass basic
validation but indicate unreliable or manipulated data. Runs after the
basic quality gate (engine._filter_quotes_for_quality) and before
opportunity detection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from arb_bot.models import BinaryQuote

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FirewallConfig:
    """Configuration for quote sanity firewall.

    Parameters
    ----------
    max_complement_sum_deviation:
        Reject if |yes_buy + no_buy - 1.0| exceeds this. Catches
        micro-price artifacts and degenerate books. Default 0.15.
    min_price_floor:
        Reject quotes with yes or no price below this floor.
        Prevents trading on near-zero prices. Default 0.01.
    max_price_ceiling:
        Reject quotes with yes or no price above this ceiling.
        Prevents trading at near-certain prices. Default 0.99.
    max_price_jump:
        If a quote's price changed by more than this since the
        previous observation, flag it. Default 0.0 (disabled).
    max_quote_age_seconds:
        Reject quotes older than this. Default 0.0 (disabled).
    min_book_depth_contracts:
        Reject if either side has depth below this. Default 0.0 (disabled).
    reject_locked_book:
        Reject if bid >= ask on either side (locked/crossed book).
        Default True.
    reject_inverted_prices:
        Reject if yes_buy > 1.0 - no_buy (buying both sides costs > $1).
        This indicates data error or extreme illiquidity. Default False.
    """

    max_complement_sum_deviation: float = 0.15
    min_price_floor: float = 0.01
    max_price_ceiling: float = 0.99
    max_price_jump: float = 0.0
    max_quote_age_seconds: float = 0.0
    min_book_depth_contracts: float = 0.0
    reject_locked_book: bool = True
    reject_inverted_prices: bool = False


@dataclass
class FirewallStats:
    """Tracks firewall rejection statistics."""

    total_checked: int = 0
    total_rejected: int = 0
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.total_rejected += 1
        self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1

    def reset(self) -> None:
        self.total_checked = 0
        self.total_rejected = 0
        self.rejections_by_reason.clear()


class QuoteFirewall:
    """Second-pass quote filter for subtle anomalies.

    Returns a filtered list of quotes plus rejection reasons for each
    rejected quote. Optionally tracks price history for jump detection.
    """

    def __init__(self, config: FirewallConfig | None = None) -> None:
        self._config = config or FirewallConfig()
        self._last_prices: Dict[tuple[str, str], tuple[float, float]] = {}
        self._stats = FirewallStats()

    @property
    def config(self) -> FirewallConfig:
        return self._config

    @property
    def stats(self) -> FirewallStats:
        return self._stats

    def filter(
        self,
        quotes: list[BinaryQuote],
        now: datetime | None = None,
    ) -> tuple[list[BinaryQuote], Dict[str, list[str]]]:
        """Filter quotes through the firewall.

        Parameters
        ----------
        quotes:
            List of BinaryQuote objects to check.
        now:
            Current time. Defaults to utcnow.

        Returns
        -------
        Tuple of (accepted quotes, rejected dict mapping market_key → [reasons]).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        accepted: list[BinaryQuote] = []
        rejected: Dict[str, list[str]] = {}

        for quote in quotes:
            self._stats.total_checked += 1
            reasons = self._check(quote, now)
            market_key = f"{quote.venue}/{quote.market_id}"

            if reasons:
                rejected[market_key] = reasons
                for reason in reasons:
                    self._stats.record_rejection(reason)
                LOGGER.debug(
                    "Firewall rejected %s: %s",
                    market_key,
                    "; ".join(reasons),
                )
            else:
                accepted.append(quote)
                # Update price history for jump detection.
                key = (quote.venue, quote.market_id)
                self._last_prices[key] = (quote.yes_buy_price, quote.no_buy_price)

        return accepted, rejected

    def _check(self, quote: BinaryQuote, now: datetime) -> list[str]:
        reasons: list[str] = []
        cfg = self._config

        yes = quote.yes_buy_price
        no = quote.no_buy_price

        # 1. Complement sum deviation
        if cfg.max_complement_sum_deviation > 0:
            deviation = abs(yes + no - 1.0)
            if deviation > cfg.max_complement_sum_deviation:
                reasons.append(
                    f"complement_deviation({yes:.4f}+{no:.4f}={yes+no:.4f}, "
                    f"dev={deviation:.4f})"
                )

        # 2. Price floor
        if cfg.min_price_floor > 0:
            if yes < cfg.min_price_floor:
                reasons.append(f"below_price_floor(yes={yes:.4f})")
            if no < cfg.min_price_floor:
                reasons.append(f"below_price_floor(no={no:.4f})")

        # 3. Price ceiling
        if cfg.max_price_ceiling < 1.0:
            if yes > cfg.max_price_ceiling:
                reasons.append(f"above_price_ceiling(yes={yes:.4f})")
            if no > cfg.max_price_ceiling:
                reasons.append(f"above_price_ceiling(no={no:.4f})")

        # 4. Price jump
        if cfg.max_price_jump > 0:
            key = (quote.venue, quote.market_id)
            prev = self._last_prices.get(key)
            if prev is not None:
                prev_yes, prev_no = prev
                yes_jump = abs(yes - prev_yes)
                no_jump = abs(no - prev_no)
                if yes_jump > cfg.max_price_jump:
                    reasons.append(f"price_jump(yes={prev_yes:.4f}→{yes:.4f}, Δ={yes_jump:.4f})")
                if no_jump > cfg.max_price_jump:
                    reasons.append(f"price_jump(no={prev_no:.4f}→{no:.4f}, Δ={no_jump:.4f})")

        # 5. Quote age
        if cfg.max_quote_age_seconds > 0:
            age = (now - quote.observed_at).total_seconds()
            if age > cfg.max_quote_age_seconds:
                reasons.append(f"stale(age={age:.1f}s)")

        # 6. Book depth
        if cfg.min_book_depth_contracts > 0:
            if quote.yes_buy_size < cfg.min_book_depth_contracts:
                reasons.append(f"thin_book(yes_size={quote.yes_buy_size:.1f})")
            if quote.no_buy_size < cfg.min_book_depth_contracts:
                reasons.append(f"thin_book(no_size={quote.no_buy_size:.1f})")

        # 7. Locked/crossed book
        if cfg.reject_locked_book:
            if quote.yes_bid_price is not None and quote.yes_bid_price >= yes:
                reasons.append(f"locked_book(yes_bid={quote.yes_bid_price:.4f}>=ask={yes:.4f})")
            if quote.no_bid_price is not None and quote.no_bid_price >= no:
                reasons.append(f"locked_book(no_bid={quote.no_bid_price:.4f}>=ask={no:.4f})")

        # 8. Inverted prices (buying both sides costs > $1)
        if cfg.reject_inverted_prices:
            if yes + no > 1.0:
                reasons.append(f"inverted_prices(yes={yes:.4f}+no={no:.4f}={yes+no:.4f}>1.0)")

        return reasons

    def reset_price_history(self) -> None:
        """Clear price history (e.g., after reconnect)."""
        self._last_prices.clear()

    def reset_stats(self) -> None:
        self._stats.reset()
