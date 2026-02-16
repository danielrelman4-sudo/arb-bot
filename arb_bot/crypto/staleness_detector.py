"""Quote staleness detection for Kalshi crypto markets.

Detects when Kalshi quotes lag behind Binance spot price movements,
identifying stale quotes that represent exploitable mispricings.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StalenessResult:
    """Result of a staleness check for a single ticker."""
    is_stale: bool
    spot_delta_pct: float       # How much spot has moved (signed %)
    quote_delta_pct: float      # How much the Kalshi quote has moved (absolute)
    staleness_score: float      # 0.0 = fresh, 1.0 = very stale
    seconds_since_update: float # Time since quote last changed


class StalenessDetector:
    """Detects stale Kalshi quotes by comparing spot movement to quote changes.

    A quote is "stale" when the underlying Binance spot price has moved
    significantly but the Kalshi bid/ask hasn't updated to reflect it.

    Parameters
    ----------
    spot_move_threshold:
        Minimum absolute spot price change (as fraction) to consider significant.
        Default 0.003 = 0.3%.
    quote_change_threshold:
        Maximum absolute change in Kalshi yes_ask to still consider the quote stale.
        Default 0.005 = 0.5 cents.
    lookback_seconds:
        How far back to compare spot prices. Default 120 = 2 minutes.
    max_age_seconds:
        Discard quote snapshots older than this. Default 300 = 5 minutes.
    edge_bonus:
        Extra edge credit for stale quotes. Default 0.02 = 2%.
    """

    def __init__(
        self,
        spot_move_threshold: float = 0.003,
        quote_change_threshold: float = 0.005,
        lookback_seconds: int = 120,
        max_age_seconds: int = 300,
        edge_bonus: float = 0.02,
    ) -> None:
        self._spot_move_threshold = spot_move_threshold
        self._quote_change_threshold = quote_change_threshold
        self._lookback_seconds = lookback_seconds
        self._max_age_seconds = max_age_seconds
        self._edge_bonus = edge_bonus
        # {ticker: (yes_ask, no_ask, timestamp)}
        self._snapshots: Dict[str, tuple] = {}

    @property
    def edge_bonus(self) -> float:
        return self._edge_bonus

    def record_quote_snapshot(
        self,
        ticker: str,
        yes_ask: float,
        no_ask: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record the current Kalshi quote for future staleness comparison.

        Should be called every cycle for every quoted market.
        """
        ts = timestamp if timestamp is not None else time.time()
        prev = self._snapshots.get(ticker)

        if prev is not None:
            prev_yes, prev_no, prev_ts = prev
            # Only update timestamp if the quote actually changed
            if abs(yes_ask - prev_yes) > 1e-6 or abs(no_ask - prev_no) > 1e-6:
                self._snapshots[ticker] = (yes_ask, no_ask, ts)
            # If quote didn't change, keep old timestamp (tracks staleness)
        else:
            self._snapshots[ticker] = (yes_ask, no_ask, ts)

    def compute_staleness(
        self,
        ticker: str,
        current_spot: float,
        spot_at_lookback: Optional[float],
        current_yes_ask: float,
        current_no_ask: float,
        now: Optional[float] = None,
    ) -> StalenessResult:
        """Compute staleness score for a ticker.

        Parameters
        ----------
        ticker: Market ticker
        current_spot: Current Binance spot price
        spot_at_lookback: Spot price ``lookback_seconds`` ago (None if unavailable)
        current_yes_ask: Current Kalshi YES ask price
        current_no_ask: Current Kalshi NO ask price
        now: Current timestamp (defaults to time.time())

        Returns
        -------
        StalenessResult with staleness assessment
        """
        ts = now if now is not None else time.time()

        # Compute spot movement
        if spot_at_lookback is None or spot_at_lookback <= 0 or current_spot <= 0:
            return StalenessResult(
                is_stale=False,
                spot_delta_pct=0.0,
                quote_delta_pct=0.0,
                staleness_score=0.0,
                seconds_since_update=0.0,
            )

        spot_delta_pct = (current_spot - spot_at_lookback) / spot_at_lookback

        # Compute quote change from stored snapshot
        prev = self._snapshots.get(ticker)
        if prev is None:
            return StalenessResult(
                is_stale=False,
                spot_delta_pct=spot_delta_pct,
                quote_delta_pct=0.0,
                staleness_score=0.0,
                seconds_since_update=0.0,
            )

        prev_yes, prev_no, prev_ts = prev

        # How much has the quote changed?
        quote_delta = abs(current_yes_ask - prev_yes)
        seconds_since = ts - prev_ts

        # Discard if snapshot is too old
        if seconds_since > self._max_age_seconds:
            return StalenessResult(
                is_stale=False,
                spot_delta_pct=spot_delta_pct,
                quote_delta_pct=quote_delta,
                staleness_score=0.0,
                seconds_since_update=seconds_since,
            )

        # Is the quote stale?
        spot_moved_enough = abs(spot_delta_pct) >= self._spot_move_threshold
        quote_unchanged = quote_delta <= self._quote_change_threshold
        is_stale = spot_moved_enough and quote_unchanged and seconds_since > 5.0

        # Compute staleness score (0-1)
        # Score increases with spot movement and quote age
        if is_stale:
            move_factor = min(abs(spot_delta_pct) / self._spot_move_threshold, 3.0) / 3.0
            age_factor = min(seconds_since / self._lookback_seconds, 1.0)
            staleness_score = 0.5 * move_factor + 0.5 * age_factor
        else:
            staleness_score = 0.0

        return StalenessResult(
            is_stale=is_stale,
            spot_delta_pct=spot_delta_pct,
            quote_delta_pct=quote_delta,
            staleness_score=staleness_score,
            seconds_since_update=seconds_since,
        )

    def prune_old_snapshots(self, now: Optional[float] = None) -> int:
        """Remove snapshots older than max_age_seconds. Returns count removed."""
        ts = now if now is not None else time.time()
        cutoff = ts - self._max_age_seconds * 2  # Keep 2x max_age for safety
        to_remove = [
            k for k, (_, _, snap_ts) in self._snapshots.items()
            if snap_ts < cutoff
        ]
        for k in to_remove:
            del self._snapshots[k]
        return len(to_remove)
