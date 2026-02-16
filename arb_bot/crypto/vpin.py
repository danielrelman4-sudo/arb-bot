"""VPIN -- Volume-Synchronized Probability of Informed Trading.

Computes order-flow toxicity using equal-volume buckets rather than
equal-time windows.  High VPIN (> 0.7) predicts imminent volatility.
Signed VPIN indicates directional pressure.

Reference: Easley, Lopez de Prado, O'Hara (2012).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VPINBucket:
    """One completed volume bucket."""
    buy_volume: float
    sell_volume: float
    timestamp: float  # Timestamp when bucket completed


class VPINCalculator:
    """Computes VPIN from a stream of classified trades.

    Parameters
    ----------
    bucket_volume:
        Volume per bucket.  Set to 0 for auto-calibration.
    num_buckets:
        Number of completed buckets used for the VPIN moving average.
    """

    def __init__(
        self,
        bucket_volume: float = 0.0,
        num_buckets: int = 50,
        staleness_seconds: float = 120.0,
    ) -> None:
        self._bucket_volume = bucket_volume
        self._num_buckets = num_buckets
        self._staleness_seconds = staleness_seconds

        # Current (in-progress) bucket accumulators
        self._cur_buy: float = 0.0
        self._cur_sell: float = 0.0
        self._cur_total: float = 0.0

        # Completed buckets ring buffer
        self._buckets: Deque[VPINBucket] = deque(maxlen=num_buckets)

        # Track when we last received a trade for staleness detection
        self._last_trade_time: float = 0.0

    # -- public properties ------------------------------------------------

    @property
    def bucket_volume(self) -> float:
        return self._bucket_volume

    @bucket_volume.setter
    def bucket_volume(self, value: float) -> None:
        self._bucket_volume = value

    @property
    def num_completed_buckets(self) -> int:
        return len(self._buckets)

    # -- trade ingestion --------------------------------------------------

    def process_trade(
        self,
        price: float,
        volume: float,
        is_buy: bool,
        timestamp: float,
    ) -> None:
        """Ingest a single classified trade.

        If the trade causes the current bucket to overflow, the bucket
        is closed and excess volume rolls into the next bucket.

        Parameters
        ----------
        price:
            Trade price (not used directly, kept for future features).
        volume:
            Trade size in base-asset units.
        is_buy:
            True if buyer-initiated (aggressor).
        timestamp:
            Unix timestamp of the trade.
        """
        # Track wall-clock time of last trade ingestion for staleness detection
        self._last_trade_time = time.monotonic()

        if self._bucket_volume <= 0:
            # Auto-calibration hasn't happened yet; accumulate but don't bucket
            return

        remaining = volume
        while remaining > 0:
            space = self._bucket_volume - self._cur_total
            fill = min(remaining, space)

            if is_buy:
                self._cur_buy += fill
            else:
                self._cur_sell += fill
            self._cur_total += fill
            remaining -= fill

            # Check if bucket is full
            if self._cur_total >= self._bucket_volume - 1e-12:
                self._buckets.append(VPINBucket(
                    buy_volume=self._cur_buy,
                    sell_volume=self._cur_sell,
                    timestamp=timestamp,
                ))
                self._cur_buy = 0.0
                self._cur_sell = 0.0
                self._cur_total = 0.0

    # -- staleness detection -----------------------------------------------

    @property
    def is_stale(self) -> bool:
        """True if no trades have been received within the staleness window.

        When stale, VPIN values are unreliable because the underlying
        data is no longer being refreshed.
        """
        if self._last_trade_time <= 0:
            return True  # Never received a trade
        return (time.monotonic() - self._last_trade_time) > self._staleness_seconds

    @property
    def seconds_since_last_trade(self) -> float:
        """Seconds elapsed since the last trade was processed.

        Returns ``float('inf')`` if no trade has ever been processed.
        """
        if self._last_trade_time <= 0:
            return float("inf")
        return time.monotonic() - self._last_trade_time

    # -- VPIN computation -------------------------------------------------

    def get_vpin(self) -> Optional[float]:
        """Unsigned VPIN over the last ``num_buckets`` completed buckets.

        VPIN = mean(|buy_vol - sell_vol| / bucket_volume) across buckets.
        Returns a value in [0, 1].  Returns ``None`` if fewer than 5
        buckets have been completed **or if data is stale** (no trades
        received within ``staleness_seconds``).
        """
        if len(self._buckets) < 5:
            return None
        if self.is_stale:
            return None
        total = 0.0
        for b in self._buckets:
            bv = b.buy_volume + b.sell_volume
            if bv > 0:
                total += abs(b.buy_volume - b.sell_volume) / bv
        return total / len(self._buckets)

    def get_signed_vpin(self) -> Optional[float]:
        """Signed VPIN: positive = net buying pressure, negative = net selling.

        signed_vpin = mean((buy_vol - sell_vol) / bucket_volume).
        Returns a value in [-1, 1].  ``None`` if < 5 buckets or stale data.
        """
        if len(self._buckets) < 5:
            return None
        if self.is_stale:
            return None
        total = 0.0
        for b in self._buckets:
            bv = b.buy_volume + b.sell_volume
            if bv > 0:
                total += (b.buy_volume - b.sell_volume) / bv
        return total / len(self._buckets)

    def get_vpin_trend(self, lookback_buckets: int = 10) -> Optional[float]:
        """Slope of unsigned VPIN over recent buckets.

        Positive = VPIN is increasing (volatility building).
        Negative = VPIN is decreasing (volatility subsiding).
        Returns ``None`` if insufficient data.
        """
        n = min(lookback_buckets, len(self._buckets))
        if n < 3:
            return None

        recent = list(self._buckets)[-n:]
        # Simple linear regression slope of per-bucket |OI|
        vals = []
        for b in recent:
            bv = b.buy_volume + b.sell_volume
            if bv > 0:
                vals.append(abs(b.buy_volume - b.sell_volume) / bv)
            else:
                vals.append(0.0)

        n_vals = len(vals)
        x_mean = (n_vals - 1) / 2.0
        y_mean = sum(vals) / n_vals
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n_vals))
        if den == 0:
            return 0.0
        return num / den

    def auto_calibrate_bucket_size(
        self,
        total_volume: float,
        window_minutes: float = 60.0,
        target_buckets_per_hour: int = 50,
    ) -> float:
        """Set bucket volume from recent trading activity.

        Parameters
        ----------
        total_volume:
            Total volume observed over *window_minutes*.
        window_minutes:
            The window used to measure *total_volume*.
        target_buckets_per_hour:
            Desired number of buckets per hour.

        Returns
        -------
        The computed bucket volume (also stored internally).
        """
        if total_volume <= 0 or window_minutes <= 0:
            self._bucket_volume = 1.0  # fallback
            return self._bucket_volume

        vol_per_min = total_volume / window_minutes
        vol_per_hour = vol_per_min * 60.0
        bucket_vol = vol_per_hour / max(target_buckets_per_hour, 1)
        self._bucket_volume = max(bucket_vol, 0.001)  # floor to avoid tiny buckets
        LOGGER.info(
            "VPIN: auto-calibrated bucket_volume=%.4f from %.1f vol/hr (target %d buckets/hr)",
            self._bucket_volume, vol_per_hour, target_buckets_per_hour,
        )
        return self._bucket_volume
