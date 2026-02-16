"""VPIN -- Volume-Synchronized Probability of Informed Trading.

Computes order-flow toxicity using equal-volume buckets rather than
equal-time windows.  High VPIN (> 0.7) predicts imminent volatility.
Signed VPIN indicates directional pressure.

Includes adaptive thresholding: instead of hardcoded 0.85, the halt/momentum
thresholds can be derived from a rolling percentile of recent VPIN readings
so that gates adapt to current market conditions.

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
        adaptive_history_size: int = 500,
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

        # Rolling VPIN history for adaptive thresholding
        self._vpin_history: Deque[float] = deque(maxlen=adaptive_history_size)

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
                # Record VPIN snapshot for adaptive thresholding
                vpin_snap = self._compute_vpin_raw()
                if vpin_snap is not None:
                    self._vpin_history.append(vpin_snap)

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

    # -- internal VPIN (no staleness check) ----------------------------------

    def _compute_vpin_raw(self) -> Optional[float]:
        """Unsigned VPIN without staleness check (for internal recording)."""
        if len(self._buckets) < 5:
            return None
        total = 0.0
        for b in self._buckets:
            bv = b.buy_volume + b.sell_volume
            if bv > 0:
                total += abs(b.buy_volume - b.sell_volume) / bv
        return total / len(self._buckets)

    # -- adaptive thresholding ------------------------------------------------

    def get_adaptive_threshold(
        self,
        percentile: float = 75.0,
        floor: float = 0.50,
        ceiling: float = 0.95,
        min_history: int = 30,
    ) -> Optional[float]:
        """Compute an adaptive VPIN threshold from recent VPIN history.

        Instead of a hardcoded threshold like 0.85, this returns a value
        based on the rolling distribution of VPIN readings.  For example,
        the 75th percentile means "current VPIN is above 75% of recent
        readings" — i.e., unusually elevated for THIS market's conditions.

        Parameters
        ----------
        percentile:
            Which percentile of the VPIN distribution to use (0-100).
            75 = moderate sensitivity, 90 = only extreme readings trigger.
        floor:
            Minimum threshold regardless of history (safety floor).
        ceiling:
            Maximum threshold — never go above this.
        min_history:
            Minimum VPIN readings required before returning a value.
            Returns ``None`` if fewer readings are available.

        Returns
        -------
        The adaptive threshold, or ``None`` if insufficient history.
        """
        if len(self._vpin_history) < min_history:
            return None

        sorted_vals = sorted(self._vpin_history)
        n = len(sorted_vals)
        # Percentile index (linear interpolation)
        idx = (percentile / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        pctl_val = sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

        # Clamp to [floor, ceiling]
        return max(floor, min(ceiling, pctl_val))

    def get_adaptive_momentum_thresholds(
        self,
        halt_percentile: float = 90.0,
        momentum_percentile: float = 75.0,
        halt_floor: float = 0.70,
        halt_ceiling: float = 0.98,
        momentum_floor: float = 0.50,
        momentum_ceiling: float = 0.95,
        min_history: int = 30,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get adaptive (momentum_floor, halt_ceiling) thresholds.

        Returns a tuple of (momentum_threshold, halt_threshold) derived
        from the rolling VPIN distribution.  The momentum threshold defines
        where the momentum zone starts; the halt threshold defines where
        all trading is halted.

        Returns (None, None) if insufficient history.
        """
        if len(self._vpin_history) < min_history:
            return (None, None)

        momentum = self.get_adaptive_threshold(
            percentile=momentum_percentile,
            floor=momentum_floor,
            ceiling=momentum_ceiling,
            min_history=min_history,
        )
        halt = self.get_adaptive_threshold(
            percentile=halt_percentile,
            floor=halt_floor,
            ceiling=halt_ceiling,
            min_history=min_history,
        )

        # Ensure halt > momentum (with at least 0.05 gap)
        if momentum is not None and halt is not None:
            if halt <= momentum + 0.05:
                halt = min(momentum + 0.10, halt_ceiling)

        return (momentum, halt)

    @property
    def vpin_history_size(self) -> int:
        """Number of VPIN readings in the adaptive history buffer."""
        return len(self._vpin_history)

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
