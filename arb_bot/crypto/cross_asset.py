"""Cross-asset feature construction.

BTC is the dominant crypto asset whose microstructure dynamics (OFI,
volatility, momentum) predict price movements in ETH, SOL, and other
altcoins.  This module extracts leader-asset features and computes
adjustments that feed into the follower-asset probability models.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from arb_bot.crypto.price_feed import PriceFeed

LOGGER = logging.getLogger(__name__)

_MINUTES_PER_YEAR_CRYPTO = 525_960  # 365.25 * 24 * 60


@dataclass(frozen=True)
class CrossAssetSignal:
    """Cross-asset features and adjustments for a follower symbol."""
    leader_ofi: float            # Leader's current OFI
    leader_return_5m: float      # Leader's 5-min log return
    leader_vol_ratio: float      # Leader short/long vol ratio
    cross_ofi_divergence: float  # |leader OFI - target OFI|
    drift_adjustment: float      # Weighted drift modifier for follower
    vol_adjustment: float        # Vol multiplier for follower (>1 when leader vol is elevated)


class CrossAssetFeatures:
    """Computes cross-asset features using a leader symbol (typically BTC).

    Parameters
    ----------
    leader_symbol:
        Binance symbol of the leader asset (e.g. "btcusdt").
    ofi_weight:
        Weight of leader OFI in drift adjustment.
    return_weight:
        Weight of leader recent return in drift adjustment.
    vol_weight:
        Weight of leader vol ratio in vol adjustment.
    vol_baseline:
        Expected normal vol ratio. Deviations from this scale the adjustment.
    """

    def __init__(
        self,
        leader_symbol: str = "btcusdt",
        ofi_weight: float = 0.3,
        return_weight: float = 0.2,
        vol_weight: float = 0.2,
        vol_baseline: float = 1.0,
        return_scale: float = 20.0,
        max_drift: float = 2.0,
    ) -> None:
        self._leader = leader_symbol
        self._ofi_weight = ofi_weight
        self._return_weight = return_weight
        self._vol_weight = vol_weight
        self._vol_baseline = vol_baseline
        self._return_scale = return_scale
        self._max_drift = max_drift

    @property
    def leader_symbol(self) -> str:
        return self._leader

    def compute_features(
        self,
        price_feed: "PriceFeed",
        target_symbol: str,
    ) -> Optional[CrossAssetSignal]:
        """Compute cross-asset signal for a follower symbol.

        Parameters
        ----------
        price_feed:
            The PriceFeed instance with live data for all symbols.
        target_symbol:
            The follower symbol being priced (e.g. "solusdt").

        Returns
        -------
        CrossAssetSignal or None if leader data is unavailable.
        """
        if target_symbol == self._leader:
            return None  # Don't cross-asset BTC against itself

        # Leader OFI
        leader_ofi = price_feed.get_ofi(self._leader, window_seconds=300)

        # Leader recent return (5-min)
        leader_returns = price_feed.get_returns(
            self._leader, interval_seconds=60, window_minutes=5,
        )
        if leader_returns:
            leader_return_5m = sum(leader_returns)  # cumulative 5-min log return
        else:
            leader_return_5m = 0.0

        # Leader volatility ratio (short vs long)
        short_rate = price_feed.get_volume_flow_rate(self._leader, window_seconds=300)
        long_rate = price_feed.get_volume_flow_rate(self._leader, window_seconds=3600)
        if long_rate > 0:
            leader_vol_ratio = short_rate / long_rate
        else:
            leader_vol_ratio = 1.0

        # Target OFI for divergence
        target_ofi = price_feed.get_ofi(target_symbol, window_seconds=300)
        cross_ofi_divergence = abs(leader_ofi - target_ofi)

        # Drift adjustment: leader momentum propagates to follower
        # Positive leader OFI + positive leader return -> positive drift for follower
        drift_from_ofi = leader_ofi * self._ofi_weight
        drift_from_return = leader_return_5m * self._return_weight * self._return_scale
        # Scale return to annualized drift (return is over 5 min)
        drift_adjustment = drift_from_ofi + drift_from_return
        drift_adjustment = max(-self._max_drift, min(self._max_drift, drift_adjustment))

        # Vol adjustment: when leader vol is elevated, follower vol tends to be too
        vol_deviation = leader_vol_ratio / max(self._vol_baseline, 0.01) - 1.0
        # Clamp the multiplier to [0.8, 1.5] to prevent extreme adjustments
        vol_multiplier = 1.0 + vol_deviation * self._vol_weight
        vol_adjustment = max(0.8, min(1.5, vol_multiplier))

        return CrossAssetSignal(
            leader_ofi=leader_ofi,
            leader_return_5m=leader_return_5m,
            leader_vol_ratio=leader_vol_ratio,
            cross_ofi_divergence=cross_ofi_divergence,
            drift_adjustment=drift_adjustment,
            vol_adjustment=vol_adjustment,
        )
