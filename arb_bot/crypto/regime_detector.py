"""Market regime detection for the crypto prediction engine.

Classifies the current market into one of four regimes based on
real-time microstructure signals.  The regime label is stored in
the feature store so the classifier can learn regime-conditional
trading rules.

Regimes
-------
- **trending_up**   – sustained bullish move, positive OFI across assets
- **trending_down** – sustained bearish move, negative OFI across assets
- **mean_reverting** – choppy, oscillating OFI, low autocorrelation
- **high_vol**      – sudden vol expansion, VPIN spike, jump cluster

The detector is intentionally simple and fast (no ML, no lookback
tables).  It runs once per engine cycle and produces a label +
confidence score that the classifier consumes as a feature.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── Regime labels ────────────────────────────────────────────────

TRENDING_UP = "trending_up"
TRENDING_DOWN = "trending_down"
MEAN_REVERTING = "mean_reverting"
HIGH_VOL = "high_vol"

ALL_REGIMES = [TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING, HIGH_VOL]


# ── Data classes ─────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeSnapshot:
    """Point-in-time regime classification for one underlying."""

    symbol: str
    timestamp: float
    regime: str                 # one of ALL_REGIMES
    confidence: float           # [0, 1] — how certain are we?
    trend_score: float          # [-1, 1] — directional strength
    vol_score: float            # [0, 1] — vol expansion indicator
    mean_reversion_score: float # [0, 1] — choppiness indicator
    ofi_alignment: float        # [0, 1] — cross-timescale OFI agreement
    is_transitioning: bool = False  # True when regime not confirmed by 3/5 majority


@dataclass(frozen=True)
class MarketRegime:
    """Aggregate regime across all underlyings."""

    timestamp: float
    regime: str                 # dominant regime
    confidence: float           # aggregate confidence
    per_symbol: Dict[str, RegimeSnapshot] = field(default_factory=dict)

    @property
    def is_trending(self) -> bool:
        return self.regime in (TRENDING_UP, TRENDING_DOWN)

    @property
    def is_transitioning(self) -> bool:
        """True if ANY per-symbol regime is transitioning."""
        return any(snap.is_transitioning for snap in self.per_symbol.values())

    @property
    def trend_direction(self) -> int:
        """1 = up, -1 = down, 0 = not trending."""
        if self.regime == TRENDING_UP:
            return 1
        elif self.regime == TRENDING_DOWN:
            return -1
        return 0


# ── Detector ─────────────────────────────────────────────────────

class RegimeDetector:
    """Classifies market regime from microstructure signals.

    Parameters
    ----------
    ofi_trend_threshold : float
        Minimum OFI alignment score to classify as trending.
    vol_expansion_threshold : float
        Minimum vol ratio (short/long) to classify as high_vol.
    autocorr_window : int
        Number of recent returns to use for autocorrelation.
    min_returns : int
        Minimum returns needed to compute regime (else → mean_reverting).
    """

    def __init__(
        self,
        ofi_trend_threshold: float = 0.3,
        vol_expansion_threshold: float = 2.0,
        autocorr_window: int = 15,
        min_returns: int = 10,
        vpin_spike_threshold: float = 0.85,
    ) -> None:
        self._ofi_trend_threshold = ofi_trend_threshold
        self._vol_expansion_threshold = vol_expansion_threshold
        self._autocorr_window = autocorr_window
        self._min_returns = min_returns
        self._vpin_spike_threshold = vpin_spike_threshold

        # History for smoothing (prevents flipping every cycle)
        self._regime_history: Dict[str, List[str]] = {}
        self._history_maxlen = 5

    def classify(
        self,
        symbol: str,
        ofi_multiscale: Dict[int, float],
        returns_1m: List[float],
        vol_short: float,
        vol_long: float,
        vpin: Optional[float] = None,
        signed_vpin: Optional[float] = None,
    ) -> RegimeSnapshot:
        """Classify the regime for a single underlying.

        Parameters
        ----------
        symbol : str
            Binance symbol (e.g. "btcusdt").
        ofi_multiscale : dict
            OFI values keyed by window_seconds (e.g. {30: 0.5, 60: 0.3, ...}).
        returns_1m : list[float]
            Recent 1-minute log returns.
        vol_short : float
            Short-window annualized vol (e.g. 15-min).
        vol_long : float
            Long-window annualized vol (e.g. 120-min).
        vpin : float or None
            Unsigned VPIN [0, 1].
        signed_vpin : float or None
            Signed VPIN [-1, 1].
        """
        now = time.time()

        # ── Compute component scores ─────────────────────────────

        # 1. OFI alignment: do all timescales agree on direction?
        ofi_alignment, ofi_direction = self._compute_ofi_alignment(ofi_multiscale)

        # 2. Trend score: OFI direction weighted by alignment
        trend_score = ofi_direction * ofi_alignment
        # Boost with signed VPIN if available
        if signed_vpin is not None:
            trend_score = 0.6 * trend_score + 0.4 * signed_vpin

        # 3. Vol expansion score
        vol_score = self._compute_vol_score(vol_short, vol_long, vpin)

        # 4. Mean reversion score (autocorrelation of returns)
        mean_reversion_score = self._compute_mean_reversion_score(returns_1m)

        # ── Decision logic ───────────────────────────────────────

        # Priority: high_vol > trending > mean_reverting
        # High vol takes priority because it changes everything
        if vol_score > 0.7:
            regime = HIGH_VOL
            confidence = vol_score
        elif (ofi_alignment > self._ofi_trend_threshold
              and abs(trend_score) > 0.25):
            if trend_score > 0:
                regime = TRENDING_UP
            else:
                regime = TRENDING_DOWN
            confidence = min(1.0, ofi_alignment * abs(trend_score) * 2)
        else:
            regime = MEAN_REVERTING
            confidence = max(0.3, mean_reversion_score)

        # ── Smoothing: require persistence to change regime ──────
        regime, is_transitioning = self._smooth_regime(symbol, regime)

        return RegimeSnapshot(
            symbol=symbol,
            timestamp=now,
            regime=regime,
            confidence=confidence,
            trend_score=trend_score,
            vol_score=vol_score,
            mean_reversion_score=mean_reversion_score,
            ofi_alignment=ofi_alignment,
            is_transitioning=is_transitioning,
        )

    def classify_market(
        self,
        snapshots: Dict[str, RegimeSnapshot],
    ) -> MarketRegime:
        """Aggregate per-symbol regimes into a market-wide regime.

        Uses majority vote weighted by confidence.
        """
        now = time.time()

        if not snapshots:
            return MarketRegime(
                timestamp=now,
                regime=MEAN_REVERTING,
                confidence=0.0,
                per_symbol={},
            )

        # Weighted vote
        regime_scores: Dict[str, float] = {r: 0.0 for r in ALL_REGIMES}
        for snap in snapshots.values():
            regime_scores[snap.regime] += snap.confidence

        best_regime = max(regime_scores, key=regime_scores.get)
        total_weight = sum(s.confidence for s in snapshots.values())
        if total_weight > 0:
            best_confidence = regime_scores[best_regime] / total_weight
        else:
            best_confidence = 0.0

        return MarketRegime(
            timestamp=now,
            regime=best_regime,
            confidence=best_confidence,
            per_symbol=dict(snapshots),
        )

    # ── Component score computation ──────────────────────────────

    def _compute_ofi_alignment(
        self, ofi_multiscale: Dict[int, float],
    ) -> tuple:
        """Compute OFI cross-timescale alignment and direction.

        Returns (alignment_score, direction):
        - alignment_score: [0, 1] — 1.0 means all windows agree
        - direction: [-1, 1] — average OFI direction
        """
        if not ofi_multiscale:
            return 0.0, 0.0

        values = list(ofi_multiscale.values())
        if not values:
            return 0.0, 0.0

        # Direction: weighted average (shorter windows get more weight)
        # Weights: 30s=4, 60s=3, 120s=2, 300s=1
        weight_map = {30: 4.0, 60: 3.0, 120: 2.0, 300: 1.0}
        weighted_sum = 0.0
        weight_total = 0.0
        for window, ofi in ofi_multiscale.items():
            w = weight_map.get(window, 1.0)
            weighted_sum += ofi * w
            weight_total += w

        direction = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Alignment: what fraction of windows agree on sign?
        signs = [1 if v > 0.05 else (-1 if v < -0.05 else 0) for v in values]
        non_zero = [s for s in signs if s != 0]
        if not non_zero:
            return 0.0, direction

        # Count how many agree with the majority sign
        majority_sign = 1 if sum(non_zero) > 0 else -1
        agreement = sum(1 for s in non_zero if s == majority_sign)
        alignment = agreement / len(values)  # fraction of ALL windows

        return alignment, direction

    def _compute_vol_score(
        self,
        vol_short: float,
        vol_long: float,
        vpin: Optional[float],
    ) -> float:
        """Compute vol expansion score [0, 1].

        High score = vol regime (sudden expansion or VPIN spike).
        """
        # Vol ratio: short/long > 2.0 means vol is expanding
        if vol_long > 0:
            vol_ratio = vol_short / vol_long
        else:
            vol_ratio = 1.0

        # Normalize to [0, 1]: ratio of 1.0 → 0, ratio of 3.0+ → 1.0
        vol_component = max(0.0, min(1.0, (vol_ratio - 1.0) / 2.0))

        # VPIN spike component
        vpin_component = 0.0
        if vpin is not None and vpin > self._vpin_spike_threshold:
            vpin_component = (vpin - self._vpin_spike_threshold) / (
                1.0 - self._vpin_spike_threshold
            )

        # Combine: vol expansion OR VPIN spike
        return max(vol_component, vpin_component)

    def _compute_mean_reversion_score(
        self, returns: List[float],
    ) -> float:
        """Compute mean reversion score from return autocorrelation.

        Negative autocorrelation → mean reverting (score near 1.0).
        Positive autocorrelation → trending (score near 0.0).
        Zero autocorrelation → neutral (score ~0.5).
        """
        if len(returns) < self._min_returns:
            return 0.5  # insufficient data → neutral

        # Use last N returns
        r = returns[-self._autocorr_window:]
        if len(r) < 3:
            return 0.5

        arr = np.array(r, dtype=np.float64)
        mean = arr.mean()
        var = arr.var()
        if var < 1e-20:
            return 0.5

        # Lag-1 autocorrelation
        shifted = arr[1:] - mean
        original = arr[:-1] - mean
        autocorr = np.sum(shifted * original) / (len(shifted) * var)

        # Clamp to [-1, 1]
        autocorr = max(-1.0, min(1.0, autocorr))

        # Map: autocorr=-1 → score=1.0 (mean reverting)
        #       autocorr=0  → score=0.5 (neutral)
        #       autocorr=+1 → score=0.0 (trending)
        return 0.5 * (1.0 - autocorr)

    # ── Smoothing ────────────────────────────────────────────────

    def _smooth_regime(self, symbol: str, raw_regime: str) -> tuple:
        """Prevent rapid regime flipping by requiring persistence.

        A regime must appear in at least 3 of the last 5 cycles
        to become the active regime.  Otherwise, hold the previous.

        Returns (regime, is_transitioning) where is_transitioning is
        True when the smoothed regime is not yet confirmed by majority
        or the raw input disagrees with the smoothed output.
        """
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []

        history = self._regime_history[symbol]
        history.append(raw_regime)

        # Trim to max length
        if len(history) > self._history_maxlen:
            history[:] = history[-self._history_maxlen:]

        # Count occurrences in recent history
        counts = {}
        for r in history:
            counts[r] = counts.get(r, 0) + 1

        # Find most frequent
        most_frequent = max(counts, key=counts.get)

        # Require at least 3/5 persistence (or if we have < 5 history,
        # require majority)
        threshold = min(3, max(1, len(history) // 2 + 1))
        if counts[most_frequent] >= threshold:
            # Confirmed regime, but transitioning if raw disagrees
            return (most_frequent, raw_regime != most_frequent)

        # Not enough persistence — hold previous regime if we have one
        # (return the second-to-last if different from raw)
        if len(history) >= 2:
            return (history[-2], True)

        return (raw_regime, True)

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear regime history (e.g. on restart)."""
        if symbol is not None:
            self._regime_history.pop(symbol, None)
        else:
            self._regime_history.clear()
