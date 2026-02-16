"""Composite confidence scoring: multi-signal agreement gating.

Combines staleness, VPIN, OFI, funding rate, vol regime, cross-asset, and
model edge signals into a single confidence score. Only trades when >=3
independent signals agree on direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

LOGGER = logging.getLogger(__name__)


@dataclass
class ConfidenceComponents:
    """Per-signal scores aligned with trade direction.

    Each score ranges from -1 to +1:
    - Positive means signal supports the trade direction
    - Negative means signal opposes the trade direction
    - Zero means no signal available or neutral
    """

    staleness_signal: float = 0.0    # Stale quote in trade direction -> positive
    vpin_signal: float = 0.0          # Signed VPIN aligned with trade -> positive
    ofi_signal: float = 0.0           # OFI flow in trade direction -> positive
    funding_signal: float = 0.0       # Funding rate drift aligned -> positive
    vol_regime_signal: float = 0.0    # Vol regime favorable -> positive
    cross_asset_signal: float = 0.0   # Cross-asset momentum aligned -> positive
    model_edge_signal: float = 0.0    # Model edge strength -> positive


@dataclass
class ConfidenceResult:
    """Result of composite confidence scoring."""

    score: float           # 0 to 1, overall confidence
    signal_agreement: int  # Number of signals that agree (same sign as trade)
    total_signals: int     # Number of signals with non-zero values
    reasons: List[str]     # Human-readable reasons
    components: ConfidenceComponents  # Raw component scores


class ConfidenceScorer:
    """Scores trade confidence from multiple independent signals.

    Parameters
    ----------
    min_score: Minimum confidence score to allow a trade (default 0.65)
    min_agreement: Minimum number of agreeing signals (default 3)
    staleness_weight: Weight for staleness signal (default 0.25)
    vpin_weight: Weight for VPIN signal (default 0.20)
    ofi_weight: Weight for OFI signal (default 0.15)
    funding_weight: Weight for funding rate signal (default 0.10)
    vol_regime_weight: Weight for vol regime signal (default 0.10)
    cross_asset_weight: Weight for cross-asset signal (default 0.10)
    model_edge_weight: Weight for model edge signal (default 0.10)
    """

    def __init__(
        self,
        min_score: float = 0.65,
        min_agreement: int = 3,
        staleness_weight: float = 0.25,
        vpin_weight: float = 0.20,
        ofi_weight: float = 0.15,
        funding_weight: float = 0.10,
        vol_regime_weight: float = 0.10,
        cross_asset_weight: float = 0.10,
        model_edge_weight: float = 0.10,
    ) -> None:
        self._min_score = min_score
        self._min_agreement = min_agreement
        self._weights = {
            "staleness": staleness_weight,
            "vpin": vpin_weight,
            "ofi": ofi_weight,
            "funding": funding_weight,
            "vol_regime": vol_regime_weight,
            "cross_asset": cross_asset_weight,
            "model_edge": model_edge_weight,
        }

    @property
    def min_score(self) -> float:
        return self._min_score

    @property
    def min_agreement(self) -> int:
        return self._min_agreement

    def score(self, components: ConfidenceComponents) -> ConfidenceResult:
        """Score trade confidence from signal components.

        Algorithm:
        1. Count how many signals are non-zero (total_signals)
        2. Count how many are positive (signal_agreement) - since components
           are pre-aligned with trade direction, positive = supportive
        3. Weighted sum of positive signals -> raw score (0 to 1)
        4. If fewer than min_agreement signals agree, cap score at 0.5

        Returns ConfidenceResult with score, agreement count, and reasons.
        """
        signal_values = {
            "staleness": components.staleness_signal,
            "vpin": components.vpin_signal,
            "ofi": components.ofi_signal,
            "funding": components.funding_signal,
            "vol_regime": components.vol_regime_signal,
            "cross_asset": components.cross_asset_signal,
            "model_edge": components.model_edge_signal,
        }

        # Count non-zero and agreeing (positive) signals
        total_signals = sum(1 for v in signal_values.values() if v != 0.0)
        signal_agreement = sum(1 for v in signal_values.values() if v > 0.0)

        # Weighted sum of clamped signal contributions
        # Each signal is clamped to [-1, 1], then we take the positive part
        # and weight it. A negative signal contributes 0 (it just doesn't help).
        total_weight = sum(self._weights.values())
        if total_weight <= 0:
            total_weight = 1.0

        weighted_sum = 0.0
        negative_penalty = 0.0
        reasons: List[str] = []

        for name, value in signal_values.items():
            clamped = max(-1.0, min(1.0, value))
            weight = self._weights.get(name, 0.0)

            if clamped > 0:
                weighted_sum += clamped * weight
                reasons.append(f"{name}=+{clamped:.2f}")
            elif clamped < 0:
                # Negative signals reduce confidence
                negative_penalty += abs(clamped) * weight * 0.5  # Half penalty
                reasons.append(f"{name}={clamped:.2f}")

        # Normalize to [0, 1]
        raw_score = (weighted_sum - negative_penalty) / total_weight
        raw_score = max(0.0, min(1.0, raw_score))

        # Cap at 0.5 if insufficient agreement
        if signal_agreement < self._min_agreement:
            raw_score = min(raw_score, 0.5)
            reasons.append(
                f"capped: only {signal_agreement}/{self._min_agreement} agree"
            )

        return ConfidenceResult(
            score=raw_score,
            signal_agreement=signal_agreement,
            total_signals=total_signals,
            reasons=reasons,
            components=components,
        )

    def passes(self, result: ConfidenceResult) -> bool:
        """Check if a confidence result passes the minimum threshold."""
        return result.score >= self._min_score
