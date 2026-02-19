"""Tail-risk-aware Kelly sizing (Phase 4A).

Extends the basic Kelly fraction with uncertainty haircuts that
reduce position size when model confidence is low or variance
is high. Prevents over-betting in uncertain regimes.

Usage::

    sizer = TailRiskKelly(config)
    fraction = sizer.compute(
        edge=0.03, cost=0.55, fill_prob=0.8,
        model_uncertainty=0.3, lane_variance=0.1,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailRiskKellyConfig:
    """Configuration for tail-risk-aware Kelly sizing.

    Parameters
    ----------
    base_kelly_fraction:
        Maximum Kelly fraction to use (fractional Kelly). Default 0.25.
    uncertainty_haircut_factor:
        Legacy linear haircut factor.  Only used when
        ``use_baker_mchale_shrinkage`` is False.  Default 1.0.
    variance_haircut_factor:
        How much to reduce sizing per unit of lane variance.
        Applied as: fraction *= (1 - variance * factor). Default 0.5.
    min_confidence:
        Below this confidence level, sizing is zero. Default 0.1.
    max_model_uncertainty:
        Above this uncertainty, sizing is zero. Default 0.8.
    lane_variance_window:
        Number of recent outcomes to consider for variance. Default 50.
    use_baker_mchale_shrinkage:
        When True, replace the linear uncertainty haircut with the
        Baker-McHale (2013) closed-form shrinkage:
            k* = 1 / (1 + b² × σ²)
        where b = edge/cost (the odds) and σ² is the variance of
        the estimated edge (derived from model_uncertainty).
        Reference: Baker & McHale, "Optimal Betting Under Parameter
        Uncertainty", Decision Analysis 10(3):189-199, 2013.
    """

    base_kelly_fraction: float = 0.25
    uncertainty_haircut_factor: float = 1.0
    variance_haircut_factor: float = 0.5
    min_confidence: float = 0.1
    max_model_uncertainty: float = 0.8
    lane_variance_window: int = 50
    use_baker_mchale_shrinkage: bool = True


# ---------------------------------------------------------------------------
# Sizing result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KellySizingResult:
    """Result from tail-risk-aware Kelly sizing."""

    raw_kelly: float
    adjusted_fraction: float
    uncertainty_haircut: float
    variance_haircut: float
    confidence: float
    blocked: bool
    block_reason: str


# ---------------------------------------------------------------------------
# Tail-risk Kelly sizer
# ---------------------------------------------------------------------------


class TailRiskKelly:
    """Kelly sizing with model uncertainty and variance haircuts."""

    def __init__(self, config: TailRiskKellyConfig | None = None) -> None:
        self._config = config or TailRiskKellyConfig()
        self._lane_outcomes: Dict[str, list[float]] = {}

    @property
    def config(self) -> TailRiskKellyConfig:
        return self._config

    def compute(
        self,
        edge: float,
        cost: float,
        fill_prob: float,
        model_uncertainty: float = 0.0,
        lane: str = "",
        failure_loss: float | None = None,
    ) -> KellySizingResult:
        """Compute tail-risk-adjusted Kelly fraction.

        Parameters
        ----------
        edge:
            Expected edge per contract.
        cost:
            Cost per contract.
        fill_prob:
            Probability of full fill.
        model_uncertainty:
            Model uncertainty score (0=certain, 1=no idea). Default 0.
        lane:
            Lane identifier for variance lookup. Default "".
        failure_loss:
            Loss per contract on failure. Default = cost.
        """
        cfg = self._config

        # Block on high uncertainty.
        if model_uncertainty > cfg.max_model_uncertainty:
            return KellySizingResult(
                raw_kelly=0.0, adjusted_fraction=0.0,
                uncertainty_haircut=1.0, variance_haircut=0.0,
                confidence=1.0 - model_uncertainty, blocked=True,
                block_reason="model_uncertainty_too_high",
            )

        # Confidence = 1 - uncertainty.
        confidence = max(0.0, 1.0 - model_uncertainty)
        if confidence < cfg.min_confidence:
            return KellySizingResult(
                raw_kelly=0.0, adjusted_fraction=0.0,
                uncertainty_haircut=1.0, variance_haircut=0.0,
                confidence=confidence, blocked=True,
                block_reason="confidence_below_minimum",
            )

        # Raw Kelly fraction.
        raw = _raw_kelly(edge, cost, fill_prob, failure_loss)
        if raw <= 0.0:
            return KellySizingResult(
                raw_kelly=0.0, adjusted_fraction=0.0,
                uncertainty_haircut=0.0, variance_haircut=0.0,
                confidence=confidence, blocked=True,
                block_reason="negative_edge",
            )

        # Apply fractional Kelly cap.
        fraction = min(raw, cfg.base_kelly_fraction)

        # Uncertainty haircut — Baker-McHale shrinkage or legacy linear.
        if cfg.use_baker_mchale_shrinkage and cost > 0:
            # Baker-McHale (2013): k* = 1 / (1 + b² σ²)
            # b = edge/cost (the odds ratio)
            # σ² = variance of estimated edge ≈ (model_uncertainty × edge)²
            #   We interpret model_uncertainty as the relative std dev of
            #   our edge estimate, so σ = uncertainty × edge.
            b = edge / cost
            sigma_sq = (model_uncertainty * edge) ** 2
            shrinkage = 1.0 / (1.0 + b * b * sigma_sq) if (b * b * sigma_sq) > 0 else 1.0
            u_haircut = 1.0 - shrinkage
            fraction *= shrinkage
        else:
            # Legacy linear haircut.
            u_haircut = min(1.0, model_uncertainty * cfg.uncertainty_haircut_factor)
            fraction *= (1.0 - u_haircut)

        # Variance haircut.
        lane_var = self._lane_variance(lane)
        v_haircut = min(1.0, lane_var * cfg.variance_haircut_factor)
        fraction *= (1.0 - v_haircut)

        fraction = max(0.0, min(1.0, fraction))

        return KellySizingResult(
            raw_kelly=raw,
            adjusted_fraction=fraction,
            uncertainty_haircut=u_haircut,
            variance_haircut=v_haircut,
            confidence=confidence,
            blocked=False,
            block_reason="",
        )

    def record_outcome(self, lane: str, pnl: float) -> None:
        """Record a realized PnL outcome for lane variance tracking."""
        if lane not in self._lane_outcomes:
            self._lane_outcomes[lane] = []
        outcomes = self._lane_outcomes[lane]
        outcomes.append(pnl)
        if len(outcomes) > self._config.lane_variance_window:
            self._lane_outcomes[lane] = outcomes[-self._config.lane_variance_window:]

    def _lane_variance(self, lane: str) -> float:
        """Compute normalized variance for a lane."""
        outcomes = self._lane_outcomes.get(lane, [])
        if len(outcomes) < 2:
            return 0.0
        mean = sum(outcomes) / len(outcomes)
        variance = sum((x - mean) ** 2 for x in outcomes) / (len(outcomes) - 1)
        return math.sqrt(variance)

    def clear(self) -> None:
        """Clear all lane outcomes."""
        self._lane_outcomes.clear()


# ---------------------------------------------------------------------------
# Raw Kelly helper
# ---------------------------------------------------------------------------


def _raw_kelly(
    edge: float,
    cost: float,
    fill_prob: float,
    failure_loss: float | None = None,
) -> float:
    """Compute raw Kelly fraction (same formula as PositionSizer)."""
    if edge <= 0.0 or cost <= 0.0:
        return 0.0
    p = max(0.0, min(1.0, fill_prob))
    q = 1.0 - p
    b = edge / cost
    if b <= 0.0:
        return 0.0
    if failure_loss is None:
        failure_loss = cost
    failure_loss = max(0.0, min(cost, failure_loss))
    a = failure_loss / cost
    if a <= 0.0:
        raw = 1.0
    else:
        raw = max(0.0, (b * p - a * q) / (a * b))
    adjusted = raw * math.sqrt(p)
    return max(0.0, min(1.0, adjusted))


# ---------------------------------------------------------------------------
# Rust dispatch for TailRiskKelly.compute().
# When arb_engine_rs is installed and ARB_USE_RUST_KELLY=1 (or
# ARB_USE_RUST_ALL=1), replaces compute() with Rust implementation.
# The lane_variance is still computed in Python (stateful tracking).
# Set env var to "0" for instant rollback.
# ---------------------------------------------------------------------------


def _try_rust_dispatch() -> bool:
    """Attempt to replace TailRiskKelly.compute with Rust implementation."""
    import json
    import os

    if os.environ.get("ARB_USE_RUST_KELLY", "") != "1" and \
       os.environ.get("ARB_USE_RUST_ALL", "") != "1":
        return False

    try:
        import arb_engine_rs  # type: ignore[import-untyped]
    except ImportError:
        return False

    _py_compute = TailRiskKelly.compute

    def _rs_compute(
        self: TailRiskKelly,
        edge: float,
        cost: float,
        fill_prob: float,
        model_uncertainty: float = 0.0,
        lane: str = "",
        failure_loss: float | None = None,
    ) -> KellySizingResult:
        cfg = self._config
        lane_var = self._lane_variance(lane)
        rs_json = arb_engine_rs.compute_kelly(
            edge, cost, fill_prob,
            model_uncertainty, lane_var, failure_loss,
            cfg.base_kelly_fraction,
            cfg.uncertainty_haircut_factor,
            cfg.variance_haircut_factor,
            cfg.min_confidence,
            cfg.max_model_uncertainty,
            cfg.use_baker_mchale_shrinkage,
        )
        d = json.loads(rs_json)
        return KellySizingResult(
            raw_kelly=d["raw_kelly"],
            adjusted_fraction=d["adjusted_fraction"],
            uncertainty_haircut=d["uncertainty_haircut"],
            variance_haircut=d["variance_haircut"],
            confidence=d["confidence"],
            blocked=d["blocked"],
            block_reason=d["block_reason"],
        )

    TailRiskKelly.compute = _rs_compute  # type: ignore[assignment]

    # Also dispatch _raw_kelly module-level function.
    import sys
    this = sys.modules[__name__]

    def _rs_raw_kelly(
        edge: float, cost: float, fill_prob: float,
        failure_loss: float | None = None,
    ) -> float:
        return arb_engine_rs.execution_aware_kelly_fraction(
            edge, cost, fill_prob, failure_loss,
        )

    setattr(this, "_raw_kelly", _rs_raw_kelly)

    return True


_RUST_ACTIVE = _try_rust_dispatch()
