"""Confidence-scaled max-loss cap (Phase 4C).

Limits the maximum allowable loss per trade as a function of
model confidence and bankroll. Higher confidence allows larger
loss budgets; lower confidence restricts to smaller caps.

Usage::

    cap = LossCapManager(config)
    result = cap.compute(
        bankroll=10_000, confidence=0.8,
        cost_per_contract=0.55, contracts=20,
    )
    if result.capped:
        contracts = result.allowed_contracts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossCapConfig:
    """Configuration for confidence-scaled loss caps.

    Parameters
    ----------
    base_loss_fraction:
        Maximum fraction of bankroll at risk for a perfect-confidence
        trade. Default 0.02 (2%).
    min_loss_fraction:
        Minimum fraction of bankroll allowed at risk even at very low
        confidence. Default 0.002 (0.2%).
    confidence_exponent:
        How steeply confidence scales the loss budget. 1.0 = linear,
        >1.0 = sub-linear (conservative), <1.0 = concave (aggressive
        at low confidence). Default 1.0.
    absolute_max_loss:
        Hard dollar cap regardless of bankroll or confidence.
        Default 500.0.
    absolute_min_loss:
        Minimum dollar loss threshold — trades below this are not
        worth the overhead. Default 1.0.
    """

    base_loss_fraction: float = 0.02
    min_loss_fraction: float = 0.002
    confidence_exponent: float = 1.0
    absolute_max_loss: float = 500.0
    absolute_min_loss: float = 1.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossCapResult:
    """Result of loss cap computation."""

    max_loss_dollars: float
    loss_fraction: float
    confidence: float
    bankroll: float
    requested_contracts: int
    allowed_contracts: int
    max_loss_per_contract: float
    capped: bool
    cap_reason: str


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class LossCapManager:
    """Computes confidence-scaled loss caps for trade sizing.

    Scales maximum allowable loss between min_loss_fraction and
    base_loss_fraction based on confidence^exponent.
    """

    def __init__(self, config: LossCapConfig | None = None) -> None:
        self._config = config or LossCapConfig()
        self._history: List[LossCapResult] = []

    @property
    def config(self) -> LossCapConfig:
        return self._config

    def compute(
        self,
        bankroll: float,
        confidence: float,
        cost_per_contract: float,
        contracts: int,
        failure_loss_fraction: float = 1.0,
    ) -> LossCapResult:
        """Compute the loss cap for a trade.

        Parameters
        ----------
        bankroll:
            Current bankroll in dollars.
        confidence:
            Model confidence in [0, 1]. Higher = more allowed risk.
        cost_per_contract:
            Dollar cost per contract.
        contracts:
            Requested number of contracts.
        failure_loss_fraction:
            Fraction of cost lost on total failure. Default 1.0 (full loss).
        """
        cfg = self._config
        confidence = max(0.0, min(1.0, confidence))
        failure_loss_fraction = max(0.0, min(1.0, failure_loss_fraction))

        # Scale loss fraction by confidence.
        conf_scale = confidence ** cfg.confidence_exponent
        loss_fraction = (
            cfg.min_loss_fraction
            + (cfg.base_loss_fraction - cfg.min_loss_fraction) * conf_scale
        )

        # Dollar budget from bankroll.
        max_loss_from_bankroll = bankroll * loss_fraction

        # Apply absolute caps.
        max_loss = min(max_loss_from_bankroll, cfg.absolute_max_loss)
        max_loss = max(max_loss, 0.0)

        # Max loss per contract.
        max_loss_per_contract = cost_per_contract * failure_loss_fraction
        if max_loss_per_contract <= 0:
            result = LossCapResult(
                max_loss_dollars=max_loss,
                loss_fraction=loss_fraction,
                confidence=confidence,
                bankroll=bankroll,
                requested_contracts=contracts,
                allowed_contracts=0,
                max_loss_per_contract=0.0,
                capped=True,
                cap_reason="zero_loss_per_contract",
            )
            self._history.append(result)
            return result

        # How many contracts fit in the budget?
        max_allowed = int(max_loss / max_loss_per_contract)

        # Check absolute minimum loss threshold.
        if max_loss < cfg.absolute_min_loss:
            result = LossCapResult(
                max_loss_dollars=max_loss,
                loss_fraction=loss_fraction,
                confidence=confidence,
                bankroll=bankroll,
                requested_contracts=contracts,
                allowed_contracts=0,
                max_loss_per_contract=max_loss_per_contract,
                capped=True,
                cap_reason="below_absolute_min_loss",
            )
            self._history.append(result)
            return result

        allowed = min(contracts, max_allowed)
        capped = allowed < contracts
        cap_reason = "loss_budget_exceeded" if capped else ""

        result = LossCapResult(
            max_loss_dollars=max_loss,
            loss_fraction=loss_fraction,
            confidence=confidence,
            bankroll=bankroll,
            requested_contracts=contracts,
            allowed_contracts=allowed,
            max_loss_per_contract=max_loss_per_contract,
            capped=capped,
            cap_reason=cap_reason,
        )
        self._history.append(result)
        return result

    def recent_cap_rate(self, n: int = 50) -> float:
        """Fraction of recent trades that were capped."""
        recent = self._history[-n:] if self._history else []
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.capped) / len(recent)

    def clear(self) -> None:
        """Clear history."""
        self._history.clear()
