"""Hawkes self-exciting jump intensity process.

Models volatility clustering: large returns spike the jump intensity,
which decays exponentially. Based on Filimonov & Sornette (2012).

    lambda(t) = mu + sum_{t_i < t} alpha * m_i * exp(-beta * (t - t_i))

where m_i is the magnitude of shock i.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple


@dataclass
class HawkesIntensity:
    """Self-exciting jump intensity tracker.

    Parameters
    ----------
    mu:
        Baseline intensity (jumps per day). This is the long-run
        average when no recent shocks have occurred.
    alpha:
        Excitation amplitude. Controls how much each shock raises
        the intensity.
    beta:
        Decay rate per second. ``ln(2)/600`` gives a 10-minute
        half-life.
    max_history_seconds:
        Shocks older than this are pruned from the deque.
    return_threshold_sigma:
        A log return must exceed this many standard deviations
        (z-score) to trigger a shock.
    """

    mu: float = 3.0
    alpha: float = 5.0
    beta: float = 0.00115  # ln(2)/600 ~ 10-min half-life
    max_history_seconds: float = 1800
    return_threshold_sigma: float = 4.0

    _shocks: Deque[Tuple[float, float]] = field(
        default_factory=deque, init=False, repr=False,
    )

    # ── Public API ────────────────────────────────────────────────

    def record_shock(self, t: float, magnitude: float) -> None:
        """Append a shock event.

        Parameters
        ----------
        t:
            Timestamp (e.g. ``time.monotonic()``).
        magnitude:
            Shock magnitude (positive). Typically ``z_score / threshold``.
            Ignored if zero.
        """
        if magnitude == 0.0:
            return
        self._shocks.append((t, magnitude))

    def record_return(
        self, t: float, log_return: float, realized_vol: float,
    ) -> None:
        """Evaluate a log return and trigger a shock if it is extreme.

        Parameters
        ----------
        t:
            Timestamp.
        log_return:
            Observed log return for one interval.
        realized_vol:
            Per-interval realized volatility (standard deviation of
            returns at the same frequency).
        """
        if realized_vol <= 0.0:
            return
        z_score = abs(log_return) / realized_vol
        if z_score >= self.return_threshold_sigma:
            magnitude = z_score / self.return_threshold_sigma
            self.record_shock(t, magnitude)

    def intensity(self, t: float) -> float:
        """Compute current Hawkes intensity.

        .. math::

            \\lambda(t) = \\mu + \\sum_{t_i < t} \\alpha \\cdot m_i
            \\cdot \\exp(-\\beta \\cdot (t - t_i))

        Parameters
        ----------
        t:
            Current timestamp.

        Returns
        -------
        float
            Instantaneous jump intensity (jumps per day).
        """
        self._prune(t)
        excitation = 0.0
        for t_i, m_i in self._shocks:
            dt = t - t_i
            excitation += self.alpha * m_i * math.exp(-self.beta * dt)
        return self.mu + excitation

    # ── Internal ──────────────────────────────────────────────────

    def _prune(self, t: float) -> None:
        """Remove shocks older than ``max_history_seconds``."""
        cutoff = t - self.max_history_seconds
        while self._shocks and self._shocks[0][0] < cutoff:
            self._shocks.popleft()
