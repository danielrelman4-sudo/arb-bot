"""OFI-to-drift calibrator: fits alpha for mu = alpha * OFI.

Performs rolling OLS regression of Order Flow Imbalance against
forward returns to determine the optimal drift coefficient.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass(frozen=True)
class OFICalibrationResult:
    """Result of OFI alpha calibration."""
    alpha: float       # Fitted drift coefficient
    r_squared: float   # Goodness of fit
    n_samples: int     # Number of samples used


class OFICalibrator:
    """Calibrates OFI alpha using rolling OLS.

    Fits: forward_return = alpha * OFI + epsilon

    Parameters
    ----------
    max_samples:
        Maximum number of (OFI, return) pairs to keep in rolling window.
    min_samples:
        Minimum samples needed before calibrating (otherwise returns alpha=0).
    min_r_squared:
        Minimum R² to accept the signal as predictive. Below this, alpha=0.
    """

    def __init__(
        self,
        max_samples: int = 5000,
        min_samples: int = 30,
        min_r_squared: float = 0.01,
    ) -> None:
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._min_r_squared = min_r_squared
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=max_samples)

    def record_sample(self, ofi: float, forward_return: float) -> None:
        """Record an (OFI, forward_return) observation."""
        self._samples.append((ofi, forward_return))

    def calibrate(self) -> OFICalibrationResult:
        """Fit alpha via OLS and return calibration result.

        Returns alpha=0.0 if:
        - Not enough samples (< min_samples)
        - R² below min_r_squared (signal is noise)
        - Degenerate data (all OFI values are zero)
        """
        n = len(self._samples)
        if n < self._min_samples:
            return OFICalibrationResult(alpha=0.0, r_squared=0.0, n_samples=n)

        arr = np.array(list(self._samples))
        x = arr[:, 0]  # OFI values
        y = arr[:, 1]  # forward returns

        # OLS without intercept: y = alpha * x + epsilon
        # alpha = sum(x*y) / sum(x^2)
        x_sq_sum = float(np.sum(x * x))
        if x_sq_sum < 1e-12:
            return OFICalibrationResult(alpha=0.0, r_squared=0.0, n_samples=n)

        alpha = float(np.sum(x * y)) / x_sq_sum

        # R-squared (uncentered, appropriate for no-intercept regression)
        # For y = alpha*x (no intercept), the correct R² uses uncentered
        # SS_tot = sum(y²) rather than centered SS_tot = sum((y - mean(y))²).
        # See: Kvalseth (1985), "Cautionary Note about R²".
        y_pred = alpha * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum(y * y))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_sq = max(0.0, r_sq)

        # If R² below threshold, signal is not predictive
        if r_sq < self._min_r_squared:
            return OFICalibrationResult(alpha=0.0, r_squared=r_sq, n_samples=n)

        return OFICalibrationResult(alpha=alpha, r_squared=r_sq, n_samples=n)

    @property
    def sample_count(self) -> int:
        """Number of samples currently stored."""
        return len(self._samples)
