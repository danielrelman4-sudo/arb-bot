"""OFI-to-drift calibrator: fits alpha for mu = alpha * sgn(OFI) * |OFI|^theta.

Performs rolling nonlinear regression of Order Flow Imbalance against
forward returns to determine the optimal drift coefficient and power-law
exponent (square-root law from Kyle/Cont).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OFICalibrationResult:
    """Result of OFI alpha calibration."""
    alpha: float       # Fitted drift coefficient
    theta: float       # Power-law exponent
    r_squared: float   # Goodness of fit
    n_samples: int     # Number of samples used


class OFICalibrator:
    """Calibrates OFI alpha and theta using nonlinear power-law fitting.

    Fits: forward_return = alpha * sgn(OFI) * |OFI|^theta + epsilon

    Falls back to OLS linear fit (theta=1.0) if scipy is unavailable
    or curve_fit fails.

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

    @staticmethod
    def _power_law_model(x: np.ndarray, alpha: float, theta: float) -> np.ndarray:
        """Power-law impact model: y = alpha * sgn(x) * |x|^theta."""
        return alpha * np.sign(x) * np.abs(x) ** theta

    def calibrate(self) -> OFICalibrationResult:
        """Fit alpha and theta via nonlinear curve_fit, with OLS fallback.

        Returns alpha=0.0, theta=0.5 if:
        - Not enough samples (< min_samples)
        - R² below min_r_squared (signal is noise)
        - Degenerate data (all OFI values are zero)
        """
        n = len(self._samples)
        if n < self._min_samples:
            return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=0.0, n_samples=n)

        arr = np.array(list(self._samples))
        x = arr[:, 0]  # OFI values
        y = arr[:, 1]  # forward returns

        # Check for degenerate data
        x_sq_sum = float(np.sum(x * x))
        if x_sq_sum < 1e-12:
            return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=0.0, n_samples=n)

        # Try nonlinear power-law fit via scipy curve_fit
        alpha, theta = self._fit_power_law(x, y)

        if alpha is None:
            # Fallback: OLS linear fit (theta=1.0)
            alpha_lin, theta_lin = self._fit_ols_linear(x, y)
            alpha = alpha_lin
            theta = theta_lin

        # Compute R² (uncentered, appropriate for no-intercept model)
        y_pred = alpha * np.sign(x) * np.abs(x) ** theta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum(y * y))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_sq = max(0.0, r_sq)

        # If R² below threshold, signal is not predictive
        if r_sq < self._min_r_squared:
            return OFICalibrationResult(alpha=0.0, theta=0.5, r_squared=r_sq, n_samples=n)

        return OFICalibrationResult(alpha=alpha, theta=theta, r_squared=r_sq, n_samples=n)

    def _fit_power_law(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """Attempt nonlinear power-law fit. Returns (alpha, theta) or (None, None)."""
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            return None, None

        try:
            popt, _ = curve_fit(
                self._power_law_model,
                x,
                y,
                p0=[0.1, 0.5],
                bounds=([-10, 0.1], [10, 1.5]),
                maxfev=5000,
            )
            return float(popt[0]), float(popt[1])
        except (RuntimeError, ValueError, TypeError):
            return None, None

    def _fit_ols_linear(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """OLS linear fallback: y = alpha * x (theta=1.0)."""
        x_sq_sum = float(np.sum(x * x))
        if x_sq_sum < 1e-12:
            return 0.0, 1.0
        alpha = float(np.sum(x * y)) / x_sq_sum
        return alpha, 1.0

    @property
    def sample_count(self) -> int:
        """Number of samples currently stored."""
        return len(self._samples)
