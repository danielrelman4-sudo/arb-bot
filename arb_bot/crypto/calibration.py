"""Model calibration: tracks prediction accuracy, applies Platt scaling.

Accumulates (predicted_probability, actual_outcome) pairs from settled
trades and fits a logistic calibration function to map raw model
probabilities to better-calibrated probabilities.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationRecord:
    """A single prediction-outcome pair."""
    predicted_prob: float
    outcome: bool  # True if YES settled
    timestamp: float = 0.0
    ticker: str = ""
    market_implied_prob: float = 0.5  # Market's implied probability at trade entry


@dataclass
class CalibrationBin:
    """A calibration curve bin."""
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_realized: float
    count: int


class ModelCalibrator:
    """Tracks prediction accuracy and applies Platt scaling.

    Platt scaling fits: P_calibrated = 1 / (1 + exp(-a * P_raw - b))
    using logistic regression on historical (prediction, outcome) pairs.

    Parameters
    ----------
    min_samples_for_calibration:
        Minimum number of settled outcomes before applying calibration.
    recalibrate_every:
        Re-fit Platt parameters after this many new outcomes.
    """

    def __init__(
        self,
        min_samples_for_calibration: int = 50,
        recalibrate_every: int = 20,
        method: str = "platt",
        isotonic_min_samples: int = 20,
    ) -> None:
        self._min_samples = min_samples_for_calibration
        self._recalibrate_every = recalibrate_every
        self._method = method  # "platt" or "isotonic"
        self._isotonic_min_samples = isotonic_min_samples
        self._records: List[CalibrationRecord] = []
        self._platt_a: float = 1.0  # Identity mapping by default
        self._platt_b: float = 0.0
        self._is_calibrated: bool = False
        self._samples_since_calibration: int = 0
        # Isotonic regression state
        self._isotonic_x: Optional[np.ndarray] = None  # breakpoint x values
        self._isotonic_y: Optional[np.ndarray] = None  # breakpoint y values

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def num_records(self) -> int:
        return len(self._records)

    @property
    def platt_params(self) -> Tuple[float, float]:
        return (self._platt_a, self._platt_b)

    def record_outcome(
        self,
        predicted_prob: float,
        outcome: bool,
        timestamp: float = 0.0,
        ticker: str = "",
        market_implied_prob: float = 0.5,
    ) -> None:
        """Record a prediction-outcome pair for calibration."""
        self._records.append(CalibrationRecord(
            predicted_prob=predicted_prob,
            outcome=outcome,
            timestamp=timestamp,
            ticker=ticker,
            market_implied_prob=market_implied_prob,
        ))
        self._samples_since_calibration += 1

        # Re-calibrate if we have enough data
        if self._method == "isotonic":
            if (len(self._records) >= self._isotonic_min_samples
                    and self._samples_since_calibration >= self._recalibrate_every):
                self._fit_isotonic()
        else:
            if (len(self._records) >= self._min_samples
                    and self._samples_since_calibration >= self._recalibrate_every):
                self._fit_platt()

    def calibrate(self, raw_prob: float) -> float:
        """Apply calibration (Platt or isotonic) to a raw model probability.

        Returns calibrated probability, or raw_prob if not yet calibrated.
        """
        if not self._is_calibrated:
            return raw_prob

        if self._method == "isotonic" and self._isotonic_x is not None:
            return self._interpolate_isotonic(raw_prob)

        # Platt sigmoid: 1 / (1 + exp(-a*x - b))
        z = self._platt_a * raw_prob + self._platt_b
        # Clamp to avoid overflow
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _interpolate_isotonic(self, raw_prob: float) -> float:
        """Linearly interpolate the isotonic calibration curve."""
        assert self._isotonic_x is not None and self._isotonic_y is not None
        x = self._isotonic_x
        y = self._isotonic_y
        # Clamp to breakpoint range
        clamped = max(float(x[0]), min(float(x[-1]), raw_prob))
        result = float(np.interp(clamped, x, y))
        return max(0.0, min(1.0, result))

    def _fit_isotonic(self) -> None:
        """Fit isotonic regression (PAVA) from accumulated records."""
        if len(self._records) < self._isotonic_min_samples:
            return

        x = np.array([r.predicted_prob for r in self._records])
        y = np.array([1.0 if r.outcome else 0.0 for r in self._records])

        # Sort by predicted prob
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # Group by unique x values (average y for ties)
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        grouped_y = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        for i, idx in enumerate(inverse):
            grouped_y[idx] += y_sorted[i]
            counts[idx] += 1
        grouped_y /= counts

        # PAVA (Pool Adjacent Violators Algorithm)
        iso_y = self._pava(grouped_y)

        self._isotonic_x = unique_x
        self._isotonic_y = iso_y
        self._is_calibrated = True
        self._samples_since_calibration = 0

        LOGGER.info(
            "ModelCalibrator: fitted isotonic with %d breakpoints from %d samples",
            len(unique_x), len(self._records),
        )

    @staticmethod
    def _pava(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm for isotonic regression.

        Produces a monotonically non-decreasing sequence that minimizes
        the sum of squared deviations from the input.
        """
        n = len(y)
        result = y.astype(float).copy()
        # Weighted PAVA
        w = np.ones(n, dtype=float)

        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # Pool: merge i and i+1
                total_w = w[i] + w[i + 1]
                result[i] = (w[i] * result[i] + w[i + 1] * result[i + 1]) / total_w
                result[i + 1] = result[i]
                w[i] = total_w
                w[i + 1] = total_w
                # Check backwards
                j = i
                while j > 0 and result[j - 1] > result[j]:
                    total_w = w[j - 1] + w[j]
                    result[j - 1] = (w[j - 1] * result[j - 1] + w[j] * result[j]) / total_w
                    result[j] = result[j - 1]
                    w[j - 1] = total_w
                    w[j] = total_w
                    j -= 1
                i = j + 1
            else:
                i += 1

        return result

    def _fit_platt(self) -> None:
        """Fit Platt scaling parameters using Newton's method.

        Minimizes negative log-likelihood of logistic regression:
        L(a, b) = -sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
        where p_i = sigmoid(a * x_i + b)
        """
        if len(self._records) < self._min_samples:
            return

        x = np.array([r.predicted_prob for r in self._records])
        y = np.array([1.0 if r.outcome else 0.0 for r in self._records])

        # Simple gradient descent for logistic regression
        a, b = self._platt_a, self._platt_b
        lr = 0.1
        for _ in range(100):
            z = a * x + b
            z = np.clip(z, -20, 20)
            p = 1.0 / (1.0 + np.exp(-z))
            # Gradients
            err = p - y
            grad_a = np.mean(err * x)
            grad_b = np.mean(err)
            a -= lr * grad_a
            b -= lr * grad_b

        self._platt_a = float(a)
        self._platt_b = float(b)
        self._is_calibrated = True
        self._samples_since_calibration = 0

        LOGGER.info(
            "ModelCalibrator: fitted Platt params a=%.3f b=%.3f from %d samples",
            self._platt_a, self._platt_b, len(self._records),
        )

    def compute_brier_score(self) -> float:
        """Brier score = mean((predicted - outcome)^2). Lower is better."""
        if not self._records:
            return 1.0
        total = 0.0
        for r in self._records:
            outcome_val = 1.0 if r.outcome else 0.0
            total += (r.predicted_prob - outcome_val) ** 2
        return total / len(self._records)

    def compute_market_brier_score(self) -> float:
        """Brier score using market-implied probabilities. Lower is better."""
        if not self._records:
            return 1.0
        total = 0.0
        for r in self._records:
            outcome_val = 1.0 if r.outcome else 0.0
            total += (r.market_implied_prob - outcome_val) ** 2
        return total / len(self._records)

    def compute_brier_skill_score(self) -> float:
        """BSS = 1 - (model_brier / market_brier). Positive = model beats market."""
        market_brier = self.compute_market_brier_score()
        if market_brier <= 0:
            return 0.0
        return 1.0 - (self.compute_brier_score() / market_brier)

    def compute_calibration_curve(self, num_bins: int = 10) -> List[CalibrationBin]:
        """Bin predictions by model_prob, compare vs realized frequency."""
        if not self._records:
            return []

        bins: List[CalibrationBin] = []
        bin_width = 1.0 / num_bins

        for i in range(num_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width
            bin_records = [
                r for r in self._records
                if lower <= r.predicted_prob < upper
            ]
            if not bin_records:
                continue
            mean_pred = sum(r.predicted_prob for r in bin_records) / len(bin_records)
            mean_real = sum(1.0 if r.outcome else 0.0 for r in bin_records) / len(bin_records)
            bins.append(CalibrationBin(
                bin_lower=lower,
                bin_upper=upper,
                mean_predicted=mean_pred,
                mean_realized=mean_real,
                count=len(bin_records),
            ))
        return bins
