"""Outlier-robust sizing inputs (Phase 4D).

Provides robust statistical estimators and outlier filters for
the data that feeds into position sizing — edge estimates,
fill probabilities, cost estimates, and slippage.

Usage::

    filt = RobustInputFilter(config)
    clean_edge = filt.filter_edge(raw_edges)
    clean_cost = filt.filter_cost(raw_costs)
    report = filt.diagnose(raw_edges, "edge")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustInputConfig:
    """Configuration for outlier-robust input filtering.

    Parameters
    ----------
    iqr_multiplier:
        Multiplier for IQR-based outlier fencing. Values outside
        [Q1 - k*IQR, Q3 + k*IQR] are flagged. Default 1.5.
    mad_threshold:
        Modified Z-score threshold using median absolute deviation.
        Values with |modified_z| > threshold are flagged. Default 3.5.
    min_samples:
        Minimum samples needed before filtering applies. Below this
        all values pass through. Default 5.
    winsorize_fraction:
        Fraction of each tail to winsorize (clip). Default 0.05 (5%).
    max_consecutive_outliers:
        If this many consecutive samples are outliers, raise an alert.
        Default 3.
    """

    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5
    min_samples: int = 5
    winsorize_fraction: float = 0.05
    max_consecutive_outliers: int = 3


# ---------------------------------------------------------------------------
# Diagnosis report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosisReport:
    """Diagnosis of a sample for outlier contamination."""

    field_name: str
    sample_count: int
    median: float
    mad: float
    q1: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
    outlier_count: int
    outlier_fraction: float
    consecutive_outlier_alert: bool


# ---------------------------------------------------------------------------
# Robust input filter
# ---------------------------------------------------------------------------


class RobustInputFilter:
    """Filters outliers from sizing inputs using robust statistics.

    Combines IQR fencing and MAD-based z-scores. Supports
    winsorization and consecutive outlier alerts.
    """

    def __init__(self, config: RobustInputConfig | None = None) -> None:
        self._config = config or RobustInputConfig()
        self._consecutive_outliers: Dict[str, int] = {}
        self._alert_history: Dict[str, List[bool]] = {}

    @property
    def config(self) -> RobustInputConfig:
        return self._config

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def filter_values(
        self,
        values: Sequence[float],
        field_name: str = "",
    ) -> List[float]:
        """Remove outliers and return cleaned values.

        Uses IQR fencing. Values below min_samples pass through.
        """
        if len(values) < self._config.min_samples:
            return list(values)

        q1, q3 = self._quartiles(values)
        iqr = q3 - q1
        k = self._config.iqr_multiplier
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        cleaned = [v for v in values if lower <= v <= upper]
        outlier_count = len(values) - len(cleaned)
        self._track_consecutive(field_name, outlier_count > 0)
        return cleaned

    def winsorize(self, values: Sequence[float]) -> List[float]:
        """Winsorize (clip) extreme tails.

        Clips the bottom and top `winsorize_fraction` of values
        to the fence values.
        """
        if len(values) < self._config.min_samples:
            return list(values)

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        frac = self._config.winsorize_fraction
        low_idx = max(0, int(math.floor(n * frac)))
        high_idx = min(n - 1, int(math.ceil(n * (1.0 - frac))) - 1)
        low_val = sorted_vals[low_idx]
        high_val = sorted_vals[high_idx]

        return [max(low_val, min(high_val, v)) for v in values]

    def is_outlier(self, value: float, reference: Sequence[float]) -> bool:
        """Check if a single value is an outlier relative to reference."""
        if len(reference) < self._config.min_samples:
            return False

        med = self._median(reference)
        mad = self._mad(reference)
        if mad == 0.0:
            return value != med

        modified_z = 0.6745 * (value - med) / mad
        return abs(modified_z) > self._config.mad_threshold

    def robust_mean(self, values: Sequence[float]) -> float:
        """Compute outlier-robust mean (winsorized mean)."""
        if not values:
            return 0.0
        if len(values) < self._config.min_samples:
            return sum(values) / len(values)
        cleaned = self.winsorize(values)
        return sum(cleaned) / len(cleaned) if cleaned else 0.0

    def robust_std(self, values: Sequence[float]) -> float:
        """Compute outlier-robust standard deviation (MAD-based)."""
        if len(values) < 2:
            return 0.0
        mad = self._mad(values)
        # MAD to std conversion factor for normal distribution.
        return mad * 1.4826

    def diagnose(
        self,
        values: Sequence[float],
        field_name: str = "",
    ) -> DiagnosisReport:
        """Produce a diagnostic report on outlier contamination."""
        n = len(values)
        if n == 0:
            return DiagnosisReport(
                field_name=field_name, sample_count=0,
                median=0.0, mad=0.0, q1=0.0, q3=0.0, iqr=0.0,
                lower_fence=0.0, upper_fence=0.0,
                outlier_count=0, outlier_fraction=0.0,
                consecutive_outlier_alert=False,
            )

        med = self._median(values)
        mad = self._mad(values)
        q1, q3 = self._quartiles(values)
        iqr = q3 - q1
        k = self._config.iqr_multiplier
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        outlier_count = sum(1 for v in values if v < lower or v > upper)
        outlier_fraction = outlier_count / n if n > 0 else 0.0

        consec_alert = self._consecutive_outliers.get(
            field_name, 0
        ) >= self._config.max_consecutive_outliers

        return DiagnosisReport(
            field_name=field_name,
            sample_count=n,
            median=med,
            mad=mad,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_fence=lower,
            upper_fence=upper,
            outlier_count=outlier_count,
            outlier_fraction=outlier_fraction,
            consecutive_outlier_alert=consec_alert,
        )

    def has_consecutive_alert(self, field_name: str) -> bool:
        """Check if a field has triggered consecutive outlier alert."""
        return (
            self._consecutive_outliers.get(field_name, 0)
            >= self._config.max_consecutive_outliers
        )

    def clear(self) -> None:
        """Reset all tracked state."""
        self._consecutive_outliers.clear()
        self._alert_history.clear()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _track_consecutive(self, field_name: str, is_outlier: bool) -> None:
        if not field_name:
            return
        if is_outlier:
            self._consecutive_outliers[field_name] = (
                self._consecutive_outliers.get(field_name, 0) + 1
            )
        else:
            self._consecutive_outliers[field_name] = 0

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0

    @staticmethod
    def _mad(values: Sequence[float]) -> float:
        """Median absolute deviation."""
        if len(values) < 2:
            return 0.0
        med = RobustInputFilter._median(values)
        deviations = [abs(v - med) for v in values]
        return RobustInputFilter._median(deviations)

    @staticmethod
    def _quartiles(values: Sequence[float]) -> Tuple[float, float]:
        """Compute Q1 and Q3 using linear interpolation."""
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return s[0], s[0]

        def _percentile(pct: float) -> float:
            idx = pct * (n - 1)
            lo = int(math.floor(idx))
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return s[lo] + frac * (s[hi] - s[lo])

        return _percentile(0.25), _percentile(0.75)
