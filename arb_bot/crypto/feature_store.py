"""Feature store: records feature vectors at trade entry, labels at settlement.

Stores ~30-field FeatureVectors as CSV rows, labeled with binary outcomes
when trades settle. Provides load_training_data() to feed the classifier.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Features captured at trade entry for classifier training.

    All fields are floats for easy numpy conversion.
    The label field is set when the trade settles.
    """
    # Identifiers
    ticker: str = ""
    timestamp: float = 0.0

    # Market structure
    strike_distance_pct: float = 0.0      # (strike - spot) / spot
    time_to_expiry_minutes: float = 0.0
    implied_probability: float = 0.0       # Market-implied prob
    spread_cents: float = 0.0              # yes_ask - yes_bid or similar
    book_depth_yes: int = 0
    book_depth_no: int = 0

    # Model outputs
    model_probability: float = 0.0         # MC model prob
    model_uncertainty: float = 0.0         # Wilson CI half-width
    edge_cents: float = 0.0                # model_prob - market_prob
    blended_probability: float = 0.0

    # Signal features
    staleness_score: float = 0.0           # 0=fresh, 1=stale
    vpin: float = 0.0                      # Unsigned VPIN [0,1]
    signed_vpin: float = 0.0              # Directional [-1,1]
    vpin_trend: float = 0.0               # VPIN slope
    ofi_30s: float = 0.0                  # Order flow at 30s
    ofi_60s: float = 0.0
    ofi_120s: float = 0.0
    ofi_300s: float = 0.0
    aggressor_ratio: float = 0.0          # Buy vol / total vol [0,1]
    volume_acceleration: float = 0.0
    funding_rate: float = 0.0
    funding_rate_8h_avg: float = 0.0
    funding_rate_change: float = 0.0
    student_t_probability: float = 0.0       # Student-t model prob (0=not computed)
    student_t_nu: float = 0.0               # Fitted degrees of freedom
    empirical_probability: float = 0.0      # Empirical CDF bootstrap prob (0=not computed)

    # Volatility features
    realized_vol_1m: float = 0.0          # Realized vol from 1-min returns
    realized_vol_5m: float = 0.0          # From 5-min returns
    vol_ratio: float = 0.0               # short_vol / long_vol

    # Cross-asset features
    leader_ofi: float = 0.0
    leader_return_5m: float = 0.0
    leader_vol_ratio: float = 0.0

    # Regime features
    regime: str = ""                      # trending_up/trending_down/mean_reverting/high_vol
    regime_confidence: float = 0.0        # [0, 1]
    regime_trend_score: float = 0.0       # [-1, 1] directional strength
    regime_vol_score: float = 0.0         # [0, 1] vol expansion
    regime_mean_reversion_score: float = 0.0  # [0, 1] choppiness
    regime_ofi_alignment: float = 0.0     # [0, 1] cross-timescale OFI agreement

    # Regime decision features (logged per trade for classifier training)
    regime_is_transitioning: int = 0     # 0=confirmed, 1=transitioning
    regime_kelly_multiplier: float = 1.0 # Applied regime Kelly multiplier
    regime_min_edge_applied: float = 0.0 # Regime min edge threshold used
    vpin_at_entry: float = 0.0           # VPIN value when trade was placed

    # ── v42 Tier 1: High-impact features ──────────────────────────
    hour_utc_sin: float = 0.0           # sin(2π × hour/24) cyclical time encoding
    hour_utc_cos: float = 0.0           # cos(2π × hour/24) cyclical time encoding
    day_of_week: float = 0.0            # 0-6 (Monday=0)
    inv_sqrt_tte: float = 0.0           # 1/sqrt(time_to_expiry_minutes) — gamma acceleration
    model_disagreement: float = 0.0     # max - min across ensemble model probs
    hawkes_intensity: float = 0.0       # Self-exciting jump intensity at entry
    volume_flow_rate: float = 0.0       # Avg volume per minute (5-min window)

    # ── v42 Tier 2: Medium-impact features ────────────────────────
    cell_rolling_win_rate: float = 0.0  # Win rate of last N trades in this cell
    cell_rolling_pnl: float = 0.0       # Sum PnL of last N trades in this cell
    cell_trades_last_24h: int = 0       # Count of trades in this cell in last 24h
    cross_ofi_divergence: float = 0.0   # |leader_ofi - target_ofi|
    realized_vol_15m: float = 0.0       # Std of 15-min returns (contract-horizon vol)
    realized_vol_1h: float = 0.0        # Std of 1-hour returns (daily-horizon vol)

    # ── v42 Tier 3: New-data-source features ──────────────────────
    iv_rv_ratio: float = 0.0            # Implied vol / realized vol
    funding_rate_persistence: int = 0   # Consecutive same-sign funding intervals

    # Trade direction
    side: str = ""                        # "yes" or "no"
    entry_price: float = 0.0

    # Strategy identifier (for classifier training)
    strategy: str = "model"            # "model" (v17) or "momentum" (v18)
    strategy_cell: str = ""            # "yes_15min", "yes_daily", "no_15min", "no_daily"

    # Label (set at settlement)
    outcome: int = -1                     # -1=unsettled, 0=loss, 1=win


# Regime string → numeric encoding (XGBoost needs numeric features)
_REGIME_ENCODE = {
    "mean_reverting": 0, "trending_up": 1, "trending_down": 2, "high_vol": 3,
}

# Feature columns used for training (excludes identifiers and label)
FEATURE_COLUMNS = [
    "strike_distance_pct", "time_to_expiry_minutes", "implied_probability",
    "spread_cents", "book_depth_yes", "book_depth_no",
    "model_probability", "model_uncertainty", "edge_cents", "blended_probability",
    "staleness_score", "vpin", "signed_vpin", "vpin_trend",
    "ofi_30s", "ofi_60s", "ofi_120s", "ofi_300s",
    "aggressor_ratio", "volume_acceleration",
    "funding_rate", "funding_rate_8h_avg", "funding_rate_change",
    "student_t_probability", "student_t_nu",
    "empirical_probability",
    "realized_vol_1m", "realized_vol_5m", "vol_ratio",
    "leader_ofi", "leader_return_5m", "leader_vol_ratio",
    "regime", "regime_confidence", "regime_trend_score",
    "regime_vol_score", "regime_mean_reversion_score", "regime_ofi_alignment",
    "regime_is_transitioning", "regime_kelly_multiplier",
    "regime_min_edge_applied", "vpin_at_entry",
    # v42 Tier 1
    "hour_utc_sin", "hour_utc_cos", "day_of_week",
    "inv_sqrt_tte", "model_disagreement",
    "hawkes_intensity", "volume_flow_rate",
    # v42 Tier 2
    "cell_rolling_win_rate", "cell_rolling_pnl", "cell_trades_last_24h",
    "cross_ofi_divergence", "realized_vol_15m", "realized_vol_1h",
    # v42 Tier 3
    "iv_rv_ratio", "funding_rate_persistence",
]

# All CSV columns (features + identifiers + label)
ALL_COLUMNS = [
    "ticker", "timestamp", "side", "entry_price", "strategy", "strategy_cell",
] + FEATURE_COLUMNS + ["outcome"]


class FeatureStore:
    """Stores training data for the classifier.

    Records FeatureVectors at trade entry to a CSV file.
    Labels are updated when trades settle.

    Parameters
    ----------
    path: Path to the CSV file for storage.
    min_samples_for_classifier: Minimum settled samples before classifier can train.
    """

    def __init__(
        self,
        path: str = "feature_store.csv",
        min_samples_for_classifier: int = 200,
        flush_interval: int = 50,
    ) -> None:
        self._path = Path(path)
        self._min_samples = min_samples_for_classifier
        self._flush_interval = flush_interval
        self._write_buffer: list[dict] = []
        self._entries: dict[str, FeatureVector] = {}  # ticker -> pending FV
        self._ensure_file()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def min_samples(self) -> int:
        return self._min_samples

    def _ensure_file(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                writer.writeheader()

    def record_entry(self, fv: FeatureVector) -> None:
        """Record a feature vector at trade entry.

        The feature vector is buffered in memory and flushed to CSV when
        the buffer reaches ``flush_interval`` entries.
        Also stored in memory for later labeling.
        """
        if not fv.ticker:
            LOGGER.warning("FeatureStore: ignoring entry with empty ticker")
            return

        self._entries[fv.ticker] = fv

        # Buffer the row for periodic flush
        row = self._fv_to_row(fv)
        self._write_buffer.append(row)

        if len(self._write_buffer) >= self._flush_interval:
            self.flush()

        LOGGER.debug("FeatureStore: recorded entry for %s", fv.ticker)

    def flush(self) -> None:
        """Write buffered entries to CSV."""
        if not self._write_buffer:
            return
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            for row in self._write_buffer:
                writer.writerow(row)
        self._write_buffer.clear()

    def record_outcome(self, ticker: str, won: bool) -> None:
        """Update the outcome label for a settled trade.

        Reads the CSV, updates the matching row's outcome, rewrites.
        """
        outcome = 1 if won else 0

        # Update in-memory entry
        self._entries.pop(ticker, None)

        # Update CSV file: find the last row with this ticker and outcome=-1
        if not self._path.exists():
            return

        rows = self._read_all_rows()
        updated = False
        # Search backwards to find the most recent unsettled entry
        for i in range(len(rows) - 1, -1, -1):
            if rows[i].get("ticker") == ticker and str(rows[i].get("outcome", "")) == "-1":
                rows[i]["outcome"] = str(outcome)
                updated = True
                break

        if updated:
            self._write_all_rows(rows)
            LOGGER.debug("FeatureStore: recorded outcome=%d for %s", outcome, ticker)
        else:
            LOGGER.debug("FeatureStore: no pending entry found for %s", ticker)

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load settled training samples as (X, y) arrays.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (only FEATURE_COLUMNS).
        y : ndarray of shape (n_samples,)
            Binary labels (0 or 1).
        """
        if not self._path.exists():
            return np.empty((0, len(FEATURE_COLUMNS))), np.empty(0)

        rows = self._read_all_rows()

        # Filter to settled rows only
        settled = [r for r in rows if str(r.get("outcome", "-1")) in ("0", "1")]

        if not settled:
            return np.empty((0, len(FEATURE_COLUMNS))), np.empty(0)

        X = np.zeros((len(settled), len(FEATURE_COLUMNS)))
        y = np.zeros(len(settled))

        for i, row in enumerate(settled):
            for j, col in enumerate(FEATURE_COLUMNS):
                val = row.get(col, 0.0)
                if col == "regime":
                    X[i, j] = float(_REGIME_ENCODE.get(str(val), 0))
                else:
                    try:
                        X[i, j] = float(val)
                    except (ValueError, TypeError):
                        X[i, j] = 0.0
            y[i] = float(row["outcome"])

        return X, y

    def count_settled(self) -> int:
        """Count the number of settled training samples."""
        if not self._path.exists():
            return 0
        rows = self._read_all_rows()
        return sum(1 for r in rows if str(r.get("outcome", "-1")) in ("0", "1"))

    def count_pending(self) -> int:
        """Count unsettled (pending) entries."""
        if not self._path.exists():
            return 0
        rows = self._read_all_rows()
        return sum(1 for r in rows if str(r.get("outcome", "-1")) == "-1")

    def has_enough_samples(self) -> bool:
        """Check if we have enough settled samples for classifier training."""
        return self.count_settled() >= self._min_samples

    def _fv_to_row(self, fv: FeatureVector) -> dict:
        """Convert FeatureVector to CSV row dict."""
        row = {}
        for col in ALL_COLUMNS:
            val = getattr(fv, col, 0)
            row[col] = val
        return row

    def _read_all_rows(self) -> List[dict]:
        """Read all rows from CSV."""
        with open(self._path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _write_all_rows(self, rows: List[dict]) -> None:
        """Rewrite entire CSV with updated rows."""
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


# ── Multi-file utilities (for offline training) ─────────────────────────


def load_training_data_from_files(
    paths: List[str],
    strategy_filter: str = "model",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load and merge training data from multiple feature store CSVs.

    Parameters
    ----------
    paths: List of CSV file paths to load.
    strategy_filter: "model" or "momentum". Rows with matching strategy field
        are included. Rows with empty strategy but a non-empty strategy_cell
        are treated as "model" (legacy rows from before the field was added).

    Returns
    -------
    X : Feature matrix (n_samples, n_features + 2) — FEATURE_COLUMNS + side_numeric + is_daily
    y : Binary labels (n_samples,)
    metadata : List of dicts with ticker, side, strategy_cell, entry_price per row
    """
    all_rows: List[dict] = []

    for path in paths:
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
        except (FileNotFoundError, OSError):
            LOGGER.warning("load_training_data_from_files: skipping %s", path)
            continue

    # Filter to settled rows matching strategy
    settled: List[dict] = []
    for row in all_rows:
        outcome = str(row.get("outcome", "-1"))
        if outcome not in ("0", "1"):
            continue
        strategy = row.get("strategy", "")
        cell = row.get("strategy_cell", "")
        # Match strategy filter
        if strategy == strategy_filter:
            settled.append(row)
        elif not strategy and cell and strategy_filter == "model":
            # Legacy rows: no strategy field but have cell → model-path
            settled.append(row)

    if not settled:
        n_extra = 2  # side_numeric, is_daily
        return (
            np.empty((0, len(FEATURE_COLUMNS) + n_extra)),
            np.empty(0),
            [],
        )

    n_extra = 2  # side_numeric, is_daily
    n_features = len(FEATURE_COLUMNS) + n_extra
    X = np.zeros((len(settled), n_features))
    y = np.zeros(len(settled))
    metadata: List[dict] = []

    for i, row in enumerate(settled):
        # Encode standard FEATURE_COLUMNS
        for j, col in enumerate(FEATURE_COLUMNS):
            val = row.get(col, 0.0)
            if col == "regime":
                X[i, j] = float(_REGIME_ENCODE.get(str(val), 0))
            else:
                try:
                    X[i, j] = float(val)
                except (ValueError, TypeError):
                    X[i, j] = 0.0

        # Extra derived features
        base = len(FEATURE_COLUMNS)
        X[i, base + 0] = 1.0 if row.get("side", "") == "yes" else 0.0  # side_numeric
        cell = row.get("strategy_cell", "")
        X[i, base + 1] = 1.0 if "daily" in cell else 0.0  # is_daily

        y[i] = float(row["outcome"])
        metadata.append({
            "ticker": row.get("ticker", ""),
            "side": row.get("side", ""),
            "strategy_cell": cell,
            "entry_price": row.get("entry_price", ""),
        })

    return X, y, metadata


def get_extended_feature_names() -> List[str]:
    """Return feature names including the extra derived columns."""
    return FEATURE_COLUMNS + ["side_numeric", "is_daily"]
