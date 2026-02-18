"""XGBoost binary classifier with isotonic post-hoc calibration.

Replaces GBM Monte Carlo with a trained classifier that directly predicts
P(YES outcome | features). Uses FeatureStore training data.

XGBoost is an optional dependency -- when not installed, the classifier
is unavailable and the engine falls back to Monte Carlo.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# Check XGBoost availability
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None  # type: ignore


@dataclass
class ClassifierReport:
    """Report from a classifier training run."""
    num_samples: int = 0
    num_features: int = 0
    accuracy: float = 0.0
    brier_score: float = 1.0
    log_loss: float = float("inf")
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    feature_importances: Dict[str, float] = field(default_factory=dict)
    trained_at: float = 0.0


@dataclass
class ProbabilityEstimate:
    """Calibrated probability estimate from the classifier."""
    probability: float
    raw_probability: float  # Before calibration
    uncertainty: float      # Ensemble uncertainty or calibration spread
    is_classifier: bool = True  # True if from classifier, False if fallback


class BinaryClassifier:
    """XGBoost binary classifier with isotonic calibration.

    Parameters
    ----------
    max_depth: Maximum tree depth (default 4)
    n_estimators: Number of boosting rounds (default 100)
    learning_rate: Learning rate (default 0.1)
    min_child_weight: Minimum sum of instance weight in child (default 5)
    subsample: Row subsampling ratio (default 0.8)
    use_isotonic_calibration: Whether to apply isotonic post-hoc calibration
    model_path: Optional path to save/load model
    """

    def __init__(
        self,
        max_depth: int = 4,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        use_isotonic_calibration: bool = True,
        model_path: Optional[str] = None,
    ) -> None:
        self._max_depth = max_depth
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._min_child_weight = min_child_weight
        self._subsample = subsample
        self._use_isotonic = use_isotonic_calibration
        self._model_path = Path(model_path) if model_path else None

        self._model: Any = None  # xgb.XGBClassifier when trained
        self._is_trained: bool = False
        self._last_report: Optional[ClassifierReport] = None
        self._trained_at: float = 0.0

        # Isotonic calibration breakpoints
        self._iso_x: Optional[np.ndarray] = None
        self._iso_y: Optional[np.ndarray] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def last_report(self) -> Optional[ClassifierReport]:
        return self._last_report

    @property
    def trained_at(self) -> float:
        return self._trained_at

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ClassifierReport:
        """Train the classifier on feature matrix X and labels y.

        Uses a hold-out split (80/20) for calibration and evaluation.
        Applies isotonic calibration on the hold-out set if enabled.

        Parameters
        ----------
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        feature_names: Optional list of feature names for importance reporting

        Returns ClassifierReport with training metrics.
        """
        if not HAS_XGBOOST:
            LOGGER.warning("XGBoost not installed -- cannot train classifier")
            return ClassifierReport()

        n_samples = len(y)
        if n_samples < 20:
            LOGGER.warning("Too few samples (%d) to train classifier", n_samples)
            return ClassifierReport(num_samples=n_samples)

        # Shuffle and split 80/20
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_samples)
        split = int(0.8 * n_samples)
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train XGBoost
        model = xgb.XGBClassifier(
            max_depth=self._max_depth,
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            min_child_weight=self._min_child_weight,
            subsample=self._subsample,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        self._model = model
        self._is_trained = True
        self._trained_at = time.time()

        # Isotonic calibration on validation set
        if self._use_isotonic and len(X_val) >= 10:
            raw_probs = model.predict_proba(X_val)[:, 1]
            self._fit_isotonic(raw_probs, y_val)

        # Compute metrics
        val_probs = self._predict_raw(X_val)
        val_preds = (val_probs >= 0.5).astype(int)
        accuracy = float(np.mean(val_preds == y_val))
        brier = float(np.mean((val_probs - y_val) ** 2))

        # Log loss (with clipping for numerical stability)
        eps = 1e-15
        clipped = np.clip(val_probs, eps, 1 - eps)
        ll = -float(np.mean(y_val * np.log(clipped) + (1 - y_val) * np.log(1 - clipped)))

        # Feature importances
        importances: Dict[str, float] = {}
        if feature_names and hasattr(model, "feature_importances_"):
            for name, imp in zip(feature_names, model.feature_importances_):
                importances[name] = float(imp)

        report = ClassifierReport(
            num_samples=n_samples,
            num_features=X.shape[1],
            accuracy=accuracy,
            brier_score=brier,
            log_loss=ll,
            feature_importances=importances,
            trained_at=self._trained_at,
        )
        self._last_report = report

        LOGGER.info(
            "Classifier trained: %d samples, accuracy=%.3f, brier=%.4f, logloss=%.4f",
            n_samples, accuracy, brier, ll,
        )

        # Auto-save if path configured
        if self._model_path:
            self.save_model(self._model_path)

        return report

    def predict(self, features: np.ndarray) -> ProbabilityEstimate:
        """Predict calibrated probability for a single feature vector.

        Parameters
        ----------
        features: 1D array of feature values (n_features,) or 2D (1, n_features)

        Returns ProbabilityEstimate with calibrated probability.
        """
        if not self._is_trained or self._model is None:
            return ProbabilityEstimate(
                probability=0.5,
                raw_probability=0.5,
                uncertainty=1.0,
                is_classifier=False,
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Handle feature dimension mismatch: model trained with N features
        # but current code may produce M features (M > N after adding new
        # columns, or M < N if columns were removed).
        # XGBoost stores the expected feature count in model.n_features_in_.
        expected_n = getattr(self._model, "n_features_in_", features.shape[1])
        if features.shape[1] != expected_n:
            if features.shape[1] > expected_n:
                # Truncate: only pass the first N features the model knows
                features = features[:, :expected_n]
            else:
                # Pad with zeros
                pad = np.zeros((features.shape[0], expected_n - features.shape[1]))
                features = np.hstack([features, pad])

        raw_prob = float(self._model.predict_proba(features)[:, 1][0])

        # Apply isotonic calibration if available
        if self._iso_x is not None and self._use_isotonic:
            calibrated = float(np.interp(raw_prob, self._iso_x, self._iso_y))
        else:
            calibrated = raw_prob

        # Estimate uncertainty from distance to calibration boundary
        uncertainty = min(
            abs(calibrated - 0.5) * 2,  # Higher confidence = lower uncertainty
            0.5,
        )
        uncertainty = 0.5 - uncertainty  # Invert: 0 = confident, 0.5 = uncertain

        return ProbabilityEstimate(
            probability=calibrated,
            raw_probability=raw_prob,
            uncertainty=uncertainty,
            is_classifier=True,
        )

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Get raw (uncalibrated) probabilities for a batch."""
        if self._model is None:
            return np.full(len(X), 0.5)
        raw = self._model.predict_proba(X)[:, 1]
        if self._iso_x is not None and self._use_isotonic:
            return np.interp(raw, self._iso_x, self._iso_y)
        return raw

    def _fit_isotonic(self, raw_probs: np.ndarray, y: np.ndarray) -> None:
        """Fit isotonic calibration using Pool Adjacent Violators Algorithm.

        Same PAVA implementation as in calibration.py.
        """
        order = np.argsort(raw_probs)
        x_sorted = raw_probs[order].copy()
        y_sorted = y[order].copy().astype(float)

        n = len(y_sorted)
        weights = np.ones(n, dtype=float)
        values = y_sorted.copy()

        i = 0
        while i < n - 1:
            if values[i] > values[i + 1]:
                pooled = (
                    (values[i] * weights[i] + values[i + 1] * weights[i + 1])
                    / (weights[i] + weights[i + 1])
                )
                pooled_w = weights[i] + weights[i + 1]
                values[i] = pooled
                weights[i] = pooled_w
                values = np.delete(values, i + 1)
                weights = np.delete(weights, i + 1)
                x_sorted = np.delete(x_sorted, i + 1)
                n -= 1
                if i > 0:
                    i -= 1
            else:
                i += 1

        self._iso_x = x_sorted
        self._iso_y = values
        LOGGER.info("Classifier: fitted isotonic calibration with %d breakpoints", len(x_sorted))

    def save_model(self, path: Optional[Path] = None) -> None:
        """Save the trained model to disk."""
        save_path = path or self._model_path
        if save_path is None or self._model is None:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self._model.save_model(str(save_path))

        # Save isotonic calibration alongside
        if self._iso_x is not None:
            iso_path = save_path.with_suffix(".iso.npz")
            np.savez(str(iso_path), x=self._iso_x, y=self._iso_y)

        LOGGER.info("Classifier: saved model to %s", save_path)

    def load_model(self, path: Optional[Path] = None) -> bool:
        """Load a trained model from disk. Returns True if successful."""
        if not HAS_XGBOOST:
            return False

        load_path = path or self._model_path
        if load_path is None:
            return False

        load_path = Path(load_path)
        if not load_path.exists():
            return False

        try:
            model = xgb.XGBClassifier()
            model.load_model(str(load_path))
            self._model = model
            self._is_trained = True

            # Load isotonic calibration
            iso_path = load_path.with_suffix(".iso.npz")
            if iso_path.exists():
                data = np.load(str(iso_path))
                self._iso_x = data["x"]
                self._iso_y = data["y"]

            LOGGER.info("Classifier: loaded model from %s", load_path)
            return True
        except Exception as exc:
            LOGGER.warning("Classifier: failed to load model: %s", exc)
            return False
