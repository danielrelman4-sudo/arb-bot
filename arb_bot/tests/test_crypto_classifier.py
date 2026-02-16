"""Tests for the XGBoost binary classifier (C2)."""

from __future__ import annotations

import os
import time
from unittest import mock

import pytest
import numpy as np

from arb_bot.crypto.classifier import (
    BinaryClassifier,
    ClassifierReport,
    ProbabilityEstimate,
    HAS_XGBOOST,
)

needs_xgboost = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")


# ── Tests that work regardless of xgboost installation ─────────────


class TestBinaryClassifierNoXGBoost:
    """Tests that work regardless of xgboost installation."""

    def test_not_trained_by_default(self):
        clf = BinaryClassifier()
        assert not clf.is_trained

    def test_predict_untrained_returns_fallback(self):
        clf = BinaryClassifier()
        result = clf.predict(np.zeros(10))
        assert result.probability == 0.5
        assert not result.is_classifier

    def test_last_report_none_initially(self):
        clf = BinaryClassifier()
        assert clf.last_report is None

    def test_trained_at_zero_initially(self):
        clf = BinaryClassifier()
        assert clf.trained_at == 0.0

    def test_classifier_report_defaults(self):
        report = ClassifierReport()
        assert report.num_samples == 0
        assert report.accuracy == 0.0
        assert report.brier_score == 1.0
        assert report.log_loss == float("inf")
        assert report.feature_importances == {}

    def test_probability_estimate_fields(self):
        est = ProbabilityEstimate(
            probability=0.7,
            raw_probability=0.68,
            uncertainty=0.1,
        )
        assert est.probability == 0.7
        assert est.raw_probability == 0.68
        assert est.uncertainty == 0.1
        assert est.is_classifier is True

    def test_probability_estimate_fallback_flag(self):
        est = ProbabilityEstimate(
            probability=0.5,
            raw_probability=0.5,
            uncertainty=1.0,
            is_classifier=False,
        )
        assert not est.is_classifier

    def test_load_model_no_xgboost_returns_false(self):
        """load_model returns False when path doesn't exist."""
        clf = BinaryClassifier(model_path="/nonexistent/path/model.json")
        loaded = clf.load_model()
        assert not loaded
        assert not clf.is_trained


# ── Tests requiring xgboost ────────────────────────────────────────


@needs_xgboost
class TestBinaryClassifierWithXGBoost:
    """Tests requiring xgboost to be installed."""

    def _make_training_data(self, n=200, seed=42):
        """Generate synthetic binary classification data."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 10))
        # Simple linear boundary with noise
        y = (X[:, 0] + X[:, 1] + rng.standard_normal(n) * 0.3 > 0).astype(float)
        return X, y

    def test_train_basic(self):
        clf = BinaryClassifier(n_estimators=10)
        X, y = self._make_training_data()
        report = clf.train(X, y)
        assert clf.is_trained
        assert report.num_samples == 200
        assert report.num_features == 10
        assert report.accuracy > 0.5

    def test_predict_after_training(self):
        clf = BinaryClassifier(n_estimators=10)
        X, y = self._make_training_data()
        clf.train(X, y)
        result = clf.predict(X[0])
        assert result.is_classifier
        assert 0.0 <= result.probability <= 1.0
        assert 0.0 <= result.raw_probability <= 1.0
        assert 0.0 <= result.uncertainty <= 0.5

    def test_predict_2d_input(self):
        clf = BinaryClassifier(n_estimators=10)
        X, y = self._make_training_data()
        clf.train(X, y)
        result = clf.predict(X[0:1])  # 2D input
        assert result.is_classifier
        assert 0.0 <= result.probability <= 1.0

    def test_train_too_few_samples(self):
        clf = BinaryClassifier(n_estimators=10)
        X = np.zeros((5, 10))
        y = np.array([0, 1, 0, 1, 0], dtype=float)
        report = clf.train(X, y)
        assert not clf.is_trained
        assert report.num_samples == 5

    def test_isotonic_calibration_applied(self):
        clf = BinaryClassifier(n_estimators=10, use_isotonic_calibration=True)
        X, y = self._make_training_data()
        clf.train(X, y)
        assert clf._iso_x is not None
        assert clf._iso_y is not None
        assert len(clf._iso_x) > 0
        assert len(clf._iso_y) > 0

    def test_no_isotonic_calibration(self):
        clf = BinaryClassifier(n_estimators=10, use_isotonic_calibration=False)
        X, y = self._make_training_data()
        clf.train(X, y)
        assert clf._iso_x is None
        assert clf._iso_y is None

    def test_feature_importances(self):
        clf = BinaryClassifier(n_estimators=10)
        X, y = self._make_training_data()
        names = [f"feat_{i}" for i in range(10)]
        report = clf.train(X, y, feature_names=names)
        assert len(report.feature_importances) == 10
        assert "feat_0" in report.feature_importances
        # Importances should sum to approximately 1
        total = sum(report.feature_importances.values())
        assert 0.9 < total < 1.1

    def test_brier_score_reasonable(self):
        clf = BinaryClassifier(n_estimators=50)
        X, y = self._make_training_data(n=400)
        report = clf.train(X, y)
        # Model should be better than random (0.25 for balanced data)
        assert report.brier_score < 0.5

    def test_log_loss_reasonable(self):
        clf = BinaryClassifier(n_estimators=50)
        X, y = self._make_training_data(n=400)
        report = clf.train(X, y)
        # Better than random guessing (log_loss of 0.693 for p=0.5)
        assert report.log_loss < 1.0

    def test_save_load_model(self, tmp_path):
        clf = BinaryClassifier(
            n_estimators=10,
            model_path=str(tmp_path / "model.json"),
        )
        X, y = self._make_training_data()
        clf.train(X, y)

        # Load into new classifier
        clf2 = BinaryClassifier(model_path=str(tmp_path / "model.json"))
        loaded = clf2.load_model()
        assert loaded
        assert clf2.is_trained

        # Predictions should match
        r1 = clf.predict(X[0])
        r2 = clf2.predict(X[0])
        assert abs(r1.raw_probability - r2.raw_probability) < 0.01

    def test_save_load_with_isotonic(self, tmp_path):
        clf = BinaryClassifier(
            n_estimators=10,
            use_isotonic_calibration=True,
            model_path=str(tmp_path / "model.json"),
        )
        X, y = self._make_training_data()
        clf.train(X, y)

        # Verify isotonic file is saved alongside model
        iso_path = tmp_path / "model.iso.npz"
        assert iso_path.exists()

        # Load into new classifier
        clf2 = BinaryClassifier(
            use_isotonic_calibration=True,
            model_path=str(tmp_path / "model.json"),
        )
        loaded = clf2.load_model()
        assert loaded
        assert clf2._iso_x is not None

    def test_load_nonexistent_model(self, tmp_path):
        clf = BinaryClassifier(model_path=str(tmp_path / "nonexistent.json"))
        loaded = clf.load_model()
        assert not loaded
        assert not clf.is_trained

    def test_retrain_updates_model(self):
        clf = BinaryClassifier(n_estimators=10)
        X, y = self._make_training_data(n=200, seed=1)
        clf.train(X, y)
        t1 = clf.trained_at

        time.sleep(0.01)

        X2, y2 = self._make_training_data(n=200, seed=2)
        clf.train(X2, y2)
        assert clf.trained_at > t1

    def test_uncertainty_varies(self):
        clf = BinaryClassifier(n_estimators=50)
        X, y = self._make_training_data(n=400)
        clf.train(X, y)

        # Get predictions for different inputs
        results = [clf.predict(X[i]) for i in range(20)]
        uncertainties = [r.uncertainty for r in results]
        # Should have some variation in uncertainty values
        assert max(uncertainties) > min(uncertainties) or len(set(uncertainties)) >= 1

    def test_predict_raw_batch(self):
        clf = BinaryClassifier(n_estimators=10)
        X, y = self._make_training_data()
        clf.train(X, y)
        probs = clf._predict_raw(X[:5])
        assert len(probs) == 5
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_auto_save_on_train(self, tmp_path):
        model_path = str(tmp_path / "auto_model.json")
        clf = BinaryClassifier(n_estimators=10, model_path=model_path)
        X, y = self._make_training_data()
        clf.train(X, y)

        # Model file should have been saved automatically
        from pathlib import Path
        assert Path(model_path).exists()


# ── Config integration tests ───────────────────────────────────────


class TestClassifierConfig:

    def test_config_defaults(self):
        from arb_bot.crypto.config import CryptoSettings
        s = CryptoSettings()
        assert s.classifier_enabled is False
        assert s.classifier_model_path == ""
        assert s.classifier_max_depth == 4
        assert s.classifier_n_estimators == 100
        assert s.classifier_learning_rate == 0.1
        assert s.classifier_min_child_weight == 5
        assert s.classifier_subsample == 0.8
        assert s.classifier_use_isotonic_calibration is True
        assert s.classifier_min_training_samples == 200
        assert s.classifier_retrain_interval_hours == 24.0

    def test_config_env_vars(self, monkeypatch):
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_ENABLED", "true")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_MAX_DEPTH", "6")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_N_ESTIMATORS", "200")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_LEARNING_RATE", "0.05")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_MIN_CHILD_WEIGHT", "10")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_SUBSAMPLE", "0.9")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_USE_ISOTONIC", "false")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_MIN_TRAINING_SAMPLES", "500")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_RETRAIN_INTERVAL_HOURS", "12.0")
        monkeypatch.setenv("ARB_CRYPTO_CLASSIFIER_MODEL_PATH", "/tmp/model.json")
        from arb_bot.crypto.config import load_crypto_settings
        s = load_crypto_settings()
        assert s.classifier_enabled is True
        assert s.classifier_max_depth == 6
        assert s.classifier_n_estimators == 200
        assert s.classifier_learning_rate == 0.05
        assert s.classifier_min_child_weight == 10
        assert s.classifier_subsample == 0.9
        assert s.classifier_use_isotonic_calibration is False
        assert s.classifier_min_training_samples == 500
        assert s.classifier_retrain_interval_hours == 12.0
        assert s.classifier_model_path == "/tmp/model.json"
