"""Tests for the market regime detector."""

from __future__ import annotations

import time

import numpy as np
import pytest

from arb_bot.crypto.regime_detector import (
    HIGH_VOL,
    MEAN_REVERTING,
    TRENDING_DOWN,
    TRENDING_UP,
    MarketRegime,
    RegimeDetector,
    RegimeSnapshot,
)


# ── TestOFIAlignment ──────────────────────────────────────────────

class TestOFIAlignment:
    """Test OFI cross-timescale alignment computation."""

    def test_all_positive_full_alignment(self) -> None:
        """All OFI windows positive → high alignment, positive direction."""
        detector = RegimeDetector()
        ofi = {30: 0.5, 60: 0.3, 120: 0.2, 300: 0.1}
        alignment, direction = detector._compute_ofi_alignment(ofi)
        assert alignment == 1.0  # All 4/4 agree
        assert direction > 0

    def test_all_negative_full_alignment(self) -> None:
        """All OFI windows negative → high alignment, negative direction."""
        detector = RegimeDetector()
        ofi = {30: -0.5, 60: -0.3, 120: -0.2, 300: -0.1}
        alignment, direction = detector._compute_ofi_alignment(ofi)
        assert alignment == 1.0
        assert direction < 0

    def test_mixed_signals_lower_alignment(self) -> None:
        """Mixed OFI signals → lower alignment."""
        detector = RegimeDetector()
        ofi = {30: 0.5, 60: -0.3, 120: 0.2, 300: -0.1}
        alignment, direction = detector._compute_ofi_alignment(ofi)
        assert alignment < 1.0

    def test_empty_ofi_zero(self) -> None:
        """Empty OFI → zero alignment and direction."""
        detector = RegimeDetector()
        alignment, direction = detector._compute_ofi_alignment({})
        assert alignment == 0.0
        assert direction == 0.0

    def test_near_zero_excluded_from_sign(self) -> None:
        """Values near zero (within ±0.05) don't count toward sign agreement."""
        detector = RegimeDetector()
        ofi = {30: 0.5, 60: 0.02, 120: 0.3, 300: -0.01}
        alignment, direction = detector._compute_ofi_alignment(ofi)
        # Two non-zero agree (30, 120 both positive), two are zero
        # alignment = 2/4 = 0.5
        assert alignment == 0.5

    def test_weighted_direction(self) -> None:
        """Shorter windows get more weight in direction computation."""
        detector = RegimeDetector()
        # 30s has weight 4, 300s has weight 1
        # If 30s is strongly positive and 300s is weakly negative,
        # direction should still be positive
        ofi = {30: 0.5, 300: -0.3}
        _, direction = detector._compute_ofi_alignment(ofi)
        assert direction > 0  # 30s dominates


# ── TestVolScore ──────────────────────────────────────────────────

class TestVolScore:
    """Test vol expansion score computation."""

    def test_equal_vol_zero_score(self) -> None:
        """Short vol = long vol → score near 0."""
        detector = RegimeDetector()
        score = detector._compute_vol_score(0.02, 0.02, None)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_vol_expansion_high_score(self) -> None:
        """Short vol >> long vol → high score."""
        detector = RegimeDetector()
        score = detector._compute_vol_score(0.06, 0.02, None)
        # ratio = 3.0, score = (3.0-1.0)/2.0 = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_vpin_spike_high_score(self) -> None:
        """VPIN above spike threshold → high score."""
        detector = RegimeDetector(vpin_spike_threshold=0.85)
        score = detector._compute_vol_score(0.02, 0.02, 0.95)
        # vpin_component = (0.95 - 0.85) / (1.0 - 0.85) = 0.667
        assert score > 0.5

    def test_zero_long_vol_default(self) -> None:
        """Zero long vol → ratio defaults to 1.0 → score 0."""
        detector = RegimeDetector()
        score = detector._compute_vol_score(0.05, 0.0, None)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_vpin_below_threshold_no_contribution(self) -> None:
        """VPIN below threshold → no contribution from VPIN."""
        detector = RegimeDetector(vpin_spike_threshold=0.85)
        score_no_vpin = detector._compute_vol_score(0.02, 0.02, None)
        score_low_vpin = detector._compute_vol_score(0.02, 0.02, 0.50)
        assert score_low_vpin == score_no_vpin


# ── TestMeanReversionScore ────────────────────────────────────────

class TestMeanReversionScore:
    """Test mean reversion score (autocorrelation-based)."""

    def test_insufficient_data_neutral(self) -> None:
        """Fewer than min_returns → score 0.5 (neutral)."""
        detector = RegimeDetector(min_returns=10)
        score = detector._compute_mean_reversion_score([0.01, 0.02, 0.01])
        assert score == 0.5

    def test_trending_returns_low_score(self) -> None:
        """Positively autocorrelated returns → low mean reversion score."""
        detector = RegimeDetector(min_returns=5, autocorr_window=15)
        # Trending: each return is similar to the previous
        returns = [0.01] * 20  # Constant positive → autocorr ≈ 0 (but > 0 due to lag structure)
        score = detector._compute_mean_reversion_score(returns)
        # Constant returns have zero variance in shifted vs original → score ≈ 0.5
        # Let's use a clearly trending sequence instead
        rng = np.random.default_rng(42)
        trending = list(np.cumsum(rng.normal(0.001, 0.0001, 20)))  # Cumsum creates positive autocorr
        # Actually, for lag-1 autocorr we want returns that tend to follow each other
        trending = [0.01, 0.012, 0.015, 0.013, 0.016, 0.018, 0.020,
                    0.019, 0.021, 0.023, 0.025, 0.024, 0.027, 0.029, 0.030]
        score = detector._compute_mean_reversion_score(trending)
        # Positive autocorrelation → score < 0.5
        assert score < 0.5

    def test_mean_reverting_returns_high_score(self) -> None:
        """Negatively autocorrelated returns → high mean reversion score."""
        detector = RegimeDetector(min_returns=5, autocorr_window=15)
        # Alternating: negative autocorrelation
        mean_reverting = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01,
                         0.01, -0.01, 0.01, -0.01, 0.01, -0.01,
                         0.01, -0.01, 0.01]
        score = detector._compute_mean_reversion_score(mean_reverting)
        assert score > 0.7

    def test_constant_returns_neutral(self) -> None:
        """Constant returns → zero variance → score 0.5."""
        detector = RegimeDetector(min_returns=5)
        score = detector._compute_mean_reversion_score([0.01] * 20)
        assert score == 0.5


# ── TestClassify ──────────────────────────────────────────────────

class TestClassify:
    """Test the full classification logic."""

    def test_high_vol_regime(self) -> None:
        """Vol expansion + VPIN spike → HIGH_VOL regime."""
        detector = RegimeDetector(vpin_spike_threshold=0.85)
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={30: 0.2, 60: 0.1, 120: 0.05, 300: 0.02},
            returns_1m=[0.01] * 15,
            vol_short=0.06,
            vol_long=0.02,  # ratio = 3.0 → vol_score = 1.0
            vpin=0.90,
            signed_vpin=0.3,
        )
        assert snap.regime == HIGH_VOL
        assert snap.vol_score > 0.7
        assert snap.confidence > 0.7

    def test_trending_up_regime(self) -> None:
        """Strong positive OFI alignment → TRENDING_UP."""
        detector = RegimeDetector(ofi_trend_threshold=0.3)
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={30: 0.5, 60: 0.4, 120: 0.3, 300: 0.2},
            returns_1m=[0.002] * 15,
            vol_short=0.02,
            vol_long=0.02,
            vpin=0.3,
            signed_vpin=0.6,
        )
        assert snap.regime == TRENDING_UP
        assert snap.trend_score > 0

    def test_trending_down_regime(self) -> None:
        """Strong negative OFI alignment → TRENDING_DOWN."""
        detector = RegimeDetector(ofi_trend_threshold=0.3)
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={30: -0.5, 60: -0.4, 120: -0.3, 300: -0.2},
            returns_1m=[-0.002] * 15,
            vol_short=0.02,
            vol_long=0.02,
            vpin=0.3,
            signed_vpin=-0.6,
        )
        assert snap.regime == TRENDING_DOWN
        assert snap.trend_score < 0

    def test_mean_reverting_regime(self) -> None:
        """Low OFI alignment, low vol → MEAN_REVERTING."""
        detector = RegimeDetector(ofi_trend_threshold=0.3)
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={30: 0.1, 60: -0.1, 120: 0.05, 300: -0.05},
            returns_1m=[0.001, -0.001] * 8,
            vol_short=0.02,
            vol_long=0.02,
            vpin=0.3,
            signed_vpin=0.0,
        )
        assert snap.regime == MEAN_REVERTING

    def test_no_ofi_data_mean_reverting(self) -> None:
        """No OFI data → defaults to MEAN_REVERTING."""
        detector = RegimeDetector()
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={},
            returns_1m=[0.001] * 15,
            vol_short=0.02,
            vol_long=0.02,
        )
        assert snap.regime == MEAN_REVERTING

    def test_high_vol_overrides_trend(self) -> None:
        """HIGH_VOL takes priority over trending signals."""
        detector = RegimeDetector()
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={30: 0.5, 60: 0.4, 120: 0.3, 300: 0.2},
            returns_1m=[0.005] * 15,
            vol_short=0.08,
            vol_long=0.02,  # ratio = 4.0 → vol_score = 1.0
            vpin=0.95,
            signed_vpin=0.8,
        )
        assert snap.regime == HIGH_VOL

    def test_snapshot_has_all_fields(self) -> None:
        """Snapshot has all expected fields."""
        detector = RegimeDetector()
        snap = detector.classify(
            symbol="btcusdt",
            ofi_multiscale={30: 0.1},
            returns_1m=[0.001] * 15,
            vol_short=0.02,
            vol_long=0.02,
        )
        assert snap.symbol == "btcusdt"
        assert snap.timestamp > 0
        assert snap.regime in (TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING, HIGH_VOL)
        assert 0.0 <= snap.confidence <= 1.0
        assert -1.0 <= snap.trend_score <= 1.0
        assert 0.0 <= snap.vol_score <= 1.0
        assert 0.0 <= snap.mean_reversion_score <= 1.0
        assert 0.0 <= snap.ofi_alignment <= 1.0


# ── TestSmoothing ─────────────────────────────────────────────────

class TestSmoothing:
    """Test regime smoothing (persistence requirement)."""

    def test_initial_regime_passthrough(self) -> None:
        """First classification passes through directly."""
        detector = RegimeDetector()
        regime, is_transitioning = detector._smooth_regime("btc", TRENDING_UP)
        assert regime == TRENDING_UP

    def test_persistence_required_to_switch(self) -> None:
        """New regime needs 3/5 persistence to become active."""
        detector = RegimeDetector()
        # Start with trending_up for 3 cycles
        for _ in range(3):
            detector._smooth_regime("btc", TRENDING_UP)

        # Single flip doesn't change regime
        regime, is_transitioning = detector._smooth_regime("btc", HIGH_VOL)
        assert regime == TRENDING_UP  # Still trending_up (3/4 vs 1/4)

    def test_sustained_switch(self) -> None:
        """Sustained new regime eventually takes over."""
        detector = RegimeDetector()
        # Start with trending_up
        for _ in range(3):
            detector._smooth_regime("btc", TRENDING_UP)

        # Now high_vol for 3 cycles
        detector._smooth_regime("btc", HIGH_VOL)
        detector._smooth_regime("btc", HIGH_VOL)
        regime, is_transitioning = detector._smooth_regime("btc", HIGH_VOL)
        assert regime == HIGH_VOL  # Now 3/5 high_vol

    def test_symbols_independent(self) -> None:
        """Different symbols have independent regime history."""
        detector = RegimeDetector()
        for _ in range(3):
            detector._smooth_regime("btc", TRENDING_UP)
        for _ in range(3):
            detector._smooth_regime("eth", TRENDING_DOWN)

        r1, _ = detector._smooth_regime("btc", TRENDING_UP)
        r2, _ = detector._smooth_regime("eth", TRENDING_DOWN)
        assert r1 == TRENDING_UP
        assert r2 == TRENDING_DOWN

    def test_reset_clears_history(self) -> None:
        """Reset clears regime history."""
        detector = RegimeDetector()
        for _ in range(3):
            detector._smooth_regime("btc", TRENDING_UP)
        detector.reset("btc")
        # After reset, first classification should pass through
        regime, _ = detector._smooth_regime("btc", HIGH_VOL)
        assert regime == HIGH_VOL

    def test_reset_all_clears_all(self) -> None:
        """Reset without symbol clears all history."""
        detector = RegimeDetector()
        for _ in range(3):
            detector._smooth_regime("btc", TRENDING_UP)
            detector._smooth_regime("eth", TRENDING_DOWN)
        detector.reset()
        r1, _ = detector._smooth_regime("btc", HIGH_VOL)
        r2, _ = detector._smooth_regime("eth", MEAN_REVERTING)
        assert r1 == HIGH_VOL
        assert r2 == MEAN_REVERTING


# ── TestClassifyMarket ────────────────────────────────────────────

class TestClassifyMarket:
    """Test aggregate market regime classification."""

    def test_unanimous_regime(self) -> None:
        """All symbols agree → aggregate matches."""
        detector = RegimeDetector()
        now = time.time()
        snaps = {
            "btcusdt": RegimeSnapshot("btcusdt", now, TRENDING_UP, 0.8, 0.5, 0.1, 0.3, 0.9),
            "ethusdt": RegimeSnapshot("ethusdt", now, TRENDING_UP, 0.7, 0.4, 0.1, 0.3, 0.8),
        }
        market = detector.classify_market(snaps)
        assert market.regime == TRENDING_UP
        assert market.confidence > 0.5

    def test_mixed_regime_confidence_weighted(self) -> None:
        """Mixed regimes → highest weighted score wins."""
        detector = RegimeDetector()
        now = time.time()
        snaps = {
            "btcusdt": RegimeSnapshot("btcusdt", now, TRENDING_UP, 0.9, 0.5, 0.1, 0.3, 0.9),
            "ethusdt": RegimeSnapshot("ethusdt", now, MEAN_REVERTING, 0.3, 0.0, 0.1, 0.7, 0.2),
        }
        market = detector.classify_market(snaps)
        assert market.regime == TRENDING_UP  # Higher confidence

    def test_empty_snapshots_default(self) -> None:
        """No snapshots → MEAN_REVERTING with 0 confidence."""
        detector = RegimeDetector()
        market = detector.classify_market({})
        assert market.regime == MEAN_REVERTING
        assert market.confidence == 0.0

    def test_market_regime_properties(self) -> None:
        """Test MarketRegime convenience properties."""
        now = time.time()
        regime = MarketRegime(now, TRENDING_UP, 0.8, {})
        assert regime.is_trending is True
        assert regime.trend_direction == 1

        regime_down = MarketRegime(now, TRENDING_DOWN, 0.8, {})
        assert regime_down.is_trending is True
        assert regime_down.trend_direction == -1

        regime_mr = MarketRegime(now, MEAN_REVERTING, 0.8, {})
        assert regime_mr.is_trending is False
        assert regime_mr.trend_direction == 0


# ── TestConfigIntegration ─────────────────────────────────────────

class TestConfigIntegration:
    """Test regime detection config settings."""

    def test_config_defaults(self) -> None:
        """Config defaults are sensible."""
        from arb_bot.crypto.config import CryptoSettings
        settings = CryptoSettings()
        assert settings.regime_detection_enabled is True
        assert settings.regime_ofi_trend_threshold == 0.3
        assert settings.regime_vol_expansion_threshold == 2.0
        assert settings.regime_autocorr_window == 15
        assert settings.regime_min_returns == 10
        assert settings.regime_vpin_spike_threshold == 0.85

    def test_env_var_loading(self, monkeypatch) -> None:
        """Regime settings load from env vars."""
        from arb_bot.crypto.config import load_crypto_settings
        monkeypatch.setenv("ARB_CRYPTO_REGIME_DETECTION_ENABLED", "false")
        monkeypatch.setenv("ARB_CRYPTO_REGIME_OFI_TREND_THRESHOLD", "0.5")
        monkeypatch.setenv("ARB_CRYPTO_REGIME_VPIN_SPIKE_THRESHOLD", "0.90")
        settings = load_crypto_settings()
        assert settings.regime_detection_enabled is False
        assert settings.regime_ofi_trend_threshold == 0.5
        assert settings.regime_vpin_spike_threshold == 0.90

    def test_disabled_no_detector(self) -> None:
        """When disabled, engine should have no regime detector."""
        from arb_bot.crypto.config import CryptoSettings
        from arb_bot.crypto.engine import CryptoEngine
        settings = CryptoSettings(regime_detection_enabled=False)
        engine = CryptoEngine(settings)
        assert engine._regime_detector is None
        assert engine._current_regime is None

    def test_enabled_has_detector(self) -> None:
        """When enabled, engine should have a regime detector."""
        from arb_bot.crypto.config import CryptoSettings
        from arb_bot.crypto.engine import CryptoEngine
        settings = CryptoSettings(regime_detection_enabled=True)
        engine = CryptoEngine(settings)
        assert engine._regime_detector is not None
        assert engine._current_regime is None  # Not yet classified
