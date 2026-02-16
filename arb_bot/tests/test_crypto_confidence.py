"""Tests for composite confidence scoring (B2)."""

from __future__ import annotations

import pytest

from arb_bot.crypto.confidence_scorer import (
    ConfidenceComponents,
    ConfidenceResult,
    ConfidenceScorer,
)


class TestConfidenceScorer:
    """Core scoring logic tests."""

    def test_all_positive_signals_high_score(self):
        """All 7 signals positive should yield a score near maximum."""
        scorer = ConfidenceScorer()
        components = ConfidenceComponents(
            staleness_signal=0.8,
            vpin_signal=0.7,
            ofi_signal=0.6,
            funding_signal=0.5,
            vol_regime_signal=0.5,
            cross_asset_signal=0.4,
            model_edge_signal=0.9,
        )
        result = scorer.score(components)
        assert result.score > 0.5
        assert result.signal_agreement == 7
        assert result.total_signals == 7
        assert scorer.passes(result)

    def test_all_negative_signals_zero_score(self):
        """All negative signals should produce a score of 0 or very low."""
        scorer = ConfidenceScorer()
        components = ConfidenceComponents(
            staleness_signal=-0.8,
            vpin_signal=-0.7,
            ofi_signal=-0.6,
            funding_signal=-0.5,
            vol_regime_signal=-0.5,
            cross_asset_signal=-0.4,
            model_edge_signal=-0.9,
        )
        result = scorer.score(components)
        assert result.score == 0.0
        assert result.signal_agreement == 0
        assert result.total_signals == 7
        assert not scorer.passes(result)

    def test_mixed_signals_agreement(self):
        """4 positive 3 negative should pass agreement check (>= 3)."""
        scorer = ConfidenceScorer(min_score=0.10, min_agreement=3)
        components = ConfidenceComponents(
            staleness_signal=0.8,
            vpin_signal=0.7,
            ofi_signal=0.6,
            funding_signal=0.5,
            vol_regime_signal=-0.3,
            cross_asset_signal=-0.2,
            model_edge_signal=-0.1,
        )
        result = scorer.score(components)
        assert result.signal_agreement == 4
        assert result.total_signals == 7
        # 4 agree >= min_agreement=3, so no cap at 0.5
        # Should have positive score since weighted positive > penalty
        assert result.score > 0.0

    def test_insufficient_agreement_caps_score(self):
        """Only 2 agreeing signals should cap score at 0.5."""
        scorer = ConfidenceScorer(min_agreement=3)
        components = ConfidenceComponents(
            staleness_signal=1.0,
            vpin_signal=1.0,
            ofi_signal=0.0,
            funding_signal=0.0,
            vol_regime_signal=0.0,
            cross_asset_signal=0.0,
            model_edge_signal=0.0,
        )
        result = scorer.score(components)
        assert result.signal_agreement == 2
        assert result.score <= 0.5
        assert "capped" in result.reasons[-1]

    def test_min_agreement_threshold(self):
        """Exactly 3 agreeing signals with min_agreement=3 should not be capped."""
        scorer = ConfidenceScorer(min_agreement=3)
        components = ConfidenceComponents(
            staleness_signal=1.0,
            vpin_signal=1.0,
            ofi_signal=1.0,
            funding_signal=0.0,
            vol_regime_signal=0.0,
            cross_asset_signal=0.0,
            model_edge_signal=0.0,
        )
        result = scorer.score(components)
        assert result.signal_agreement == 3
        # Should not be capped -- no "capped" in reasons
        assert not any("capped" in r for r in result.reasons)

    def test_zero_signals_score(self):
        """All zero signals should yield score = 0."""
        scorer = ConfidenceScorer()
        components = ConfidenceComponents()  # All defaults (0.0)
        result = scorer.score(components)
        assert result.score == 0.0
        assert result.signal_agreement == 0
        assert result.total_signals == 0

    def test_weight_priorities(self):
        """Staleness + VPIN dominate due to higher weights."""
        scorer = ConfidenceScorer(min_agreement=1)
        # Only staleness at max
        comp_staleness = ConfidenceComponents(staleness_signal=1.0)
        # Only funding at max (lower weight)
        comp_funding = ConfidenceComponents(funding_signal=1.0)

        result_staleness = scorer.score(comp_staleness)
        result_funding = scorer.score(comp_funding)

        # staleness_weight=0.25 > funding_weight=0.10
        assert result_staleness.score > result_funding.score

    def test_passes_above_threshold(self):
        """Score >= min_score should pass."""
        scorer = ConfidenceScorer(min_score=0.50, min_agreement=3)
        components = ConfidenceComponents(
            staleness_signal=1.0,
            vpin_signal=1.0,
            ofi_signal=1.0,
            funding_signal=1.0,
            vol_regime_signal=1.0,
            cross_asset_signal=1.0,
            model_edge_signal=1.0,
        )
        result = scorer.score(components)
        assert result.score >= 0.50
        assert scorer.passes(result)

    def test_fails_below_threshold(self):
        """Score < min_score should not pass."""
        scorer = ConfidenceScorer(min_score=0.90)
        # Single weak signal -- won't reach 0.90
        components = ConfidenceComponents(
            staleness_signal=0.3,
            vpin_signal=0.0,
            ofi_signal=0.0,
            funding_signal=0.0,
            vol_regime_signal=0.0,
            cross_asset_signal=0.0,
            model_edge_signal=0.0,
        )
        result = scorer.score(components)
        assert result.score < 0.90
        assert not scorer.passes(result)

    def test_custom_weights(self):
        """Non-default weights should affect scoring correctly."""
        # Give all weight to model_edge
        scorer = ConfidenceScorer(
            min_agreement=1,
            staleness_weight=0.0,
            vpin_weight=0.0,
            ofi_weight=0.0,
            funding_weight=0.0,
            vol_regime_weight=0.0,
            cross_asset_weight=0.0,
            model_edge_weight=1.0,
        )
        components = ConfidenceComponents(model_edge_signal=0.8)
        result = scorer.score(components)
        # Score should be 0.8 (all weight on model_edge at 0.8)
        assert abs(result.score - 0.8) < 0.01

    def test_custom_min_agreement(self):
        """Setting min_agreement=2 relaxes the requirement."""
        scorer = ConfidenceScorer(min_agreement=2)
        components = ConfidenceComponents(
            staleness_signal=1.0,
            vpin_signal=1.0,
            ofi_signal=0.0,
            funding_signal=0.0,
            vol_regime_signal=0.0,
            cross_asset_signal=0.0,
            model_edge_signal=0.0,
        )
        result = scorer.score(components)
        assert result.signal_agreement == 2
        # With min_agreement=2, 2 signals should NOT be capped
        assert not any("capped" in r for r in result.reasons)

    def test_signal_clamping(self):
        """Signals > 1.0 or < -1.0 should be clamped."""
        scorer = ConfidenceScorer(min_agreement=1)
        # Signal at 5.0 should be clamped to 1.0
        components = ConfidenceComponents(staleness_signal=5.0)
        result = scorer.score(components)
        # The effective contribution should be same as signal=1.0
        comp_clamped = ConfidenceComponents(staleness_signal=1.0)
        result_clamped = scorer.score(comp_clamped)
        assert abs(result.score - result_clamped.score) < 0.01

        # Signal at -5.0 should be clamped to -1.0
        components_neg = ConfidenceComponents(staleness_signal=-5.0)
        result_neg = scorer.score(components_neg)
        comp_neg_clamped = ConfidenceComponents(staleness_signal=-1.0)
        result_neg_clamped = scorer.score(comp_neg_clamped)
        assert abs(result_neg.score - result_neg_clamped.score) < 0.01

    def test_single_strong_signal(self):
        """One signal at 1.0 others zero should be capped at 0.5 (low agreement)."""
        scorer = ConfidenceScorer(min_agreement=3)
        components = ConfidenceComponents(staleness_signal=1.0)
        result = scorer.score(components)
        assert result.signal_agreement == 1
        assert result.score <= 0.5
        assert "capped" in result.reasons[-1]

    def test_reasons_list(self):
        """Reasons list should include signal names and values."""
        scorer = ConfidenceScorer(min_agreement=1)
        components = ConfidenceComponents(
            staleness_signal=0.5,
            vpin_signal=-0.3,
        )
        result = scorer.score(components)
        reasons_str = " ".join(result.reasons)
        assert "staleness" in reasons_str
        assert "vpin" in reasons_str

    def test_confidence_result_fields(self):
        """All fields should be populated correctly."""
        scorer = ConfidenceScorer()
        components = ConfidenceComponents(
            staleness_signal=0.6,
            vpin_signal=0.5,
            ofi_signal=0.4,
            funding_signal=0.3,
            vol_regime_signal=0.0,
            cross_asset_signal=0.0,
            model_edge_signal=0.0,
        )
        result = scorer.score(components)

        assert isinstance(result, ConfidenceResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.signal_agreement, int)
        assert isinstance(result.total_signals, int)
        assert isinstance(result.reasons, list)
        assert result.components is components
        assert result.total_signals == 4
        assert result.signal_agreement == 4

    def test_default_config_values(self):
        """ConfidenceScorer default params should match plan."""
        scorer = ConfidenceScorer()
        assert scorer.min_score == 0.65
        assert scorer.min_agreement == 3
        assert scorer._weights["staleness"] == 0.25
        assert scorer._weights["vpin"] == 0.20
        assert scorer._weights["ofi"] == 0.15
        assert scorer._weights["funding"] == 0.10
        assert scorer._weights["vol_regime"] == 0.10
        assert scorer._weights["cross_asset"] == 0.10
        assert scorer._weights["model_edge"] == 0.10


class TestConfidenceConfig:
    """Verify config integration."""

    def test_config_defaults(self):
        """Default CryptoSettings should have confidence scoring disabled."""
        from arb_bot.crypto.config import CryptoSettings

        settings = CryptoSettings()
        assert settings.confidence_scoring_enabled is False
        assert settings.confidence_min_score == 0.65
        assert settings.confidence_min_agreement == 3
        assert settings.confidence_staleness_weight == 0.25
        assert settings.confidence_vpin_weight == 0.20
        assert settings.confidence_ofi_weight == 0.15
        assert settings.confidence_funding_weight == 0.10
        assert settings.confidence_vol_regime_weight == 0.10
        assert settings.confidence_cross_asset_weight == 0.10
        assert settings.confidence_model_edge_weight == 0.10

    def test_config_env_parsing(self, monkeypatch):
        """Env var parsing should populate confidence fields."""
        from arb_bot.crypto.config import load_crypto_settings

        monkeypatch.setenv("ARB_CRYPTO_CONFIDENCE_SCORING_ENABLED", "true")
        monkeypatch.setenv("ARB_CRYPTO_CONFIDENCE_MIN_SCORE", "0.70")
        monkeypatch.setenv("ARB_CRYPTO_CONFIDENCE_MIN_AGREEMENT", "4")
        monkeypatch.setenv("ARB_CRYPTO_CONFIDENCE_STALENESS_WEIGHT", "0.30")

        settings = load_crypto_settings()
        assert settings.confidence_scoring_enabled is True
        assert settings.confidence_min_score == 0.70
        assert settings.confidence_min_agreement == 4
        assert settings.confidence_staleness_weight == 0.30
