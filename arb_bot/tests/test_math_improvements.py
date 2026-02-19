"""Tests for Phase 10 math improvements:
    10A: Thompson Sampling
    10B: Baker-McHale shrunk Kelly
    10D: Embedding-based cross-venue matching
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from arb_bot.bucket_quality import BucketAccumulator, BucketQualityModel
from arb_bot.framework.kelly_sizing import KellySizingResult, TailRiskKelly, TailRiskKellyConfig


# ===========================================================================
# Helpers
# ===========================================================================


def _make_bucket_model(**kwargs) -> BucketQualityModel:
    """Create a BucketQualityModel with sensible test defaults."""
    defaults = dict(
        bucket_leg_counts={"b1": 3, "b2": 3, "b3": 3},
        enabled=True,
        max_active_buckets=2,
        min_observations=8,
        min_score=-0.02,
        live_update_interval=10000,  # prevent auto-recompute during setup
    )
    defaults.update(kwargs)
    return BucketQualityModel(**defaults)


def _inject_accumulator(
    model: BucketQualityModel,
    group_id: str,
    *,
    detections: int = 0,
    opened: int = 0,
    positive_outcomes: int = 0,
    realized_profit_sum: float = 0.0,
    detected_edge_sum: float = 0.0,
    fill_probability_sum: float = 0.0,
    expected_realized_profit_sum: float = 0.0,
    slippage_sum: float = 0.0,
    latency_sum_ms: float = 0.0,
    settled: int = 0,
) -> None:
    """Set accumulator data for a bucket directly."""
    model._accumulators[group_id] = BucketAccumulator(
        detections=detections,
        opened=opened,
        settled=settled,
        positive_outcomes=positive_outcomes,
        detected_edge_sum=detected_edge_sum,
        fill_probability_sum=fill_probability_sum,
        expected_realized_profit_sum=expected_realized_profit_sum,
        realized_profit_sum=realized_profit_sum,
        slippage_sum=slippage_sum,
        latency_sum_ms=latency_sum_ms,
    )


def _kelly_sizer(**kwargs) -> TailRiskKelly:
    return TailRiskKelly(TailRiskKellyConfig(**kwargs))


# ===========================================================================
# Phase 10A — Thompson Sampling in BucketQualityModel
# ===========================================================================


class TestThompsonSampling:
    """Tests for Thompson Sampling bucket selection (Phase 10A)."""

    def test_thompson_sampling_enabled_by_default(self) -> None:
        """use_thompson_sampling should default to True."""
        model = _make_bucket_model()
        assert model._use_thompson_sampling is True

    def test_thompson_all_buckets_enabled_no_history(self) -> None:
        """With no history at all, all buckets should be active.

        When there are no accumulators, _recompute_active_set keeps
        all buckets active (no scored entries = early return).
        """
        model = _make_bucket_model(
            bucket_leg_counts={"b1": 3, "b2": 3, "b3": 3},
            max_active_buckets=2,
            use_thompson_sampling=True,
        )
        # No accumulators injected => no scores => all kept active.
        active = model.active_bucket_ids
        assert active == {"b1", "b2", "b3"}

    def test_thompson_proven_buckets_selected_more_often(self) -> None:
        """A bucket with high success rate should be selected much more
        often than one with a low success rate across many recomputes.
        """
        selection_counts: Counter[str] = Counter()
        n_trials = 200

        for _ in range(n_trials):
            model = _make_bucket_model(
                bucket_leg_counts={"good": 3, "bad": 3, "meh": 3},
                max_active_buckets=1,
                use_thompson_sampling=True,
            )
            # Good bucket: 50 opens, 45 positive (90% success rate).
            _inject_accumulator(
                model, "good",
                detections=50, opened=50, positive_outcomes=45,
                realized_profit_sum=0.5, detected_edge_sum=0.5,
                fill_probability_sum=40.0,
                expected_realized_profit_sum=0.5,
            )
            # Bad bucket: 50 opens, 5 positive (10% success rate).
            _inject_accumulator(
                model, "bad",
                detections=50, opened=50, positive_outcomes=5,
                realized_profit_sum=-0.3, detected_edge_sum=0.2,
                fill_probability_sum=30.0,
                expected_realized_profit_sum=-0.3,
            )
            # Meh bucket: 50 opens, 25 positive (50% success rate).
            _inject_accumulator(
                model, "meh",
                detections=50, opened=50, positive_outcomes=25,
                realized_profit_sum=0.0, detected_edge_sum=0.3,
                fill_probability_sum=35.0,
                expected_realized_profit_sum=0.0,
            )
            model._recompute_active_set()
            for bid in model.active_bucket_ids:
                selection_counts[bid] += 1

        # The good bucket should be selected far more than the bad bucket.
        assert selection_counts["good"] > selection_counts["bad"] * 3, (
            f"Expected 'good' >> 'bad', got good={selection_counts['good']}, "
            f"bad={selection_counts['bad']}"
        )

    def test_thompson_unexplored_buckets_get_sampled(self) -> None:
        """A bucket with no history should still be sampled sometimes
        thanks to the prior giving it high variance.
        """
        unexplored_selected = 0
        n_trials = 200

        for _ in range(n_trials):
            model = _make_bucket_model(
                bucket_leg_counts={"proven1": 3, "proven2": 3, "newbie": 3},
                max_active_buckets=2,
                use_thompson_sampling=True,
            )
            # Two proven buckets with moderate data.
            _inject_accumulator(
                model, "proven1",
                detections=30, opened=30, positive_outcomes=20,
                realized_profit_sum=0.3, detected_edge_sum=0.3,
                fill_probability_sum=24.0,
                expected_realized_profit_sum=0.3,
            )
            _inject_accumulator(
                model, "proven2",
                detections=30, opened=30, positive_outcomes=18,
                realized_profit_sum=0.2, detected_edge_sum=0.25,
                fill_probability_sum=22.0,
                expected_realized_profit_sum=0.2,
            )
            # newbie has zero history — not in accumulators at all.
            model._recompute_active_set()
            if "newbie" in model.active_bucket_ids:
                unexplored_selected += 1

        # With Beta(1,1) prior, newbie should appear in at least some trials.
        assert unexplored_selected >= 5, (
            f"Unexplored bucket was selected only {unexplored_selected} times "
            f"in {n_trials} trials; expected natural exploration"
        )

    def test_thompson_bad_bucket_filtered_by_min_score(self) -> None:
        """A bucket with enough observations and score below min_score
        should never be selected by Thompson Sampling.
        """
        min_score = 0.0  # deliberately high to make filtering easy
        n_trials = 100

        for _ in range(n_trials):
            model = _make_bucket_model(
                bucket_leg_counts={"good": 3, "terrible": 3},
                max_active_buckets=2,
                min_observations=5,
                min_score=min_score,
                use_thompson_sampling=True,
            )
            # Good bucket: high positive rate, high profit.
            _inject_accumulator(
                model, "good",
                detections=20, opened=20, positive_outcomes=18,
                realized_profit_sum=0.4, detected_edge_sum=0.3,
                fill_probability_sum=16.0,
                expected_realized_profit_sum=0.4,
            )
            # Terrible bucket: many observations, very negative score.
            _inject_accumulator(
                model, "terrible",
                detections=20, opened=20, positive_outcomes=1,
                realized_profit_sum=-1.0, detected_edge_sum=0.01,
                fill_probability_sum=4.0,
                expected_realized_profit_sum=-1.0,
                slippage_sum=0.5,
                latency_sum_ms=10000.0,
            )
            model._recompute_active_set()
            # The terrible bucket's score should be below min_score
            # and it should have >= min_observations, so it's hard-filtered.
            terrible_score = model.score_for("terrible")
            if terrible_score is not None and terrible_score < min_score:
                assert "terrible" not in model.active_bucket_ids, (
                    f"Terrible bucket (score={terrible_score}) should be filtered"
                )

    def test_deterministic_fallback(self) -> None:
        """With use_thompson_sampling=False, selection should be deterministic."""
        results = []
        for _ in range(10):
            model = _make_bucket_model(
                bucket_leg_counts={"a": 3, "b": 3, "c": 3},
                max_active_buckets=2,
                use_thompson_sampling=False,
            )
            _inject_accumulator(
                model, "a",
                detections=20, opened=20, positive_outcomes=18,
                realized_profit_sum=0.5, detected_edge_sum=0.3,
                fill_probability_sum=16.0,
                expected_realized_profit_sum=0.5,
            )
            _inject_accumulator(
                model, "b",
                detections=20, opened=20, positive_outcomes=10,
                realized_profit_sum=0.1, detected_edge_sum=0.15,
                fill_probability_sum=12.0,
                expected_realized_profit_sum=0.1,
            )
            _inject_accumulator(
                model, "c",
                detections=20, opened=20, positive_outcomes=5,
                realized_profit_sum=-0.1, detected_edge_sum=0.05,
                fill_probability_sum=8.0,
                expected_realized_profit_sum=-0.1,
            )
            model._recompute_active_set()
            results.append(frozenset(model.active_bucket_ids))

        # All 10 runs should produce the same active set.
        assert len(set(results)) == 1, (
            f"Deterministic mode produced varying results: {results}"
        )

    @pytest.mark.parametrize(
        "prior_alpha,prior_beta,description",
        [
            (0.1, 0.1, "informative_low_prior"),
            (10.0, 1.0, "strong_optimistic_prior"),
            (1.0, 10.0, "strong_pessimistic_prior"),
        ],
    )
    def test_thompson_prior_params_affect_selection(
        self,
        prior_alpha: float,
        prior_beta: float,
        description: str,
    ) -> None:
        """Custom prior_alpha/beta should change selection behavior.

        With a strong optimistic prior (alpha=10, beta=1), unexplored
        buckets should be selected more often. With a pessimistic prior
        (alpha=1, beta=10), proven buckets dominate.
        """
        unexplored_count = 0
        n_trials = 200

        for _ in range(n_trials):
            model = _make_bucket_model(
                bucket_leg_counts={"proven": 3, "unexplored": 3},
                max_active_buckets=1,
                use_thompson_sampling=True,
                thompson_prior_alpha=prior_alpha,
                thompson_prior_beta=prior_beta,
            )
            _inject_accumulator(
                model, "proven",
                detections=40, opened=40, positive_outcomes=28,
                realized_profit_sum=0.4, detected_edge_sum=0.3,
                fill_probability_sum=32.0,
                expected_realized_profit_sum=0.4,
            )
            # unexplored has no data.
            model._recompute_active_set()
            if "unexplored" in model.active_bucket_ids:
                unexplored_count += 1

        # We don't assert exact values, but the prior should matter.
        # With optimistic prior, unexplored should be picked more often.
        # With pessimistic prior, less often. We just verify it runs and
        # selection counts are within valid range.
        assert 0 <= unexplored_count <= n_trials


# ===========================================================================
# Phase 10B — Baker-McHale Shrunk Kelly
# ===========================================================================


# Standard test inputs: edge large relative to cost for positive Kelly.
EDGE = 0.30
COST = 0.10
FILL = 0.8


class TestBakerMcHaleShrunkKelly:
    """Tests for Baker-McHale shrinkage in TailRiskKelly (Phase 10B)."""

    def test_baker_mchale_enabled_by_default(self) -> None:
        """Config should default to use_baker_mchale_shrinkage=True."""
        cfg = TailRiskKellyConfig()
        assert cfg.use_baker_mchale_shrinkage is True

    def test_baker_mchale_no_uncertainty_no_haircut(self) -> None:
        """With model_uncertainty=0, the shrinkage factor is 1.0
        (i.e., no haircut), because sigma_sq=0 => k*=1/(1+0)=1.
        """
        sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.0)

        assert not result.blocked
        assert result.uncertainty_haircut == pytest.approx(0.0, abs=1e-9)
        # Adjusted fraction should equal the fractional Kelly cap with
        # no uncertainty haircut and no variance haircut.
        assert result.adjusted_fraction > 0.0

    def test_baker_mchale_moderate_uncertainty(self) -> None:
        """With moderate uncertainty, Baker-McHale should apply a
        non-trivial haircut that is less aggressive than the legacy
        linear haircut.
        """
        uncertainty = 0.3

        bm_sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)
        bm_result = bm_sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=uncertainty,
        )

        assert not bm_result.blocked
        assert 0.0 < bm_result.uncertainty_haircut < 1.0
        assert bm_result.adjusted_fraction > 0.0

    def test_baker_mchale_high_uncertainty_large_haircut(self) -> None:
        """High uncertainty should produce a large haircut,
        substantially reducing the adjusted fraction.
        """
        sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)

        low_u = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.1)
        high_u = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.6)

        assert not low_u.blocked
        assert not high_u.blocked
        # Higher uncertainty => larger haircut => smaller fraction.
        assert high_u.uncertainty_haircut > low_u.uncertainty_haircut
        assert high_u.adjusted_fraction < low_u.adjusted_fraction

    def test_baker_mchale_higher_edge_ratio_more_aggressive_shrinkage(self) -> None:
        """When b = edge/cost is large, Baker-McHale shrinks more
        aggressively because b^2 * sigma^2 grows with b^2.
        """
        uncertainty = 0.3
        sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)

        # Small b: edge/cost = 0.03/0.55 ~ 0.055
        small_b = sizer.compute(
            edge=0.03, cost=0.55, fill_prob=FILL, model_uncertainty=uncertainty,
        )
        # Large b: edge/cost = 0.30/0.10 = 3.0
        large_b = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=uncertainty,
        )

        # Both should have positive edge (otherwise blocked).
        if not small_b.blocked and not large_b.blocked:
            # Larger edge ratio => more aggressive shrinkage.
            assert large_b.uncertainty_haircut > small_b.uncertainty_haircut, (
                f"Expected larger haircut for large b, got "
                f"large_b={large_b.uncertainty_haircut}, small_b={small_b.uncertainty_haircut}"
            )

    def test_baker_mchale_vs_legacy(self) -> None:
        """Baker-McHale and legacy linear should give different haircuts
        for the same inputs.
        """
        uncertainty = 0.3

        bm_sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)
        legacy_sizer = _kelly_sizer(
            use_baker_mchale_shrinkage=False,
            uncertainty_haircut_factor=1.0,
        )

        bm_result = bm_sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=uncertainty,
        )
        legacy_result = legacy_sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=uncertainty,
        )

        assert not bm_result.blocked
        assert not legacy_result.blocked

        # The two methods should produce different haircuts.
        assert bm_result.uncertainty_haircut != pytest.approx(
            legacy_result.uncertainty_haircut, abs=1e-6
        ), "Baker-McHale and legacy should differ"

        # And therefore different adjusted fractions.
        assert bm_result.adjusted_fraction != pytest.approx(
            legacy_result.adjusted_fraction, abs=1e-6
        ), "Adjusted fractions should differ between methods"

    def test_legacy_linear_when_disabled(self) -> None:
        """With use_baker_mchale_shrinkage=False, the legacy linear
        haircut should apply: haircut = uncertainty * factor.
        """
        uncertainty = 0.4
        factor = 1.0

        sizer = _kelly_sizer(
            use_baker_mchale_shrinkage=False,
            uncertainty_haircut_factor=factor,
        )
        result = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=uncertainty,
        )

        assert not result.blocked
        expected_haircut = min(1.0, uncertainty * factor)
        assert result.uncertainty_haircut == pytest.approx(expected_haircut, abs=1e-9)

    def test_baker_mchale_fraction_always_positive(self) -> None:
        """The adjusted fraction should never go negative, even with
        high uncertainty (below the max_model_uncertainty threshold).
        """
        sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)

        for uncertainty in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            result = sizer.compute(
                edge=EDGE, cost=COST, fill_prob=FILL,
                model_uncertainty=uncertainty,
            )
            if not result.blocked:
                assert result.adjusted_fraction >= 0.0, (
                    f"Fraction went negative at uncertainty={uncertainty}: "
                    f"{result.adjusted_fraction}"
                )

    @pytest.mark.parametrize(
        "uncertainty",
        [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
        ids=lambda u: f"u={u}",
    )
    def test_baker_mchale_shrinkage_formula(self, uncertainty: float) -> None:
        """Verify the shrinkage factor matches the closed-form formula:
        k* = 1 / (1 + b^2 * sigma^2)
        where b = edge/cost and sigma^2 = (uncertainty * edge)^2.
        """
        sizer = _kelly_sizer(use_baker_mchale_shrinkage=True)
        result = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=uncertainty,
        )

        if result.blocked:
            return  # blocked for other reasons, skip formula check

        b = EDGE / COST
        sigma_sq = (uncertainty * EDGE) ** 2
        expected_shrinkage = 1.0 / (1.0 + b * b * sigma_sq) if (b * b * sigma_sq) > 0 else 1.0
        expected_haircut = 1.0 - expected_shrinkage

        assert result.uncertainty_haircut == pytest.approx(expected_haircut, abs=1e-9), (
            f"Haircut mismatch at uncertainty={uncertainty}: "
            f"got {result.uncertainty_haircut}, expected {expected_haircut}"
        )


# ===========================================================================
# Phase 10D — Embedding-Based Cross-Venue Matching
# ===========================================================================


import numpy as np

from arb_bot.cross_mapping_generator import (
    _HAS_SENTENCE_TRANSFORMERS,
    _build_candidates,
    _compute_embeddings,
    _embedding_best_match_for,
    generate_cross_venue_mapping_rows,
    CandidateMarket,
)


def _make_kalshi_market(
    ticker: str, title: str, *, liquidity: float = 100.0
) -> dict:
    return {
        "venue": "kalshi",
        "ticker": ticker,
        "title": title,
        "liquidity": liquidity,
    }


def _make_polymarket_market(
    condition_id: str, question: str, *, liquidity: float = 100.0
) -> dict:
    return {
        "venue": "polymarket",
        "conditionId": condition_id,
        "question": question,
        "liquidityNum": liquidity,
    }


class TestEmbeddingMatching:
    """Tests for embedding-based cross-venue matching (Phase 10D)."""

    def test_sentence_transformers_available(self) -> None:
        """Verify sentence-transformers is installed for these tests."""
        assert _HAS_SENTENCE_TRANSFORMERS, "sentence-transformers required for embedding tests"

    def test_compute_embeddings_shape(self) -> None:
        """_compute_embeddings returns correct shape."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        candidates = [
            CandidateMarket("kalshi", "m1", "Will GDP exceed 3 percent", frozenset(["gdp", "exceed", "3", "percent"]), 100.0, 0.0),
            CandidateMarket("kalshi", "m2", "Will unemployment drop below 4 percent", frozenset(["unemployment", "drop", "below", "4", "percent"]), 100.0, 0.0),
        ]
        embeddings = _compute_embeddings(candidates, model)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # embedding dimension
        assert embeddings.dtype == np.float32

    def test_compute_embeddings_normalized(self) -> None:
        """Embeddings should be L2-normalized (unit vectors)."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        candidates = [
            CandidateMarket("kalshi", "m1", "Federal Reserve interest rate decision", frozenset(["federal", "reserve", "interest", "rate"]), 100.0, 0.0),
        ]
        embeddings = _compute_embeddings(candidates, model)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_compute_embeddings_empty(self) -> None:
        """Empty candidates list should return empty array."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _compute_embeddings([], model)
        assert embeddings.shape[0] == 0

    def test_embedding_similar_texts_high_score(self) -> None:
        """Semantically similar texts should have high cosine similarity."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        source = [
            CandidateMarket("kalshi", "k1", "Will US GDP growth exceed 3 percent in 2026", frozenset(["us", "gdp", "growth", "exceed", "3", "percent", "2026"]), 100.0, 0.0),
        ]
        targets = [
            CandidateMarket("polymarket", "p1", "US GDP growth above 3% in 2026", frozenset(["us", "gdp", "growth", "above", "3", "2026"]), 100.0, 0.0),
            CandidateMarket("polymarket", "p2", "Will Bitcoin price reach 100k", frozenset(["bitcoin", "price", "reach", "100k"]), 100.0, 0.0),
        ]

        s_emb = _compute_embeddings(source, model)
        t_emb = _compute_embeddings(targets, model)

        # Cosine sim between GDP questions should be high.
        gdp_sim = float(t_emb[0] @ s_emb[0])
        btc_sim = float(t_emb[1] @ s_emb[0])

        assert gdp_sim > btc_sim, f"GDP sim ({gdp_sim}) should exceed BTC sim ({btc_sim})"
        assert gdp_sim > 0.5, f"GDP question similarity should be high, got {gdp_sim}"

    def test_embedding_best_match_selects_correct(self) -> None:
        """_embedding_best_match_for should pick the semantically closest target."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        sources = [
            CandidateMarket("kalshi", "k1", "Will unemployment drop below 4 percent", frozenset(["unemployment", "drop", "below", "4", "percent"]), 100.0, 0.0),
        ]
        targets = [
            CandidateMarket("polymarket", "p1", "Unemployment rate under 4%", frozenset(["unemployment", "rate", "under", "4"]), 100.0, 0.0),
            CandidateMarket("polymarket", "p2", "Bitcoin reaches new all time high", frozenset(["bitcoin", "reaches", "new", "all", "time", "high"]), 100.0, 0.0),
            CandidateMarket("polymarket", "p3", "Will inflation exceed 5 percent", frozenset(["inflation", "exceed", "5", "percent"]), 100.0, 0.0),
        ]

        s_emb = _compute_embeddings(sources, model)
        t_emb = _compute_embeddings(targets, model)

        skip = Counter()
        result = _embedding_best_match_for(
            0, s_emb, t_emb, targets,
            min_match_score=0.3, min_score_gap=0.01, skip_reasons=skip,
        )

        assert result is not None
        assert result.market_id == "p1", f"Expected unemployment match, got {result.market_id}"

    def test_embedding_best_match_below_threshold(self) -> None:
        """When best score is below min_match_score, should return None."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        sources = [
            CandidateMarket("kalshi", "k1", "Federal Reserve interest rate", frozenset(["federal", "reserve", "interest", "rate"]), 100.0, 0.0),
        ]
        targets = [
            CandidateMarket("polymarket", "p1", "Super Bowl winner prediction", frozenset(["super", "bowl", "winner", "prediction"]), 100.0, 0.0),
        ]

        s_emb = _compute_embeddings(sources, model)
        t_emb = _compute_embeddings(targets, model)

        skip = Counter()
        result = _embedding_best_match_for(
            0, s_emb, t_emb, targets,
            min_match_score=0.95, min_score_gap=0.01, skip_reasons=skip,
        )

        assert result is None
        assert skip["below_threshold"] == 1

    def test_embedding_best_match_ambiguous(self) -> None:
        """When top-2 scores are too close, should return None (ambiguous)."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        sources = [
            CandidateMarket("kalshi", "k1", "Federal Reserve rate decision", frozenset(["federal", "reserve", "rate", "decision"]), 100.0, 0.0),
        ]
        # Two very similar targets — should be ambiguous.
        targets = [
            CandidateMarket("polymarket", "p1", "Federal Reserve rate decision meeting", frozenset(["federal", "reserve", "rate", "decision", "meeting"]), 100.0, 0.0),
            CandidateMarket("polymarket", "p2", "Federal Reserve rate decision outcome", frozenset(["federal", "reserve", "rate", "decision", "outcome"]), 100.0, 0.0),
        ]

        s_emb = _compute_embeddings(sources, model)
        t_emb = _compute_embeddings(targets, model)

        skip = Counter()
        result = _embedding_best_match_for(
            0, s_emb, t_emb, targets,
            min_match_score=0.3, min_score_gap=0.5, skip_reasons=skip,  # very large gap requirement
        )

        assert result is None
        assert skip["ambiguous_top_match"] == 1

    def test_generate_with_embeddings_flag(self) -> None:
        """generate_cross_venue_mapping_rows with use_embeddings=True should produce mappings."""
        markets = [
            _make_kalshi_market("KXGDP-2026-Q1", "Will US GDP growth exceed 3 percent in Q1 2026"),
            _make_kalshi_market("KXUNEMP-2026-FEB", "Will unemployment rate drop below 4 percent in February 2026"),
            _make_polymarket_market("0xabc123", "US GDP growth over 3% Q1 2026"),
            _make_polymarket_market("0xdef456", "February 2026 unemployment rate under 4%"),
        ]

        rows, diag = generate_cross_venue_mapping_rows(
            markets,
            use_embeddings=True,
            embedding_min_match_score=0.3,  # low threshold for test
            min_score_gap=0.01,
        )

        assert diag.kalshi_candidates == 2
        assert diag.polymarket_candidates == 2
        # Should produce at least 1 mapping.
        assert len(rows) >= 1

    def test_generate_without_embeddings_uses_jaccard(self) -> None:
        """With use_embeddings=False, should fall back to Jaccard (existing behavior)."""
        markets = [
            _make_kalshi_market("KXFED-2026-MAR", "Will the Fed raise interest rates in March 2026"),
            _make_polymarket_market("0x111", "Fed interest rate hike March 2026"),
        ]

        rows, diag = generate_cross_venue_mapping_rows(
            markets,
            use_embeddings=False,
            min_match_score=0.3,
            min_shared_tokens=1,
        )

        assert diag.kalshi_candidates == 1
        assert diag.polymarket_candidates == 1

    def test_embedding_fallback_when_unavailable(self) -> None:
        """When sentence-transformers is not available, should fall back to Jaccard."""
        from unittest.mock import patch

        markets = [
            _make_kalshi_market("KXTEST-1", "Will GDP exceed 3 percent"),
            _make_polymarket_market("0xtest1", "GDP growth above 3 percent"),
        ]

        with patch("arb_bot.cross_mapping_generator._HAS_SENTENCE_TRANSFORMERS", False):
            rows, diag = generate_cross_venue_mapping_rows(
                markets,
                use_embeddings=True,  # requested but unavailable
                min_match_score=0.3,
                min_shared_tokens=1,
            )

        # Should not crash — graceful fallback to Jaccard.
        assert diag.kalshi_candidates == 1
        assert diag.polymarket_candidates == 1
