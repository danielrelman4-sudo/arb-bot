"""Tests for LLM-as-offline-curator mapping verification pipeline (B1)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from arb_bot.mapping_verifier import (
    VerificationResult,
    VerificationSettings,
    VerificationDiagnostics,
    _build_verification_prompt,
    _call_mock,
    _heuristic_rejection,
    _parse_llm_response,
    edge_sanity_check,
    verify_mappings,
)


# ---------------------------------------------------------------------------
# Tests: Prompt building
# ---------------------------------------------------------------------------

class TestBuildVerificationPrompt:
    def test_basic_prompt(self):
        prompt = _build_verification_prompt(
            kalshi_text="Will Bitcoin reach $100k?",
            polymarket_text="Will BTC hit $100,000?",
        )
        assert "Will Bitcoin reach $100k?" in prompt
        assert "Will BTC hit $100,000?" in prompt
        assert "Market A (Kalshi)" in prompt
        assert "Market B (Polymarket)" in prompt

    def test_with_descriptions(self):
        prompt = _build_verification_prompt(
            kalshi_text="Will X happen?",
            polymarket_text="Will X occur?",
            kalshi_extra="Resolution: by end of 2026",
            polymarket_extra="Resolves YES if X happens before Jan 2027",
        )
        assert "Resolution: by end of 2026" in prompt
        assert "Resolves YES if X happens before Jan 2027" in prompt

    def test_empty_extra_not_in_prompt(self):
        prompt = _build_verification_prompt(
            kalshi_text="Will X happen?",
            polymarket_text="Will X occur?",
        )
        assert "Description:" not in prompt


# ---------------------------------------------------------------------------
# Tests: LLM response parsing
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    def test_parse_valid_json(self):
        response = '{"same_question": true, "confidence": 0.95, "reason": "Same market"}'
        same, confidence, reason = _parse_llm_response(response)
        assert same is True
        assert confidence == 0.95
        assert reason == "Same market"

    def test_parse_false_response(self):
        response = '{"same_question": false, "confidence": 0.90, "reason": "Different thresholds"}'
        same, confidence, reason = _parse_llm_response(response)
        assert same is False
        assert confidence == 0.90
        assert reason == "Different thresholds"

    def test_parse_json_embedded_in_text(self):
        response = 'Here is my analysis:\n{"same_question": true, "confidence": 0.85, "reason": "Both ask about GDP"}\nLet me explain...'
        same, confidence, reason = _parse_llm_response(response)
        assert same is True
        assert confidence == 0.85

    def test_parse_invalid_response(self):
        response = "This is not valid JSON at all"
        same, confidence, reason = _parse_llm_response(response)
        assert same is False
        assert confidence == 0.0
        assert "failed to parse" in reason

    def test_confidence_clamped(self):
        response = '{"same_question": true, "confidence": 1.5, "reason": "test"}'
        _, confidence, _ = _parse_llm_response(response)
        assert confidence == 1.0

        response = '{"same_question": true, "confidence": -0.5, "reason": "test"}'
        _, confidence, _ = _parse_llm_response(response)
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Tests: Heuristic rejection
# ---------------------------------------------------------------------------

class TestHeuristicRejection:
    def test_run_vs_win_mismatch(self):
        """Should catch the KX2028RRUN bug."""
        reason = _heuristic_rejection(
            kalshi_text="Will Ted Cruz run for president?",
            polymarket_text="Will Ted Cruz win the Republican nomination?",
            settings=VerificationSettings(),
        )
        assert reason == "run_vs_win_mismatch"

    def test_both_say_win(self):
        """Should NOT reject when both ask about winning."""
        reason = _heuristic_rejection(
            kalshi_text="Will Ted Cruz win the Republican nomination?",
            polymarket_text="Will Ted Cruz win the GOP primary?",
            settings=VerificationSettings(),
        )
        assert reason is None

    def test_both_say_run(self):
        """Should NOT reject when both ask about running."""
        reason = _heuristic_rejection(
            kalshi_text="Will Ted Cruz run for president?",
            polymarket_text="Will Ted Cruz run in 2028?",
            settings=VerificationSettings(),
        )
        assert reason is None

    def test_missing_text(self):
        reason = _heuristic_rejection("", "something", VerificationSettings())
        assert reason == "missing_market_text"

        reason = _heuristic_rejection("something", "", VerificationSettings())
        assert reason == "missing_market_text"

    def test_normal_pair_passes(self):
        reason = _heuristic_rejection(
            kalshi_text="Will the Fed cut rates in March 2026?",
            polymarket_text="Will the Federal Reserve reduce interest rates in March 2026?",
            settings=VerificationSettings(),
        )
        assert reason is None


# ---------------------------------------------------------------------------
# Tests: Mock LLM
# ---------------------------------------------------------------------------

class TestMockLLM:
    def test_matching_markets(self):
        settings = VerificationSettings(dry_run=True)
        prompt = _build_verification_prompt(
            kalshi_text="Will Bitcoin reach $100k by December 2026?",
            polymarket_text="Will Bitcoin reach $100,000 by December 2026?",
        )
        response = _call_mock(prompt, settings)
        data = json.loads(response)
        assert data["same_question"] is True
        assert data["confidence"] > 0.5

    def test_different_markets(self):
        settings = VerificationSettings(dry_run=True)
        prompt = _build_verification_prompt(
            kalshi_text="Will GDP grow above 3%?",
            polymarket_text="Will inflation fall below 2%?",
        )
        response = _call_mock(prompt, settings)
        data = json.loads(response)
        assert data["same_question"] is False

    def test_run_vs_win_detection(self):
        settings = VerificationSettings(dry_run=True)
        prompt = _build_verification_prompt(
            kalshi_text="Will Cruz run for president in 2028?",
            polymarket_text="Will Cruz win the 2028 presidential election?",
        )
        response = _call_mock(prompt, settings)
        data = json.loads(response)
        assert data["same_question"] is False
        assert "run" in data["reason"].lower() or "win" in data["reason"].lower()


# ---------------------------------------------------------------------------
# Tests: Edge sanity check
# ---------------------------------------------------------------------------

class TestEdgeSanityCheck:
    def test_normal_edge_passes(self):
        assert edge_sanity_check(0.05, max_edge=0.30) is True
        assert edge_sanity_check(0.25, max_edge=0.30) is True
        assert edge_sanity_check(-0.10, max_edge=0.30) is True

    def test_extreme_edge_fails(self):
        assert edge_sanity_check(0.50, max_edge=0.30) is False
        assert edge_sanity_check(0.65, max_edge=0.30) is False
        assert edge_sanity_check(-0.65, max_edge=0.30) is False

    def test_boundary(self):
        assert edge_sanity_check(0.30, max_edge=0.30) is True
        assert edge_sanity_check(0.31, max_edge=0.30) is False


# ---------------------------------------------------------------------------
# Tests: Full pipeline (dry run)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_dry_run_pipeline(self, tmp_path: Path):
        """Full pipeline with mock LLM."""
        # Create candidate CSV.
        candidates_csv = tmp_path / "candidates.csv"
        with open(candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group_id", "kalshi_market_id", "polymarket_market_id"])
            writer.writeheader()
            writer.writerow({
                "group_id": "will_bitcoin_reach_100k",
                "kalshi_market_id": "KXBTC-100K",
                "polymarket_market_id": "0xabc123",
            })
            writer.writerow({
                "group_id": "will_cruz_run_for_president",
                "kalshi_market_id": "KXCRUZ-RUN",
                "polymarket_market_id": "0xdef456",
            })

        # Create mock market data.
        market_data = {
            "KXBTC-100K": {
                "title": "Will Bitcoin reach $100,000 by end of 2026?",
                "venue": "kalshi",
            },
            "0xabc123": {
                "question": "Will Bitcoin reach $100,000 by December 2026?",
                "venue": "polymarket",
            },
            "KXCRUZ-RUN": {
                "title": "Will Ted Cruz run for president in 2028?",
                "venue": "kalshi",
            },
            "0xdef456": {
                "question": "Will Ted Cruz win the 2028 Republican presidential nomination?",
                "venue": "polymarket",
            },
        }
        market_data_json = tmp_path / "market_data.json"
        with open(market_data_json, "w") as f:
            json.dump(market_data, f)

        output_csv = tmp_path / "verified.csv"

        settings = VerificationSettings(dry_run=True, min_confidence=0.50)
        results, diagnostics = verify_mappings(
            candidates_path=str(candidates_csv),
            output_path=str(output_csv),
            market_data_path=str(market_data_json),
            settings=settings,
        )

        assert diagnostics.total_candidates == 2
        # Bitcoin pair should be verified, Cruz pair should be rejected.
        assert diagnostics.verified_count >= 1
        assert diagnostics.rejected_count >= 1

        # Check that the output CSV only has verified pairs.
        assert output_csv.exists()
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == diagnostics.verified_count

        # Check audit JSON was created.
        audit_path = tmp_path / "verified.audit.json"
        assert audit_path.exists()
        with open(audit_path) as f:
            audit = json.load(f)
        assert "diagnostics" in audit
        assert "results" in audit
        assert len(audit["results"]) == 2

    def test_pipeline_missing_market_data(self, tmp_path: Path):
        """Pipeline should work without market data (uses group_id as text)."""
        candidates_csv = tmp_path / "candidates.csv"
        with open(candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group_id", "kalshi_market_id", "polymarket_market_id"])
            writer.writeheader()
            writer.writerow({
                "group_id": "will_bitcoin_reach_100k",
                "kalshi_market_id": "K1",
                "polymarket_market_id": "P1",
            })

        output_csv = tmp_path / "verified.csv"
        settings = VerificationSettings(dry_run=True)
        results, diagnostics = verify_mappings(
            candidates_path=str(candidates_csv),
            output_path=str(output_csv),
            settings=settings,
        )
        assert diagnostics.total_candidates == 1

    def test_pipeline_empty_candidates(self, tmp_path: Path):
        """Pipeline should handle empty candidate file."""
        candidates_csv = tmp_path / "candidates.csv"
        with open(candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group_id", "kalshi_market_id", "polymarket_market_id"])
            writer.writeheader()

        output_csv = tmp_path / "verified.csv"
        settings = VerificationSettings(dry_run=True)
        results, diagnostics = verify_mappings(
            candidates_path=str(candidates_csv),
            output_path=str(output_csv),
            settings=settings,
        )
        assert diagnostics.total_candidates == 0
        assert diagnostics.verified_count == 0
        assert output_csv.exists()

    def test_pipeline_skips_missing_polymarket(self, tmp_path: Path):
        """Pairs without polymarket_market_id should be skipped."""
        candidates_csv = tmp_path / "candidates.csv"
        with open(candidates_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group_id", "kalshi_market_id", "polymarket_market_id"])
            writer.writeheader()
            writer.writerow({
                "group_id": "test_pair",
                "kalshi_market_id": "K1",
                "polymarket_market_id": "",
            })

        output_csv = tmp_path / "verified.csv"
        settings = VerificationSettings(dry_run=True)
        results, diagnostics = verify_mappings(
            candidates_path=str(candidates_csv),
            output_path=str(output_csv),
            settings=settings,
        )
        assert diagnostics.skipped_no_data == 1
        assert diagnostics.verified_count == 0


# ---------------------------------------------------------------------------
# Tests: VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_verified_result(self):
        r = VerificationResult(
            group_id="test",
            kalshi_market_id="K1",
            polymarket_market_id="P1",
            forecastex_market_id="",
            verified=True,
            confidence=0.95,
            rejection_reason="",
            llm_explanation="Same question",
            kalshi_text="Test K",
            polymarket_text="Test P",
        )
        assert r.verified
        assert r.confidence == 0.95

    def test_rejected_result(self):
        r = VerificationResult(
            group_id="test",
            kalshi_market_id="K1",
            polymarket_market_id="P1",
            forecastex_market_id="",
            verified=False,
            confidence=0.90,
            rejection_reason="run_vs_win_mismatch",
            llm_explanation="Different questions",
            kalshi_text="Will X run?",
            polymarket_text="Will X win?",
        )
        assert not r.verified
        assert r.rejection_reason == "run_vs_win_mismatch"


# ---------------------------------------------------------------------------
# Tests: VerificationSettings
# ---------------------------------------------------------------------------

class TestVerificationSettings:
    def test_defaults(self):
        s = VerificationSettings()
        assert s.max_edge_sanity == 0.30
        assert s.min_confidence == 0.70
        assert s.dry_run is False
        assert s.llm_provider == "anthropic"

    def test_dry_run(self):
        s = VerificationSettings(dry_run=True)
        assert s.dry_run is True
