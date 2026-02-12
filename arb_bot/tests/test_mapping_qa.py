"""Tests for Phase 2F: Market mapping QA pipeline."""

from __future__ import annotations

import pytest

from arb_bot.mapping_qa import (
    IssueSeverity,
    MappingEntry,
    MappingIssue,
    MappingQAConfig,
    MappingQAPipeline,
    MappingQAReport,
    _jaccard_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _m(
    group_id: str = "g1",
    kalshi_id: str = "K1",
    polymarket_id: str = "P1",
) -> MappingEntry:
    return MappingEntry(group_id=group_id, kalshi_id=kalshi_id, polymarket_id=polymarket_id)


# ---------------------------------------------------------------------------
# MappingQAConfig
# ---------------------------------------------------------------------------


class TestMappingQAConfig:
    def test_defaults(self) -> None:
        cfg = MappingQAConfig()
        assert cfg.check_cardinality is True
        assert cfg.check_stale_links is True
        assert cfg.check_duplicate_group_ids is True
        assert cfg.check_empty_fields is True
        assert cfg.min_text_similarity == 0.0


# ---------------------------------------------------------------------------
# MappingQAReport
# ---------------------------------------------------------------------------


class TestMappingQAReport:
    def test_no_issues(self) -> None:
        report = MappingQAReport(total_mappings=5)
        assert report.has_errors is False
        assert report.has_warnings is False
        assert "OK" in report.summary

    def test_with_errors(self) -> None:
        report = MappingQAReport(
            total_mappings=5,
            total_issues=1,
            issues_by_severity={"error": 1},
        )
        assert report.has_errors is True
        assert "1 error(s)" in report.summary

    def test_with_warnings(self) -> None:
        report = MappingQAReport(
            total_mappings=5,
            total_issues=1,
            issues_by_severity={"warning": 1},
        )
        assert report.has_warnings is True


# ---------------------------------------------------------------------------
# MappingIssue
# ---------------------------------------------------------------------------


class TestMappingIssue:
    def test_fields(self) -> None:
        issue = MappingIssue(
            group_id="g1",
            check_name="empty_kalshi_id",
            severity=IssueSeverity.ERROR,
            message="test",
        )
        assert issue.group_id == "g1"
        assert issue.severity == IssueSeverity.ERROR


# ---------------------------------------------------------------------------
# Empty fields check
# ---------------------------------------------------------------------------


class TestEmptyFields:
    def test_all_valid(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m()])
        issues = [i for i in report.issues if "empty" in i.check_name]
        assert len(issues) == 0

    def test_empty_group_id(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(group_id="")])
        assert report.has_errors is True
        assert any(i.check_name == "empty_group_id" for i in report.issues)

    def test_empty_kalshi_id(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(kalshi_id="")])
        assert any(i.check_name == "empty_kalshi_id" for i in report.issues)

    def test_empty_polymarket_id(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(polymarket_id="")])
        assert any(i.check_name == "empty_polymarket_id" for i in report.issues)

    def test_disabled(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(check_empty_fields=False))
        report = pipeline.run([_m(kalshi_id="")])
        assert report.total_issues == 0


# ---------------------------------------------------------------------------
# Duplicate group IDs
# ---------------------------------------------------------------------------


class TestDuplicateGroupIds:
    def test_no_duplicates(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(group_id="g1"), _m(group_id="g2", kalshi_id="K2", polymarket_id="P2")])
        dupes = [i for i in report.issues if i.check_name == "duplicate_group_id"]
        assert len(dupes) == 0

    def test_duplicate_detected(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([
            _m(group_id="g1"),
            _m(group_id="g1", kalshi_id="K2", polymarket_id="P2"),
        ])
        dupes = [i for i in report.issues if i.check_name == "duplicate_group_id"]
        assert len(dupes) == 1
        assert "2 times" in dupes[0].message

    def test_disabled(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(check_duplicate_group_ids=False))
        report = pipeline.run([_m(group_id="g1"), _m(group_id="g1", kalshi_id="K2", polymarket_id="P2")])
        dupes = [i for i in report.issues if i.check_name == "duplicate_group_id"]
        assert len(dupes) == 0


# ---------------------------------------------------------------------------
# Cardinality checks
# ---------------------------------------------------------------------------


class TestCardinality:
    def test_one_to_one_passes(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([
            _m(group_id="g1", kalshi_id="K1", polymarket_id="P1"),
            _m(group_id="g2", kalshi_id="K2", polymarket_id="P2"),
        ])
        card_issues = [i for i in report.issues if "cardinality" in i.check_name or "one_to_many" in i.check_name or "many_to_one" in i.check_name]
        assert len(card_issues) == 0

    def test_one_to_many_kalshi(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([
            _m(group_id="g1", kalshi_id="K1", polymarket_id="P1"),
            _m(group_id="g2", kalshi_id="K1", polymarket_id="P2"),  # Same K1
        ])
        issues = [i for i in report.issues if i.check_name == "one_to_many_kalshi"]
        assert len(issues) == 1
        assert "K1" in issues[0].message

    def test_many_to_one_polymarket(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([
            _m(group_id="g1", kalshi_id="K1", polymarket_id="P1"),
            _m(group_id="g2", kalshi_id="K2", polymarket_id="P1"),  # Same P1
        ])
        issues = [i for i in report.issues if i.check_name == "many_to_one_polymarket"]
        assert len(issues) == 1
        assert "P1" in issues[0].message

    def test_disabled(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(check_cardinality=False))
        report = pipeline.run([
            _m(group_id="g1", kalshi_id="K1", polymarket_id="P1"),
            _m(group_id="g2", kalshi_id="K1", polymarket_id="P2"),
        ])
        card_issues = [i for i in report.issues if "one_to_many" in i.check_name]
        assert len(card_issues) == 0


# ---------------------------------------------------------------------------
# Stale links
# ---------------------------------------------------------------------------


class TestStaleLinks:
    def test_all_live(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run(
            [_m(kalshi_id="K1", polymarket_id="P1")],
            live_kalshi_ids={"K1"},
            live_polymarket_ids={"P1"},
        )
        stale = [i for i in report.issues if "stale" in i.check_name]
        assert len(stale) == 0

    def test_stale_kalshi(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run(
            [_m(kalshi_id="K_OLD", polymarket_id="P1")],
            live_kalshi_ids={"K1", "K2"},
            live_polymarket_ids={"P1"},
        )
        stale = [i for i in report.issues if i.check_name == "stale_kalshi"]
        assert len(stale) == 1
        assert "K_OLD" in stale[0].message

    def test_stale_polymarket(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run(
            [_m(kalshi_id="K1", polymarket_id="P_OLD")],
            live_kalshi_ids={"K1"},
            live_polymarket_ids={"P1", "P2"},
        )
        stale = [i for i in report.issues if i.check_name == "stale_polymarket"]
        assert len(stale) == 1

    def test_no_live_data_skips_check(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(kalshi_id="K_OLD")])
        stale = [i for i in report.issues if "stale" in i.check_name]
        assert len(stale) == 0

    def test_disabled(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(check_stale_links=False))
        report = pipeline.run(
            [_m(kalshi_id="K_OLD")],
            live_kalshi_ids={"K1"},
            live_polymarket_ids={"P1"},
        )
        stale = [i for i in report.issues if "stale" in i.check_name]
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def test_similar_texts_pass(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(min_text_similarity=0.3))
        report = pipeline.run(
            [_m(kalshi_id="K1", polymarket_id="P1")],
            market_texts={
                "K1": "Will Bitcoin reach 100k by end of 2025",
                "P1": "Will Bitcoin reach 100k before 2026",
            },
        )
        sim_issues = [i for i in report.issues if i.check_name == "low_text_similarity"]
        assert len(sim_issues) == 0

    def test_dissimilar_texts_flagged(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(min_text_similarity=0.5))
        report = pipeline.run(
            [_m(kalshi_id="K1", polymarket_id="P1")],
            market_texts={
                "K1": "Will the temperature exceed 100 degrees",
                "P1": "Who will win the presidential election",
            },
        )
        sim_issues = [i for i in report.issues if i.check_name == "low_text_similarity"]
        assert len(sim_issues) == 1

    def test_missing_text_skipped(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(min_text_similarity=0.5))
        report = pipeline.run(
            [_m(kalshi_id="K1", polymarket_id="P1")],
            market_texts={"K1": "some text"},  # P1 missing
        )
        sim_issues = [i for i in report.issues if i.check_name == "low_text_similarity"]
        assert len(sim_issues) == 0

    def test_disabled_by_default(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run(
            [_m()],
            market_texts={"K1": "aaa", "P1": "zzz"},
        )
        sim_issues = [i for i in report.issues if i.check_name == "low_text_similarity"]
        assert len(sim_issues) == 0


# ---------------------------------------------------------------------------
# Jaccard similarity helper
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical(self) -> None:
        assert _jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _jaccard_similarity("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        sim = _jaccard_similarity("the quick brown fox", "the slow brown dog")
        # Tokens: {the, quick, brown, fox} ∩ {the, slow, brown, dog} = {the, brown}
        # Union = {the, quick, brown, fox, slow, dog} = 6
        assert sim == pytest.approx(2.0 / 6.0)

    def test_empty_string(self) -> None:
        assert _jaccard_similarity("", "hello") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Report summary
# ---------------------------------------------------------------------------


class TestReportSummary:
    def test_clean_summary(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m()])
        assert "OK" in report.summary

    def test_issue_summary(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(kalshi_id="")])
        assert "error" in report.summary

    def test_issues_by_check(self) -> None:
        pipeline = MappingQAPipeline()
        report = pipeline.run([_m(kalshi_id=""), _m(group_id="g2", kalshi_id="", polymarket_id="P2")])
        assert report.issues_by_check.get("empty_kalshi_id", 0) == 2


# ---------------------------------------------------------------------------
# Multiple mappings
# ---------------------------------------------------------------------------


class TestMultipleMappings:
    def test_mixed_issues(self) -> None:
        pipeline = MappingQAPipeline()
        mappings = [
            _m(group_id="g1", kalshi_id="K1", polymarket_id="P1"),  # Valid
            _m(group_id="g2", kalshi_id="", polymarket_id="P2"),    # Empty kalshi
            _m(group_id="g1", kalshi_id="K3", polymarket_id="P3"),  # Dup group_id
        ]
        report = pipeline.run(mappings)
        assert report.total_mappings == 3
        assert report.total_issues > 0
        assert report.has_errors is True  # Empty field
        assert report.has_warnings is True  # Dup group_id


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline(self) -> None:
        pipeline = MappingQAPipeline(MappingQAConfig(min_text_similarity=0.3))
        mappings = [
            _m(group_id="g1", kalshi_id="K1", polymarket_id="P1"),
            _m(group_id="g2", kalshi_id="K2", polymarket_id="P2"),
            _m(group_id="g3", kalshi_id="K3", polymarket_id="P3"),
        ]
        report = pipeline.run(
            mappings,
            live_kalshi_ids={"K1", "K2", "K3"},
            live_polymarket_ids={"P1", "P2", "P3"},
            market_texts={
                "K1": "Will BTC reach 100k",
                "P1": "Will Bitcoin reach 100k",
                "K2": "US presidential election 2028",
                "P2": "US presidential election 2028",
                "K3": "Temperature above 100F tomorrow",
                "P3": "Temperature above 100F tomorrow",
            },
        )
        assert report.total_mappings == 3
        # All valid, no stale, good similarity.
        assert report.total_issues == 0

    def test_config_property(self) -> None:
        cfg = MappingQAConfig(min_text_similarity=0.42)
        pipeline = MappingQAPipeline(cfg)
        assert pipeline.config.min_text_similarity == 0.42
