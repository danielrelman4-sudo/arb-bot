"""Tests for Phase 3F: Low-latency execution path optimization."""

from __future__ import annotations

import pytest

from arb_bot.framework.execution_pipeline import (
    ExecutionPipeline,
    ExecutionPipelineConfig,
    PipelineStage,
    PipelineStats,
    PipelineToken,
    PrebuiltRequest,
    StageStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline(**kw) -> ExecutionPipeline:
    return ExecutionPipeline(ExecutionPipelineConfig(**kw))


def _run_full_pipeline(
    pipeline: ExecutionPipeline,
    detect: float = 100.0,
    plan: float = 100.05,
    submit: float = 100.15,
    ack: float = 100.30,
) -> str:
    """Run a token through all stages and return its ID."""
    token_id = pipeline.start_detection(now=detect)
    pipeline.mark_plan_ready(token_id, now=plan)
    pipeline.mark_submitted(token_id, now=submit)
    pipeline.mark_acknowledged(token_id, now=ack)
    return token_id


# ---------------------------------------------------------------------------
# ExecutionPipelineConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = ExecutionPipelineConfig()
        assert cfg.max_inflight == 10
        assert cfg.detect_to_ack_slo_seconds == 0.5
        assert cfg.history_size == 500

    def test_custom(self) -> None:
        cfg = ExecutionPipelineConfig(max_inflight=5)
        assert cfg.max_inflight == 5


# ---------------------------------------------------------------------------
# Start detection
# ---------------------------------------------------------------------------


class TestStartDetection:
    def test_creates_token(self) -> None:
        p = _pipeline()
        token_id = p.start_detection(now=100.0)
        assert len(token_id) == 16
        assert p.inflight_count == 1

    def test_max_inflight(self) -> None:
        p = _pipeline(max_inflight=2)
        p.start_detection(now=100.0)
        p.start_detection(now=100.1)
        with pytest.raises(ValueError, match="max capacity"):
            p.start_detection(now=100.2)

    def test_can_accept(self) -> None:
        p = _pipeline(max_inflight=1)
        assert p.can_accept() is True
        p.start_detection(now=100.0)
        assert p.can_accept() is False

    def test_metadata(self) -> None:
        p = _pipeline()
        token_id = p.start_detection(now=100.0, venue="kalshi")
        token = p.get_token(token_id)
        assert token.metadata["venue"] == "kalshi"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


class TestPipelineStages:
    def test_mark_plan_ready(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        p.mark_plan_ready(tid, now=100.05)
        token = p.get_token(tid)
        assert token.stage == PipelineStage.PLANNING
        assert token.detect_to_plan == pytest.approx(0.05)

    def test_mark_submitted(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        p.mark_plan_ready(tid, now=100.05)
        p.mark_submitted(tid, now=100.15)
        token = p.get_token(tid)
        assert token.stage == PipelineStage.SUBMISSION
        assert token.plan_to_submit == pytest.approx(0.10)

    def test_mark_acknowledged(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        p.mark_plan_ready(tid, now=100.05)
        p.mark_submitted(tid, now=100.15)
        p.mark_acknowledged(tid, now=100.30)
        assert p.inflight_count == 0
        assert p.completed_count == 1
        token = p.get_token(tid)
        assert token.stage == PipelineStage.ACKNOWLEDGED
        assert token.submit_to_ack == pytest.approx(0.15)
        assert token.detect_to_ack == pytest.approx(0.30)

    def test_mark_failed(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        p.mark_failed(tid, reason="exchange_error")
        assert p.inflight_count == 0
        token = p.get_token(tid)
        assert token.stage == PipelineStage.FAILED
        assert token.metadata["fail_reason"] == "exchange_error"

    def test_unknown_token_raises(self) -> None:
        p = _pipeline()
        with pytest.raises(KeyError):
            p.mark_plan_ready("nonexistent")


# ---------------------------------------------------------------------------
# PipelineToken properties
# ---------------------------------------------------------------------------


class TestTokenProperties:
    def test_incomplete_latencies(self) -> None:
        token = PipelineToken(token_id="t1", detect_time=100.0)
        assert token.detect_to_plan is None
        assert token.plan_to_submit is None
        assert token.submit_to_ack is None
        assert token.detect_to_ack is None

    def test_is_complete(self) -> None:
        token = PipelineToken(
            token_id="t1", detect_time=100.0,
            stage=PipelineStage.DETECTION,
        )
        assert token.is_complete is False

        token.stage = PipelineStage.ACKNOWLEDGED
        assert token.is_complete is True

        token.stage = PipelineStage.FAILED
        assert token.is_complete is True


# ---------------------------------------------------------------------------
# Pre-built request components
# ---------------------------------------------------------------------------


class TestPrebuild:
    def test_attach_and_retrieve(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        req = PrebuiltRequest(
            venue="kalshi", market_id="M1",
            side="yes", price=0.55, contracts=10,
            headers={"Authorization": "Bearer xxx"},
            built_at=100.0,
        )
        p.prebuild_request(tid, req)
        retrieved = p.get_prebuilt(tid, "kalshi:M1")
        assert retrieved is not None
        assert retrieved.price == 0.55
        assert retrieved.contracts == 10

    def test_missing_prebuilt(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        assert p.get_prebuilt(tid, "kalshi:M1") is None


# ---------------------------------------------------------------------------
# Pipeline stats — empty
# ---------------------------------------------------------------------------


class TestStatsEmpty:
    def test_no_completed(self) -> None:
        p = _pipeline()
        stats = p.pipeline_stats()
        assert stats.total_completed == 0
        assert stats.total_failed == 0
        assert stats.detect_to_ack_p95 == 0.0

    def test_only_inflight(self) -> None:
        p = _pipeline()
        p.start_detection(now=100.0)
        stats = p.pipeline_stats()
        assert stats.inflight_count == 1
        assert stats.total_completed == 0


# ---------------------------------------------------------------------------
# Pipeline stats — with data
# ---------------------------------------------------------------------------


class TestStatsWithData:
    def test_basic_stats(self) -> None:
        p = _pipeline()
        for i in range(10):
            _run_full_pipeline(
                p,
                detect=float(i * 10),
                plan=float(i * 10 + 0.05),
                submit=float(i * 10 + 0.15),
                ack=float(i * 10 + 0.30),
            )
        stats = p.pipeline_stats()
        assert stats.total_completed == 10
        assert "detect_to_plan" in stats.stages
        assert "plan_to_submit" in stats.stages
        assert "submit_to_ack" in stats.stages
        assert "detect_to_ack" in stats.stages

    def test_stage_latencies(self) -> None:
        p = _pipeline()
        _run_full_pipeline(p, detect=100.0, plan=100.05, submit=100.15, ack=100.30)
        stats = p.pipeline_stats()
        assert stats.stages["detect_to_plan"].mean == pytest.approx(0.05)
        assert stats.stages["plan_to_submit"].mean == pytest.approx(0.10)
        assert stats.stages["submit_to_ack"].mean == pytest.approx(0.15)
        assert stats.stages["detect_to_ack"].mean == pytest.approx(0.30)

    def test_slo_breach_rate(self) -> None:
        p = _pipeline(detect_to_ack_slo_seconds=0.4)
        # 5 within SLO (0.30).
        for i in range(5):
            _run_full_pipeline(p, detect=float(i*10), plan=float(i*10+.05),
                              submit=float(i*10+.15), ack=float(i*10+.30))
        # 5 breach SLO (0.50).
        for i in range(5, 10):
            _run_full_pipeline(p, detect=float(i*10), plan=float(i*10+.10),
                              submit=float(i*10+.25), ack=float(i*10+.50))
        stats = p.pipeline_stats()
        assert stats.slo_breach_rate == pytest.approx(0.5)

    def test_failed_count(self) -> None:
        p = _pipeline()
        tid1 = p.start_detection(now=100.0)
        p.mark_failed(tid1)
        _run_full_pipeline(p)
        stats = p.pipeline_stats()
        assert stats.total_completed == 1
        assert stats.total_failed == 1


# ---------------------------------------------------------------------------
# History trimming
# ---------------------------------------------------------------------------


class TestHistoryTrimming:
    def test_trims_to_history_size(self) -> None:
        p = _pipeline(history_size=5)
        for i in range(10):
            _run_full_pipeline(p, detect=float(i*10), plan=float(i*10+.05),
                              submit=float(i*10+.15), ack=float(i*10+.30))
        assert p.completed_count == 5


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        p = _pipeline()
        p.start_detection(now=100.0)
        _run_full_pipeline(p)
        p.clear()
        assert p.inflight_count == 0
        assert p.completed_count == 0


# ---------------------------------------------------------------------------
# Get token
# ---------------------------------------------------------------------------


class TestGetToken:
    def test_inflight(self) -> None:
        p = _pipeline()
        tid = p.start_detection(now=100.0)
        assert p.get_token(tid) is not None

    def test_completed(self) -> None:
        p = _pipeline()
        tid = _run_full_pipeline(p)
        assert p.get_token(tid) is not None
        assert p.get_token(tid).stage == PipelineStage.ACKNOWLEDGED

    def test_unknown(self) -> None:
        p = _pipeline()
        assert p.get_token("nonexistent") is None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Simulate a realistic execution pipeline."""
        cfg = ExecutionPipelineConfig(
            max_inflight=5,
            detect_to_ack_slo_seconds=0.5,
            detect_to_plan_slo_seconds=0.1,
            plan_to_submit_slo_seconds=0.2,
            submit_to_ack_slo_seconds=0.3,
        )
        p = ExecutionPipeline(cfg)

        # Detection phase.
        tid = p.start_detection(now=100.0, venue="kalshi", market="M1")
        assert p.inflight_count == 1

        # Prebuild request.
        req = PrebuiltRequest(
            venue="kalshi", market_id="M1", side="yes",
            price=0.55, contracts=10, built_at=100.0,
        )
        p.prebuild_request(tid, req)

        # Plan ready.
        p.mark_plan_ready(tid, now=100.05, edge=0.03)

        # Retrieve prebuilt for submission.
        prebuilt = p.get_prebuilt(tid, "kalshi:M1")
        assert prebuilt is not None
        assert prebuilt.price == 0.55

        # Submit.
        p.mark_submitted(tid, now=100.15)

        # Ack.
        p.mark_acknowledged(tid, now=100.30)
        assert p.inflight_count == 0

        # Stats.
        stats = p.pipeline_stats()
        assert stats.total_completed == 1
        assert stats.detect_to_ack_p95 == pytest.approx(0.30)
        assert stats.slo_breach_rate == pytest.approx(0.0)

    def test_config_property(self) -> None:
        cfg = ExecutionPipelineConfig(max_inflight=3)
        p = ExecutionPipeline(cfg)
        assert p.config.max_inflight == 3
