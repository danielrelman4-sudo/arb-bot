"""Low-latency execution path optimization (Phase 3F).

Separates detection from execution into distinct pipeline stages
with pre-built request components and timing instrumentation.

The pipeline measures detect-to-plan, plan-to-submit, and
submit-to-ack latencies independently, enabling targeted
optimization of the critical path.

Usage::

    pipeline = ExecutionPipeline(config)
    token = pipeline.start_detection()
    pipeline.mark_plan_ready(token, plan_data)
    pipeline.mark_submitted(token)
    pipeline.mark_acknowledged(token)
    stats = pipeline.stage_stats()
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


class PipelineStage(Enum):
    DETECTION = "detection"
    PLANNING = "planning"
    SUBMISSION = "submission"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPipelineConfig:
    """Configuration for the execution pipeline.

    Parameters
    ----------
    max_inflight:
        Maximum number of concurrent in-flight executions.
        Default 10.
    detect_to_ack_slo_seconds:
        Target for full pipeline latency. Default 0.5.
    detect_to_plan_slo_seconds:
        Target for detection-to-plan latency. Default 0.1.
    plan_to_submit_slo_seconds:
        Target for plan-to-submit latency. Default 0.2.
    submit_to_ack_slo_seconds:
        Target for submit-to-ack latency. Default 0.3.
    history_size:
        Number of completed pipeline runs to retain for stats.
        Default 500.
    enable_prebuild:
        Enable pre-building of request components. Default True.
    """

    max_inflight: int = 10
    detect_to_ack_slo_seconds: float = 0.5
    detect_to_plan_slo_seconds: float = 0.1
    plan_to_submit_slo_seconds: float = 0.2
    submit_to_ack_slo_seconds: float = 0.3
    history_size: int = 500
    enable_prebuild: bool = True


# ---------------------------------------------------------------------------
# Pipeline token
# ---------------------------------------------------------------------------


@dataclass
class PipelineToken:
    """Tracks timing for a single pipeline execution."""

    token_id: str
    detect_time: float
    plan_time: float | None = None
    submit_time: float | None = None
    ack_time: float | None = None
    stage: PipelineStage = PipelineStage.DETECTION
    metadata: Dict[str, Any] = field(default_factory=dict)
    prebuilt_components: Dict[str, Any] = field(default_factory=dict)

    @property
    def detect_to_plan(self) -> float | None:
        if self.plan_time is None:
            return None
        return self.plan_time - self.detect_time

    @property
    def plan_to_submit(self) -> float | None:
        if self.plan_time is None or self.submit_time is None:
            return None
        return self.submit_time - self.plan_time

    @property
    def submit_to_ack(self) -> float | None:
        if self.submit_time is None or self.ack_time is None:
            return None
        return self.ack_time - self.submit_time

    @property
    def detect_to_ack(self) -> float | None:
        if self.ack_time is None:
            return None
        return self.ack_time - self.detect_time

    @property
    def is_complete(self) -> bool:
        return self.stage in (PipelineStage.ACKNOWLEDGED, PipelineStage.FAILED)


# ---------------------------------------------------------------------------
# Stage stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageStats:
    """Statistics for a single pipeline stage."""

    stage_name: str
    sample_count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max_val: float
    slo_seconds: float
    slo_breaches: int
    breach_rate: float


@dataclass(frozen=True)
class PipelineStats:
    """Overall pipeline statistics."""

    total_completed: int
    total_failed: int
    inflight_count: int
    stages: Dict[str, StageStats]
    detect_to_ack_p95: float
    slo_breach_rate: float


# ---------------------------------------------------------------------------
# Pre-built request component
# ---------------------------------------------------------------------------


@dataclass
class PrebuiltRequest:
    """A pre-built request component ready for fast submission."""

    venue: str
    market_id: str
    side: str
    price: float
    contracts: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)
    built_at: float = 0.0


# ---------------------------------------------------------------------------
# Execution pipeline
# ---------------------------------------------------------------------------


class ExecutionPipeline:
    """Low-latency execution pipeline with stage timing.

    Separates detection, planning, submission, and acknowledgment
    into discrete stages with independent timing and SLO tracking.
    """

    def __init__(self, config: ExecutionPipelineConfig | None = None) -> None:
        self._config = config or ExecutionPipelineConfig()
        self._inflight: Dict[str, PipelineToken] = {}
        self._completed: list[PipelineToken] = []
        self._failed_count: int = 0

    @property
    def config(self) -> ExecutionPipelineConfig:
        return self._config

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def can_accept(self) -> bool:
        """Check if the pipeline can accept a new execution."""
        return len(self._inflight) < self._config.max_inflight

    def start_detection(self, now: float | None = None, **metadata: Any) -> str:
        """Start a new pipeline execution at the detection stage.

        Returns a token ID for tracking.
        Raises ValueError if at max inflight capacity.
        """
        if not self.can_accept():
            raise ValueError(
                f"Pipeline at max capacity ({self._config.max_inflight})"
            )

        token_id = uuid.uuid4().hex[:16]
        token = PipelineToken(
            token_id=token_id,
            detect_time=now or time.monotonic(),
            metadata=dict(metadata),
        )
        self._inflight[token_id] = token
        return token_id

    def mark_plan_ready(
        self,
        token_id: str,
        now: float | None = None,
        **plan_data: Any,
    ) -> None:
        """Mark that planning is complete for a token."""
        token = self._get_inflight(token_id)
        token.plan_time = now or time.monotonic()
        token.stage = PipelineStage.PLANNING
        token.metadata.update(plan_data)

    def prebuild_request(
        self,
        token_id: str,
        request: PrebuiltRequest,
    ) -> None:
        """Attach a pre-built request component to a token."""
        token = self._get_inflight(token_id)
        key = f"{request.venue}:{request.market_id}"
        token.prebuilt_components[key] = request

    def get_prebuilt(self, token_id: str, key: str) -> PrebuiltRequest | None:
        """Retrieve a pre-built request component."""
        token = self._get_inflight(token_id)
        return token.prebuilt_components.get(key)

    def mark_submitted(self, token_id: str, now: float | None = None) -> None:
        """Mark that the order has been submitted."""
        token = self._get_inflight(token_id)
        token.submit_time = now or time.monotonic()
        token.stage = PipelineStage.SUBMISSION

    def mark_acknowledged(self, token_id: str, now: float | None = None) -> None:
        """Mark that the exchange has acknowledged the order."""
        token = self._inflight.pop(token_id, None)
        if token is None:
            return
        token.ack_time = now or time.monotonic()
        token.stage = PipelineStage.ACKNOWLEDGED
        self._add_completed(token)

    def mark_failed(self, token_id: str, reason: str = "") -> None:
        """Mark a pipeline execution as failed."""
        token = self._inflight.pop(token_id, None)
        if token is None:
            return
        token.stage = PipelineStage.FAILED
        token.metadata["fail_reason"] = reason
        self._failed_count += 1
        self._add_completed(token)

    def get_token(self, token_id: str) -> PipelineToken | None:
        """Get a token (inflight or completed)."""
        if token_id in self._inflight:
            return self._inflight[token_id]
        for t in self._completed:
            if t.token_id == token_id:
                return t
        return None

    def pipeline_stats(self) -> PipelineStats:
        """Compute pipeline statistics from completed runs."""
        completed = [
            t for t in self._completed
            if t.stage == PipelineStage.ACKNOWLEDGED
        ]

        if not completed:
            return PipelineStats(
                total_completed=0,
                total_failed=self._failed_count,
                inflight_count=len(self._inflight),
                stages={},
                detect_to_ack_p95=0.0,
                slo_breach_rate=0.0,
            )

        cfg = self._config

        # Stage latencies.
        d2p = [t.detect_to_plan for t in completed if t.detect_to_plan is not None]
        p2s = [t.plan_to_submit for t in completed if t.plan_to_submit is not None]
        s2a = [t.submit_to_ack for t in completed if t.submit_to_ack is not None]
        d2a = [t.detect_to_ack for t in completed if t.detect_to_ack is not None]

        stages: Dict[str, StageStats] = {}
        if d2p:
            stages["detect_to_plan"] = _compute_stage_stats(
                "detect_to_plan", d2p, cfg.detect_to_plan_slo_seconds,
            )
        if p2s:
            stages["plan_to_submit"] = _compute_stage_stats(
                "plan_to_submit", p2s, cfg.plan_to_submit_slo_seconds,
            )
        if s2a:
            stages["submit_to_ack"] = _compute_stage_stats(
                "submit_to_ack", s2a, cfg.submit_to_ack_slo_seconds,
            )
        if d2a:
            stages["detect_to_ack"] = _compute_stage_stats(
                "detect_to_ack", d2a, cfg.detect_to_ack_slo_seconds,
            )

        d2a_p95 = _percentile(d2a, 95.0) if d2a else 0.0
        slo_breaches = sum(1 for v in d2a if v > cfg.detect_to_ack_slo_seconds)
        breach_rate = slo_breaches / len(d2a) if d2a else 0.0

        return PipelineStats(
            total_completed=len(completed),
            total_failed=self._failed_count,
            inflight_count=len(self._inflight),
            stages=stages,
            detect_to_ack_p95=d2a_p95,
            slo_breach_rate=breach_rate,
        )

    def clear(self) -> None:
        """Clear all state."""
        self._inflight.clear()
        self._completed.clear()
        self._failed_count = 0

    def _get_inflight(self, token_id: str) -> PipelineToken:
        token = self._inflight.get(token_id)
        if token is None:
            raise KeyError(f"Token {token_id} not found in inflight")
        return token

    def _add_completed(self, token: PipelineToken) -> None:
        self._completed.append(token)
        if len(self._completed) > self._config.history_size:
            self._completed = self._completed[-self._config.history_size:]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _compute_stage_stats(
    name: str,
    values: list[float],
    slo: float,
) -> StageStats:
    import statistics as stats_mod

    n = len(values)
    mean = stats_mod.mean(values)
    breaches = sum(1 for v in values if v > slo)

    return StageStats(
        stage_name=name,
        sample_count=n,
        mean=mean,
        p50=_percentile(values, 50.0),
        p95=_percentile(values, 95.0),
        p99=_percentile(values, 99.0),
        max_val=max(values),
        slo_seconds=slo,
        slo_breaches=breaches,
        breach_rate=breaches / n if n > 0 else 0.0,
    )


def _percentile(values: list[float], pct: float) -> float:
    import math

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = min(lower + 1, n - 1)
    fraction = rank - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])
