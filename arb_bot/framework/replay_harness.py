"""Replay and deterministic simulation harness (Phase 5N).

Record raw stream/poll events and decisions, then replay them
deterministically for A/B comparison, debugging, and regression
testing.

Usage::

    # Recording phase.
    recorder = EventRecorder(config)
    recorder.record_event("stream", {"venue": "kalshi", "price": 0.55}, ts=100.0)
    recorder.record_decision("skip", {"reason": "stale"}, ts=100.1)
    recorder.save(path="/tmp/recording.json")

    # Replay phase.
    recording = EventRecorder.load(path="/tmp/recording.json")
    harness = ReplayHarness(config)
    harness.load_recording(recording)
    result = harness.replay(handler_fn)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventKind(str, Enum):
    """Kind of recorded event."""

    STREAM = "stream"
    POLL = "poll"
    DECISION = "decision"
    EXECUTION = "execution"
    ERROR = "error"
    MARKER = "marker"


# ---------------------------------------------------------------------------
# Recorded event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedEvent:
    """A single recorded event."""

    kind: EventKind
    timestamp: float
    data: Dict[str, Any]
    sequence: int  # Global sequence number.
    source: str = ""  # e.g., venue name, lane name.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecorderConfig:
    """Configuration for event recorder.

    Parameters
    ----------
    max_events:
        Maximum events to record. Default 100000.
    include_decisions:
        Whether to record decision events. Default True.
    include_executions:
        Whether to record execution events. Default True.
    """

    max_events: int = 100000
    include_decisions: bool = True
    include_executions: bool = True


@dataclass(frozen=True)
class ReplayHarnessConfig:
    """Configuration for replay harness.

    Parameters
    ----------
    speed_multiplier:
        Replay speed (1.0 = real-time, 0.0 = instant). Default 0.0.
    stop_on_error:
        Stop replay on handler error. Default False.
    max_events:
        Maximum events to replay (0 = unlimited). Default 0.
    record_outputs:
        Record handler outputs for comparison. Default True.
    """

    speed_multiplier: float = 0.0
    stop_on_error: bool = False
    max_events: int = 0
    record_outputs: bool = True


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    """A captured recording of events."""

    events: List[RecordedEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def duration(self) -> float:
        if len(self.events) < 2:
            return 0.0
        return self.events[-1].timestamp - self.events[0].timestamp

    def events_of_kind(self, kind: EventKind) -> List[RecordedEvent]:
        return [e for e in self.events if e.kind == kind]

    def time_range(self) -> Tuple[float, float]:
        if not self.events:
            return (0.0, 0.0)
        return (self.events[0].timestamp, self.events[-1].timestamp)


# ---------------------------------------------------------------------------
# Replay output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayOutput:
    """Output from processing a single event during replay."""

    event: RecordedEvent
    handler_result: Any
    error: str = ""


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """Result from a complete replay run."""

    events_processed: int
    events_skipped: int
    errors: int
    outputs: Tuple[ReplayOutput, ...]
    elapsed_seconds: float
    stopped_early: bool = False


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Result from comparing two replay runs."""

    total_events: int
    matching_outputs: int
    differing_outputs: int
    differences: Tuple[Dict[str, Any], ...]


# ---------------------------------------------------------------------------
# Event recorder
# ---------------------------------------------------------------------------


class EventRecorder:
    """Records events for later replay.

    Captures stream updates, poll results, decisions, and execution
    events with timestamps and sequence numbers for deterministic replay.
    """

    def __init__(self, config: EventRecorderConfig | None = None) -> None:
        self._config = config or EventRecorderConfig()
        self._events: List[RecordedEvent] = []
        self._metadata: Dict[str, Any] = {}
        self._sequence: int = 0
        self._recording: bool = False
        self._created_at: float = 0.0

    @property
    def config(self) -> EventRecorderConfig:
        return self._config

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, now: float | None = None) -> None:
        """Start recording."""
        if now is None:
            now = time.time()
        self._recording = True
        self._created_at = now

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False

    def record_event(
        self,
        kind: str,
        data: Dict[str, Any],
        ts: float | None = None,
        source: str = "",
    ) -> bool:
        """Record a raw event.

        Returns False if max events exceeded or not recording.
        """
        if not self._recording:
            return False
        if len(self._events) >= self._config.max_events:
            return False

        # Filter by config.
        event_kind = EventKind(kind)
        if event_kind == EventKind.DECISION and not self._config.include_decisions:
            return False
        if event_kind == EventKind.EXECUTION and not self._config.include_executions:
            return False

        if ts is None:
            ts = time.time()

        self._events.append(RecordedEvent(
            kind=event_kind,
            timestamp=ts,
            data=dict(data),
            sequence=self._sequence,
            source=source,
        ))
        self._sequence += 1
        return True

    def record_decision(
        self,
        action: str,
        data: Dict[str, Any] | None = None,
        ts: float | None = None,
        source: str = "",
    ) -> bool:
        """Record a decision event (convenience method)."""
        combined = dict(data or {})
        combined["action"] = action
        return self.record_event("decision", combined, ts=ts, source=source)

    def event_count(self) -> int:
        """Number of recorded events."""
        return len(self._events)

    def get_recording(self) -> Recording:
        """Get the current recording."""
        return Recording(
            events=list(self._events),
            metadata=dict(self._metadata),
            created_at=self._created_at,
        )

    def set_metadata(self, key: str, value: Any) -> None:
        """Set recording metadata."""
        self._metadata[key] = value

    def save(self, path: str) -> None:
        """Save the recording to disk as JSON."""
        recording = self.get_recording()
        data = {
            "created_at": recording.created_at,
            "metadata": recording.metadata,
            "events": [
                {
                    "kind": e.kind.value,
                    "timestamp": e.timestamp,
                    "data": e.data,
                    "sequence": e.sequence,
                    "source": e.source,
                }
                for e in recording.events
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load(path: str) -> Recording | None:
        """Load a recording from disk."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        events = [
            RecordedEvent(
                kind=EventKind(e["kind"]),
                timestamp=e["timestamp"],
                data=e["data"],
                sequence=e["sequence"],
                source=e.get("source", ""),
            )
            for e in data.get("events", [])
        ]

        return Recording(
            events=events,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", 0.0),
        )

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
        self._metadata.clear()
        self._sequence = 0
        self._recording = False
        self._created_at = 0.0


# ---------------------------------------------------------------------------
# Replay harness
# ---------------------------------------------------------------------------


class ReplayHarness:
    """Deterministic replay harness for recorded events.

    Feeds recorded events through a handler function in order,
    collecting outputs for comparison and analysis.
    """

    def __init__(self, config: ReplayHarnessConfig | None = None) -> None:
        self._config = config or ReplayHarnessConfig()
        self._recording: Recording | None = None

    @property
    def config(self) -> ReplayHarnessConfig:
        return self._config

    def load_recording(self, recording: Recording) -> None:
        """Load a recording for replay."""
        self._recording = recording

    def has_recording(self) -> bool:
        """Whether a recording is loaded."""
        return self._recording is not None

    def replay(
        self,
        handler: Callable[[RecordedEvent], Any],
        event_filter: Callable[[RecordedEvent], bool] | None = None,
    ) -> ReplayResult:
        """Replay the loaded recording through a handler.

        Parameters
        ----------
        handler:
            Function called for each event. Receives RecordedEvent,
            returns arbitrary output.
        event_filter:
            Optional filter — only events where filter returns True
            are replayed. Default: replay all events.

        Returns
        -------
        ReplayResult with outputs and stats.
        """
        if self._recording is None:
            return ReplayResult(
                events_processed=0,
                events_skipped=0,
                errors=0,
                outputs=(),
                elapsed_seconds=0.0,
            )

        cfg = self._config
        outputs: List[ReplayOutput] = []
        processed = 0
        skipped = 0
        errors = 0
        stopped_early = False
        start_time = time.monotonic()

        for event in self._recording.events:
            # Check max events limit.
            if cfg.max_events > 0 and processed >= cfg.max_events:
                stopped_early = True
                break

            # Apply filter.
            if event_filter is not None and not event_filter(event):
                skipped += 1
                continue

            # Process event.
            try:
                result = handler(event)
                if cfg.record_outputs:
                    outputs.append(ReplayOutput(
                        event=event,
                        handler_result=result,
                    ))
                processed += 1
            except Exception as e:
                errors += 1
                if cfg.record_outputs:
                    outputs.append(ReplayOutput(
                        event=event,
                        handler_result=None,
                        error=str(e),
                    ))
                if cfg.stop_on_error:
                    stopped_early = True
                    break
                processed += 1

        elapsed = time.monotonic() - start_time

        return ReplayResult(
            events_processed=processed,
            events_skipped=skipped,
            errors=errors,
            outputs=tuple(outputs),
            elapsed_seconds=elapsed,
            stopped_early=stopped_early,
        )

    @staticmethod
    def compare_runs(
        run_a: ReplayResult,
        run_b: ReplayResult,
    ) -> ComparisonResult:
        """Compare outputs of two replay runs.

        Matches events by sequence number and compares handler results.
        """
        # Build lookup by sequence.
        a_by_seq: Dict[int, ReplayOutput] = {
            o.event.sequence: o for o in run_a.outputs
        }
        b_by_seq: Dict[int, ReplayOutput] = {
            o.event.sequence: o for o in run_b.outputs
        }

        all_seqs = sorted(set(a_by_seq.keys()) | set(b_by_seq.keys()))
        matching = 0
        differing = 0
        diffs: List[Dict[str, Any]] = []

        for seq in all_seqs:
            out_a = a_by_seq.get(seq)
            out_b = b_by_seq.get(seq)

            if out_a is None or out_b is None:
                differing += 1
                diffs.append({
                    "sequence": seq,
                    "reason": "missing_in_a" if out_a is None else "missing_in_b",
                })
            elif out_a.handler_result == out_b.handler_result:
                matching += 1
            else:
                differing += 1
                diffs.append({
                    "sequence": seq,
                    "reason": "output_mismatch",
                    "result_a": out_a.handler_result,
                    "result_b": out_b.handler_result,
                })

        return ComparisonResult(
            total_events=len(all_seqs),
            matching_outputs=matching,
            differing_outputs=differing,
            differences=tuple(diffs),
        )
