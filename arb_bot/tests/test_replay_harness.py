"""Tests for replay_harness module (Phase 5N)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest

from arb_bot.replay_harness import (
    ComparisonResult,
    EventKind,
    EventRecorder,
    EventRecorderConfig,
    Recording,
    RecordedEvent,
    ReplayHarness,
    ReplayHarnessConfig,
    ReplayOutput,
    ReplayResult,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestEventRecorderConfig:
    def test_defaults(self) -> None:
        cfg = EventRecorderConfig()
        assert cfg.max_events == 100000
        assert cfg.include_decisions is True
        assert cfg.include_executions is True

    def test_custom(self) -> None:
        cfg = EventRecorderConfig(
            max_events=500, include_decisions=False, include_executions=False
        )
        assert cfg.max_events == 500
        assert cfg.include_decisions is False
        assert cfg.include_executions is False

    def test_frozen(self) -> None:
        cfg = EventRecorderConfig()
        with pytest.raises(AttributeError):
            cfg.max_events = 10  # type: ignore[misc]


class TestReplayHarnessConfig:
    def test_defaults(self) -> None:
        cfg = ReplayHarnessConfig()
        assert cfg.speed_multiplier == 0.0
        assert cfg.stop_on_error is False
        assert cfg.max_events == 0
        assert cfg.record_outputs is True

    def test_custom(self) -> None:
        cfg = ReplayHarnessConfig(
            speed_multiplier=2.0, stop_on_error=True, max_events=50
        )
        assert cfg.speed_multiplier == 2.0
        assert cfg.stop_on_error is True
        assert cfg.max_events == 50


# ---------------------------------------------------------------------------
# EventKind
# ---------------------------------------------------------------------------


class TestEventKind:
    def test_values(self) -> None:
        assert EventKind.STREAM == "stream"
        assert EventKind.POLL == "poll"
        assert EventKind.DECISION == "decision"
        assert EventKind.EXECUTION == "execution"
        assert EventKind.ERROR == "error"
        assert EventKind.MARKER == "marker"


# ---------------------------------------------------------------------------
# RecordedEvent
# ---------------------------------------------------------------------------


class TestRecordedEvent:
    def test_basic(self) -> None:
        e = RecordedEvent(
            kind=EventKind.STREAM,
            timestamp=100.0,
            data={"price": 0.5},
            sequence=0,
            source="kalshi",
        )
        assert e.kind == EventKind.STREAM
        assert e.timestamp == 100.0
        assert e.data == {"price": 0.5}
        assert e.sequence == 0
        assert e.source == "kalshi"

    def test_frozen(self) -> None:
        e = RecordedEvent(
            kind=EventKind.POLL, timestamp=1.0, data={}, sequence=0
        )
        with pytest.raises(AttributeError):
            e.timestamp = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_empty(self) -> None:
        r = Recording()
        assert r.event_count == 0
        assert r.duration == 0.0
        assert r.time_range() == (0.0, 0.0)

    def test_single_event(self) -> None:
        e = RecordedEvent(
            kind=EventKind.STREAM, timestamp=100.0, data={}, sequence=0
        )
        r = Recording(events=[e])
        assert r.event_count == 1
        assert r.duration == 0.0  # Single event = 0 duration.

    def test_duration(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=100.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.POLL, timestamp=150.0, data={}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=200.0, data={}, sequence=2),
        ]
        r = Recording(events=events)
        assert r.duration == 100.0
        assert r.time_range() == (100.0, 200.0)

    def test_events_of_kind(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.DECISION, timestamp=2.0, data={}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=3.0, data={}, sequence=2),
        ]
        r = Recording(events=events)
        streams = r.events_of_kind(EventKind.STREAM)
        assert len(streams) == 2
        decisions = r.events_of_kind(EventKind.DECISION)
        assert len(decisions) == 1


# ---------------------------------------------------------------------------
# EventRecorder — start/stop
# ---------------------------------------------------------------------------


class TestEventRecorderStartStop:
    def test_not_recording_by_default(self) -> None:
        rec = EventRecorder()
        assert rec.is_recording is False

    def test_start_stop(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        assert rec.is_recording is True
        rec.stop()
        assert rec.is_recording is False

    def test_no_recording_when_stopped(self) -> None:
        rec = EventRecorder()
        ok = rec.record_event("stream", {"price": 0.5}, ts=1.0)
        assert ok is False
        assert rec.event_count() == 0


# ---------------------------------------------------------------------------
# EventRecorder — record events
# ---------------------------------------------------------------------------


class TestRecordEvent:
    def test_record_stream(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        ok = rec.record_event("stream", {"price": 0.55}, ts=101.0, source="kalshi")
        assert ok is True
        assert rec.event_count() == 1

    def test_record_poll(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        rec.record_event("poll", {"bid": 0.4, "ask": 0.6}, ts=102.0)
        assert rec.event_count() == 1

    def test_record_decision(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        ok = rec.record_decision("skip", {"reason": "stale"}, ts=103.0)
        assert ok is True
        assert rec.event_count() == 1

    def test_record_decision_adds_action(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        rec.record_decision("buy", {"edge": 0.03}, ts=105.0)
        recording = rec.get_recording()
        assert recording.events[0].data["action"] == "buy"
        assert recording.events[0].data["edge"] == 0.03

    def test_max_events_exceeded(self) -> None:
        cfg = EventRecorderConfig(max_events=3)
        rec = EventRecorder(cfg)
        rec.start(now=100.0)
        assert rec.record_event("stream", {}, ts=1.0) is True
        assert rec.record_event("stream", {}, ts=2.0) is True
        assert rec.record_event("stream", {}, ts=3.0) is True
        assert rec.record_event("stream", {}, ts=4.0) is False
        assert rec.event_count() == 3

    def test_filter_decisions(self) -> None:
        cfg = EventRecorderConfig(include_decisions=False)
        rec = EventRecorder(cfg)
        rec.start(now=100.0)
        ok = rec.record_event("decision", {"action": "buy"}, ts=1.0)
        assert ok is False
        assert rec.event_count() == 0

    def test_filter_executions(self) -> None:
        cfg = EventRecorderConfig(include_executions=False)
        rec = EventRecorder(cfg)
        rec.start(now=100.0)
        ok = rec.record_event("execution", {"fill": 0.5}, ts=1.0)
        assert ok is False
        # But stream still works.
        ok = rec.record_event("stream", {"price": 0.5}, ts=2.0)
        assert ok is True

    def test_sequence_numbers(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        rec.record_event("stream", {}, ts=1.0)
        rec.record_event("poll", {}, ts=2.0)
        rec.record_event("stream", {}, ts=3.0)
        recording = rec.get_recording()
        seqs = [e.sequence for e in recording.events]
        assert seqs == [0, 1, 2]


# ---------------------------------------------------------------------------
# EventRecorder — metadata
# ---------------------------------------------------------------------------


class TestRecorderMetadata:
    def test_set_metadata(self) -> None:
        rec = EventRecorder()
        rec.set_metadata("version", "1.0")
        rec.set_metadata("run_id", "abc")
        recording = rec.get_recording()
        assert recording.metadata["version"] == "1.0"
        assert recording.metadata["run_id"] == "abc"


# ---------------------------------------------------------------------------
# EventRecorder — save/load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load(self, tmp_path: object) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        rec.record_event("stream", {"price": 0.55}, ts=101.0, source="kalshi")
        rec.record_event("poll", {"bid": 0.4}, ts=102.0, source="poly")
        rec.record_decision("buy", {"edge": 0.03}, ts=103.0, source="engine")
        rec.set_metadata("session", "test-1")

        path = os.path.join(str(tmp_path), "recording.json")
        rec.save(path)

        loaded = EventRecorder.load(path)
        assert loaded is not None
        assert loaded.event_count == 3
        assert loaded.metadata["session"] == "test-1"
        assert loaded.created_at == 100.0
        assert loaded.events[0].kind == EventKind.STREAM
        assert loaded.events[0].source == "kalshi"
        assert loaded.events[1].kind == EventKind.POLL
        assert loaded.events[2].kind == EventKind.DECISION
        assert loaded.events[2].data["action"] == "buy"

    def test_save_creates_directories(self, tmp_path: object) -> None:
        rec = EventRecorder()
        rec.start(now=1.0)
        rec.record_event("marker", {}, ts=2.0)
        path = os.path.join(str(tmp_path), "nested", "dir", "rec.json")
        rec.save(path)
        assert os.path.exists(path)

    def test_load_nonexistent(self) -> None:
        result = EventRecorder.load("/nonexistent/path.json")
        assert result is None

    def test_load_invalid_json(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        result = EventRecorder.load(path)
        assert result is None

    def test_round_trip_preserves_sequence(self, tmp_path: object) -> None:
        rec = EventRecorder()
        rec.start(now=50.0)
        for i in range(5):
            rec.record_event("stream", {"i": i}, ts=50.0 + i)

        path = os.path.join(str(tmp_path), "seq.json")
        rec.save(path)
        loaded = EventRecorder.load(path)
        assert loaded is not None
        seqs = [e.sequence for e in loaded.events]
        assert seqs == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# EventRecorder — clear
# ---------------------------------------------------------------------------


class TestRecorderClear:
    def test_clear(self) -> None:
        rec = EventRecorder()
        rec.start(now=100.0)
        rec.record_event("stream", {}, ts=101.0)
        rec.set_metadata("k", "v")
        rec.clear()

        assert rec.event_count() == 0
        assert rec.is_recording is False
        recording = rec.get_recording()
        assert recording.metadata == {}


# ---------------------------------------------------------------------------
# ReplayHarness — basic replay
# ---------------------------------------------------------------------------


def _echo_handler(event: RecordedEvent) -> Any:
    """Handler that echoes event kind."""
    return event.kind.value


def _transform_handler(event: RecordedEvent) -> Dict[str, Any]:
    """Handler that transforms event data."""
    return {"processed": True, "kind": event.kind.value, "seq": event.sequence}


class TestReplayBasic:
    def test_no_recording(self) -> None:
        harness = ReplayHarness()
        result = harness.replay(_echo_handler)
        assert result.events_processed == 0
        assert result.events_skipped == 0
        assert result.errors == 0

    def test_has_recording(self) -> None:
        harness = ReplayHarness()
        assert harness.has_recording() is False
        harness.load_recording(Recording())
        assert harness.has_recording() is True

    def test_replay_events(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.POLL, timestamp=2.0, data={}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=3.0, data={}, sequence=2),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness()
        harness.load_recording(recording)
        result = harness.replay(_echo_handler)

        assert result.events_processed == 3
        assert result.events_skipped == 0
        assert result.errors == 0
        assert len(result.outputs) == 3
        assert result.outputs[0].handler_result == "stream"
        assert result.outputs[1].handler_result == "poll"
        assert result.outputs[2].handler_result == "stream"

    def test_replay_with_transform(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.DECISION, timestamp=1.0, data={}, sequence=5),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness()
        harness.load_recording(recording)
        result = harness.replay(_transform_handler)

        assert result.events_processed == 1
        out = result.outputs[0].handler_result
        assert out == {"processed": True, "kind": "decision", "seq": 5}


# ---------------------------------------------------------------------------
# ReplayHarness — filter
# ---------------------------------------------------------------------------


class TestReplayFilter:
    def test_filter_by_kind(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.DECISION, timestamp=2.0, data={}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=3.0, data={}, sequence=2),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness()
        harness.load_recording(recording)

        result = harness.replay(
            _echo_handler,
            event_filter=lambda e: e.kind == EventKind.STREAM,
        )
        assert result.events_processed == 2
        assert result.events_skipped == 1

    def test_filter_all(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness()
        harness.load_recording(recording)

        result = harness.replay(
            _echo_handler,
            event_filter=lambda e: False,
        )
        assert result.events_processed == 0
        assert result.events_skipped == 1


# ---------------------------------------------------------------------------
# ReplayHarness — error handling
# ---------------------------------------------------------------------------


def _error_handler(event: RecordedEvent) -> Any:
    """Handler that always raises."""
    raise ValueError("test error")


def _conditional_error(event: RecordedEvent) -> Any:
    """Handler that errors on sequence 1."""
    if event.sequence == 1:
        raise ValueError("seq 1 error")
    return "ok"


class TestReplayErrors:
    def test_continue_on_error(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.STREAM, timestamp=2.0, data={}, sequence=1),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness(ReplayHarnessConfig(stop_on_error=False))
        harness.load_recording(recording)
        result = harness.replay(_error_handler)

        assert result.events_processed == 2
        assert result.errors == 2
        assert result.stopped_early is False
        assert result.outputs[0].error == "test error"

    def test_stop_on_error(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.STREAM, timestamp=2.0, data={}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=3.0, data={}, sequence=2),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness(ReplayHarnessConfig(stop_on_error=True))
        harness.load_recording(recording)
        result = harness.replay(_error_handler)

        assert result.errors == 1
        assert result.stopped_early is True
        # Only first event processed before stop.
        assert len(result.outputs) == 1

    def test_partial_error(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.STREAM, timestamp=2.0, data={}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=3.0, data={}, sequence=2),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness()
        harness.load_recording(recording)
        result = harness.replay(_conditional_error)

        assert result.events_processed == 3
        assert result.errors == 1
        assert result.outputs[0].handler_result == "ok"
        assert result.outputs[1].error == "seq 1 error"
        assert result.outputs[2].handler_result == "ok"


# ---------------------------------------------------------------------------
# ReplayHarness — max events
# ---------------------------------------------------------------------------


class TestReplayMaxEvents:
    def test_max_events_limit(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=float(i), data={}, sequence=i)
            for i in range(10)
        ]
        recording = Recording(events=events)
        harness = ReplayHarness(ReplayHarnessConfig(max_events=3))
        harness.load_recording(recording)
        result = harness.replay(_echo_handler)

        assert result.events_processed == 3
        assert result.stopped_early is True
        assert len(result.outputs) == 3


# ---------------------------------------------------------------------------
# ReplayHarness — no output recording
# ---------------------------------------------------------------------------


class TestReplayNoOutput:
    def test_no_output_recording(self) -> None:
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0, data={}, sequence=0),
            RecordedEvent(kind=EventKind.STREAM, timestamp=2.0, data={}, sequence=1),
        ]
        recording = Recording(events=events)
        harness = ReplayHarness(ReplayHarnessConfig(record_outputs=False))
        harness.load_recording(recording)
        result = harness.replay(_echo_handler)

        assert result.events_processed == 2
        assert len(result.outputs) == 0  # No outputs recorded.


# ---------------------------------------------------------------------------
# Compare runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def _make_outputs(
        self, results: list[tuple[int, Any]]
    ) -> tuple[ReplayOutput, ...]:
        return tuple(
            ReplayOutput(
                event=RecordedEvent(
                    kind=EventKind.STREAM, timestamp=float(seq), data={}, sequence=seq
                ),
                handler_result=res,
            )
            for seq, res in results
        )

    def test_identical_runs(self) -> None:
        outputs = self._make_outputs([(0, "a"), (1, "b"), (2, "c")])
        run_a = ReplayResult(
            events_processed=3, events_skipped=0, errors=0,
            outputs=outputs, elapsed_seconds=0.1,
        )
        run_b = ReplayResult(
            events_processed=3, events_skipped=0, errors=0,
            outputs=outputs, elapsed_seconds=0.2,
        )
        cmp = ReplayHarness.compare_runs(run_a, run_b)
        assert cmp.total_events == 3
        assert cmp.matching_outputs == 3
        assert cmp.differing_outputs == 0
        assert len(cmp.differences) == 0

    def test_different_outputs(self) -> None:
        out_a = self._make_outputs([(0, "a"), (1, "b")])
        out_b = self._make_outputs([(0, "a"), (1, "X")])
        run_a = ReplayResult(
            events_processed=2, events_skipped=0, errors=0,
            outputs=out_a, elapsed_seconds=0.1,
        )
        run_b = ReplayResult(
            events_processed=2, events_skipped=0, errors=0,
            outputs=out_b, elapsed_seconds=0.1,
        )
        cmp = ReplayHarness.compare_runs(run_a, run_b)
        assert cmp.total_events == 2
        assert cmp.matching_outputs == 1
        assert cmp.differing_outputs == 1
        assert cmp.differences[0]["reason"] == "output_mismatch"
        assert cmp.differences[0]["result_a"] == "b"
        assert cmp.differences[0]["result_b"] == "X"

    def test_missing_events(self) -> None:
        out_a = self._make_outputs([(0, "a"), (1, "b"), (2, "c")])
        out_b = self._make_outputs([(0, "a"), (2, "c")])
        run_a = ReplayResult(
            events_processed=3, events_skipped=0, errors=0,
            outputs=out_a, elapsed_seconds=0.1,
        )
        run_b = ReplayResult(
            events_processed=2, events_skipped=0, errors=0,
            outputs=out_b, elapsed_seconds=0.1,
        )
        cmp = ReplayHarness.compare_runs(run_a, run_b)
        assert cmp.total_events == 3
        assert cmp.matching_outputs == 2
        assert cmp.differing_outputs == 1
        assert cmp.differences[0]["reason"] == "missing_in_b"

    def test_empty_runs(self) -> None:
        run = ReplayResult(
            events_processed=0, events_skipped=0, errors=0,
            outputs=(), elapsed_seconds=0.0,
        )
        cmp = ReplayHarness.compare_runs(run, run)
        assert cmp.total_events == 0
        assert cmp.matching_outputs == 0
        assert cmp.differing_outputs == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_record_save_load_replay(self, tmp_path: object) -> None:
        """Full workflow: record → save → load → replay → verify."""
        # Record.
        rec = EventRecorder()
        rec.start(now=100.0)
        rec.record_event("stream", {"price": 0.55}, ts=101.0, source="kalshi")
        rec.record_event("stream", {"price": 0.45}, ts=102.0, source="poly")
        rec.record_decision("buy", {"edge": 0.03}, ts=103.0, source="engine")
        rec.record_event("execution", {"fill": 1.0}, ts=104.0, source="engine")
        rec.set_metadata("session", "integration-test")

        # Save.
        path = os.path.join(str(tmp_path), "integration.json")
        rec.save(path)

        # Load.
        loaded = EventRecorder.load(path)
        assert loaded is not None
        assert loaded.event_count == 4

        # Replay.
        harness = ReplayHarness()
        harness.load_recording(loaded)
        result = harness.replay(_echo_handler)

        assert result.events_processed == 4
        assert result.errors == 0
        outputs = [o.handler_result for o in result.outputs]
        assert outputs == ["stream", "stream", "decision", "execution"]

    def test_ab_comparison(self) -> None:
        """A/B replay comparison: same events, different handlers."""
        events = [
            RecordedEvent(kind=EventKind.STREAM, timestamp=1.0,
                          data={"price": 0.5}, sequence=0),
            RecordedEvent(kind=EventKind.STREAM, timestamp=2.0,
                          data={"price": 0.6}, sequence=1),
            RecordedEvent(kind=EventKind.STREAM, timestamp=3.0,
                          data={"price": 0.7}, sequence=2),
        ]
        recording = Recording(events=events)

        # Handler A: buy if price > 0.55.
        def handler_a(event: RecordedEvent) -> str:
            return "buy" if event.data.get("price", 0) > 0.55 else "skip"

        # Handler B: buy if price > 0.65.
        def handler_b(event: RecordedEvent) -> str:
            return "buy" if event.data.get("price", 0) > 0.65 else "skip"

        harness = ReplayHarness()
        harness.load_recording(recording)
        run_a = harness.replay(handler_a)
        run_b = harness.replay(handler_b)

        cmp = ReplayHarness.compare_runs(run_a, run_b)
        assert cmp.total_events == 3
        # price=0.5: both skip. price=0.6: A=buy, B=skip. price=0.7: both buy.
        assert cmp.matching_outputs == 2
        assert cmp.differing_outputs == 1
        assert cmp.differences[0]["sequence"] == 1
        assert cmp.differences[0]["result_a"] == "buy"
        assert cmp.differences[0]["result_b"] == "skip"
