"""Tests for replay_bundle module (Phase 5M)."""

from __future__ import annotations

import json
import os

import pytest

from arb_bot.framework.replay_bundle import (
    BundleManifest,
    LogEntry,
    ReplayBundler,
    ReplayBundlerConfig,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestReplayBundlerConfig:
    def test_defaults(self) -> None:
        cfg = ReplayBundlerConfig()
        assert cfg.output_dir == "replay_bundles"
        assert cfg.max_log_entries == 10000
        assert cfg.max_artifacts == 50
        assert cfg.include_timestamps is True
        assert cfg.bundle_format == "json"

    def test_custom(self) -> None:
        cfg = ReplayBundlerConfig(
            output_dir="/tmp/bundles",
            max_log_entries=500,
            max_artifacts=10,
            include_timestamps=False,
            bundle_format="jsonl",
        )
        assert cfg.output_dir == "/tmp/bundles"
        assert cfg.max_log_entries == 500
        assert cfg.max_artifacts == 10
        assert cfg.include_timestamps is False
        assert cfg.bundle_format == "jsonl"

    def test_frozen(self) -> None:
        cfg = ReplayBundlerConfig()
        with pytest.raises(AttributeError):
            cfg.output_dir = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------


class TestLogEntry:
    def test_basic(self) -> None:
        entry = LogEntry(level="INFO", message="hello", timestamp=100.0)
        assert entry.level == "INFO"
        assert entry.message == "hello"
        assert entry.timestamp == 100.0
        assert entry.data == {}

    def test_with_data(self) -> None:
        entry = LogEntry(
            level="ERROR", message="fail", timestamp=200.0, data={"code": 500}
        )
        assert entry.data == {"code": 500}

    def test_frozen(self) -> None:
        entry = LogEntry(level="INFO", message="hi", timestamp=1.0)
        with pytest.raises(AttributeError):
            entry.level = "DEBUG"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BundleManifest
# ---------------------------------------------------------------------------


class TestBundleManifest:
    def test_basic(self) -> None:
        m = BundleManifest(
            bundle_id="b1",
            created_at=100.0,
            artifact_count=3,
            log_entry_count=10,
            artifact_names=("config", "decisions"),
            output_path="/tmp/b1.json",
        )
        assert m.bundle_id == "b1"
        assert m.created_at == 100.0
        assert m.artifact_count == 3
        assert m.log_entry_count == 10
        assert m.artifact_names == ("config", "decisions")
        assert m.output_path == "/tmp/b1.json"

    def test_frozen(self) -> None:
        m = BundleManifest(
            bundle_id="b1",
            created_at=100.0,
            artifact_count=0,
            log_entry_count=0,
            artifact_names=(),
            output_path="/tmp/b1.json",
        )
        with pytest.raises(AttributeError):
            m.bundle_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReplayBundler — initialization
# ---------------------------------------------------------------------------


class TestReplayBundlerInit:
    def test_default_config(self) -> None:
        b = ReplayBundler()
        assert b.config == ReplayBundlerConfig()
        assert b.artifact_count() == 0
        assert b.log_count() == 0
        assert b.artifact_names() == []

    def test_custom_config(self) -> None:
        cfg = ReplayBundlerConfig(max_artifacts=5)
        b = ReplayBundler(cfg)
        assert b.config.max_artifacts == 5


# ---------------------------------------------------------------------------
# ReplayBundler — add_artifact
# ---------------------------------------------------------------------------


class TestAddArtifact:
    def test_add_single(self) -> None:
        b = ReplayBundler()
        ok = b.add_artifact("config", {"key": "val"})
        assert ok is True
        assert b.artifact_count() == 1
        assert b.artifact_names() == ["config"]

    def test_add_multiple(self) -> None:
        b = ReplayBundler()
        b.add_artifact("config", {})
        b.add_artifact("decisions", [1, 2])
        b.add_artifact("state", "snapshot")
        assert b.artifact_count() == 3
        assert b.artifact_names() == ["config", "decisions", "state"]

    def test_overwrite(self) -> None:
        b = ReplayBundler()
        b.add_artifact("config", {"v": 1})
        b.add_artifact("config", {"v": 2})
        assert b.artifact_count() == 1  # Same name, overwritten.

    def test_max_artifacts_exceeded(self) -> None:
        cfg = ReplayBundlerConfig(max_artifacts=2)
        b = ReplayBundler(cfg)
        assert b.add_artifact("a", 1) is True
        assert b.add_artifact("b", 2) is True
        assert b.add_artifact("c", 3) is False
        assert b.artifact_count() == 2

    def test_various_data_types(self) -> None:
        b = ReplayBundler()
        b.add_artifact("dict", {"a": 1})
        b.add_artifact("list", [1, 2, 3])
        b.add_artifact("string", "hello")
        b.add_artifact("number", 42)
        b.add_artifact("none", None)
        assert b.artifact_count() == 5


# ---------------------------------------------------------------------------
# ReplayBundler — add_log_entry
# ---------------------------------------------------------------------------


class TestAddLogEntry:
    def test_add_with_timestamp(self) -> None:
        b = ReplayBundler()
        ok = b.add_log_entry("INFO", "started", ts=100.0)
        assert ok is True
        assert b.log_count() == 1

    def test_add_without_timestamp(self) -> None:
        b = ReplayBundler()
        ok = b.add_log_entry("DEBUG", "auto-ts")
        assert ok is True
        assert b.log_count() == 1

    def test_add_with_data(self) -> None:
        b = ReplayBundler()
        ok = b.add_log_entry("ERROR", "fail", ts=50.0, data={"code": 500})
        assert ok is True
        assert b.log_count() == 1

    def test_max_log_entries_exceeded(self) -> None:
        cfg = ReplayBundlerConfig(max_log_entries=3)
        b = ReplayBundler(cfg)
        assert b.add_log_entry("INFO", "a", ts=1.0) is True
        assert b.add_log_entry("INFO", "b", ts=2.0) is True
        assert b.add_log_entry("INFO", "c", ts=3.0) is True
        assert b.add_log_entry("INFO", "d", ts=4.0) is False
        assert b.log_count() == 3

    def test_multiple_entries(self) -> None:
        b = ReplayBundler()
        for i in range(10):
            b.add_log_entry("INFO", f"msg-{i}", ts=float(i))
        assert b.log_count() == 10


# ---------------------------------------------------------------------------
# ReplayBundler — set_metadata
# ---------------------------------------------------------------------------


class TestSetMetadata:
    def test_set_and_overwrite(self) -> None:
        b = ReplayBundler()
        b.set_metadata("version", "1.0")
        b.set_metadata("run_id", "abc")
        b.set_metadata("version", "2.0")  # Overwrite.
        # Metadata verified via export.

    def test_metadata_in_export(self, tmp_path: object) -> None:
        b = ReplayBundler()
        b.set_metadata("version", "1.0")
        b.set_metadata("session", "test-123")
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="meta_test", now=100.0
        )
        data = ReplayBundler.load_bundle(manifest.output_path)
        assert data is not None
        assert data["metadata"]["version"] == "1.0"
        assert data["metadata"]["session"] == "test-123"


# ---------------------------------------------------------------------------
# ReplayBundler — export
# ---------------------------------------------------------------------------


class TestExport:
    def test_basic_export(self, tmp_path: object) -> None:
        b = ReplayBundler()
        b.add_artifact("config", {"max_pos": 100})
        b.add_log_entry("INFO", "started", ts=10.0)
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="test_bundle", now=100.0
        )

        assert isinstance(manifest, BundleManifest)
        assert manifest.bundle_id == "test_bundle"
        assert manifest.created_at == 100.0
        assert manifest.artifact_count == 1
        assert manifest.log_entry_count == 1
        assert manifest.artifact_names == ("config",)
        assert os.path.exists(manifest.output_path)

    def test_export_creates_directory(self, tmp_path: object) -> None:
        out = os.path.join(str(tmp_path), "nested", "dir")
        b = ReplayBundler()
        manifest = b.export(output_dir=out, bundle_id="b1", now=50.0)
        assert os.path.exists(manifest.output_path)

    def test_export_default_bundle_id(self, tmp_path: object) -> None:
        b = ReplayBundler()
        manifest = b.export(output_dir=str(tmp_path), now=1234567890.0)
        assert manifest.bundle_id == "bundle_1234567890"

    def test_export_uses_config_output_dir(self, tmp_path: object) -> None:
        out = os.path.join(str(tmp_path), "from_config")
        cfg = ReplayBundlerConfig(output_dir=out)
        b = ReplayBundler(cfg)
        manifest = b.export(bundle_id="cfg_test", now=100.0)
        assert manifest.output_path.startswith(out)
        assert os.path.exists(manifest.output_path)

    def test_export_json_content(self, tmp_path: object) -> None:
        b = ReplayBundler()
        b.add_artifact("decisions", [{"action": "buy"}])
        b.add_log_entry("WARN", "risk alert", ts=50.0, data={"level": 3})
        b.set_metadata("run_id", "r1")
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="content_test", now=200.0
        )

        with open(manifest.output_path) as f:
            data = json.load(f)

        assert data["bundle_id"] == "content_test"
        assert data["created_at"] == 200.0
        assert data["metadata"]["run_id"] == "r1"
        assert data["artifacts"]["decisions"] == [{"action": "buy"}]
        assert len(data["logs"]) == 1
        assert data["logs"][0]["level"] == "WARN"
        assert data["logs"][0]["message"] == "risk alert"
        assert data["logs"][0]["timestamp"] == 50.0
        assert data["logs"][0]["data"] == {"level": 3}

    def test_export_empty_bundle(self, tmp_path: object) -> None:
        b = ReplayBundler()
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="empty", now=1.0
        )
        assert manifest.artifact_count == 0
        assert manifest.log_entry_count == 0
        assert manifest.artifact_names == ()
        data = ReplayBundler.load_bundle(manifest.output_path)
        assert data is not None
        assert data["artifacts"] == {}
        assert data["logs"] == []

    def test_export_multiple_artifacts_and_logs(self, tmp_path: object) -> None:
        b = ReplayBundler()
        b.add_artifact("config", {"a": 1})
        b.add_artifact("state", {"b": 2})
        b.add_artifact("coverage", [0.8, 0.9])
        for i in range(5):
            b.add_log_entry("INFO", f"step-{i}", ts=float(i * 10))
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="multi", now=500.0
        )
        assert manifest.artifact_count == 3
        assert manifest.log_entry_count == 5
        assert set(manifest.artifact_names) == {"config", "state", "coverage"}


# ---------------------------------------------------------------------------
# ReplayBundler — load_bundle
# ---------------------------------------------------------------------------


class TestLoadBundle:
    def test_load_exported_bundle(self, tmp_path: object) -> None:
        b = ReplayBundler()
        b.add_artifact("data", [1, 2, 3])
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="load_test", now=100.0
        )
        loaded = ReplayBundler.load_bundle(manifest.output_path)
        assert loaded is not None
        assert loaded["bundle_id"] == "load_test"
        assert loaded["artifacts"]["data"] == [1, 2, 3]

    def test_load_nonexistent_file(self) -> None:
        result = ReplayBundler.load_bundle("/nonexistent/path.json")
        assert result is None

    def test_load_invalid_json(self, tmp_path: object) -> None:
        bad_file = os.path.join(str(tmp_path), "bad.json")
        with open(bad_file, "w") as f:
            f.write("not valid json {{{")
        result = ReplayBundler.load_bundle(bad_file)
        assert result is None


# ---------------------------------------------------------------------------
# ReplayBundler — clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_all(self) -> None:
        b = ReplayBundler()
        b.add_artifact("config", {})
        b.add_log_entry("INFO", "msg", ts=1.0)
        b.set_metadata("key", "val")
        assert b.artifact_count() > 0
        assert b.log_count() > 0

        b.clear()
        assert b.artifact_count() == 0
        assert b.log_count() == 0
        assert b.artifact_names() == []

    def test_clear_then_reuse(self, tmp_path: object) -> None:
        b = ReplayBundler()
        b.add_artifact("old", "data")
        b.clear()
        b.add_artifact("new", "fresh")
        manifest = b.export(
            output_dir=str(tmp_path), bundle_id="reuse", now=100.0
        )
        data = ReplayBundler.load_bundle(manifest.output_path)
        assert data is not None
        assert "old" not in data["artifacts"]
        assert data["artifacts"]["new"] == "fresh"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self, tmp_path: object) -> None:
        """Full bundler workflow: configure, populate, export, load, verify."""
        cfg = ReplayBundlerConfig(
            output_dir=str(tmp_path),
            max_log_entries=100,
            max_artifacts=10,
        )
        bundler = ReplayBundler(cfg)

        # Add artifacts.
        bundler.add_artifact("config", {"max_position": 100, "venue": "kalshi"})
        bundler.add_artifact("decisions", [
            {"action": "buy", "price": 0.45},
            {"action": "sell", "price": 0.55},
        ])
        bundler.add_artifact("coverage", {"cross": 0.85, "parity": 0.92})

        # Add logs.
        bundler.add_log_entry("INFO", "Engine started", ts=100.0)
        bundler.add_log_entry("INFO", "Opportunity detected", ts=105.0,
                              data={"edge": 0.03})
        bundler.add_log_entry("INFO", "Trade executed", ts=106.0,
                              data={"legs": 2, "fill_ratio": 1.0})
        bundler.add_log_entry("WARN", "Rate limit warning", ts=110.0,
                              data={"venue": "kalshi", "code": 429})

        # Set metadata.
        bundler.set_metadata("session_id", "sess-abc-123")
        bundler.set_metadata("engine_version", "5.0.0")

        # Export.
        manifest = bundler.export(bundle_id="incident_2024_01", now=200.0)

        assert manifest.bundle_id == "incident_2024_01"
        assert manifest.artifact_count == 3
        assert manifest.log_entry_count == 4
        assert "config" in manifest.artifact_names
        assert "decisions" in manifest.artifact_names

        # Load and verify.
        loaded = ReplayBundler.load_bundle(manifest.output_path)
        assert loaded is not None
        assert loaded["bundle_id"] == "incident_2024_01"
        assert loaded["created_at"] == 200.0
        assert loaded["metadata"]["session_id"] == "sess-abc-123"
        assert loaded["metadata"]["engine_version"] == "5.0.0"
        assert len(loaded["artifacts"]) == 3
        assert loaded["artifacts"]["config"]["max_position"] == 100
        assert len(loaded["logs"]) == 4
        assert loaded["logs"][0]["level"] == "INFO"
        assert loaded["logs"][0]["message"] == "Engine started"
        assert loaded["logs"][3]["data"]["code"] == 429

    def test_export_clear_export_again(self, tmp_path: object) -> None:
        """Export, clear, add new data, export again — separate bundles."""
        b = ReplayBundler()
        b.add_artifact("first", "data1")
        m1 = b.export(
            output_dir=str(tmp_path), bundle_id="bundle_1", now=100.0
        )

        b.clear()
        b.add_artifact("second", "data2")
        m2 = b.export(
            output_dir=str(tmp_path), bundle_id="bundle_2", now=200.0
        )

        d1 = ReplayBundler.load_bundle(m1.output_path)
        d2 = ReplayBundler.load_bundle(m2.output_path)
        assert d1 is not None and d2 is not None
        assert "first" in d1["artifacts"]
        assert "first" not in d2["artifacts"]
        assert "second" in d2["artifacts"]
        assert "second" not in d1["artifacts"]
