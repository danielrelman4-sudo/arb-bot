"""Incident replay bundle automation (Phase 5M).

One-command export of run artifacts (logs, config, coverage,
decisions) into a consumable replay bundle.

Usage::

    bundler = ReplayBundler(config)
    bundler.add_artifact("config", {"max_position": 100})
    bundler.add_artifact("decisions", [decision_1, decision_2])
    bundler.add_log_entry("INFO", "Trade executed", ts=100.0)
    path = bundler.export(output_dir="/tmp/bundles")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayBundlerConfig:
    """Configuration for replay bundler.

    Parameters
    ----------
    output_dir:
        Base directory for bundles. Default "replay_bundles".
    max_log_entries:
        Maximum log entries per bundle. Default 10000.
    max_artifacts:
        Maximum artifacts per bundle. Default 50.
    include_timestamps:
        Include timestamps in bundle metadata. Default True.
    bundle_format:
        Output format: "json" or "jsonl". Default "json".
    """

    output_dir: str = "replay_bundles"
    max_log_entries: int = 10000
    max_artifacts: int = 50
    include_timestamps: bool = True
    bundle_format: str = "json"


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogEntry:
    """A log entry in the replay bundle."""

    level: str
    message: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bundle manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleManifest:
    """Manifest of a replay bundle."""

    bundle_id: str
    created_at: float
    artifact_count: int
    log_entry_count: int
    artifact_names: Tuple[str, ...]
    output_path: str


# ---------------------------------------------------------------------------
# Replay bundler
# ---------------------------------------------------------------------------


class ReplayBundler:
    """Exports run artifacts into replay bundles.

    Collects artifacts (config, decisions, market state, etc.)
    and log entries, then exports as a structured JSON bundle
    for incident analysis and replay.
    """

    def __init__(self, config: ReplayBundlerConfig | None = None) -> None:
        self._config = config or ReplayBundlerConfig()
        self._artifacts: Dict[str, Any] = {}
        self._logs: List[LogEntry] = []
        self._metadata: Dict[str, Any] = {}

    @property
    def config(self) -> ReplayBundlerConfig:
        return self._config

    def add_artifact(self, name: str, data: Any) -> bool:
        """Add an artifact to the bundle.

        Returns False if max artifacts exceeded.
        """
        if len(self._artifacts) >= self._config.max_artifacts:
            return False
        self._artifacts[name] = data
        return True

    def add_log_entry(
        self,
        level: str,
        message: str,
        ts: float | None = None,
        data: Dict[str, Any] | None = None,
    ) -> bool:
        """Add a log entry.

        Returns False if max log entries exceeded.
        """
        if len(self._logs) >= self._config.max_log_entries:
            return False
        if ts is None:
            ts = time.time()
        self._logs.append(LogEntry(
            level=level,
            message=message,
            timestamp=ts,
            data=data or {},
        ))
        return True

    def set_metadata(self, key: str, value: Any) -> None:
        """Set bundle metadata."""
        self._metadata[key] = value

    def artifact_count(self) -> int:
        """Number of artifacts."""
        return len(self._artifacts)

    def log_count(self) -> int:
        """Number of log entries."""
        return len(self._logs)

    def artifact_names(self) -> List[str]:
        """List artifact names."""
        return list(self._artifacts.keys())

    def export(
        self,
        output_dir: str | None = None,
        bundle_id: str | None = None,
        now: float | None = None,
    ) -> BundleManifest:
        """Export the bundle to disk.

        Parameters
        ----------
        output_dir:
            Override output directory. If None, uses config.
        bundle_id:
            Custom bundle ID. If None, generates from timestamp.
        now:
            Current timestamp.
        """
        if now is None:
            now = time.time()
        if output_dir is None:
            output_dir = self._config.output_dir
        if bundle_id is None:
            bundle_id = f"bundle_{int(now)}"

        # Build bundle.
        bundle = {
            "bundle_id": bundle_id,
            "created_at": now,
            "metadata": dict(self._metadata),
            "artifacts": dict(self._artifacts),
            "logs": [
                {
                    "level": entry.level,
                    "message": entry.message,
                    "timestamp": entry.timestamp,
                    "data": entry.data,
                }
                for entry in self._logs
            ],
        }

        # Write to disk.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, f"{bundle_id}.json")

        with open(output_path, "w") as f:
            json.dump(bundle, f, indent=2, default=str)

        return BundleManifest(
            bundle_id=bundle_id,
            created_at=now,
            artifact_count=len(self._artifacts),
            log_entry_count=len(self._logs),
            artifact_names=tuple(self._artifacts.keys()),
            output_path=output_path,
        )

    @staticmethod
    def load_bundle(path: str) -> Dict[str, Any] | None:
        """Load a bundle from disk."""
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def clear(self) -> None:
        """Clear all artifacts and logs."""
        self._artifacts.clear()
        self._logs.clear()
        self._metadata.clear()
