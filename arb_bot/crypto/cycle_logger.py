"""Per-cycle CSV data logger for offline calibration.

Logs OFI, volume rates, Hawkes intensity, model outputs, and PnL
every cycle. Use for calibrating theta (power-law exponent),
beta (Hawkes decay), and validating model predictions.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from typing import Optional, TextIO


@dataclass
class CycleSnapshot:
    """State captured at the end of each engine cycle."""

    timestamp: float
    cycle: int
    symbol: str
    price: float
    ofi: float
    volume_rate_short: float
    volume_rate_long: float
    activity_ratio: float
    realized_vol: float
    hawkes_intensity: float
    num_edges: int
    num_positions: int
    session_pnl: float
    bankroll: float
    effective_horizon: float = 0.0        # Volume-clock effective horizon (0 = not active)
    projected_volume: float = 0.0         # Total projected volume over horizon


_FIELDS = [f.name for f in fields(CycleSnapshot)]


class CycleLogger:
    """Append-only CSV logger for CycleSnapshot records."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        write_header = not os.path.exists(self._path) or os.path.getsize(self._path) == 0
        self._file = open(self._path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDS)
        if write_header:
            self._writer.writeheader()

    def log(self, snap: CycleSnapshot) -> None:
        """Append a snapshot row to the CSV."""
        self._ensure_open()
        assert self._writer is not None
        row = {f: getattr(snap, f) for f in _FIELDS}
        self._writer.writerow(row)

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
