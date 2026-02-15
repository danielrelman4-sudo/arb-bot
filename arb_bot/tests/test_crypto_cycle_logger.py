"""Tests for per-cycle CSV data logger."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pytest

from arb_bot.crypto.cycle_logger import CycleLogger, CycleSnapshot


def _make_snap(**overrides) -> CycleSnapshot:
    defaults = dict(
        timestamp=time.time(), cycle=1, symbol="btcusdt",
        price=70000.0, ofi=0.3, volume_rate_short=100.0,
        volume_rate_long=80.0, activity_ratio=1.25,
        realized_vol=0.5, hawkes_intensity=3.0,
        num_edges=1, num_positions=0,
        session_pnl=0.0, bankroll=500.0,
    )
    defaults.update(overrides)
    return CycleSnapshot(**defaults)


class TestCycleSnapshot:
    """CycleSnapshot dataclass captures per-cycle state."""

    def test_snapshot_fields(self):
        snap = _make_snap(ofi=0.35, hawkes_intensity=5.2)
        assert snap.ofi == 0.35
        assert snap.hawkes_intensity == 5.2

    def test_snapshot_all_fields_accessible(self):
        snap = _make_snap()
        from dataclasses import fields
        for f in fields(snap):
            assert hasattr(snap, f.name)


class TestCycleLogger:
    """CycleLogger writes snapshots to CSV."""

    def test_creates_csv_file(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.log(_make_snap())
        logger.flush()
        assert path.exists()

    def test_csv_has_header_and_row(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.log(_make_snap(ofi=0.3, hawkes_intensity=3.0))
        logger.flush()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "btcusdt"
        assert float(rows[0]["ofi"]) == pytest.approx(0.3)
        assert float(rows[0]["hawkes_intensity"]) == pytest.approx(3.0)

    def test_multiple_snapshots_append(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        for i in range(5):
            logger.log(_make_snap(cycle=i + 1, price=70000.0 + i * 100))
        logger.flush()

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_close_flushes(self, tmp_path):
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.log(_make_snap())
        logger.close()
        assert path.exists()
        with open(path) as f:
            assert len(list(csv.DictReader(f))) == 1

    def test_reopen_appends(self, tmp_path):
        """Closing and reopening should append without duplicate headers."""
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.log(_make_snap(cycle=1))
        logger.close()

        logger2 = CycleLogger(str(path))
        logger2.log(_make_snap(cycle=2))
        logger2.close()

        with open(path) as f:
            lines = f.readlines()
        # 1 header + 2 data rows
        assert len(lines) == 3

    def test_csv_field_order(self, tmp_path):
        """Fields should match CycleSnapshot dataclass field order."""
        from dataclasses import fields as dc_fields
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.log(_make_snap())
        logger.flush()

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        expected = [f.name for f in dc_fields(CycleSnapshot)]
        assert header == expected

    def test_flush_without_data(self, tmp_path):
        """Flushing without writing should not crash."""
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.flush()  # no crash

    def test_close_without_open(self, tmp_path):
        """Closing without opening should not crash."""
        path = tmp_path / "test_log.csv"
        logger = CycleLogger(str(path))
        logger.close()  # no crash


class TestEngineLoggerIntegration:
    """Engine's set_cycle_logger and _log_cycle integration."""

    def test_engine_has_set_cycle_logger(self):
        from arb_bot.crypto.config import CryptoSettings
        from arb_bot.crypto.engine import CryptoEngine
        settings = CryptoSettings(enabled=True, paper_mode=True)
        engine = CryptoEngine(settings)
        assert hasattr(engine, 'set_cycle_logger')
        assert hasattr(engine, '_log_cycle')

    def test_log_cycle_with_no_logger(self):
        """_log_cycle should be a no-op when no logger is set."""
        from arb_bot.crypto.config import CryptoSettings
        from arb_bot.crypto.engine import CryptoEngine
        settings = CryptoSettings(enabled=True, paper_mode=True)
        engine = CryptoEngine(settings)
        engine._log_cycle(0)  # no crash

    def test_log_cycle_writes_data(self, tmp_path):
        """_log_cycle should write data when logger is set and price exists."""
        from arb_bot.crypto.config import CryptoSettings
        from arb_bot.crypto.engine import CryptoEngine
        from arb_bot.crypto.price_feed import PriceTick
        import time

        settings = CryptoSettings(
            enabled=True, paper_mode=True,
            price_feed_symbols=["btcusdt"],
        )
        engine = CryptoEngine(settings)

        # Inject price data so _log_cycle has something to log
        now = time.time()
        for i in range(10):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0 + i, now - 10 + i, 1.0)
            )

        path = tmp_path / "cycle_log.csv"
        logger = CycleLogger(str(path))
        engine.set_cycle_logger(logger)
        engine._log_cycle(3)
        logger.close()

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["symbol"] == "btcusdt"
        assert int(rows[0]["num_edges"]) == 3
