"""Tests for Phase 2G: Long-run storage + analytics pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from arb_bot.framework.analytics_store import (
    AnalyticsConfig,
    AnalyticsStore,
    DailySummary,
    DecisionRecord,
    FillRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 2025-06-01 12:00:00 UTC as Unix timestamp
TS = 1748779200.0


def _decision(
    kind: str = "cross_venue",
    accepted: bool = True,
    net_edge: float = 0.02,
    contracts: int = 10,
    ts: float = TS,
) -> DecisionRecord:
    return DecisionRecord(
        timestamp=ts,
        kind=kind,
        execution_style="taker",
        market_key="kalshi/M1:polymarket/P1",
        gross_edge=0.03,
        net_edge=net_edge,
        contracts=contracts,
        capital_required=5.0,
        accepted=accepted,
        reject_reason="" if accepted else "risk_limit",
    )


def _fill(
    kind: str = "cross_venue",
    planned: int = 10,
    filled: int = 10,
    pnl: float = 0.20,
    ts: float = TS,
) -> FillRecord:
    return FillRecord(
        timestamp=ts,
        intent_id="intent_1",
        kind=kind,
        market_key="kalshi/M1:polymarket/P1",
        planned_contracts=planned,
        filled_contracts=filled,
        planned_edge=0.02,
        realized_pnl=pnl,
        slippage=-0.005,
        fill_rate=filled / max(planned, 1),
        venue="kalshi",
    )


@pytest.fixture
def store(tmp_path: Path) -> AnalyticsStore:
    s = AnalyticsStore(AnalyticsConfig(db_path=str(tmp_path / "test.db")))
    s.open()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# AnalyticsConfig
# ---------------------------------------------------------------------------


class TestAnalyticsConfig:
    def test_defaults(self) -> None:
        cfg = AnalyticsConfig()
        assert cfg.db_path == "analytics.db"
        assert cfg.retention_days == 90
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# DecisionRecord
# ---------------------------------------------------------------------------


class TestDecisionRecord:
    def test_fields(self) -> None:
        rec = _decision()
        assert rec.kind == "cross_venue"
        assert rec.accepted is True
        assert rec.contracts == 10


# ---------------------------------------------------------------------------
# FillRecord
# ---------------------------------------------------------------------------


class TestFillRecord:
    def test_fields(self) -> None:
        rec = _fill()
        assert rec.intent_id == "intent_1"
        assert rec.fill_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Store — open/close
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_open_creates_db(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        store = AnalyticsStore(AnalyticsConfig(db_path=db_path))
        store.open()
        assert Path(db_path).exists()
        store.close()

    def test_double_open_noop(self, store: AnalyticsStore) -> None:
        store.open()  # Already open — no error.

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        store = AnalyticsStore(AnalyticsConfig(db_path=db_path))
        store.open()
        store.record_decision(_decision())
        store.close()
        store.open()
        assert store.decision_count() == 1
        store.close()


# ---------------------------------------------------------------------------
# Store — record decisions
# ---------------------------------------------------------------------------


class TestRecordDecisions:
    def test_record_and_count(self, store: AnalyticsStore) -> None:
        store.record_decision(_decision())
        store.record_decision(_decision(accepted=False))
        assert store.decision_count() == 2

    def test_count_by_date(self, store: AnalyticsStore) -> None:
        store.record_decision(_decision())
        assert store.decision_count("2025-06-01") == 1
        assert store.decision_count("2025-06-02") == 0

    def test_disabled_no_record(self, tmp_path: Path) -> None:
        s = AnalyticsStore(AnalyticsConfig(
            db_path=str(tmp_path / "test.db"),
            enabled=False,
        ))
        s.open()
        s.record_decision(_decision())
        assert s.decision_count() == 0
        s.close()


# ---------------------------------------------------------------------------
# Store — record fills
# ---------------------------------------------------------------------------


class TestRecordFills:
    def test_record_and_count(self, store: AnalyticsStore) -> None:
        store.record_fill(_fill())
        assert store.fill_count() == 1

    def test_count_by_date(self, store: AnalyticsStore) -> None:
        store.record_fill(_fill())
        assert store.fill_count("2025-06-01") == 1
        assert store.fill_count("2025-06-02") == 0


# ---------------------------------------------------------------------------
# Store — daily summary
# ---------------------------------------------------------------------------


class TestDailySummary:
    def test_empty_day(self, store: AnalyticsStore) -> None:
        summary = store.daily_summary("2025-06-01")
        assert summary.total_decisions == 0
        assert summary.total_fills == 0

    def test_decisions_summary(self, store: AnalyticsStore) -> None:
        store.record_decision(_decision(accepted=True))
        store.record_decision(_decision(accepted=True, kind="intra_venue"))
        store.record_decision(_decision(accepted=False))
        summary = store.daily_summary("2025-06-01")
        assert summary.total_decisions == 3
        assert summary.total_accepted == 2
        assert summary.total_rejected == 1
        assert summary.decisions_by_kind["cross_venue"] == 2
        assert summary.decisions_by_kind["intra_venue"] == 1

    def test_fills_summary(self, store: AnalyticsStore) -> None:
        store.record_fill(_fill(pnl=0.20, filled=10, planned=10))
        store.record_fill(_fill(pnl=-0.05, filled=5, planned=10))
        summary = store.daily_summary("2025-06-01")
        assert summary.total_fills == 2
        assert summary.total_contracts == 15
        assert summary.total_realized_pnl == pytest.approx(0.15)
        assert summary.avg_fill_rate == pytest.approx(0.75)

    def test_pnl_by_kind(self, store: AnalyticsStore) -> None:
        store.record_fill(_fill(kind="cross_venue", pnl=0.30))
        store.record_fill(_fill(kind="intra_venue", pnl=0.10))
        summary = store.daily_summary("2025-06-01")
        assert summary.pnl_by_kind["cross_venue"] == pytest.approx(0.30)
        assert summary.pnl_by_kind["intra_venue"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Store — purge
# ---------------------------------------------------------------------------


class TestPurge:
    def test_purge_old_records(self, store: AnalyticsStore) -> None:
        # June 1 records.
        store.record_decision(_decision(ts=TS))
        store.record_fill(_fill(ts=TS))
        # June 3 records (172800 seconds = 2 days later).
        store.record_decision(_decision(ts=TS + 172800))
        store.record_fill(_fill(ts=TS + 172800))
        # Purge before June 2.
        d, f = store.purge_old_records("2025-06-02")
        assert d == 1
        assert f == 1
        assert store.decision_count() == 1
        assert store.fill_count() == 1

    def test_purge_nothing(self, store: AnalyticsStore) -> None:
        store.record_decision(_decision())
        d, f = store.purge_old_records("2025-01-01")
        assert d == 0
        assert f == 0


# ---------------------------------------------------------------------------
# DailySummary dataclass
# ---------------------------------------------------------------------------


class TestDailySummaryDataclass:
    def test_fields(self) -> None:
        s = DailySummary(
            date="2025-06-01",
            total_decisions=10,
            total_accepted=8,
            total_rejected=2,
            total_fills=5,
            total_contracts=50,
            total_realized_pnl=1.23,
            avg_fill_rate=0.9,
            avg_slippage=-0.01,
        )
        assert s.date == "2025-06-01"
        assert s.total_realized_pnl == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self, store: AnalyticsStore) -> None:
        # Record a mix of decisions and fills.
        store.record_decision(_decision(kind="cross_venue", accepted=True))
        store.record_decision(_decision(kind="cross_venue", accepted=False))
        store.record_decision(_decision(kind="intra_venue", accepted=True))
        store.record_fill(_fill(kind="cross_venue", pnl=0.50, filled=10))
        store.record_fill(_fill(kind="intra_venue", pnl=0.10, filled=5, planned=10))

        summary = store.daily_summary("2025-06-01")
        assert summary.total_decisions == 3
        assert summary.total_accepted == 2
        assert summary.total_fills == 2
        assert summary.total_realized_pnl == pytest.approx(0.60)
        assert "cross_venue" in summary.decisions_by_kind
        assert "cross_venue" in summary.pnl_by_kind

    def test_auto_open(self, tmp_path: Path) -> None:
        """Store auto-opens on first use."""
        s = AnalyticsStore(AnalyticsConfig(db_path=str(tmp_path / "auto.db")))
        s.record_decision(_decision())
        assert s.decision_count() == 1
        s.close()

    def test_config_property(self, store: AnalyticsStore) -> None:
        assert store.config.enabled is True
