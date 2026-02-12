from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

import arb_bot.paper as paper_module
from arb_bot.config import (
    AppSettings,
    FillModelSettings,
    KalshiSettings,
    OpportunityLaneSettings,
    PolymarketSettings,
    RiskSettings,
    SizingSettings,
    StrategySettings,
    UniverseRankingSettings,
)
from arb_bot.engine import CycleReport


@dataclass(frozen=True)
class _Quote:
    venue: str
    market_id: str


class _StreamingAdapter:
    venue = "streaming"

    def supports_streaming(self) -> bool:
        return True

    async def stream_quotes(self):
        yield _Quote(venue=self.venue, market_id="STREAM-1")
        while True:
            await asyncio.sleep(60)


class _PollingOnlyAdapter:
    venue = "polling-only"

    def supports_streaming(self) -> bool:
        return False

    async def stream_quotes(self):
        if False:
            yield None
        return


class _FakeEngine:
    def __init__(self, exchanges: dict[str, object]) -> None:
        self._exchanges = exchanges
        self.run_once_calls = 0
        self.evaluate_sources: list[str] = []
        self.closed = False

    async def run_once(self) -> CycleReport:
        self.run_once_calls += 1
        return self._build_report(quotes_count=1)

    async def _fetch_all_quotes(self) -> list[_Quote]:
        return [_Quote(venue="seed", market_id="SEED-1")]

    async def _evaluate_quotes(
        self,
        quotes: list[_Quote],
        started_at: datetime,
        source: str = "poll",
    ) -> CycleReport:
        self.evaluate_sources.append(source)
        return CycleReport(
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            quotes_count=len(quotes),
            opportunities_count=0,
            near_opportunities_count=0,
            decisions=(),
        )

    async def aclose(self) -> None:
        self.closed = True

    @staticmethod
    def _build_report(quotes_count: int) -> CycleReport:
        now = datetime.now(timezone.utc)
        return CycleReport(
            started_at=now,
            ended_at=now,
            quotes_count=quotes_count,
            opportunities_count=0,
            near_opportunities_count=0,
            decisions=(),
        )


class _CloseOnlyEngine:
    def __init__(self) -> None:
        self.closed = False
        self._exchanges = {}

    async def aclose(self) -> None:
        self.closed = True


def _settings(
    stream_mode: bool,
    poll_interval_seconds: int = 1,
    stream_poll_decision_clock: bool = True,
) -> AppSettings:
    return AppSettings(
        live_mode=False,
        run_once=False,
        poll_interval_seconds=poll_interval_seconds,
        dry_run=True,
        paper_strict_simulation=True,
        paper_position_lifetime_seconds=60,
        paper_dynamic_lifetime_enabled=False,
        paper_dynamic_lifetime_resolution_fraction=0.02,
        paper_dynamic_lifetime_min_seconds=60,
        paper_dynamic_lifetime_max_seconds=900,
        stream_mode=stream_mode,
        stream_recompute_cooldown_ms=0,
        stream_poll_decision_clock=stream_poll_decision_clock,
        default_bankroll_usd=1000.0,
        bankroll_by_venue={},
        log_level="INFO",
        strategy=StrategySettings(),
        lanes=OpportunityLaneSettings(),
        sizing=SizingSettings(),
        risk=RiskSettings(),
        universe=UniverseRankingSettings(),
        fill_model=FillModelSettings(),
        kalshi=KalshiSettings(enabled=False),
        polymarket=PolymarketSettings(enabled=False),
    )


def test_paper_session_stream_mode_uses_poll_clock_by_default(tmp_path, monkeypatch) -> None:
    fake_engine = _FakeEngine({"stream": _StreamingAdapter()})
    monkeypatch.setattr(paper_module, "ArbEngine", lambda _settings: fake_engine)

    summary = asyncio.run(
        paper_module.run_paper_session(
            settings=_settings(stream_mode=True, poll_interval_seconds=1),
            duration_minutes=0.5,
            output_csv=str(tmp_path / "paper_stream.csv"),
            max_cycles=2,
        )
    )

    assert summary.cycles == 2
    assert fake_engine.run_once_calls == 0
    assert fake_engine.evaluate_sources[0] == "paper-stream-init"
    assert "paper-stream-poll-refresh" in fake_engine.evaluate_sources
    assert "paper-stream-update" not in fake_engine.evaluate_sources
    assert fake_engine.closed is True


def test_paper_session_stream_mode_can_use_stream_updates(tmp_path, monkeypatch) -> None:
    fake_engine = _FakeEngine({"stream": _StreamingAdapter()})
    monkeypatch.setattr(paper_module, "ArbEngine", lambda _settings: fake_engine)

    summary = asyncio.run(
        paper_module.run_paper_session(
            settings=_settings(
                stream_mode=True,
                poll_interval_seconds=1,
                stream_poll_decision_clock=False,
            ),
            duration_minutes=0.5,
            output_csv=str(tmp_path / "paper_stream_updates.csv"),
            max_cycles=2,
        )
    )

    assert summary.cycles == 2
    assert fake_engine.run_once_calls == 0
    assert fake_engine.evaluate_sources[0] == "paper-stream-init"
    assert "paper-stream-update" in fake_engine.evaluate_sources
    assert fake_engine.closed is True


def test_paper_session_stream_mode_falls_back_to_polling(tmp_path, monkeypatch) -> None:
    fake_engine = _FakeEngine({"poll-only": _PollingOnlyAdapter()})
    monkeypatch.setattr(paper_module, "ArbEngine", lambda _settings: fake_engine)

    summary = asyncio.run(
        paper_module.run_paper_session(
            settings=_settings(stream_mode=True, poll_interval_seconds=0),
            duration_minutes=0.5,
            output_csv=str(tmp_path / "paper_stream_fallback.csv"),
            max_cycles=2,
        )
    )

    assert summary.cycles == 2
    assert fake_engine.evaluate_sources == ["paper-stream-init"]
    assert fake_engine.run_once_calls == 1
    assert fake_engine.closed is True


def test_paper_session_checkpoint_persists_partial_rows_on_failure(tmp_path, monkeypatch) -> None:
    fake_engine = _CloseOnlyEngine()
    monkeypatch.setattr(paper_module, "ArbEngine", lambda _settings: fake_engine)

    async def _failing_poll_cycles(*, acc, **_kwargs) -> None:
        fieldnames = list(acc.checkpoint.writer.fieldnames or [])
        row = {field: "" for field in fieldnames}
        row["timestamp_utc"] = "2026-02-11T00:00:00Z"
        row["cycle_started_utc"] = "2026-02-11T00:00:00Z"
        row["action"] = "dry_run"
        row["kind"] = "intra_venue"
        acc.checkpoint.write_row(row)
        raise RuntimeError("forced failure")

    monkeypatch.setattr(paper_module, "_run_poll_cycles", _failing_poll_cycles)

    output_csv = tmp_path / "paper_checkpoint_failure.csv"
    with pytest.raises(RuntimeError, match="forced failure"):
        asyncio.run(
            paper_module.run_paper_session(
                settings=_settings(stream_mode=False),
                duration_minutes=0.5,
                output_csv=str(output_csv),
                max_cycles=1,
            )
        )

    lines = output_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "timestamp_utc" in lines[0]
    assert "2026-02-11T00:00:00Z" in lines[1]
    assert fake_engine.closed is True
