from __future__ import annotations

import asyncio
import csv
import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

from arb_bot.config import AppSettings
from arb_bot.engine import ArbEngine, CycleReport, OpportunityDecision
from arb_bot.exchanges import ExchangeAdapter

LOGGER = logging.getLogger(__name__)

_STREAM_STATE_CONNECTED = "connected"
_STREAM_STATE_DEGRADED = "degraded"
_STREAM_STATE_POLL_ONLY = "poll_only"
_STREAM_STATE_RECOVERING = "recovering"


@dataclass(frozen=True)
class PaperSessionSummary:
    cycles: int
    quotes_seen: int
    opportunities_seen: int
    near_opportunities_seen: int
    decisions: int
    dry_run_trades: int
    settled_trades: int
    skipped: int
    simulated_pnl_usd: float
    output_csv: str


@dataclass
class _PaperAccumulator:
    checkpoint: "_PaperCsvCheckpoint"
    cycles: int = 0
    quotes_seen: int = 0
    opportunities_seen: int = 0
    near_opportunities_seen: int = 0
    decisions: int = 0
    dry_run_trades: int = 0
    settled_trades: int = 0
    skipped: int = 0
    simulated_pnl: float = 0.0


@dataclass
class _PaperCsvCheckpoint:
    path: Path
    handle: TextIO
    writer: csv.DictWriter
    flush_every_rows: int = 1
    fsync: bool = False
    rows_written: int = 0

    def write_row(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.rows_written += 1
        if self.flush_every_rows > 0 and (self.rows_written % self.flush_every_rows) == 0:
            self.flush()

    def flush(self) -> None:
        self.handle.flush()
        if self.fsync:
            with suppress(OSError):
                os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.flush()
        self.handle.close()


async def run_paper_session(
    settings: AppSettings,
    duration_minutes: float,
    output_csv: str,
    max_cycles: int | None = None,
) -> PaperSessionSummary:
    end_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = ArbEngine(settings)

    fieldnames = [
        "timestamp_utc",
        "cycle_started_utc",
        "action",
        "reason",
        "kind",
        "opportunity_family",
        "execution_style",
        "payout_per_contract",
        "match_key",
        "match_score",
        "gross_edge_per_contract",
        "net_edge_per_contract",
        "detected_edge_per_contract",
        "detected_profit",
        "fill_probability",
        "partial_fill_probability",
        "fill_quality_score",
        "adverse_selection_flag",
        "expected_realized_edge_per_contract",
        "expected_realized_profit",
        "expected_slippage_per_contract",
        "realized_edge_per_contract",
        "realized_profit",
        "execution_latency_ms",
        "partial_fill",
        "legs_count",
        "legs_summary",
        "contracts",
        "filled_contracts",
        "capital_required",
        "expected_profit",
        "simulated_pnl",
        "correlation_cluster",
        "constraint_min_payout",
        "constraint_adjusted_payout",
        "constraint_valid_assignments",
        "constraint_considered_markets",
        "constraint_assumptions",
        "cluster_budget_usd",
        "cluster_used_usd",
        "cluster_remaining_usd",
        "cluster_exposure_ratio",
        "kelly_fraction_raw",
        "kelly_fraction_effective",
        "kelly_confidence_multiplier",
        "kelly_relative_error_mean",
        "kelly_lane_enabled",
        "kelly_lane_multiplier",
        "kelly_lane_floor",
        "kelly_lane_cap",
        "time_regime",
        "days_to_resolution",
        "edge_threshold_active",
        "expected_profit_threshold_active",
        "fill_probability_threshold_active",
        "realized_profit_threshold_active",
        "leg1_venue",
        "leg1_market_id",
        "leg1_side",
        "leg1_price",
        "leg1_size",
        "leg2_venue",
        "leg2_market_id",
        "leg2_side",
        "leg2_price",
        "leg2_size",
        "mc_enabled",
        "mc_all_legs_filled",
        "mc_filled_leg_count",
        "mc_simulated_pnl",
        "mc_gross_edge_pnl",
        "mc_slippage_cost",
        "mc_adverse_selection_cost",
        "mc_legging_loss",
        "mc_edge_decay_cost",
        "mc_adverse_selection_hit",
        "mc_simulated_latency_ms",
    ]

    checkpoint_handle = output_path.open("w", encoding="utf-8", newline="")
    checkpoint_writer = csv.DictWriter(checkpoint_handle, fieldnames=fieldnames)
    checkpoint_writer.writeheader()
    checkpoint = _PaperCsvCheckpoint(
        path=output_path,
        handle=checkpoint_handle,
        writer=checkpoint_writer,
        flush_every_rows=max(1, int(getattr(settings, "paper_checkpoint_flush_rows", 1))),
        fsync=bool(getattr(settings, "paper_checkpoint_fsync", False)),
    )
    checkpoint.flush()

    acc = _PaperAccumulator(checkpoint=checkpoint)

    try:
        if settings.stream_mode:
            await _run_stream_cycles(
                engine=engine,
                settings=settings,
                end_at=end_at,
                max_cycles=max_cycles,
                acc=acc,
            )
        else:
            await _run_poll_cycles(
                engine=engine,
                settings=settings,
                end_at=end_at,
                max_cycles=max_cycles,
                acc=acc,
            )
    finally:
        with suppress(Exception):
            await engine.aclose()
        with suppress(Exception):
            checkpoint.close()

    summary = PaperSessionSummary(
        cycles=acc.cycles,
        quotes_seen=acc.quotes_seen,
        opportunities_seen=acc.opportunities_seen,
        near_opportunities_seen=acc.near_opportunities_seen,
        decisions=acc.decisions,
        dry_run_trades=acc.dry_run_trades,
        settled_trades=acc.settled_trades,
        skipped=acc.skipped,
        simulated_pnl_usd=acc.simulated_pnl,
        output_csv=str(output_path),
    )

    LOGGER.info(
        "paper session complete cycles=%d opportunities=%d near=%d dry_trades=%d settled=%d simulated_pnl=%.2f csv=%s",
        summary.cycles,
        summary.opportunities_seen,
        summary.near_opportunities_seen,
        summary.dry_run_trades,
        summary.settled_trades,
        summary.simulated_pnl_usd,
        summary.output_csv,
    )
    return summary


async def _run_poll_cycles(
    engine: ArbEngine,
    settings: AppSettings,
    end_at: datetime,
    max_cycles: int | None,
    acc: _PaperAccumulator,
) -> None:
    while _paper_should_continue(end_at=end_at, max_cycles=max_cycles, cycles=acc.cycles):
        cycle_started = datetime.now(timezone.utc)
        report = await engine.run_once()
        _ingest_report(acc, report)

        elapsed = (datetime.now(timezone.utc) - cycle_started).total_seconds()
        sleep_seconds = max(0.0, settings.poll_interval_seconds - elapsed)
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)


async def _run_stream_cycles(
    engine: ArbEngine,
    settings: AppSettings,
    end_at: datetime,
    max_cycles: int | None,
    acc: _PaperAccumulator,
) -> None:
    quote_cache: dict[tuple[str, str], object] = {}

    queue: asyncio.Queue = asyncio.Queue()
    stream_tasks: dict[str, tuple[ExchangeAdapter, asyncio.Task[None]]] = {}
    stream_state_by_venue: dict[str, str] = {}
    stream_state_since_ts_by_venue: dict[str, float] = {}
    last_stream_quote_ts_by_venue: dict[str, float] = {}
    last_stream_stale_warning_ts_by_venue: dict[str, float] = {}
    next_stream_restart_ts_by_venue: dict[str, float] = {}
    stream_restart_delay_seconds = 5.0
    base_poll_refresh_interval = max(5.0, float(settings.poll_interval_seconds))
    startup_fast_cycles = 8
    startup_fast_interval = max(2.0, min(5.0, base_poll_refresh_interval * 0.4))
    recovery_fast_interval = max(2.0, min(6.0, base_poll_refresh_interval * 0.5))
    stale_stream_threshold = max(30.0, base_poll_refresh_interval * 2.0)
    configured_stale_seconds = float(getattr(settings, "stream_stale_degrade_seconds", 0.0))
    if configured_stale_seconds > 0:
        stale_stream_threshold = max(stale_stream_threshold, configured_stale_seconds)
    poll_only_on_stale = bool(getattr(settings, "stream_poll_only_on_stale", True))
    stream_recovery_attempt_seconds = max(
        stream_restart_delay_seconds,
        float(getattr(settings, "stream_recovery_attempt_seconds", 30.0)),
    )
    stale_warning_interval = max(60.0, base_poll_refresh_interval)
    poll_decision_clock = bool(getattr(settings, "stream_poll_decision_clock", True))
    full_discovery_interval_seconds = max(
        0,
        int(getattr(settings, "stream_full_discovery_interval_seconds", 600)),
    )
    low_coverage_streak = 0
    recovery_scan_cycles_remaining = 0
    recovery_scan_cycles = max(1, int(getattr(settings, "stream_low_coverage_recovery_cycles", 3)))
    poll_cycle_count = 0

    def _start_stream_task(adapter: ExchangeAdapter, *, reason: str) -> None:
        async def _consume(adapter_ref: ExchangeAdapter) -> None:
            async for quote in adapter_ref.stream_quotes():
                await queue.put(quote)

        task = asyncio.create_task(_consume(adapter))
        stream_tasks[adapter.venue] = (adapter, task)
        stream_state_by_venue[adapter.venue] = _STREAM_STATE_RECOVERING
        stream_state_since_ts_by_venue[adapter.venue] = time.time()
        last_stream_stale_warning_ts_by_venue.pop(adapter.venue, None)
        LOGGER.info("paper stream enabled for %s (state=%s reason=%s)", adapter.venue, _STREAM_STATE_RECOVERING, reason)

    def _set_poll_only(venue: str, *, now_ts: float, reason: str, idle_seconds: float | None = None) -> None:
        if stream_state_by_venue.get(venue) == _STREAM_STATE_POLL_ONLY:
            return
        stream_state_by_venue[venue] = _STREAM_STATE_POLL_ONLY
        stream_state_since_ts_by_venue[venue] = now_ts
        next_stream_restart_ts_by_venue[venue] = now_ts + stream_recovery_attempt_seconds
        adapter_task = stream_tasks.get(venue)
        if adapter_task is not None:
            _, task = adapter_task
            if not task.done():
                task.cancel()
        if idle_seconds is None:
            LOGGER.warning(
                "paper stream downgraded to poll-only for %s (reason=%s recovery_in=%.1fs)",
                venue,
                reason,
                stream_recovery_attempt_seconds,
            )
        else:
            LOGGER.warning(
                "paper stream downgraded to poll-only for %s after %.1fs without updates (reason=%s recovery_in=%.1fs)",
                venue,
                idle_seconds,
                reason,
                stream_recovery_attempt_seconds,
            )

    for adapter in engine._exchanges.values():
        if adapter.supports_streaming():
            _start_stream_task(adapter, reason="startup")

    LOGGER.info(
        "paper stream config kalshi_rest_topup=%s kalshi_bootstrap_pages=%d kalshi_bootstrap_enrich_limit=%d",
        settings.kalshi.stream_allow_rest_topup,
        settings.kalshi.stream_bootstrap_scan_pages,
        settings.kalshi.stream_bootstrap_enrich_limit,
    )

    if not stream_tasks:
        LOGGER.warning("paper stream requested but no streaming adapters available; using polling cycles")
        if _paper_should_continue(end_at=end_at, max_cycles=max_cycles, cycles=acc.cycles):
            initial_quotes = await _fetch_all_quotes_compatible(engine, venues=None)
            for quote in initial_quotes:
                quote_cache[(quote.venue, quote.market_id)] = quote
            report = await engine._evaluate_quotes(
                list(quote_cache.values()),
                started_at=datetime.now(timezone.utc),
                source="paper-stream-init",
            )
            _ingest_report(acc, report)
        await _run_poll_cycles(
            engine=engine,
            settings=settings,
            end_at=end_at,
            max_cycles=max_cycles,
            acc=acc,
        )
        return
    else:
        # Don't delay stream startup on full-universe REST bootstrap.
        non_stream_venues = {
            venue
            for venue, adapter in engine._exchanges.items()
            if not adapter.supports_streaming()
        }
        stream_bootstrap_venues: set[str] = set()
        stream_bootstrap_selector = getattr(engine, "_stream_bootstrap_venues", None)
        if callable(stream_bootstrap_selector):
            with suppress(Exception):
                stream_bootstrap_venues = set(stream_bootstrap_selector())

        if _paper_should_continue(end_at=end_at, max_cycles=max_cycles, cycles=acc.cycles):
            initial_quotes = await _fetch_all_quotes_compatible_timed(
                engine,
                venues=non_stream_venues,
                timeout_seconds=_stream_fetch_timeout_seconds(base_poll_refresh_interval),
                source="paper-stream-init",
            )
            for quote in initial_quotes:
                quote_cache[(quote.venue, quote.market_id)] = quote

            if stream_bootstrap_venues:
                bootstrap_quotes = await _fetch_all_quotes_compatible_timed(
                    engine,
                    venues=stream_bootstrap_venues,
                    timeout_seconds=_stream_fetch_timeout_seconds(base_poll_refresh_interval),
                    source="paper-stream-bootstrap",
                )
                for quote in bootstrap_quotes:
                    quote_cache[(quote.venue, quote.market_id)] = quote
                LOGGER.info(
                    "paper stream bootstrap snapshot loaded venues=%s quotes=%d",
                    ",".join(sorted(stream_bootstrap_venues)),
                    len(bootstrap_quotes),
                )

            report = await engine._evaluate_quotes(
                list(quote_cache.values()),
                started_at=datetime.now(timezone.utc),
                source="paper-stream-init",
            )
            _ingest_report(acc, report)

    last_eval_ts = 0.0
    now_ts = time.time()
    # Trigger an early refresh so cross-venue/parity lanes don't spend the first
    # minute starved while only sparse stream deltas arrive.
    initial_poll_delay = min(5.0, base_poll_refresh_interval)
    last_poll_refresh_ts = now_ts - base_poll_refresh_interval + initial_poll_delay
    last_full_discovery_ts = now_ts

    try:
        while _paper_should_continue(end_at=end_at, max_cycles=max_cycles, cycles=acc.cycles):
            # Restart streams that exited unexpectedly.
            for venue, (adapter, task) in list(stream_tasks.items()):
                state = stream_state_by_venue.get(venue, _STREAM_STATE_RECOVERING)

                # Poll-only mode: keep consuming snapshots and periodically probe
                # stream recovery without flapping every loop.
                if state == _STREAM_STATE_POLL_ONLY:
                    if not task.done():
                        continue
                    now_ts = time.time()
                    if now_ts < next_stream_restart_ts_by_venue.get(venue, 0.0):
                        continue
                    next_stream_restart_ts_by_venue[venue] = now_ts + stream_recovery_attempt_seconds
                    _start_stream_task(adapter, reason="poll_only_recovery")
                    continue

                if not task.done():
                    continue

                exc = None
                with suppress(Exception):
                    exc = task.exception()
                if exc is not None:
                    LOGGER.warning("paper stream task ended for %s: %s (restarting)", venue, exc)
                else:
                    LOGGER.warning("paper stream task ended for %s without exception (restarting)", venue)
                stream_state_by_venue[venue] = _STREAM_STATE_DEGRADED
                now_ts = time.time()
                if now_ts < next_stream_restart_ts_by_venue.get(venue, 0.0):
                    continue
                next_stream_restart_ts_by_venue[venue] = now_ts + stream_restart_delay_seconds
                _start_stream_task(adapter, reason="task_end_restart")

            timeout_to_end = _stream_wait_timeout_seconds(
                end_at=end_at,
                poll_interval_seconds=settings.poll_interval_seconds,
            )
            if timeout_to_end <= 0:
                break

            now_ts = time.time()
            effective_poll_refresh_interval = base_poll_refresh_interval
            if poll_cycle_count < startup_fast_cycles:
                effective_poll_refresh_interval = min(effective_poll_refresh_interval, startup_fast_interval)
            if low_coverage_streak > 0:
                effective_poll_refresh_interval = min(effective_poll_refresh_interval, recovery_fast_interval)

            remaining_to_poll_refresh = max(
                0.0,
                effective_poll_refresh_interval - (now_ts - last_poll_refresh_ts),
            )
            timeout = min(timeout_to_end, max(0.05, remaining_to_poll_refresh))

            try:
                quote = await asyncio.wait_for(queue.get(), timeout=timeout)
                quote_cache[(quote.venue, quote.market_id)] = quote
                now_ts = time.time()
                last_stream_quote_ts_by_venue[quote.venue] = now_ts
                previous_state = stream_state_by_venue.get(quote.venue)
                if previous_state in {
                    _STREAM_STATE_RECOVERING,
                    _STREAM_STATE_DEGRADED,
                    _STREAM_STATE_POLL_ONLY,
                }:
                    stream_state_by_venue[quote.venue] = _STREAM_STATE_CONNECTED
                    stream_state_since_ts_by_venue[quote.venue] = now_ts
                    LOGGER.info(
                        "paper stream recovered for %s (state=%s)",
                        quote.venue,
                        _STREAM_STATE_CONNECTED,
                    )

                # Drain all remaining quotes without blocking so that bulk
                # pushes (e.g. background enrichment) are absorbed quickly.
                _drain_count = 0
                while not queue.empty():
                    try:
                        extra = queue.get_nowait()
                        quote_cache[(extra.venue, extra.market_id)] = extra
                        last_stream_quote_ts_by_venue[extra.venue] = now_ts
                        _drain_count += 1
                    except asyncio.QueueEmpty:
                        break
                if _drain_count > 0:
                    LOGGER.info(
                        "paper stream drained %d queued quotes in burst",
                        _drain_count,
                    )

                now_ms = time.time() * 1000
                if (
                    not poll_decision_clock
                    and (now_ms - last_eval_ts) >= settings.stream_recompute_cooldown_ms
                ):
                    report = await engine._evaluate_quotes(
                        list(quote_cache.values()),
                        started_at=datetime.now(timezone.utc),
                        source="paper-stream-update",
                    )
                    _ingest_report(acc, report)
                    last_eval_ts = now_ms
            except asyncio.TimeoutError:
                pass

            now_ts = time.time()
            if (now_ts - last_poll_refresh_ts) >= effective_poll_refresh_interval:
                poll_venues = (
                    set(engine._exchanges.keys())
                    if poll_decision_clock
                    else set(non_stream_venues)
                )
                if not poll_decision_clock:
                    poll_only_venues = {
                        venue
                        for venue, state in stream_state_by_venue.items()
                        if state in {_STREAM_STATE_POLL_ONLY, _STREAM_STATE_DEGRADED}
                    }
                    poll_venues |= poll_only_venues
                full_discovery_venues: set[str] = set()
                recovery_venues: set[str] = set()
                if (
                    full_discovery_interval_seconds > 0
                    and (now_ts - last_full_discovery_ts) >= full_discovery_interval_seconds
                ):
                    selector = getattr(engine, "_stream_full_discovery_venues", None)
                    if callable(selector):
                        with suppress(Exception):
                            full_discovery_venues = set(selector()) & poll_venues
                    if full_discovery_venues:
                        LOGGER.info(
                            "paper stream hybrid full-discovery refresh venues=%s",
                            ",".join(sorted(full_discovery_venues)),
                        )
                    last_full_discovery_ts = now_ts

                if (
                    bool(getattr(settings, "stream_low_coverage_enable_full_scan_fallback", True))
                    and recovery_scan_cycles_remaining > 0
                ):
                    selector = getattr(engine, "_stream_full_discovery_venues", None)
                    if callable(selector):
                        with suppress(Exception):
                            recovery_venues = set(selector())
                    poll_venues |= recovery_venues
                    full_discovery_venues |= recovery_venues
                    recovery_scan_cycles_remaining -= 1
                    LOGGER.info(
                        "paper stream coverage recovery active cycles_remaining=%d venues=%s",
                        recovery_scan_cycles_remaining,
                        ",".join(sorted(recovery_venues)) if recovery_venues else "none",
                    )

                if poll_venues:
                    polled = await _fetch_all_quotes_compatible_timed(
                        engine,
                        venues=poll_venues,
                        full_discovery_venues=full_discovery_venues,
                        timeout_seconds=_stream_fetch_timeout_seconds(effective_poll_refresh_interval),
                        source="paper-stream-poll-refresh",
                    )
                    for quote in polled:
                        quote_cache[(quote.venue, quote.market_id)] = quote

                if not poll_decision_clock:
                    stream_topup_venues: set[str] = set()
                    with suppress(Exception):
                        stream_topup_venues = engine._stream_rest_topup_venues(
                            quote_cache=quote_cache,
                            last_stream_quote_ts_by_venue=last_stream_quote_ts_by_venue,
                            now_ts=now_ts,
                        )
                    if stream_topup_venues:
                        polled_stream = await _fetch_all_quotes_compatible_timed(
                            engine,
                            venues=stream_topup_venues,
                            timeout_seconds=_stream_fetch_timeout_seconds(effective_poll_refresh_interval),
                            source="paper-stream-topup",
                        )
                        for quote in polled_stream:
                            quote_cache[(quote.venue, quote.market_id)] = quote

                report = await engine._evaluate_quotes(
                    list(quote_cache.values()),
                    started_at=datetime.now(timezone.utc),
                    source="paper-stream-poll-refresh",
                )
                _ingest_report(acc, report)
                last_poll_refresh_ts = now_ts
                poll_cycle_count += 1

                if hasattr(engine, "_finder") and bool(
                    getattr(settings, "stream_low_coverage_enable_full_scan_fallback", True)
                ):
                    coverage = engine._finder.coverage_snapshot(list(quote_cache.values()))
                    degraded, reason = _coverage_degraded(
                        coverage=coverage,
                        min_pairs=max(0, int(getattr(settings, "stream_low_coverage_min_cross_pairs", 1))),
                        min_parity=max(0, int(getattr(settings, "stream_low_coverage_min_parity_rules", 1))),
                    )
                    trigger_cycles = max(1, int(getattr(settings, "stream_low_coverage_trigger_cycles", 3)))
                    if degraded:
                        low_coverage_streak += 1
                    else:
                        if low_coverage_streak > 0:
                            LOGGER.info("paper stream coverage recovered after streak=%d", low_coverage_streak)
                        low_coverage_streak = 0
                    if low_coverage_streak >= trigger_cycles:
                        recovery_scan_cycles_remaining = max(
                            recovery_scan_cycles_remaining,
                            recovery_scan_cycles,
                        )
                        LOGGER.warning(
                            "paper stream coverage watchdog triggered streak=%d reason=%s recovery_cycles=%d",
                            low_coverage_streak,
                            reason,
                            recovery_scan_cycles_remaining,
                        )
                        low_coverage_streak = 0

                for venue in engine._exchanges.keys():
                    last_stream_ts = last_stream_quote_ts_by_venue.get(venue)
                    if last_stream_ts is None:
                        idle_seconds = now_ts - stream_state_since_ts_by_venue.get(venue, now_ts)
                    else:
                        idle_seconds = now_ts - last_stream_ts
                    if idle_seconds < stale_stream_threshold:
                        continue

                    if poll_only_on_stale and stream_state_by_venue.get(venue) != _STREAM_STATE_POLL_ONLY:
                        _set_poll_only(
                            venue,
                            now_ts=now_ts,
                            reason="stream_stale",
                            idle_seconds=idle_seconds,
                        )
                        continue

                    last_warn_ts = last_stream_stale_warning_ts_by_venue.get(venue, 0.0)
                    if (now_ts - last_warn_ts) < stale_warning_interval:
                        continue
                    last_stream_stale_warning_ts_by_venue[venue] = now_ts
                    LOGGER.warning(
                        "paper stream stale for %s: no stream quote updates for %.1fs; relying on periodic poll refresh",
                        venue,
                        idle_seconds,
                    )
    finally:
        for _, task in stream_tasks.values():
            task.cancel()
        await asyncio.gather(*(task for _, task in stream_tasks.values()), return_exceptions=True)


def _paper_should_continue(end_at: datetime, max_cycles: int | None, cycles: int) -> bool:
    if datetime.now(timezone.utc) >= end_at:
        return False
    if max_cycles is not None and cycles >= max_cycles:
        return False
    return True


def _stream_wait_timeout_seconds(end_at: datetime, poll_interval_seconds: int) -> float:
    remaining = (end_at - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return 0.0
    return min(float(poll_interval_seconds), remaining)


def _coverage_degraded(
    coverage: dict[str, int],
    min_pairs: int,
    min_parity: int,
) -> tuple[bool, str]:
    covered_pairs = int(coverage.get("cross_mapping_pairs_covered", 0))
    covered_parity = int(coverage.get("structural_parity_rules_covered", 0))

    deficits: list[str] = []
    if covered_pairs < min_pairs:
        deficits.append(f"cross_pairs={covered_pairs}/{min_pairs}")
    if covered_parity < min_parity:
        deficits.append(f"parity_rules={covered_parity}/{min_parity}")

    return bool(deficits), ";".join(deficits) if deficits else "ok"


def _stream_fetch_timeout_seconds(poll_refresh_interval_seconds: float) -> float:
    interval = max(1.0, float(poll_refresh_interval_seconds))
    return max(15.0, min(45.0, interval * 0.75))


async def _fetch_all_quotes_compatible(
    engine: ArbEngine,
    venues: set[str] | None,
    full_discovery_venues: set[str] | None = None,
) -> list:
    try:
        if venues is None:
            return await engine._fetch_all_quotes(full_discovery_venues=full_discovery_venues)
        return await engine._fetch_all_quotes(
            venues=venues,
            full_discovery_venues=full_discovery_venues,
        )
    except TypeError:
        # Test doubles may not yet accept the `venues` kwarg.
        return await engine._fetch_all_quotes()


async def _fetch_all_quotes_compatible_timed(
    engine: ArbEngine,
    venues: set[str] | None,
    timeout_seconds: float,
    source: str,
    full_discovery_venues: set[str] | None = None,
) -> list:
    timed_fetch = getattr(engine, "_fetch_all_quotes_timed", None)
    if callable(timed_fetch):
        with suppress(TypeError):
            return await timed_fetch(
                venues=venues,
                timeout_seconds=max(1.0, float(timeout_seconds)),
                source=source,
                full_discovery_venues=full_discovery_venues,
            )
    try:
        return await asyncio.wait_for(
            _fetch_all_quotes_compatible(
                engine,
                venues=venues,
                full_discovery_venues=full_discovery_venues,
            ),
            timeout=max(1.0, timeout_seconds),
        )
    except asyncio.TimeoutError:
        venue_label = "all" if venues is None else ",".join(sorted(venues))
        discovery_label = (
            "none" if not full_discovery_venues else ",".join(sorted(full_discovery_venues))
        )
        LOGGER.warning(
            "paper quote refresh timed out after %.1fs (source=%s venues=%s full_discovery=%s)",
            timeout_seconds,
            source,
            venue_label,
            discovery_label,
        )
        return []


def _ingest_report(acc: _PaperAccumulator, report: CycleReport) -> None:
    acc.cycles += 1
    acc.quotes_seen += report.quotes_count
    acc.opportunities_seen += report.opportunities_count
    acc.near_opportunities_seen += report.near_opportunities_count
    acc.decisions += len(report.decisions)

    for decision in report.decisions:
        row, row_sim_pnl = _decision_row(decision, report.started_at)

        if decision.action == "dry_run":
            acc.dry_run_trades += 1
        if decision.action == "settled":
            acc.settled_trades += 1
        if decision.action in {"dry_run", "settled"}:
            acc.simulated_pnl += row_sim_pnl
        if decision.action == "skipped":
            acc.skipped += 1

        acc.checkpoint.write_row(row)


def _decision_row(decision: OpportunityDecision, cycle_started_at: datetime) -> tuple[dict[str, object], float]:
    leg_1 = decision.opportunity.legs[0]
    leg_2 = decision.opportunity.legs[1]
    legs_summary = "|".join(
        f"{leg.venue}:{leg.market_id}:{leg.side.value}@{leg.buy_price:.4f}"
        for leg in decision.opportunity.legs
    )

    plan = decision.plan
    contracts = plan.contracts if plan is not None else 0
    expected_profit = plan.expected_profit if plan is not None else 0.0
    capital_required = plan.capital_required if plan is not None else 0.0
    metrics = decision.metrics

    if decision.action == "settled":
        row_sim_pnl = float(metrics.get("realized_profit") or 0.0)
    elif decision.action == "dry_run" and decision.reason == "paper_position_opened":
        row_sim_pnl = 0.0
    elif decision.action == "dry_run":
        # Use MC simulated PnL if available; otherwise fall back to EV.
        mc_pnl = metrics.get("mc_simulated_pnl")
        if mc_pnl is not None and metrics.get("monte_carlo_enabled"):
            row_sim_pnl = float(mc_pnl)
        else:
            row_sim_pnl = float(metrics.get("expected_realized_profit") or 0.0)
    else:
        row_sim_pnl = 0.0

    row = {
        "timestamp_utc": decision.timestamp.isoformat(),
        "cycle_started_utc": cycle_started_at.isoformat(),
        "action": decision.action,
        "reason": decision.reason,
        "kind": decision.opportunity.kind.value,
        "opportunity_family": _opportunity_family_from_kind(decision.opportunity.kind.value),
        "execution_style": decision.opportunity.execution_style.value,
        "payout_per_contract": round(decision.opportunity.payout_per_contract, 6),
        "match_key": decision.opportunity.match_key,
        "match_score": round(decision.opportunity.match_score, 6),
        "gross_edge_per_contract": round(decision.opportunity.gross_edge_per_contract, 6),
        "net_edge_per_contract": round(decision.opportunity.net_edge_per_contract, 6),
        "detected_edge_per_contract": round(float(metrics.get("detected_edge_per_contract") or 0.0), 6),
        "detected_profit": round(float(metrics.get("detected_profit") or 0.0), 6),
        "fill_probability": round(float(metrics.get("fill_probability") or 0.0), 6),
        "partial_fill_probability": round(float(metrics.get("partial_fill_probability") or 0.0), 6),
        "fill_quality_score": round(float(metrics.get("fill_quality_score") or 0.0), 6),
        "adverse_selection_flag": bool(metrics.get("adverse_selection_flag") or False),
        "expected_realized_edge_per_contract": round(
            float(metrics.get("expected_realized_edge_per_contract") or 0.0),
            6,
        ),
        "expected_realized_profit": round(float(metrics.get("expected_realized_profit") or 0.0), 6),
        "expected_slippage_per_contract": round(
            float(metrics.get("expected_slippage_per_contract") or 0.0),
            6,
        ),
        "realized_edge_per_contract": _round_or_none(metrics.get("realized_edge_per_contract")),
        "realized_profit": _round_or_none(metrics.get("realized_profit")),
        "execution_latency_ms": _round_or_none(metrics.get("execution_latency_ms")),
        "partial_fill": bool(metrics.get("partial_fill") or False),
        "legs_count": len(decision.opportunity.legs),
        "legs_summary": legs_summary,
        "contracts": contracts,
        "filled_contracts": decision.filled_contracts,
        "capital_required": round(capital_required, 6),
        "expected_profit": round(expected_profit, 6),
        "simulated_pnl": round(row_sim_pnl, 6),
        "correlation_cluster": str(metrics.get("correlation_cluster") or ""),
        "constraint_min_payout": _round_or_none(metrics.get("constraint_min_payout")),
        "constraint_adjusted_payout": _round_or_none(metrics.get("constraint_adjusted_payout")),
        "constraint_valid_assignments": int(metrics.get("constraint_valid_assignments") or 0),
        "constraint_considered_markets": int(metrics.get("constraint_considered_markets") or 0),
        "constraint_assumptions": str(metrics.get("constraint_assumptions") or ""),
        "cluster_budget_usd": _round_or_none(metrics.get("cluster_budget_usd")),
        "cluster_used_usd": _round_or_none(metrics.get("cluster_used_usd")),
        "cluster_remaining_usd": _round_or_none(metrics.get("cluster_remaining_usd")),
        "cluster_exposure_ratio": _round_or_none(metrics.get("cluster_exposure_ratio")),
        "kelly_fraction_raw": _round_or_none(metrics.get("kelly_fraction_raw")),
        "kelly_fraction_effective": _round_or_none(metrics.get("kelly_fraction_effective")),
        "kelly_confidence_multiplier": _round_or_none(metrics.get("kelly_confidence_multiplier")),
        "kelly_relative_error_mean": _round_or_none(metrics.get("kelly_relative_error_mean")),
        "kelly_lane_enabled": bool(metrics.get("kelly_lane_enabled")) if metrics.get("kelly_lane_enabled") is not None else None,
        "kelly_lane_multiplier": _round_or_none(metrics.get("kelly_lane_multiplier")),
        "kelly_lane_floor": _round_or_none(metrics.get("kelly_lane_floor")),
        "kelly_lane_cap": _round_or_none(metrics.get("kelly_lane_cap")),
        "time_regime": str(metrics.get("time_regime") or ""),
        "days_to_resolution": _round_or_none(metrics.get("days_to_resolution")),
        "edge_threshold_active": _round_or_none(metrics.get("edge_threshold_active")),
        "expected_profit_threshold_active": _round_or_none(metrics.get("expected_profit_threshold_active")),
        "fill_probability_threshold_active": _round_or_none(metrics.get("fill_probability_threshold_active")),
        "realized_profit_threshold_active": _round_or_none(metrics.get("realized_profit_threshold_active")),
        "leg1_venue": leg_1.venue,
        "leg1_market_id": leg_1.market_id,
        "leg1_side": leg_1.side.value,
        "leg1_price": round(leg_1.buy_price, 6),
        "leg1_size": round(leg_1.buy_size, 6),
        "leg2_venue": leg_2.venue,
        "leg2_market_id": leg_2.market_id,
        "leg2_side": leg_2.side.value,
        "leg2_price": round(leg_2.buy_price, 6),
        "leg2_size": round(leg_2.buy_size, 6),
        "mc_enabled": bool(metrics.get("monte_carlo_enabled") or False),
        "mc_all_legs_filled": bool(metrics.get("mc_all_legs_filled")) if metrics.get("mc_all_legs_filled") is not None else None,
        "mc_filled_leg_count": int(metrics.get("mc_filled_leg_count") or 0) if metrics.get("mc_filled_leg_count") is not None else None,
        "mc_simulated_pnl": _round_or_none(metrics.get("mc_simulated_pnl")),
        "mc_gross_edge_pnl": _round_or_none(metrics.get("mc_gross_edge_pnl")),
        "mc_slippage_cost": _round_or_none(metrics.get("mc_slippage_cost")),
        "mc_adverse_selection_cost": _round_or_none(metrics.get("mc_adverse_selection_cost")),
        "mc_legging_loss": _round_or_none(metrics.get("mc_legging_loss")),
        "mc_edge_decay_cost": _round_or_none(metrics.get("mc_edge_decay_cost")),
        "mc_adverse_selection_hit": bool(metrics.get("mc_adverse_selection_hit") or False),
        "mc_simulated_latency_ms": _round_or_none(metrics.get("mc_simulated_latency_ms")),
    }
    return row, row_sim_pnl


def _round_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _opportunity_family_from_kind(kind: str) -> str:
    if kind in {"cross_venue", "structural_parity"}:
        return "cross_parity"
    return kind
