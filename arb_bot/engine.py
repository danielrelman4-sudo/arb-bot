from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections import Counter
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from arb_bot.config import AppSettings, LaneTuningSettings
from arb_bot.correlation import CorrelationAssessment, CorrelationConstraintModel
from arb_bot.fill_model import CorrelationMode, FillEstimate, FillModel
from arb_bot.models import (
    ArbitrageOpportunity,
    EngineState,
    ExecutionStyle,
    LegExecutionResult,
    MultiLegExecutionResult,
    OpportunityKind,
    PlannedLegExecutionResult,
    Side,
    TradeLegPlan,
    TradePlan,
)
from arb_bot.risk import RiskManager
from arb_bot.sizing import PositionSizer
from arb_bot.strategy import ArbitrageFinder
from arb_bot.universe_ranking import rank_quotes

from .exchanges import ExchangeAdapter, ForecastExAdapter, KalshiAdapter, PolymarketAdapter

# Structural opportunity kinds eligible for same-venue correlated fill model.
_STRUCTURAL_SAME_VENUE_KINDS: frozenset[OpportunityKind] = frozenset({
    OpportunityKind.STRUCTURAL_BUCKET,
    OpportunityKind.STRUCTURAL_PARITY,
    OpportunityKind.STRUCTURAL_EVENT_TREE,
})

LOGGER = logging.getLogger(__name__)
_OPPORTUNITY_FAMILIES: tuple[str, ...] = (
    OpportunityKind.INTRA_VENUE.value,
    "cross_parity",
    OpportunityKind.STRUCTURAL_BUCKET.value,
    OpportunityKind.STRUCTURAL_EVENT_TREE.value,
)
_STREAM_STATE_CONNECTED = "connected"
_STREAM_STATE_DEGRADED = "degraded"
_STREAM_STATE_POLL_ONLY = "poll_only"
_STREAM_STATE_RECOVERING = "recovering"


@dataclass(frozen=True)
class OpportunityDecision:
    timestamp: datetime
    action: str
    reason: str
    opportunity: ArbitrageOpportunity
    plan: TradePlan | None
    filled_contracts: int = 0
    metrics: dict[str, float | int | bool | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class CycleReport:
    started_at: datetime
    ended_at: datetime
    quotes_count: int
    opportunities_count: int
    near_opportunities_count: int
    decisions: tuple[OpportunityDecision, ...]


@dataclass(frozen=True)
class PaperSimPosition:
    opened_at: datetime
    release_at: datetime
    opportunity: ArbitrageOpportunity
    plan: TradePlan
    filled_contracts: int
    committed_capital_by_venue: dict[str, float]
    expected_realized_profit: float


@dataclass(frozen=True)
class OpportunityRegimePolicy:
    name: str
    days_to_resolution: float | None
    edge_multiplier: float
    expected_profit_multiplier: float
    fill_probability_delta: float
    realized_profit_multiplier: float


@dataclass(frozen=True)
class OpportunityLane:
    kind: OpportunityKind
    settings: LaneTuningSettings


def _choose_correlation_mode(opportunity: ArbitrageOpportunity) -> CorrelationMode:
    """Select the fill correlation mode based on opportunity structure.

    Structural multi-leg opportunities (bucket, parity, event-tree) where all legs
    trade on the same venue share correlated fill dynamics — the same market-maker
    presence, liquidity regime, and volatility conditions affect all legs.  Using
    independent fill multiplication drastically underestimates the joint fill
    probability (the "multi-leg death spiral": 3 legs at 0.54 → 0.157 independent
    vs 0.374 with rho=0.7).

    Cross-venue opportunities genuinely face independent fill conditions across
    exchanges and should always use the independent product.
    """
    if opportunity.kind not in _STRUCTURAL_SAME_VENUE_KINDS:
        return CorrelationMode.INDEPENDENT

    if len(opportunity.legs) <= 1:
        return CorrelationMode.INDEPENDENT

    # Only apply same-venue correlation when ALL legs share the same venue.
    venues = {leg.venue for leg in opportunity.legs}
    if len(venues) == 1:
        return CorrelationMode.SAME_VENUE

    return CorrelationMode.INDEPENDENT


class ArbEngine:
    _PAPER_LIFETIME_KIND_MULTIPLIERS: dict[OpportunityKind, float] = {
        OpportunityKind.INTRA_VENUE: 0.75,
        OpportunityKind.CROSS_VENUE: 1.0,
        OpportunityKind.STRUCTURAL_BUCKET: 1.2,
        OpportunityKind.STRUCTURAL_EVENT_TREE: 1.35,
        OpportunityKind.STRUCTURAL_PARITY: 1.1,
    }
    _STREAM_TOPUP_STALE_SECONDS = 45.0
    _EXECUTION_KIND_PRIORITY: tuple[OpportunityKind, ...] = (
        OpportunityKind.CROSS_VENUE,
        OpportunityKind.STRUCTURAL_PARITY,
        OpportunityKind.STRUCTURAL_BUCKET,
        OpportunityKind.STRUCTURAL_EVENT_TREE,
        OpportunityKind.INTRA_VENUE,
    )

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._exchanges = self._build_exchanges()
        self._finder = ArbitrageFinder(
            min_net_edge_per_contract=settings.strategy.min_net_edge_per_contract,
            enable_cross_venue=settings.strategy.enable_cross_venue,
            cross_venue_min_match_score=settings.strategy.cross_venue_min_match_score,
            cross_venue_mapping_path=settings.strategy.cross_venue_mapping_path,
            cross_venue_mapping_required=settings.strategy.cross_venue_mapping_required,
            enable_fuzzy_cross_venue_fallback=settings.strategy.enable_fuzzy_cross_venue_fallback,
            enable_maker_estimates=settings.strategy.enable_maker_estimates,
            enable_structural_arb=settings.strategy.enable_structural_arb,
            structural_rules_path=settings.strategy.structural_rules_path,
            enable_bucket_quality_model=settings.strategy.enable_bucket_quality_model,
            bucket_quality_history_glob=settings.strategy.bucket_quality_history_glob,
            bucket_quality_history_max_files=settings.strategy.bucket_quality_history_max_files,
            bucket_quality_min_observations=settings.strategy.bucket_quality_min_observations,
            bucket_quality_max_active_buckets=settings.strategy.bucket_quality_max_active_buckets,
            bucket_quality_explore_fraction=settings.strategy.bucket_quality_explore_fraction,
            bucket_quality_prior_mean_realized_profit=settings.strategy.bucket_quality_prior_mean_realized_profit,
            bucket_quality_prior_strength=settings.strategy.bucket_quality_prior_strength,
            bucket_quality_min_score=settings.strategy.bucket_quality_min_score,
            bucket_quality_leg_count_penalty=settings.strategy.bucket_quality_leg_count_penalty,
            bucket_quality_live_update_interval=settings.strategy.bucket_quality_live_update_interval,
        )
        self._lanes: tuple[OpportunityLane, ...] = (
            OpportunityLane(OpportunityKind.INTRA_VENUE, settings.lanes.intra_venue),
            OpportunityLane(OpportunityKind.CROSS_VENUE, settings.lanes.cross_venue),
            OpportunityLane(OpportunityKind.STRUCTURAL_BUCKET, settings.lanes.structural_bucket),
            OpportunityLane(OpportunityKind.STRUCTURAL_EVENT_TREE, settings.lanes.structural_event_tree),
            OpportunityLane(OpportunityKind.STRUCTURAL_PARITY, settings.lanes.structural_parity),
        )
        self._sizer = PositionSizer(settings.sizing, settings.strategy)
        self._risk = RiskManager(settings.risk)
        self._fill_model = FillModel(settings.fill_model)
        self._correlation_model = CorrelationConstraintModel(
            settings.strategy.structural_rules_path,
            enabled=settings.strategy.enable_constraint_pricing,
            max_bruteforce_markets=settings.strategy.constraint_max_bruteforce_markets,
            cross_venue_equivalence_min_match_score=settings.strategy.cross_venue_equivalence_min_match_score,
            assume_structural_buckets_exhaustive=settings.strategy.assume_structural_buckets_exhaustive,
        )
        self._state = EngineState(cash_by_venue=self._initial_cash_map())
        self._paper_positions: list[PaperSimPosition] = []
        window = max(1, self._settings.sizing.kelly_confidence_window)
        self._kelly_relative_error_window: deque[float] = deque(maxlen=window)
        self._lane_dynamic_kelly_floor: dict[OpportunityKind, float] = {
            kind: 0.0 for kind in OpportunityKind
        }
        self._lane_dynamic_fill_probability_delta: dict[OpportunityKind, float] = {
            kind: 0.0 for kind in OpportunityKind
        }

    def _initial_cash_map(self) -> dict[str, float]:
        venues = {adapter.venue for adapter in self._exchanges.values()}
        cash_map = {venue: self._settings.default_bankroll_usd for venue in venues}
        for venue, value in self._settings.bankroll_by_venue.items():
            cash_map[venue] = value
        return cash_map

    def _build_exchanges(self) -> dict[str, ExchangeAdapter]:
        exchanges: dict[str, ExchangeAdapter] = {}

        if self._settings.kalshi.enabled:
            exchanges["kalshi"] = KalshiAdapter(self._settings.kalshi)

        if self._settings.polymarket.enabled:
            exchanges["polymarket"] = PolymarketAdapter(self._settings.polymarket)

        if self._settings.forecastex.enabled:
            exchanges["forecastex"] = ForecastExAdapter(self._settings.forecastex)

        if not exchanges:
            raise ValueError(
                "No exchanges enabled. Set KALSHI_ENABLED, POLYMARKET_ENABLED, "
                "or FORECASTEX_ENABLED."
            )

        return exchanges

    async def run_forever(self, run_once: bool | None = None) -> None:
        single = self._settings.run_once if run_once is None else run_once

        try:
            if self._settings.stream_mode and not single:
                await self._run_stream_loop()
                return

            while True:
                loop_start = time.perf_counter()
                await self.run_once()

                if single:
                    return

                elapsed = time.perf_counter() - loop_start
                sleep_seconds = max(0.0, self._settings.poll_interval_seconds - elapsed)
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
        finally:
            await self.aclose()

    async def run_once(self) -> CycleReport:
        started_at = datetime.now(timezone.utc)
        if self._settings.live_mode:
            await self._refresh_venue_balances()

        quotes = await self._fetch_all_quotes()
        return await self._evaluate_quotes(quotes, started_at)

    async def _run_stream_loop(self) -> None:
        quote_cache: dict[tuple[str, str], object] = {}

        queue: asyncio.Queue = asyncio.Queue()
        stream_tasks: dict[str, tuple[ExchangeAdapter, asyncio.Task[None]]] = {}
        stream_state_by_venue: dict[str, str] = {}
        stream_state_since_ts_by_venue: dict[str, float] = {}
        last_stream_quote_ts_by_venue: dict[str, float] = {}
        last_stream_stale_warning_ts_by_venue: dict[str, float] = {}
        next_stream_restart_ts_by_venue: dict[str, float] = {}
        stream_restart_delay_seconds = 5.0
        base_poll_refresh_interval = max(5.0, float(self._settings.poll_interval_seconds))
        startup_fast_cycles = 8
        startup_fast_interval = max(2.0, min(5.0, base_poll_refresh_interval * 0.4))
        recovery_fast_interval = max(2.0, min(6.0, base_poll_refresh_interval * 0.5))
        stale_stream_threshold = max(30.0, base_poll_refresh_interval * 2.0)
        configured_stale_seconds = float(getattr(self._settings, "stream_stale_degrade_seconds", 0.0))
        if configured_stale_seconds > 0:
            stale_stream_threshold = max(stale_stream_threshold, configured_stale_seconds)
        poll_only_on_stale = bool(getattr(self._settings, "stream_poll_only_on_stale", True))
        stream_recovery_attempt_seconds = max(
            stream_restart_delay_seconds,
            float(getattr(self._settings, "stream_recovery_attempt_seconds", 30.0)),
        )
        stale_warning_interval = max(60.0, base_poll_refresh_interval)
        poll_decision_clock = bool(self._settings.stream_poll_decision_clock)
        full_discovery_interval_seconds = max(0, int(self._settings.stream_full_discovery_interval_seconds))
        low_coverage_streak = 0
        recovery_scan_cycles_remaining = 0
        recovery_scan_cycles = max(1, int(self._settings.stream_low_coverage_recovery_cycles))
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
            LOGGER.info("stream enabled for %s (state=%s reason=%s)", adapter.venue, _STREAM_STATE_RECOVERING, reason)

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
                    "stream downgraded to poll-only for %s (reason=%s recovery_in=%.1fs)",
                    venue,
                    reason,
                    stream_recovery_attempt_seconds,
                )
            else:
                LOGGER.warning(
                    "stream downgraded to poll-only for %s after %.1fs without updates (reason=%s recovery_in=%.1fs)",
                    venue,
                    idle_seconds,
                    reason,
                    stream_recovery_attempt_seconds,
                )

        for adapter in self._exchanges.values():
            if adapter.supports_streaming():
                _start_stream_task(adapter, reason="startup")

        LOGGER.info(
            "stream config kalshi_rest_topup=%s kalshi_bootstrap_pages=%d kalshi_bootstrap_enrich_limit=%d",
            self._settings.kalshi.stream_allow_rest_topup,
            self._settings.kalshi.stream_bootstrap_scan_pages,
            self._settings.kalshi.stream_bootstrap_enrich_limit,
        )

        if not stream_tasks:
            LOGGER.warning("stream mode enabled but no streaming adapters; falling back to polling")
            while True:
                await self.run_once()
                await asyncio.sleep(self._settings.poll_interval_seconds)
        else:
            # Avoid blocking stream startup on expensive full-universe REST discovery.
            # Bootstrap only non-stream venues here; stream venues will populate
            # quote_cache via websocket and periodic poll refresh.
            non_stream_venues = {
                venue
                for venue, adapter in self._exchanges.items()
                if not adapter.supports_streaming()
            }
            stream_bootstrap_venues = self._stream_bootstrap_venues()
            if non_stream_venues:
                initial_quotes = await self._fetch_all_quotes_timed(
                    venues=non_stream_venues,
                    timeout_seconds=self._stream_fetch_timeout_seconds(base_poll_refresh_interval),
                    source="stream-init",
                )
                for quote in initial_quotes:
                    quote_cache[(quote.venue, quote.market_id)] = quote
            if stream_bootstrap_venues:
                bootstrap_quotes = await self._fetch_all_quotes_timed(
                    venues=stream_bootstrap_venues,
                    timeout_seconds=self._stream_fetch_timeout_seconds(base_poll_refresh_interval),
                    source="stream-bootstrap",
                )
                for quote in bootstrap_quotes:
                    quote_cache[(quote.venue, quote.market_id)] = quote
                LOGGER.info(
                    "stream bootstrap snapshot loaded venues=%s quotes=%d",
                    ",".join(sorted(stream_bootstrap_venues)),
                    len(bootstrap_quotes),
                )
            await self._evaluate_quotes(
                list(quote_cache.values()),
                datetime.now(timezone.utc),
                source="stream-init",
            )

        last_eval_ts = 0.0
        now_ts = time.time()
        # Seed full-venue coverage quickly after stream startup; avoids a long
        # zero-cross/parity warmup while waiting a full poll interval.
        initial_poll_delay = min(5.0, base_poll_refresh_interval)
        last_poll_refresh_ts = now_ts - base_poll_refresh_interval + initial_poll_delay
        last_full_discovery_ts = now_ts

        try:
            while True:
                # Restart streams that exited unexpectedly so one venue cannot silently degrade forever.
                for venue, (adapter, task) in list(stream_tasks.items()):
                    state = stream_state_by_venue.get(venue, _STREAM_STATE_RECOVERING)

                    # Poll-only mode keeps a deterministic degraded path and probes
                    # stream recovery on a bounded cadence.
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
                        LOGGER.warning("stream task ended for %s: %s (restarting)", venue, exc)
                    else:
                        LOGGER.warning("stream task ended for %s without exception (restarting)", venue)
                    stream_state_by_venue[venue] = _STREAM_STATE_DEGRADED
                    now_ts = time.time()
                    if now_ts < next_stream_restart_ts_by_venue.get(venue, 0.0):
                        continue
                    next_stream_restart_ts_by_venue[venue] = now_ts + stream_restart_delay_seconds
                    _start_stream_task(adapter, reason="task_end_restart")

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
                wait_timeout = max(0.05, remaining_to_poll_refresh)

                try:
                    quote = await asyncio.wait_for(queue.get(), timeout=wait_timeout)
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
                        LOGGER.info("stream recovered for %s (state=%s)", quote.venue, _STREAM_STATE_CONNECTED)

                    now_ms = time.time() * 1000
                    if (
                        not poll_decision_clock
                        and (now_ms - last_eval_ts) >= self._settings.stream_recompute_cooldown_ms
                    ):
                        await self._evaluate_quotes(
                            list(quote_cache.values()),
                            started_at=datetime.now(timezone.utc),
                            source="stream-update",
                        )
                        last_eval_ts = now_ms
                except asyncio.TimeoutError:
                    pass

                now_ts = time.time()
                if (now_ts - last_poll_refresh_ts) >= effective_poll_refresh_interval:
                    poll_venues = (
                        set(self._exchanges.keys())
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
                        full_discovery_venues = self._stream_full_discovery_venues() & poll_venues
                        if full_discovery_venues:
                            LOGGER.info(
                                "stream hybrid full-discovery refresh venues=%s",
                                ",".join(sorted(full_discovery_venues)),
                            )
                        last_full_discovery_ts = now_ts

                    if (
                        self._settings.stream_low_coverage_enable_full_scan_fallback
                        and recovery_scan_cycles_remaining > 0
                    ):
                        recovery_venues = self._stream_full_discovery_venues()
                        poll_venues |= recovery_venues
                        full_discovery_venues |= recovery_venues
                        recovery_scan_cycles_remaining -= 1
                        LOGGER.info(
                            "stream coverage recovery active cycles_remaining=%d venues=%s",
                            recovery_scan_cycles_remaining,
                            ",".join(sorted(recovery_venues)) if recovery_venues else "none",
                        )

                    if poll_venues:
                        polled = await self._fetch_all_quotes_timed(
                            venues=poll_venues,
                            full_discovery_venues=full_discovery_venues,
                            timeout_seconds=self._stream_fetch_timeout_seconds(effective_poll_refresh_interval),
                            source="stream-poll-refresh",
                        )
                        for quote in polled:
                            quote_cache[(quote.venue, quote.market_id)] = quote

                    if not poll_decision_clock:
                        stream_topup_venues = self._stream_rest_topup_venues(
                            quote_cache=quote_cache,
                            last_stream_quote_ts_by_venue=last_stream_quote_ts_by_venue,
                            now_ts=now_ts,
                        )
                        if stream_topup_venues:
                            polled_stream = await self._fetch_all_quotes_timed(
                                venues=stream_topup_venues,
                                timeout_seconds=self._stream_fetch_timeout_seconds(effective_poll_refresh_interval),
                                source="stream-topup",
                            )
                            for quote in polled_stream:
                                quote_cache[(quote.venue, quote.market_id)] = quote

                    await self._evaluate_quotes(
                        list(quote_cache.values()),
                        started_at=datetime.now(timezone.utc),
                        source="stream-poll-refresh",
                    )
                    last_poll_refresh_ts = now_ts
                    poll_cycle_count += 1

                    if self._settings.stream_low_coverage_enable_full_scan_fallback:
                        coverage = self._finder.coverage_snapshot(list(quote_cache.values()))
                        degraded, reason = self._stream_coverage_is_degraded(coverage)
                        trigger_cycles = max(1, int(self._settings.stream_low_coverage_trigger_cycles))
                        if degraded:
                            low_coverage_streak += 1
                        else:
                            if low_coverage_streak > 0:
                                LOGGER.info("stream coverage recovered after streak=%d", low_coverage_streak)
                            low_coverage_streak = 0
                        if low_coverage_streak >= trigger_cycles:
                            recovery_scan_cycles_remaining = max(
                                recovery_scan_cycles_remaining,
                                recovery_scan_cycles,
                            )
                            LOGGER.warning(
                                "stream coverage watchdog triggered streak=%d reason=%s recovery_cycles=%d",
                                low_coverage_streak,
                                reason,
                                recovery_scan_cycles_remaining,
                            )
                            low_coverage_streak = 0

                    # Alert on stream-side staleness, even if polling keeps the bot alive.
                    for venue in self._exchanges.keys():
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
                            "stream stale for %s: no stream quote updates for %.1fs; relying on periodic poll refresh",
                            venue,
                            idle_seconds,
                        )
        finally:
            for _, task in stream_tasks.values():
                task.cancel()
            await asyncio.gather(*(task for _, task in stream_tasks.values()), return_exceptions=True)

    async def _fetch_all_quotes(
        self,
        venues: set[str] | None = None,
        full_discovery_venues: set[str] | None = None,
        per_adapter_timeout_seconds: float | None = None,
    ) -> list:
        full_scan = full_discovery_venues or set()
        adapters = [
            (venue, adapter)
            for venue, adapter in self._exchanges.items()
            if venues is None or venue in venues
        ]
        if not adapters:
            return []

        tasks = []
        selected_adapters: list[ExchangeAdapter] = []
        for venue, adapter in adapters:
            fetch_full_scan = venue in full_scan
            fetch_full_fn = getattr(adapter, "fetch_quotes_full_scan", None)
            if fetch_full_scan and callable(fetch_full_fn):
                fetch_coro = fetch_full_fn()
            else:
                fetch_coro = adapter.fetch_quotes()

            if per_adapter_timeout_seconds is not None and per_adapter_timeout_seconds > 0:
                fetch_coro = asyncio.wait_for(fetch_coro, timeout=per_adapter_timeout_seconds)

            tasks.append(fetch_coro)
            selected_adapters.append(adapter)

        quote_batches = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        quotes = []
        for adapter, batch in zip(selected_adapters, quote_batches):
            if isinstance(batch, Exception):
                endpoint_hint = ""
                if isinstance(adapter, KalshiAdapter):
                    active_api_url = adapter._settings.api_base_url
                    active_api_getter = getattr(adapter, "_active_api_base_url", None)
                    if callable(active_api_getter):
                        with suppress(Exception):
                            active_api_url = active_api_getter()
                    endpoint_hint = f" api_base_url={active_api_url}"
                elif isinstance(adapter, PolymarketAdapter):
                    endpoint_hint = f" clob_base_url={adapter._settings.clob_base_url}"
                if isinstance(batch, asyncio.TimeoutError):
                    LOGGER.warning(
                        "quote fetch timed out for %s%s (per_adapter_timeout=%.1fs)",
                        adapter.venue,
                        endpoint_hint,
                        float(per_adapter_timeout_seconds or 0.0),
                    )
                    continue
                LOGGER.warning("quote fetch failed for %s%s: %s", adapter.venue, endpoint_hint, batch)
                continue
            quotes.extend(batch)

        return quotes

    def _stream_rest_topup_venues(
        self,
        quote_cache: dict[tuple[str, str], object],
        last_stream_quote_ts_by_venue: dict[str, float],
        now_ts: float,
    ) -> set[str]:
        cache_counts: dict[str, int] = {}
        for venue, _ in quote_cache.keys():
            cache_counts[venue] = cache_counts.get(venue, 0) + 1

        venues: set[str] = set()
        for venue, adapter in self._exchanges.items():
            if not adapter.supports_streaming():
                continue
            if not isinstance(adapter, KalshiAdapter):
                continue
            if not self._settings.kalshi.stream_allow_rest_topup:
                continue

            # Target the bounded priority refresh window; do not tie top-up pressure
            # to full-universe limits, which can cause persistent REST churn.
            min_quotes = max(
                1,
                min(
                    int(self._settings.kalshi.market_limit),
                    int(self._settings.kalshi.stream_priority_refresh_limit),
                ),
            )
            cached = cache_counts.get(venue, 0)
            last_stream_ts = last_stream_quote_ts_by_venue.get(venue)
            stale = last_stream_ts is None or (now_ts - last_stream_ts) >= self._STREAM_TOPUP_STALE_SECONDS
            if cached < min_quotes or stale:
                venues.add(venue)
        return venues

    def _stream_bootstrap_venues(self) -> set[str]:
        venues: set[str] = set()
        for venue, adapter in self._exchanges.items():
            if not adapter.supports_streaming():
                continue
            # Always bootstrap Kalshi from the bounded priority refresh path so
            # cross/parity lanes are not starved at startup.
            if isinstance(adapter, KalshiAdapter):
                venues.add(venue)
                continue
            # Polymarket bootstrap materially improves structural bucket coverage.
            venues.add(venue)
        return venues

    def _stream_full_discovery_venues(self) -> set[str]:
        venues: set[str] = set()
        for venue, adapter in self._exchanges.items():
            fetch_full_fn = getattr(adapter, "fetch_quotes_full_scan", None)
            if callable(fetch_full_fn):
                venues.add(venue)
                continue
            if isinstance(adapter, KalshiAdapter):
                venues.add(venue)
        return venues

    def _stream_coverage_is_degraded(self, coverage: dict[str, int]) -> tuple[bool, str]:
        covered_pairs = int(coverage.get("cross_mapping_pairs_covered", 0))
        min_pairs = max(0, int(self._settings.stream_low_coverage_min_cross_pairs))
        covered_parity = int(coverage.get("structural_parity_rules_covered", 0))
        min_parity = max(0, int(self._settings.stream_low_coverage_min_parity_rules))

        deficits: list[str] = []
        if covered_pairs < min_pairs:
            deficits.append(f"cross_pairs={covered_pairs}/{min_pairs}")
        if covered_parity < min_parity:
            deficits.append(f"parity_rules={covered_parity}/{min_parity}")

        return bool(deficits), ";".join(deficits) if deficits else "ok"

    @staticmethod
    def _stream_fetch_timeout_seconds(poll_refresh_interval_seconds: float) -> float:
        interval = max(1.0, float(poll_refresh_interval_seconds))
        return max(8.0, min(30.0, interval * 0.75))

    async def _fetch_all_quotes_timed(
        self,
        venues: set[str] | None,
        timeout_seconds: float,
        source: str,
        full_discovery_venues: set[str] | None = None,
    ) -> list:
        timeout_value = max(1.0, float(timeout_seconds))
        per_adapter_timeout = max(3.0, min(20.0, timeout_value))
        try:
            quotes = await self._fetch_all_quotes(
                venues=venues,
                full_discovery_venues=full_discovery_venues,
                per_adapter_timeout_seconds=per_adapter_timeout,
            )
            return quotes
        except asyncio.TimeoutError:
            venue_label = "all" if venues is None else ",".join(sorted(venues))
            discovery_label = (
                "none" if not full_discovery_venues else ",".join(sorted(full_discovery_venues))
            )
            LOGGER.warning(
                "quote refresh timed out after %.1fs (source=%s venues=%s full_discovery=%s)",
                timeout_value,
                source,
                venue_label,
                discovery_label,
            )
            return []

    async def _evaluate_quotes(
        self,
        quotes: list,
        started_at: datetime,
        source: str = "poll",
    ) -> CycleReport:
        decisions: list[OpportunityDecision] = []
        if self._settings.dry_run and self._settings.paper_strict_simulation:
            decisions.extend(self._process_paper_settlements(started_at))

        if not quotes:
            LOGGER.info("No quotes received this cycle (source=%s)", source)
            ended_at = datetime.now(timezone.utc)
            self._finder.observe_bucket_decisions(decisions)
            return CycleReport(started_at, ended_at, 0, 0, 0, tuple(decisions))
        raw_quote_count = len(quotes)
        quotes, quote_reject_counts = self._filter_quotes_for_quality(quotes, now=started_at)
        if quote_reject_counts:
            LOGGER.info(
                "quote quality gate rejected %d/%d quotes (source=%s reasons=%s)",
                raw_quote_count - len(quotes),
                raw_quote_count,
                source,
                self._format_reject_counts(quote_reject_counts),
            )
        if not quotes:
            LOGGER.info(
                "No quotes passed quality gate (source=%s rejected=%d/%d reasons=%s)",
                source,
                raw_quote_count,
                raw_quote_count,
                self._format_reject_counts(quote_reject_counts),
            )
            ended_at = datetime.now(timezone.utc)
            self._finder.observe_bucket_decisions(decisions)
            return CycleReport(started_at, ended_at, 0, 0, 0, tuple(decisions))
        venue_counts = self._quote_venue_counts(quotes)
        coverage = self._finder.coverage_snapshot(quotes)

        min_edge_threshold = self._active_min_edge_threshold()
        min_expected_profit = self._active_min_expected_profit()

        ranked_quotes = rank_quotes(quotes, self._settings.universe, now=started_at)
        strict_windowed_quotes = self._strict_mapping_window_quotes(ranked_quotes, started_at)
        scan_scope = "full"
        if self._settings.universe.enabled and self._settings.universe.hotset_size > 0:
            hotset_size = self._settings.universe.hotset_size
            hot_quotes = ranked_quotes[:hotset_size]
            opportunities = self._detect_lane_opportunities(
                hot_quotes,
                default_min_edge=min_edge_threshold,
                coverage=coverage,
                strict_windowed_quotes=strict_windowed_quotes,
            )
            near_opportunities = self._detect_lane_near_opportunities(
                hot_quotes,
                coverage=coverage,
                strict_windowed_quotes=strict_windowed_quotes,
            )
            scan_scope = "hotset"

            if not opportunities and self._settings.universe.enable_cold_scan_fallback:
                opportunities = self._detect_lane_opportunities(
                    ranked_quotes,
                    default_min_edge=min_edge_threshold,
                    coverage=coverage,
                    strict_windowed_quotes=strict_windowed_quotes,
                )
                near_opportunities = self._detect_lane_near_opportunities(
                    ranked_quotes,
                    coverage=coverage,
                    strict_windowed_quotes=strict_windowed_quotes,
                )
                scan_scope = "full_fallback"
        else:
            opportunities = self._detect_lane_opportunities(
                ranked_quotes,
                default_min_edge=min_edge_threshold,
                coverage=coverage,
                strict_windowed_quotes=strict_windowed_quotes,
            )
            near_opportunities = self._detect_lane_near_opportunities(
                ranked_quotes,
                coverage=coverage,
                strict_windowed_quotes=strict_windowed_quotes,
            )

        opportunities = self._prioritize_opportunities_for_expected_realized_pnl(
            opportunities=opportunities,
            min_expected_profit=min_expected_profit,
        )

        near_only_count = self._near_only_count(opportunities, near_opportunities)
        opportunities_by_kind = self._opportunity_kind_counts(opportunities)
        opportunities_by_family = self._opportunity_family_counts(opportunities)
        opportunities_cross_parity_split = self._cross_parity_split_counts(opportunities)
        near_by_kind = self._opportunity_kind_counts(near_opportunities)
        near_by_family = self._opportunity_family_counts(near_opportunities)
        near_cross_parity_split = self._cross_parity_split_counts(near_opportunities)

        if not opportunities:
            LOGGER.info(
                "No qualifying arbitrage opportunities (source=%s scan=%s quotes_by_venue=%s coverage=%s near_by_family=%s near_cross_parity_split=%s near_by_kind=%s)",
                source,
                scan_scope,
                self._format_venue_counts(venue_counts),
                self._format_coverage_counts(coverage),
                self._format_family_counts(near_by_family),
                self._format_cross_parity_split_counts(near_cross_parity_split),
                self._format_kind_counts(near_by_kind),
            )
            ended_at = datetime.now(timezone.utc)
            self._finder.observe_bucket_decisions(decisions)
            return CycleReport(started_at, ended_at, len(quotes), 0, near_only_count, tuple(decisions))

        LOGGER.info(
            "Found %d opportunities (near-only=%d source=%s scan=%s quotes_by_venue=%s coverage=%s by_family=%s cross_parity_split=%s by_kind=%s near_by_family=%s near_cross_parity_split=%s near_by_kind=%s)",
            len(opportunities),
            near_only_count,
            source,
            scan_scope,
            self._format_venue_counts(venue_counts),
            self._format_coverage_counts(coverage),
            self._format_family_counts(opportunities_by_family),
            self._format_cross_parity_split_counts(opportunities_cross_parity_split),
            self._format_kind_counts(opportunities_by_kind),
            self._format_family_counts(near_by_family),
            self._format_cross_parity_split_counts(near_cross_parity_split),
            self._format_kind_counts(near_by_kind),
        )

        if opportunities_by_kind.get(OpportunityKind.CROSS_VENUE.value, 0) == 0:
            covered = coverage.get("cross_mapping_pairs_covered", 0)
            total = coverage.get("cross_mapping_pairs_total", 0)
            if total > 0 and covered == 0:
                LOGGER.warning(
                    "cross_venue lane readiness low: covered_pairs=0 total_pairs=%d (likely missing venue overlap)",
                    total,
                )
        if opportunities_by_kind.get(OpportunityKind.STRUCTURAL_PARITY.value, 0) == 0:
            covered = coverage.get("structural_parity_rules_covered", 0)
            total = coverage.get("structural_parity_rules_total", 0)
            if total > 0 and covered == 0:
                LOGGER.warning(
                    "structural_parity lane readiness low: covered_rules=0 total_rules=%d (likely missing mapped quote coverage)",
                    total,
                )

        cycle_cluster_committed: dict[str, float] = {}
        cycle_cluster_open_count: dict[str, int] = {}
        locked_cluster_capital = self._cluster_locked_capital()
        open_cluster_positions = self._open_cluster_positions_by_key()
        total_bankroll = self._total_bankroll_usd()
        cluster_budget_usd = max(
            0.0,
            total_bankroll * max(0.0, self._settings.sizing.cluster_budget_fraction),
        )
        max_open_per_cluster = max(1, self._settings.sizing.max_open_positions_per_cluster)

        for opportunity in opportunities:
            scored_opportunity, constraint_assessment = self._apply_constraint_assessment(opportunity)
            cluster_key = constraint_assessment.cluster_key
            cluster_open_positions = (
                open_cluster_positions.get(cluster_key, 0)
                + cycle_cluster_open_count.get(cluster_key, 0)
            )
            cluster_used_before = (
                locked_cluster_capital.get(cluster_key, 0.0)
                + cycle_cluster_committed.get(cluster_key, 0.0)
            )
            regime_policy = self._regime_policy_for_opportunity(scored_opportunity, now=started_at)
            lane_settings = self._lane_settings_for_kind(scored_opportunity.kind)
            lane_min_edge = (
                lane_settings.min_net_edge_per_contract
                if lane_settings.min_net_edge_per_contract is not None
                else min_edge_threshold
            )
            lane_min_expected_profit = (
                lane_settings.min_expected_profit_usd
                if lane_settings.min_expected_profit_usd is not None
                else min_expected_profit
            )
            lane_min_fill_probability = (
                lane_settings.min_fill_probability
                if lane_settings.min_fill_probability is not None
                else self._settings.fill_model.min_fill_probability
            )
            lane_dynamic_fill_delta = self._lane_dynamic_fill_probability_delta.get(
                scored_opportunity.kind,
                0.0,
            )
            lane_min_realized_profit = (
                lane_settings.min_expected_realized_profit_usd
                if lane_settings.min_expected_realized_profit_usd is not None
                else self._settings.fill_model.min_expected_realized_profit_usd
            )

            edge_threshold_for_opp = lane_min_edge * max(0.0, regime_policy.edge_multiplier)
            expected_profit_threshold_for_opp = (
                lane_min_expected_profit * max(0.0, regime_policy.expected_profit_multiplier)
            )
            fill_probability_threshold = max(
                0.0,
                min(
                    1.0,
                    lane_min_fill_probability
                    + regime_policy.fill_probability_delta
                    + lane_dynamic_fill_delta,
                ),
            )
            fill_realized_profit_threshold = (
                lane_min_realized_profit
                * max(0.0, regime_policy.realized_profit_multiplier)
            )
            context_metrics: dict[str, float | int | bool | str | None] = {}
            context_metrics.update(self._constraint_metrics(constraint_assessment))
            context_metrics.update(
                self._cluster_metrics(
                    cluster_key=cluster_key,
                    cluster_budget_usd=cluster_budget_usd,
                    cluster_used_usd=cluster_used_before,
                )
            )
            context_metrics.update(
                self._regime_metrics(
                    policy=regime_policy,
                    edge_threshold=edge_threshold_for_opp,
                    expected_profit_threshold=expected_profit_threshold_for_opp,
                    fill_probability_threshold=fill_probability_threshold,
                    realized_profit_threshold=fill_realized_profit_threshold,
                )
            )
            context_metrics["fill_probability_dynamic_delta"] = lane_dynamic_fill_delta
            if cluster_open_positions >= max_open_per_cluster:
                metrics = self._decision_metrics(
                    opportunity=scored_opportunity,
                    plan=None,
                    fill_estimate=None,
                )
                metrics.update(context_metrics)
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="skipped",
                        reason="cluster_open_position_cap_reached",
                        opportunity=scored_opportunity,
                        plan=None,
                        metrics=metrics,
                    )
                )
                continue

            if scored_opportunity.net_edge_per_contract < edge_threshold_for_opp:
                metrics = self._decision_metrics(
                    opportunity=scored_opportunity,
                    plan=None,
                    fill_estimate=None,
                )
                metrics.update(context_metrics)
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="skipped",
                        reason="edge_below_dynamic_threshold",
                        opportunity=scored_opportunity,
                        plan=None,
                        metrics=metrics,
                    )
                )
                continue

            cluster_remaining = cluster_budget_usd - cluster_used_before if cluster_budget_usd > 0 else self._settings.sizing.max_dollars_per_trade
            if cluster_budget_usd > 0 and cluster_remaining <= 0.0:
                metrics = self._decision_metrics(
                    opportunity=scored_opportunity,
                    plan=None,
                    fill_estimate=None,
                )
                metrics.update(context_metrics)
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="skipped",
                        reason="cluster_budget_exhausted",
                        opportunity=scored_opportunity,
                        plan=None,
                        metrics=metrics,
                    )
                )
                continue

            max_dollars_override = self._settings.sizing.max_dollars_per_trade
            if cluster_budget_usd > 0:
                max_dollars_override = min(max_dollars_override, cluster_remaining)

            plan = self._sizer.build_trade_plan(
                scored_opportunity,
                self._state.cash_by_venue,
                min_expected_profit_override=expected_profit_threshold_for_opp,
                max_dollars_override=max_dollars_override,
                max_liquidity_fraction_override=self._settings.sizing.max_liquidity_fraction_per_trade,
            )
            if plan is None:
                metrics = self._decision_metrics(
                    opportunity=scored_opportunity,
                    plan=None,
                    fill_estimate=None,
                )
                metrics.update(context_metrics)
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="skipped",
                        reason="position_sizer_rejected",
                        opportunity=scored_opportunity,
                        plan=None,
                        metrics=metrics,
                    )
                )
                continue

            fill_estimate = self._estimate_fill(scored_opportunity, plan)
            fill_probability_for_kelly = (
                fill_estimate.all_fill_probability if fill_estimate is not None else 1.0
            )
            lane_kelly_enabled = self._settings.sizing.enable_execution_aware_kelly
            if lane_settings.use_execution_aware_kelly is not None:
                lane_kelly_enabled = lane_settings.use_execution_aware_kelly
            lane_kelly_multiplier = (
                lane_settings.kelly_fraction_multiplier
                if lane_settings.kelly_fraction_multiplier is not None
                else 1.0
            )
            lane_kelly_multiplier = max(0.0, lane_kelly_multiplier)
            lane_kelly_cap = (
                lane_settings.kelly_fraction_cap
                if lane_settings.kelly_fraction_cap is not None
                else self._settings.sizing.kelly_fraction_cap
            )
            lane_kelly_cap = max(0.0, min(1.0, lane_kelly_cap))
            lane_kelly_floor = (
                lane_settings.kelly_fraction_floor
                if lane_settings.kelly_fraction_floor is not None
                else 0.0
            )
            lane_kelly_floor = max(
                lane_kelly_floor,
                self._lane_dynamic_kelly_floor.get(scored_opportunity.kind, 0.0),
            )
            lane_kelly_floor = max(0.0, min(lane_kelly_cap, lane_kelly_floor))
            kelly_raw = 1.0
            kelly_effective = 1.0
            kelly_confidence_multiplier = self._kelly_confidence_multiplier()
            if lane_kelly_enabled:
                expected_slippage = (
                    float(fill_estimate.expected_slippage_per_contract)
                    if fill_estimate is not None
                    else 0.0
                )
                failure_loss_floor = (
                    scored_opportunity.total_cost_per_contract
                    * max(0.0, self._settings.sizing.kelly_failure_loss_floor_fraction)
                )
                failure_loss_per_contract = max(
                    failure_loss_floor,
                    self._settings.fill_model.partial_fill_penalty_per_contract + expected_slippage,
                )
                kelly_raw = self._sizer.execution_aware_kelly_fraction(
                    edge_per_contract=scored_opportunity.net_edge_per_contract,
                    cost_per_contract=scored_opportunity.total_cost_per_contract,
                    fill_probability=fill_probability_for_kelly,
                    failure_loss_per_contract=failure_loss_per_contract,
                )
                cluster_exposure_ratio = (
                    min(1.0, cluster_used_before / cluster_budget_usd)
                    if cluster_budget_usd > 0
                    else 0.0
                )
                exposure_penalty = max(
                    0.1,
                    1.0 - self._settings.sizing.cluster_exposure_penalty_lambda * cluster_exposure_ratio,
                )
                kelly_effective = min(
                    lane_kelly_cap,
                    kelly_raw * exposure_penalty * kelly_confidence_multiplier * lane_kelly_multiplier,
                )
                if lane_kelly_floor > 0.0:
                    kelly_effective = max(kelly_effective, lane_kelly_floor)
                target_contracts = int(plan.contracts * kelly_effective)
                if kelly_effective > 0.0 and target_contracts == 0:
                    target_contracts = 1
                if target_contracts <= 0:
                    metrics = self._decision_metrics(
                        opportunity=scored_opportunity,
                        plan=plan,
                        fill_estimate=fill_estimate,
                    )
                    metrics.update(context_metrics)
                    metrics["kelly_fraction_raw"] = kelly_raw
                    metrics["kelly_fraction_effective"] = kelly_effective
                    metrics["kelly_confidence_multiplier"] = kelly_confidence_multiplier
                    metrics["kelly_relative_error_mean"] = self._kelly_relative_error_mean()
                    metrics["kelly_lane_enabled"] = lane_kelly_enabled
                    metrics["kelly_lane_multiplier"] = lane_kelly_multiplier
                    metrics["kelly_lane_floor"] = lane_kelly_floor
                    metrics["kelly_lane_cap"] = lane_kelly_cap
                    decisions.append(
                        OpportunityDecision(
                            timestamp=datetime.now(timezone.utc),
                            action="skipped",
                            reason="kelly_fraction_zero",
                            opportunity=scored_opportunity,
                            plan=plan,
                            metrics=metrics,
                        )
                    )
                    continue
                if target_contracts < plan.contracts:
                    resized_plan = self._sizer.resize_trade_plan(
                        opportunity=scored_opportunity,
                        existing_plan=plan,
                        target_contracts=target_contracts,
                        min_expected_profit_override=expected_profit_threshold_for_opp,
                    )
                    if resized_plan is None:
                        metrics = self._decision_metrics(
                            opportunity=scored_opportunity,
                            plan=plan,
                            fill_estimate=fill_estimate,
                        )
                        metrics.update(context_metrics)
                        metrics["kelly_fraction_raw"] = kelly_raw
                        metrics["kelly_fraction_effective"] = kelly_effective
                        metrics["kelly_confidence_multiplier"] = kelly_confidence_multiplier
                        metrics["kelly_relative_error_mean"] = self._kelly_relative_error_mean()
                        metrics["kelly_lane_enabled"] = lane_kelly_enabled
                        metrics["kelly_lane_multiplier"] = lane_kelly_multiplier
                        metrics["kelly_lane_floor"] = lane_kelly_floor
                        metrics["kelly_lane_cap"] = lane_kelly_cap
                        decisions.append(
                            OpportunityDecision(
                                timestamp=datetime.now(timezone.utc),
                                action="skipped",
                                reason="kelly_resize_rejected",
                                opportunity=scored_opportunity,
                                plan=plan,
                                metrics=metrics,
                            )
                        )
                        continue
                    plan = resized_plan
                    fill_estimate = self._estimate_fill(scored_opportunity, plan)

            metrics = self._decision_metrics(
                opportunity=scored_opportunity,
                plan=plan,
                fill_estimate=fill_estimate,
            )
            metrics.update(context_metrics)
            metrics["kelly_fraction_raw"] = kelly_raw
            metrics["kelly_fraction_effective"] = kelly_effective
            metrics["kelly_confidence_multiplier"] = kelly_confidence_multiplier
            metrics["kelly_relative_error_mean"] = self._kelly_relative_error_mean()
            metrics["kelly_lane_enabled"] = lane_kelly_enabled
            metrics["kelly_lane_multiplier"] = lane_kelly_multiplier
            metrics["kelly_lane_floor"] = lane_kelly_floor
            metrics["kelly_lane_cap"] = lane_kelly_cap

            if self._fill_model.enabled and fill_estimate is not None:
                if fill_estimate.all_fill_probability < fill_probability_threshold:
                    decisions.append(
                        OpportunityDecision(
                            timestamp=datetime.now(timezone.utc),
                            action="skipped",
                            reason="fill_probability_below_threshold",
                            opportunity=scored_opportunity,
                            plan=plan,
                            metrics=metrics,
                        )
                    )
                    continue

                if fill_estimate.fill_quality_score < self._settings.fill_model.min_fill_quality_score:
                    decisions.append(
                        OpportunityDecision(
                            timestamp=datetime.now(timezone.utc),
                            action="skipped",
                            reason="fill_quality_below_threshold",
                            opportunity=scored_opportunity,
                            plan=plan,
                            metrics=metrics,
                        )
                    )
                    continue

                if fill_estimate.expected_realized_profit < fill_realized_profit_threshold:
                    decisions.append(
                        OpportunityDecision(
                            timestamp=datetime.now(timezone.utc),
                            action="skipped",
                            reason="expected_realized_profit_below_threshold",
                            opportunity=scored_opportunity,
                            plan=plan,
                            metrics=metrics,
                        )
                    )
                    continue

            if self._settings.live_mode and scored_opportunity.execution_style is ExecutionStyle.MAKER_ESTIMATE:
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="skipped",
                        reason="maker_estimate_disabled_in_live",
                        opportunity=scored_opportunity,
                        plan=plan,
                        metrics=metrics,
                    )
                )
                continue

            allowed, reason = self._risk.precheck(plan, self._state)
            if not allowed:
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="skipped",
                        reason=reason,
                        opportunity=scored_opportunity,
                        plan=plan,
                        metrics=metrics,
                    )
                )
                LOGGER.debug("Skipping %s (%s)", plan.market_key, reason)
                continue

            if self._settings.dry_run:
                if self._settings.paper_strict_simulation:
                    opened = self._open_paper_position(
                        opportunity=scored_opportunity,
                        plan=plan,
                        metrics=metrics,
                        now=started_at,
                    )
                    if opened is None:
                        decisions.append(
                            OpportunityDecision(
                                timestamp=datetime.now(timezone.utc),
                                action="skipped",
                                reason="paper_no_fill",
                                opportunity=scored_opportunity,
                                plan=plan,
                                metrics=metrics,
                            )
                        )
                        continue

                    filled_contracts, entry_metrics = opened
                    LOGGER.info(
                        "[DRY-STRICT] kind=%s style=%s contracts=%d filled=%d cost=%.2f det_profit=%.2f exp_realized=%.2f fill_prob=%.3f release_in=%ds legs=%s",
                        plan.kind.value,
                        plan.execution_style.value,
                        plan.contracts,
                        filled_contracts,
                        plan.capital_required,
                        plan.expected_profit,
                        float(entry_metrics.get("expected_realized_profit", 0.0) or 0.0),
                        float(entry_metrics.get("fill_probability", 0.0) or 0.0),
                        int(entry_metrics.get("settlement_eta_seconds", 0) or 0),
                        self._format_leg_summary(plan),
                    )
                    decisions.append(
                        OpportunityDecision(
                            timestamp=datetime.now(timezone.utc),
                            action="dry_run",
                            reason="paper_position_opened",
                            opportunity=scored_opportunity,
                            plan=plan,
                            filled_contracts=filled_contracts,
                            metrics=entry_metrics,
                        )
                    )
                    cycle_cluster_committed[cluster_key] = (
                        cycle_cluster_committed.get(cluster_key, 0.0) + plan.capital_required
                    )
                    cycle_cluster_open_count[cluster_key] = (
                        cycle_cluster_open_count.get(cluster_key, 0) + 1
                    )
                    continue

                LOGGER.info(
                    "[DRY] kind=%s style=%s contracts=%d cost=%.2f det_profit=%.2f exp_realized=%.2f edge=%.4f fill_prob=%.3f legs=%s",
                    plan.kind.value,
                    plan.execution_style.value,
                    plan.contracts,
                    plan.capital_required,
                    plan.expected_profit,
                    float(metrics.get("expected_realized_profit", 0.0) or 0.0),
                    plan.edge_per_contract,
                    float(metrics.get("fill_probability", 0.0) or 0.0),
                    self._format_leg_summary(plan),
                )
                metrics["realized_edge_per_contract"] = float(metrics.get("expected_realized_edge_per_contract", 0.0) or 0.0)
                metrics["realized_profit"] = float(metrics.get("expected_realized_profit", 0.0) or 0.0)
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="dry_run",
                        reason="simulated",
                        opportunity=scored_opportunity,
                        plan=plan,
                        filled_contracts=plan.contracts,
                        metrics=metrics,
                    )
                )
                cycle_cluster_committed[cluster_key] = (
                    cycle_cluster_committed.get(cluster_key, 0.0) + plan.capital_required
                )
                cycle_cluster_open_count[cluster_key] = (
                    cycle_cluster_open_count.get(cluster_key, 0) + 1
                )
                continue

            execution_started = time.perf_counter()
            execution = await self._execute_live_plan(plan)
            execution_latency_ms = (time.perf_counter() - execution_started) * 1000.0
            metrics["execution_latency_ms"] = execution_latency_ms
            metrics["partial_fill"] = execution.filled_contracts < plan.contracts

            if not execution.success:
                reason = execution.error or "unknown execution error"
                LOGGER.warning("Execution failed %s (%s)", plan.market_key, reason)
                decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="execution_failed",
                        reason=reason,
                        opportunity=scored_opportunity,
                        plan=plan,
                        metrics=metrics,
                    )
                )
                continue

            filled = execution.filled_contracts
            realized_edge = self._realized_edge_per_contract(scored_opportunity, plan, execution)
            if realized_edge is not None:
                metrics["realized_edge_per_contract"] = realized_edge
                metrics["realized_profit"] = realized_edge * filled
                self._record_kelly_calibration(
                    expected_profit=float(metrics.get("expected_realized_profit") or 0.0),
                    realized_profit=float(metrics.get("realized_profit") or 0.0),
                )

            self._risk.record_fill(plan, self._state, filled)
            LOGGER.info(
                "Filled kind=%s style=%s contracts=%d det_profit=%.2f realized_profit=%.2f legs=%s",
                plan.kind.value,
                plan.execution_style.value,
                filled,
                plan.expected_profit,
                float(metrics.get("realized_profit", 0.0) or 0.0),
                self._format_leg_summary(plan),
            )
            decisions.append(
                    OpportunityDecision(
                        timestamp=datetime.now(timezone.utc),
                        action="filled",
                        reason="executed",
                        opportunity=scored_opportunity,
                        plan=plan,
                        filled_contracts=filled,
                        metrics=metrics,
                    )
                )
            cycle_cluster_committed[cluster_key] = (
                cycle_cluster_committed.get(cluster_key, 0.0) + plan.capital_required
            )
            cycle_cluster_open_count[cluster_key] = (
                cycle_cluster_open_count.get(cluster_key, 0) + 1
            )

        ended_at = datetime.now(timezone.utc)
        self._update_dynamic_lane_fill_threshold(
            opportunities_by_kind=opportunities_by_kind,
            decisions=decisions,
        )
        self._update_dynamic_lane_kelly(
            opportunities_by_kind=opportunities_by_kind,
            decisions=decisions,
        )
        if decisions:
            LOGGER.info(self._decision_breakdown_summary(decisions))
        self._finder.observe_bucket_decisions(decisions)
        return CycleReport(
            started_at=started_at,
            ended_at=ended_at,
            quotes_count=len(quotes),
            opportunities_count=len(opportunities),
            near_opportunities_count=near_only_count,
            decisions=tuple(decisions),
        )

    def _active_min_edge_threshold(self) -> float:
        if self._settings.strategy.discovery_mode:
            return self._settings.strategy.discovery_min_net_edge_per_contract
        return self._settings.strategy.min_net_edge_per_contract

    def _active_min_expected_profit(self) -> float:
        if self._settings.strategy.discovery_mode:
            return self._settings.strategy.discovery_min_expected_profit_usd
        return self._settings.strategy.min_expected_profit_usd

    @staticmethod
    def _near_only_count(
        opportunities: list[ArbitrageOpportunity],
        near_opportunities: list[ArbitrageOpportunity],
    ) -> int:
        active_keys = {_opp_identity(opp) for opp in opportunities}
        near_keys = {_opp_identity(opp) for opp in near_opportunities}
        return len(near_keys - active_keys)

    def _active_lanes(self) -> tuple[OpportunityLane, ...]:
        return tuple(lane for lane in self._lanes if lane.settings.enabled)

    def _lane_settings_for_kind(self, kind: OpportunityKind) -> LaneTuningSettings:
        for lane in self._lanes:
            if lane.kind is kind:
                return lane.settings
        return LaneTuningSettings()

    def _detect_lane_opportunities(
        self,
        quotes: list,
        default_min_edge: float,
        coverage: dict[str, int],
        strict_windowed_quotes: list,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []
        for lane in self._active_lanes():
            lane_quotes = self._lane_quotes_for_detection(
                lane.kind,
                quotes=quotes,
                strict_windowed_quotes=strict_windowed_quotes,
                coverage=coverage,
            )
            if not lane_quotes:
                continue
            lane_min_edge = (
                lane.settings.min_net_edge_per_contract
                if lane.settings.min_net_edge_per_contract is not None
                else default_min_edge
            )
            lane_opportunities = self._finder.find_by_kind(
                lane_quotes,
                min_net_edge_override=lane_min_edge,
                kinds={lane.kind},
            )
            opportunities.extend(lane_opportunities)
        return self._deduplicate_opportunities(opportunities)

    def _detect_lane_near_opportunities(
        self,
        quotes: list,
        coverage: dict[str, int],
        strict_windowed_quotes: list,
    ) -> list[ArbitrageOpportunity]:
        max_total_cost = self._settings.strategy.near_arb_total_cost_threshold
        near_threshold = 1.0 - max_total_cost
        near: list[ArbitrageOpportunity] = []
        for lane in self._active_lanes():
            lane_quotes = self._lane_quotes_for_detection(
                lane.kind,
                quotes=quotes,
                strict_windowed_quotes=strict_windowed_quotes,
                coverage=coverage,
            )
            if not lane_quotes:
                continue
            lane_opportunities = self._finder.find_by_kind(
                lane_quotes,
                min_net_edge_override=near_threshold,
                kinds={lane.kind},
            )
            near.extend(
                opp for opp in lane_opportunities if abs(opp.payout_per_contract - 1.0) < 1e-9
            )
        return self._deduplicate_opportunities(near)

    def _strict_mapping_window_quotes(self, quotes: list, started_at: datetime) -> list:
        window_seconds = max(0.0, self._settings.strategy.strict_mapping_temporal_join_seconds)
        if window_seconds <= 0.0:
            return quotes

        cutoff = started_at - timedelta(seconds=window_seconds)
        return [quote for quote in quotes if quote.observed_at >= cutoff]

    def _lane_quotes_for_detection(
        self,
        kind: OpportunityKind,
        quotes: list,
        strict_windowed_quotes: list,
        coverage: dict[str, int],
    ) -> list:
        if kind is OpportunityKind.CROSS_VENUE:
            min_pairs = max(0, self._settings.strategy.cross_lane_min_covered_pairs)
            covered_pairs = coverage.get("cross_mapping_pairs_covered", 0)
            if covered_pairs < min_pairs:
                return []
            return strict_windowed_quotes

        if kind is OpportunityKind.STRUCTURAL_PARITY:
            min_rules = max(0, self._settings.strategy.parity_lane_min_covered_rules)
            covered_rules = coverage.get("structural_parity_rules_covered", 0)
            if covered_rules < min_rules:
                return []
            return strict_windowed_quotes

        return quotes

    def _prioritize_opportunities_for_expected_realized_pnl(
        self,
        opportunities: list[ArbitrageOpportunity],
        min_expected_profit: float,
    ) -> list[ArbitrageOpportunity]:
        scored: list[tuple[float, float, float, ArbitrageOpportunity]] = []
        for opportunity in opportunities:
            lane = self._lane_settings_for_kind(opportunity.kind)
            min_expected = (
                lane.min_expected_profit_usd
                if lane.min_expected_profit_usd is not None
                else min_expected_profit
            )
            plan = self._sizer.build_trade_plan(
                opportunity,
                self._state.cash_by_venue,
                min_expected_profit_override=min_expected,
                max_dollars_override=self._settings.sizing.max_dollars_per_trade,
                max_liquidity_fraction_override=self._settings.sizing.max_liquidity_fraction_per_trade,
            )
            if plan is None:
                score = float("-inf")
            else:
                fill_estimate = self._estimate_fill(opportunity, plan)
                score = float(fill_estimate.expected_realized_profit if fill_estimate is not None else plan.expected_profit)

            scored.append(
                (
                    score,
                    opportunity.net_edge_per_contract,
                    opportunity.match_score,
                    opportunity,
                )
            )

        scored.sort(
            key=lambda entry: (entry[0], entry[1], entry[2]),
            reverse=True,
        )
        return self._lane_fair_order(scored)

    def _lane_fair_order(
        self,
        scored: list[tuple[float, float, float, ArbitrageOpportunity]],
    ) -> list[ArbitrageOpportunity]:
        if not scored:
            return []

        by_kind: dict[OpportunityKind, list[tuple[float, float, float, ArbitrageOpportunity]]] = {
            kind: [] for kind in OpportunityKind
        }
        for item in scored:
            by_kind[item[3].kind].append(item)

        ordered: list[ArbitrageOpportunity] = []
        seen: set[tuple[str, ...]] = set()

        # Give each active lane a first-pass execution slot so one lane cannot
        # consume the cycle before others are evaluated against risk gates.
        for kind in self._EXECUTION_KIND_PRIORITY:
            lane_items = by_kind.get(kind) or []
            if not lane_items:
                continue
            opportunity = lane_items[0][3]
            identity = _opp_identity(opportunity)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(opportunity)

        for _, _, _, opportunity in scored:
            identity = _opp_identity(opportunity)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(opportunity)

        return ordered

    @staticmethod
    def _deduplicate_opportunities(
        opportunities: list[ArbitrageOpportunity],
    ) -> list[ArbitrageOpportunity]:
        if not opportunities:
            return []

        # Keep only the strongest opportunity per exact identity, then suppress
        # cross/parity aliases that represent the same leg set.
        ranked = sorted(
            opportunities,
            key=lambda opp: (
                opp.net_edge_per_contract,
                _cross_parity_alias_rank(opp),
                opp.match_score,
                opp.gross_edge_per_contract,
            ),
            reverse=True,
        )

        kept: list[ArbitrageOpportunity] = []
        seen_identity: set[tuple[str, ...]] = set()
        seen_cross_parity_alias: set[tuple[str, ...]] = set()

        for opportunity in ranked:
            identity = _opp_identity(opportunity)
            if identity in seen_identity:
                continue

            alias = _cross_parity_alias(opportunity)
            if alias is not None and alias in seen_cross_parity_alias:
                continue

            kept.append(opportunity)
            seen_identity.add(identity)
            if alias is not None:
                seen_cross_parity_alias.add(alias)

        return kept

    @staticmethod
    def _quote_venue_counts(quotes: list) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for quote in quotes:
            counts[quote.venue] += 1
        return dict(counts)

    @staticmethod
    def _filter_quotes_for_quality(
        quotes: list,
        now: datetime,
    ) -> tuple[list, dict[str, int]]:
        filtered: list = []
        reject_counts: Counter[str] = Counter()

        for quote in quotes:
            reasons = ArbEngine._quote_reject_reasons(quote, now=now)
            if reasons:
                reject_counts.update(reasons)
                continue
            filtered.append(quote)

        return filtered, dict(reject_counts)

    @staticmethod
    def _quote_reject_reasons(
        quote: object,
        now: datetime,
    ) -> list[str]:
        reasons: list[str] = []

        venue = str(getattr(quote, "venue", "") or "").strip()
        market_id = str(getattr(quote, "market_id", "") or "").strip()
        if not venue or not market_id:
            reasons.append("missing_identity")

        observed_at = getattr(quote, "observed_at", None)
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
            reasons.append("invalid_observed_at")
        elif observed_at > (now + timedelta(minutes=5)):
            reasons.append("future_observed_at")

        for attr in ("yes_buy_price", "no_buy_price"):
            value = getattr(quote, attr, None)
            if value is None:
                reasons.append("missing_required_field")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                reasons.append("non_finite_numeric")
                continue
            if not math.isfinite(numeric):
                reasons.append("non_finite_numeric")
                continue
            if numeric < 0.0 or numeric > 1.0:
                reasons.append("price_out_of_range")

        for attr in ("yes_buy_size", "no_buy_size"):
            value = getattr(quote, attr, None)
            if value is None:
                reasons.append("missing_required_field")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                reasons.append("non_finite_numeric")
                continue
            if not math.isfinite(numeric):
                reasons.append("non_finite_numeric")
                continue
            if numeric <= 0.0:
                reasons.append("non_positive_size")

        fee = getattr(quote, "fee_per_contract", 0.0)
        try:
            fee_numeric = float(fee)
        except (TypeError, ValueError):
            reasons.append("non_finite_numeric")
        else:
            if not math.isfinite(fee_numeric):
                reasons.append("non_finite_numeric")
            elif fee_numeric < 0.0:
                reasons.append("negative_fee")

        return reasons

    @staticmethod
    def _format_reject_counts(counts: dict[str, int]) -> str:
        if not counts:
            return "none"
        return ",".join(f"{key}={counts[key]}" for key in sorted(counts))

    @staticmethod
    def _format_venue_counts(counts: dict[str, int]) -> str:
        if not counts:
            return "none"
        venues = sorted(counts.keys())
        return ",".join(f"{venue}={counts.get(venue, 0)}" for venue in venues)

    @staticmethod
    def _format_coverage_counts(counts: dict[str, int]) -> str:
        def pair(label: str, seen_key: str, total_key: str) -> str:
            seen = int(counts.get(seen_key, 0))
            total = int(counts.get(total_key, 0))
            return f"{label}={seen}/{total}"

        return ",".join(
            (
                pair("cross_pairs", "cross_mapping_pairs_covered", "cross_mapping_pairs_total"),
                pair("cross_k", "cross_mapping_kalshi_refs_seen", "cross_mapping_kalshi_refs_total"),
                pair("cross_p", "cross_mapping_polymarket_refs_seen", "cross_mapping_polymarket_refs_total"),
                pair("parity_rules", "structural_parity_rules_covered", "structural_parity_rules_total"),
                pair("parity_mkts", "structural_parity_markets_seen", "structural_parity_markets_total"),
            )
        )

    @staticmethod
    def _opportunity_kind_counts(opportunities: list[ArbitrageOpportunity]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for opportunity in opportunities:
            counts[opportunity.kind.value] += 1
        return {kind.value: int(counts.get(kind.value, 0)) for kind in OpportunityKind}

    @staticmethod
    def _opportunity_family_counts(opportunities: list[ArbitrageOpportunity]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for opportunity in opportunities:
            counts[_opportunity_family_from_kind(opportunity.kind)] += 1
        return {family: int(counts.get(family, 0)) for family in _OPPORTUNITY_FAMILIES}

    @staticmethod
    def _cross_parity_split_counts(opportunities: list[ArbitrageOpportunity]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for opportunity in opportunities:
            if opportunity.kind in {OpportunityKind.CROSS_VENUE, OpportunityKind.STRUCTURAL_PARITY}:
                counts[opportunity.kind.value] += 1
        return {
            OpportunityKind.CROSS_VENUE.value: int(counts.get(OpportunityKind.CROSS_VENUE.value, 0)),
            OpportunityKind.STRUCTURAL_PARITY.value: int(counts.get(OpportunityKind.STRUCTURAL_PARITY.value, 0)),
        }

    @staticmethod
    def _format_kind_counts(counts: dict[str, int]) -> str:
        return ",".join(f"{kind.value}={counts.get(kind.value, 0)}" for kind in OpportunityKind)

    @staticmethod
    def _format_family_counts(counts: dict[str, int]) -> str:
        return ",".join(f"{family}={counts.get(family, 0)}" for family in _OPPORTUNITY_FAMILIES)

    @staticmethod
    def _format_cross_parity_split_counts(counts: dict[str, int]) -> str:
        return ",".join(
            (
                f"{OpportunityKind.CROSS_VENUE.value}={counts.get(OpportunityKind.CROSS_VENUE.value, 0)}",
                f"{OpportunityKind.STRUCTURAL_PARITY.value}={counts.get(OpportunityKind.STRUCTURAL_PARITY.value, 0)}",
            )
        )

    def _apply_constraint_assessment(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> tuple[ArbitrageOpportunity, CorrelationAssessment]:
        assessment = self._correlation_model.assess(opportunity)
        metadata = {
            **opportunity.metadata,
            "correlation_cluster": assessment.cluster_key,
            "constraint_min_payout": assessment.min_payout_per_contract,
            "constraint_adjusted_payout": assessment.adjusted_payout_per_contract,
            "constraint_valid_assignments": assessment.valid_assignments,
            "constraint_considered_markets": assessment.considered_markets,
            "constraint_assumptions": ",".join(assessment.assumptions),
        }
        updated = replace(
            opportunity,
            gross_edge_per_contract=assessment.residual_gross_edge_per_contract,
            net_edge_per_contract=assessment.residual_net_edge_per_contract,
            payout_per_contract=assessment.adjusted_payout_per_contract,
            metadata=metadata,
        )
        return updated, assessment

    @staticmethod
    def _constraint_metrics(assessment: CorrelationAssessment) -> dict[str, float | int | bool | str | None]:
        return {
            "correlation_cluster": assessment.cluster_key,
            "constraint_min_payout": assessment.min_payout_per_contract,
            "constraint_adjusted_payout": assessment.adjusted_payout_per_contract,
            "constraint_valid_assignments": assessment.valid_assignments,
            "constraint_considered_markets": assessment.considered_markets,
            "constraint_assumptions": ",".join(assessment.assumptions),
        }

    @staticmethod
    def _cluster_metrics(
        *,
        cluster_key: str,
        cluster_budget_usd: float,
        cluster_used_usd: float,
    ) -> dict[str, float | int | bool | str | None]:
        remaining = max(0.0, cluster_budget_usd - cluster_used_usd) if cluster_budget_usd > 0 else None
        ratio = min(1.0, cluster_used_usd / cluster_budget_usd) if cluster_budget_usd > 0 else 0.0
        return {
            "correlation_cluster": cluster_key,
            "cluster_budget_usd": cluster_budget_usd if cluster_budget_usd > 0 else None,
            "cluster_used_usd": cluster_used_usd if cluster_budget_usd > 0 else 0.0,
            "cluster_remaining_usd": remaining,
            "cluster_exposure_ratio": ratio,
        }

    def _open_cluster_positions_by_key(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for position in self._paper_positions:
            cluster = str(position.plan.metadata.get("correlation_cluster") or "market:unknown")
            counts[cluster] = counts.get(cluster, 0) + 1
        return counts

    def _cluster_locked_capital(self) -> dict[str, float]:
        locked: dict[str, float] = {}
        for position in self._paper_positions:
            cluster = str(position.plan.metadata.get("correlation_cluster") or "market:unknown")
            locked[cluster] = locked.get(cluster, 0.0) + sum(position.committed_capital_by_venue.values())
        return locked

    def _total_bankroll_usd(self) -> float:
        cash = sum(self._state.cash_by_venue.values())
        locked = sum(self._state.locked_capital_by_venue.values())
        return max(0.0, cash + locked)

    def _record_kelly_calibration(self, expected_profit: float, realized_profit: float) -> None:
        if not self._settings.sizing.enable_kelly_confidence_shrinkage:
            return
        denom = max(1e-6, abs(expected_profit))
        relative_error = abs(realized_profit - expected_profit) / denom
        self._kelly_relative_error_window.append(relative_error)

    def _kelly_relative_error_mean(self) -> float:
        if not self._kelly_relative_error_window:
            return 0.0
        return sum(self._kelly_relative_error_window) / len(self._kelly_relative_error_window)

    def _kelly_confidence_multiplier(self) -> float:
        if not self._settings.sizing.enable_kelly_confidence_shrinkage:
            return 1.0
        if not self._kelly_relative_error_window:
            return 1.0

        sensitivity = max(0.0, self._settings.sizing.kelly_confidence_sensitivity)
        floor = max(0.0, min(1.0, self._settings.sizing.kelly_confidence_floor))
        mean_error = self._kelly_relative_error_mean()
        confidence = 1.0 / (1.0 + (sensitivity * mean_error))
        return max(floor, min(1.0, confidence))

    def _regime_policy_for_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        *,
        now: datetime,
    ) -> OpportunityRegimePolicy:
        if not self._settings.strategy.enable_time_regime_switching:
            return OpportunityRegimePolicy(
                name="disabled",
                days_to_resolution=None,
                edge_multiplier=1.0,
                expected_profit_multiplier=1.0,
                fill_probability_delta=0.0,
                realized_profit_multiplier=1.0,
            )

        days_to_resolution = self._days_to_resolution(opportunity, now=now)
        if days_to_resolution is None:
            return OpportunityRegimePolicy(
                name="unknown",
                days_to_resolution=None,
                edge_multiplier=1.0,
                expected_profit_multiplier=1.0,
                fill_probability_delta=0.0,
                realized_profit_multiplier=1.0,
            )

        far_days = max(0.0, self._settings.strategy.far_regime_days_threshold)
        near_days = max(0.0, self._settings.strategy.near_regime_days_threshold)
        if days_to_resolution >= far_days:
            return OpportunityRegimePolicy(
                name="far_momentum",
                days_to_resolution=days_to_resolution,
                edge_multiplier=max(0.0, self._settings.strategy.far_regime_edge_multiplier),
                expected_profit_multiplier=max(0.0, self._settings.strategy.far_regime_expected_profit_multiplier),
                fill_probability_delta=self._settings.strategy.far_regime_fill_probability_delta,
                realized_profit_multiplier=max(0.0, self._settings.strategy.far_regime_realized_profit_multiplier),
            )
        if days_to_resolution <= near_days:
            return OpportunityRegimePolicy(
                name="near_mean_reversion",
                days_to_resolution=days_to_resolution,
                edge_multiplier=max(0.0, self._settings.strategy.near_regime_edge_multiplier),
                expected_profit_multiplier=max(0.0, self._settings.strategy.near_regime_expected_profit_multiplier),
                fill_probability_delta=self._settings.strategy.near_regime_fill_probability_delta,
                realized_profit_multiplier=max(0.0, self._settings.strategy.near_regime_realized_profit_multiplier),
            )
        return OpportunityRegimePolicy(
            name="mid_neutral",
            days_to_resolution=days_to_resolution,
            edge_multiplier=1.0,
            expected_profit_multiplier=1.0,
            fill_probability_delta=0.0,
            realized_profit_multiplier=1.0,
        )

    def _days_to_resolution(
        self,
        opportunity: ArbitrageOpportunity,
        *,
        now: datetime,
    ) -> float | None:
        now_ts = now.timestamp()
        candidates: list[float] = []
        for leg in opportunity.legs:
            metadata = leg.metadata
            for key in (
                "resolution_ts",
                "close_time",
                "closeTime",
                "end_time",
                "endTime",
                "market_end_time",
                "expiration_time",
                "expirationTime",
                "resolution_time",
                "resolve_time",
                "expiry",
                "expires_at",
                "end_date",
                "endDate",
                "close_date",
            ):
                parsed = self._coerce_timestamp(metadata.get(key))
                if parsed is not None:
                    candidates.append(parsed)
                    break

        if not candidates:
            return None
        resolution_ts = min(candidates)
        return max(0.0, (resolution_ts - now_ts) / 86400.0)

    @staticmethod
    def _coerce_timestamp(value: object) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1e18:
                numeric /= 1_000_000_000.0
            elif numeric > 1e15:
                numeric /= 1_000_000.0
            elif numeric > 1e12:
                numeric /= 1_000.0
            if numeric <= 0:
                return None
            return numeric

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                numeric = float(text)
                return ArbEngine._coerce_timestamp(numeric)
            except ValueError:
                pass

            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()

        return None

    @staticmethod
    def _regime_metrics(
        *,
        policy: OpportunityRegimePolicy,
        edge_threshold: float,
        expected_profit_threshold: float,
        fill_probability_threshold: float,
        realized_profit_threshold: float,
    ) -> dict[str, float | int | bool | str | None]:
        return {
            "time_regime": policy.name,
            "days_to_resolution": policy.days_to_resolution,
            "edge_threshold_active": edge_threshold,
            "expected_profit_threshold_active": expected_profit_threshold,
            "fill_probability_threshold_active": fill_probability_threshold,
            "realized_profit_threshold_active": realized_profit_threshold,
        }

    async def _execute_live_plan(self, plan: TradePlan) -> MultiLegExecutionResult:
        market_ids = {leg.market_id for leg in plan.legs}
        has_yes = any(leg.side is Side.YES for leg in plan.legs)
        has_no = any(leg.side is Side.NO for leg in plan.legs)
        if len(plan.legs) == 2 and len(plan.venues) == 1 and len(market_ids) == 1 and has_yes and has_no:
            venue = next(iter(plan.venues))
            adapter = self._exchanges[venue]
            pair = await adapter.place_pair_order(plan)

            yes_leg = next((leg for leg in plan.legs if leg.side is Side.YES), None)
            no_leg = next((leg for leg in plan.legs if leg.side is Side.NO), None)
            if yes_leg is None or no_leg is None:
                return MultiLegExecutionResult(legs=tuple(), error="missing plan sides")

            return MultiLegExecutionResult(
                legs=(
                    PlannedLegExecutionResult(leg=yes_leg, result=pair.yes_leg),
                    PlannedLegExecutionResult(leg=no_leg, result=pair.no_leg),
                ),
                error=pair.error,
            )

        # Validate all adapters up front.
        for leg in plan.legs:
            adapter = self._exchanges.get(leg.venue)
            if adapter is None:
                error_result = LegExecutionResult(
                    success=False,
                    order_id=None,
                    requested_contracts=leg.contracts,
                    filled_contracts=0,
                    average_price=None,
                    raw={"error": f"missing adapter for {leg.venue}"},
                )
                return MultiLegExecutionResult(
                    legs=(PlannedLegExecutionResult(leg=leg, result=error_result),),
                    error=f"missing adapter for {leg.venue}",
                )

        use_sequential = (
            self._settings.risk.sequential_legs
            and len(plan.legs) >= 2
            and len(plan.venues) > 1
        )

        if use_sequential:
            return await self._execute_sequential_legs(plan)
        return await self._execute_parallel_legs(plan)

    async def _execute_parallel_legs(self, plan: TradePlan) -> MultiLegExecutionResult:
        """Original parallel execution for same-venue multi-leg plans."""
        tasks = []
        plan_legs = []
        for leg in plan.legs:
            adapter = self._exchanges[leg.venue]
            tasks.append(adapter.place_single_order(leg))
            plan_legs.append(leg)

        leg_results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[PlannedLegExecutionResult] = []
        any_failure = False
        for leg, result in zip(plan_legs, leg_results):
            if isinstance(result, Exception):
                any_failure = True
                converted = LegExecutionResult(
                    success=False,
                    order_id=None,
                    requested_contracts=leg.contracts,
                    filled_contracts=0,
                    average_price=None,
                    raw={"error": str(result)},
                )
            else:
                converted = result
                if not converted.success:
                    any_failure = True

            merged.append(PlannedLegExecutionResult(leg=leg, result=converted))

        error = "one or more legs failed" if any_failure else None
        return MultiLegExecutionResult(legs=tuple(merged), error=error)

    async def _execute_sequential_legs(self, plan: TradePlan) -> MultiLegExecutionResult:
        """Execute cross-venue legs sequentially with quote freshness re-check.

        After leg A fills, re-check leg B's quote.  If the price has drifted
        beyond ``risk.leg_quote_drift_tolerance`` or the time between legs
        exceeds ``risk.leg_max_time_window_seconds``, abort remaining legs.
        """
        tolerance = self._settings.risk.leg_quote_drift_tolerance
        max_window = self._settings.risk.leg_max_time_window_seconds

        merged: list[PlannedLegExecutionResult] = []
        first_fill_ts: float | None = None
        any_failure = False
        aborted = False

        for idx, leg in enumerate(plan.legs):
            adapter = self._exchanges[leg.venue]

            # After the first leg fills, re-check quote freshness for remaining legs.
            if first_fill_ts is not None:
                elapsed = time.perf_counter() - first_fill_ts
                if elapsed > max_window:
                    LOGGER.warning(
                        "Leg time window exceeded (%.1fs > %.1fs), aborting remaining legs for %s",
                        elapsed, max_window, plan.market_key,
                    )
                    aborted = True
                    any_failure = True
                    merged.append(PlannedLegExecutionResult(
                        leg=leg,
                        result=LegExecutionResult(
                            success=False, order_id=None,
                            requested_contracts=leg.contracts,
                            filled_contracts=0, average_price=None,
                            raw={"error": "leg time window exceeded"},
                        ),
                    ))
                    continue

                drift_ok = await self._check_leg_quote_drift(leg, tolerance)
                if not drift_ok:
                    LOGGER.warning(
                        "Quote drift exceeded tolerance (%.1f%%) for %s:%s, aborting remaining legs",
                        tolerance * 100, leg.venue, leg.market_id,
                    )
                    aborted = True
                    any_failure = True
                    merged.append(PlannedLegExecutionResult(
                        leg=leg,
                        result=LegExecutionResult(
                            success=False, order_id=None,
                            requested_contracts=leg.contracts,
                            filled_contracts=0, average_price=None,
                            raw={"error": "quote drift exceeded tolerance"},
                        ),
                    ))
                    continue

            # Submit the leg.
            try:
                result = await adapter.place_single_order(leg)
            except Exception as exc:
                result = LegExecutionResult(
                    success=False, order_id=None,
                    requested_contracts=leg.contracts,
                    filled_contracts=0, average_price=None,
                    raw={"error": str(exc)},
                )

            if not result.success:
                any_failure = True
                # If a non-first leg fails, mark remaining as aborted too.
                if idx > 0:
                    aborted = True
            elif first_fill_ts is None:
                first_fill_ts = time.perf_counter()

            merged.append(PlannedLegExecutionResult(leg=leg, result=result))

            # If any leg fails, skip remaining legs.
            if not result.success:
                for remaining_leg in plan.legs[idx + 1:]:
                    any_failure = True
                    merged.append(PlannedLegExecutionResult(
                        leg=remaining_leg,
                        result=LegExecutionResult(
                            success=False, order_id=None,
                            requested_contracts=remaining_leg.contracts,
                            filled_contracts=0, average_price=None,
                            raw={"error": "prior leg failed"},
                        ),
                    ))
                break

        error = None
        if aborted:
            error = "legging aborted: quote drift or time window exceeded"
        elif any_failure:
            error = "one or more legs failed"
        return MultiLegExecutionResult(legs=tuple(merged), error=error)

    async def _check_leg_quote_drift(self, leg: TradeLegPlan, tolerance: float) -> bool:
        """Fetch a fresh quote and check whether the price has moved beyond tolerance."""
        adapter = self._exchanges.get(leg.venue)
        if adapter is None:
            return False
        try:
            fresh_quote = await adapter.fetch_quote(leg.market_id)
        except Exception:
            LOGGER.warning("Failed to fetch fresh quote for %s:%s", leg.venue, leg.market_id)
            return False

        if fresh_quote is None:
            return False

        if leg.side is Side.YES:
            current_price = fresh_quote.yes_buy_price
        else:
            current_price = fresh_quote.no_buy_price

        drift = abs(current_price - leg.limit_price)
        return drift <= tolerance

    async def _refresh_venue_balances(self) -> None:
        for adapter in self._exchanges.values():
            with suppress(Exception):
                balance = await adapter.get_available_cash()
                if balance is not None:
                    self._state.cash_by_venue[adapter.venue] = balance

    async def aclose(self) -> None:
        await asyncio.gather(*(adapter.aclose() for adapter in self._exchanges.values()))

    @staticmethod
    def _format_leg_summary(plan: TradePlan) -> str:
        parts = []
        for leg in plan.legs:
            parts.append(f"{leg.venue}:{leg.market_id}:{leg.side.value}@{leg.limit_price:.3f}")
        return "|".join(parts)

    def _open_paper_position(
        self,
        opportunity: ArbitrageOpportunity,
        plan: TradePlan,
        metrics: dict[str, float | int | bool | str | None],
        now: datetime,
    ) -> tuple[int, dict[str, float | int | bool | str | None]] | None:
        fill_probability = float(metrics.get("fill_probability", 0.0) or 0.0)
        fill_probability = max(0.0, min(1.0, fill_probability))
        filled_contracts = int(plan.contracts * fill_probability)
        if fill_probability > 0.0 and filled_contracts == 0:
            filled_contracts = 1
        if filled_contracts <= 0:
            return None

        fill_ratio = min(1.0, filled_contracts / max(1, plan.contracts))
        committed_capital_by_venue = {
            venue: capital * fill_ratio for venue, capital in plan.capital_required_by_venue.items()
        }
        expected_realized_profit = float(metrics.get("expected_realized_profit", 0.0) or 0.0) * fill_ratio
        position_lifetime_seconds = self._paper_position_lifetime_seconds(opportunity=opportunity, now=now)
        release_at = now + timedelta(seconds=position_lifetime_seconds)

        self._risk.record_fill(plan, self._state, filled_contracts)
        self._paper_positions.append(
            PaperSimPosition(
                opened_at=now,
                release_at=release_at,
                opportunity=opportunity,
                plan=plan,
                filled_contracts=filled_contracts,
                committed_capital_by_venue=committed_capital_by_venue,
                expected_realized_profit=expected_realized_profit,
            )
        )

        opened_metrics = dict(metrics)
        opened_metrics["filled_fraction"] = fill_ratio
        opened_metrics["pending_settlement_profit"] = expected_realized_profit
        opened_metrics["settlement_eta_seconds"] = position_lifetime_seconds
        opened_metrics["realized_edge_per_contract"] = None
        opened_metrics["realized_profit"] = 0.0
        opened_metrics["partial_fill"] = filled_contracts < plan.contracts
        opened_metrics["expected_realized_profit"] = expected_realized_profit
        if filled_contracts > 0:
            opened_metrics["expected_realized_edge_per_contract"] = expected_realized_profit / filled_contracts
        return filled_contracts, opened_metrics

    def _paper_position_lifetime_seconds(
        self,
        *,
        opportunity: ArbitrageOpportunity,
        now: datetime,
    ) -> int:
        static_seconds = max(1, self._settings.paper_position_lifetime_seconds)
        if not self._settings.paper_dynamic_lifetime_enabled:
            return static_seconds

        days_to_resolution = self._days_to_resolution(opportunity, now=now)
        if days_to_resolution is None:
            return static_seconds

        resolution_seconds = max(1.0, days_to_resolution * 86400.0)
        min_seconds = max(1, self._settings.paper_dynamic_lifetime_min_seconds)
        max_seconds = max(min_seconds, self._settings.paper_dynamic_lifetime_max_seconds)
        fraction = max(0.0, self._settings.paper_dynamic_lifetime_resolution_fraction)
        kind_multiplier = self._PAPER_LIFETIME_KIND_MULTIPLIERS.get(opportunity.kind, 1.0)

        scaled_seconds = resolution_seconds * fraction * max(0.0, kind_multiplier)
        bounded_seconds = max(float(min_seconds), min(float(max_seconds), scaled_seconds))
        bounded_seconds = min(bounded_seconds, resolution_seconds)
        return max(1, int(round(bounded_seconds)))

    def _process_paper_settlements(self, now: datetime) -> list[OpportunityDecision]:
        if not self._paper_positions:
            return []

        settled: list[OpportunityDecision] = []
        still_open: list[PaperSimPosition] = []

        # Phase 8C: Rolling settlement — settle positions past the min hold time
        # when rolling mode is enabled, freeing capital for new opportunities.
        rolling_enabled = getattr(self._settings, "paper_rolling_settlement_enabled", False)
        min_hold_seconds = max(1, getattr(self._settings, "paper_rolling_settlement_min_hold_seconds", 120))
        rolling_cutoff = now - timedelta(seconds=min_hold_seconds)

        for position in self._paper_positions:
            # Always settle positions that have reached their scheduled lifetime.
            if position.release_at <= now:
                pass  # Falls through to settlement below.
            elif rolling_enabled and position.opened_at <= rolling_cutoff:
                # Rolling settlement: this position has been held long enough.
                pass  # Falls through to settlement below.
            else:
                still_open.append(position)
                continue

            committed_total = sum(position.committed_capital_by_venue.values())
            if committed_total > 0:
                for venue, committed in position.committed_capital_by_venue.items():
                    share = committed / committed_total
                    profit_share = position.expected_realized_profit * share
                    self._state.cash_by_venue[venue] = self._state.cash_for(venue) + committed + profit_share
                    self._state.locked_capital_by_venue[venue] = max(0.0, self._state.locked_for(venue) - committed)

            for leg in position.plan.legs:
                self._state.unmark_open_market(leg.venue, leg.market_id)

            realized_edge = 0.0
            if position.filled_contracts > 0:
                realized_edge = position.expected_realized_profit / position.filled_contracts

            metrics: dict[str, float | int | bool | str | None] = {
                "detected_edge_per_contract": position.opportunity.net_edge_per_contract,
                "detected_profit": position.plan.expected_profit
                * min(1.0, position.filled_contracts / max(1, position.plan.contracts)),
                "fill_probability": float(position.filled_contracts / max(1, position.plan.contracts)),
                "partial_fill_probability": 1.0
                - float(position.filled_contracts / max(1, position.plan.contracts)),
                "expected_realized_edge_per_contract": realized_edge,
                "expected_realized_profit": position.expected_realized_profit,
                "expected_slippage_per_contract": 0.0,
                "fill_quality_score": 0.0,
                "adverse_selection_flag": False,
                "realized_edge_per_contract": realized_edge,
                "realized_profit": position.expected_realized_profit,
                "execution_latency_ms": None,
                "partial_fill": position.filled_contracts < position.plan.contracts,
                "settlement_lag_seconds": max(
                    0.0,
                    (position.release_at - position.opened_at).total_seconds(),
                ),
            }
            settled.append(
                OpportunityDecision(
                    timestamp=now,
                    action="settled",
                    reason="paper_position_settled",
                    opportunity=position.opportunity,
                    plan=position.plan,
                    filled_contracts=position.filled_contracts,
                    metrics=metrics,
                )
            )
            self._record_kelly_calibration(
                expected_profit=position.expected_realized_profit,
                realized_profit=position.expected_realized_profit,
            )

        self._paper_positions = still_open
        return settled

    @property
    def state(self) -> EngineState:
        return self._state

    def _estimate_fill(
        self,
        opportunity: ArbitrageOpportunity,
        plan: TradePlan | None,
    ) -> FillEstimate | None:
        if plan is None or not self._fill_model.enabled:
            return None
        correlation_mode = _choose_correlation_mode(opportunity)
        return self._fill_model.estimate(
            opportunity, plan, now=datetime.now(timezone.utc),
            correlation_mode=correlation_mode,
        )

    def _decision_metrics(
        self,
        opportunity: ArbitrageOpportunity,
        plan: TradePlan | None,
        fill_estimate: FillEstimate | None,
    ) -> dict[str, float | int | bool | str | None]:
        detected_edge = opportunity.net_edge_per_contract
        detected_profit = (plan.expected_profit if plan is not None else 0.0)
        contracts = (plan.contracts if plan is not None else 0)

        metrics: dict[str, float | int | bool | str | None] = {
            "detected_edge_per_contract": detected_edge,
            "detected_profit": detected_profit,
            "payout_per_contract": opportunity.payout_per_contract,
            "contracts": contracts,
            "fill_probability": 1.0,
            "expected_realized_edge_per_contract": detected_edge,
            "expected_realized_profit": detected_profit,
            "expected_slippage_per_contract": 0.0,
            "fill_quality_score": 0.0,
            "adverse_selection_flag": False,
            "partial_fill_probability": 0.0,
            "execution_latency_ms": None,
            "partial_fill": False,
            "realized_edge_per_contract": None,
            "realized_profit": None,
        }
        if fill_estimate is not None:
            metrics["fill_probability"] = fill_estimate.all_fill_probability
            metrics["partial_fill_probability"] = fill_estimate.partial_fill_probability
            metrics["expected_slippage_per_contract"] = fill_estimate.expected_slippage_per_contract
            metrics["fill_quality_score"] = fill_estimate.fill_quality_score
            metrics["adverse_selection_flag"] = fill_estimate.adverse_selection_flag
            metrics["expected_realized_edge_per_contract"] = fill_estimate.expected_realized_edge_per_contract
            metrics["expected_realized_profit"] = fill_estimate.expected_realized_profit
        return metrics

    def _decision_breakdown_summary(self, decisions: list[OpportunityDecision]) -> str:
        opened_actions = {"dry_run", "filled"}
        opened_by_kind: Counter[str] = Counter()
        opened_by_family: Counter[str] = Counter()
        skipped_by_kind_reason: Counter[str] = Counter()

        for decision in decisions:
            kind = decision.opportunity.kind.value
            family = _opportunity_family_from_kind(decision.opportunity.kind)
            if decision.action in opened_actions:
                opened_by_kind[kind] += 1
                opened_by_family[family] += 1
                continue
            if decision.action == "skipped":
                key = f"{kind}:{decision.reason}"
                skipped_by_kind_reason[key] += 1

        opened_kind_text = ",".join(
            f"{kind}={opened_by_kind.get(kind.value, 0)}" for kind in OpportunityKind
        )
        opened_family_text = ",".join(
            f"{family}={opened_by_family.get(family, 0)}" for family in _OPPORTUNITY_FAMILIES
        )
        top_skips = skipped_by_kind_reason.most_common(12)
        skip_text = ",".join(f"{key}={count}" for key, count in top_skips) if top_skips else "none"

        return (
            "decision breakdown opened_by_family="
            f"{opened_family_text} opened_by_kind={opened_kind_text} skipped_top={skip_text} "
            f"dynamic_kelly_floor={self._format_dynamic_kelly_floor_text(self._lane_dynamic_kelly_floor)} "
            f"dynamic_fill_delta={self._format_dynamic_fill_delta_text(self._lane_dynamic_fill_probability_delta)}"
        )

    @staticmethod
    def _format_dynamic_kelly_floor_text(floors: dict[OpportunityKind, float] | object) -> str:
        if not isinstance(floors, dict):
            return "none"
        return ",".join(
            f"{kind.value}={max(0.0, float(floors.get(kind, 0.0))):.3f}"
            for kind in OpportunityKind
        )

    @staticmethod
    def _format_dynamic_fill_delta_text(deltas: dict[OpportunityKind, float] | object) -> str:
        if not isinstance(deltas, dict):
            return "none"
        return ",".join(
            f"{kind.value}={float(deltas.get(kind, 0.0)):.3f}"
            for kind in OpportunityKind
        )

    def _update_dynamic_lane_fill_threshold(
        self,
        opportunities_by_kind: dict[str, int],
        decisions: list[OpportunityDecision],
    ) -> None:
        if not self._settings.fill_model.enable_lane_fill_autotune:
            return

        step = max(0.0, self._settings.fill_model.lane_fill_autotune_step)
        decay = max(0.0, self._settings.fill_model.lane_fill_autotune_decay_step)
        min_detected = max(1, self._settings.fill_model.lane_fill_autotune_min_detected)
        trigger_ratio = max(0.0, min(1.0, self._settings.fill_model.lane_fill_autotune_fill_skip_ratio))
        min_fill_probability = max(
            0.0,
            min(1.0, self._settings.fill_model.lane_fill_autotune_min_fill_probability),
        )
        max_relaxation = max(0.0, self._settings.fill_model.lane_fill_autotune_max_relaxation)

        opened_by_kind: Counter[str] = Counter()
        fill_skip_by_kind: Counter[str] = Counter()
        for decision in decisions:
            kind = decision.opportunity.kind.value
            if decision.action in {"dry_run", "filled"}:
                opened_by_kind[kind] += 1
            elif decision.action == "skipped" and decision.reason == "fill_probability_below_threshold":
                fill_skip_by_kind[kind] += 1

        for kind in OpportunityKind:
            lane = self._lane_settings_for_kind(kind)
            base_threshold = (
                lane.min_fill_probability
                if lane.min_fill_probability is not None
                else self._settings.fill_model.min_fill_probability
            )
            detected = int(opportunities_by_kind.get(kind.value, 0))
            opened = int(opened_by_kind.get(kind.value, 0))
            fill_skips = int(fill_skip_by_kind.get(kind.value, 0))
            current_delta = float(self._lane_dynamic_fill_probability_delta.get(kind, 0.0))

            min_delta_by_floor = min(0.0, min_fill_probability - max(0.0, base_threshold))
            min_delta = max(-max_relaxation, min_delta_by_floor)
            trigger_count = max(1, math.ceil(detected * trigger_ratio))
            should_relax = (
                detected >= min_detected
                and opened == 0
                and fill_skips >= trigger_count
            )

            updated_delta = current_delta
            if should_relax:
                updated_delta = max(min_delta, current_delta - step)
            elif opened > 0 and current_delta < 0.0:
                updated_delta = min(0.0, current_delta + decay)

            if abs(updated_delta - current_delta) > 1e-9:
                self._lane_dynamic_fill_probability_delta[kind] = updated_delta
                LOGGER.info(
                    "lane fill autotune kind=%s detected=%d opened=%d fill_skips=%d delta=%.3f->%.3f",
                    kind.value,
                    detected,
                    opened,
                    fill_skips,
                    current_delta,
                    updated_delta,
                )

    def _update_dynamic_lane_kelly(
        self,
        opportunities_by_kind: dict[str, int],
        decisions: list[OpportunityDecision],
    ) -> None:
        if not self._settings.sizing.enable_lane_kelly_autotune:
            return

        step = max(0.0, self._settings.sizing.lane_kelly_autotune_step)
        decay = max(0.0, self._settings.sizing.lane_kelly_autotune_decay_step)
        max_floor = max(0.0, self._settings.sizing.lane_kelly_autotune_max_floor)
        min_detected = max(1, self._settings.sizing.lane_kelly_autotune_min_detected)
        trigger_ratio = max(0.0, min(1.0, self._settings.sizing.lane_kelly_autotune_kelly_zero_ratio))

        opened_by_kind: Counter[str] = Counter()
        kelly_zero_skips_by_kind: Counter[str] = Counter()
        for decision in decisions:
            kind = decision.opportunity.kind.value
            if decision.action in {"dry_run", "filled"}:
                opened_by_kind[kind] += 1
            elif decision.action == "skipped" and decision.reason == "kelly_fraction_zero":
                kelly_zero_skips_by_kind[kind] += 1

        for kind in OpportunityKind:
            detected = int(opportunities_by_kind.get(kind.value, 0))
            opened = int(opened_by_kind.get(kind.value, 0))
            kelly_zero_skips = int(kelly_zero_skips_by_kind.get(kind.value, 0))
            current_floor = float(self._lane_dynamic_kelly_floor.get(kind, 0.0))

            trigger_count = max(1, math.ceil(detected * trigger_ratio))
            should_raise = (
                detected >= min_detected
                and opened == 0
                and kelly_zero_skips >= trigger_count
            )

            updated_floor = current_floor
            if should_raise:
                updated_floor = min(max_floor, current_floor + step)
            elif opened > 0 and current_floor > 0.0:
                updated_floor = max(0.0, current_floor - decay)

            if abs(updated_floor - current_floor) > 1e-9:
                self._lane_dynamic_kelly_floor[kind] = updated_floor
                LOGGER.info(
                    "lane kelly autotune kind=%s detected=%d opened=%d kelly_zero_skips=%d floor=%.3f->%.3f",
                    kind.value,
                    detected,
                    opened,
                    kelly_zero_skips,
                    current_floor,
                    updated_floor,
                )

    @staticmethod
    def _realized_edge_per_contract(
        opportunity: ArbitrageOpportunity,
        plan: TradePlan,
        execution: MultiLegExecutionResult,
    ) -> float | None:
        if execution.filled_contracts <= 0:
            return None

        realized_cost = 0.0
        for planned in execution.legs:
            avg = planned.result.average_price
            if avg is None:
                avg = planned.leg.limit_price
            realized_cost += avg

        gross_edge = opportunity.payout_per_contract - realized_cost
        return gross_edge - opportunity.fee_per_contract


def _opp_identity(opportunity: ArbitrageOpportunity) -> tuple[str, ...]:
    values: list[str] = [opportunity.kind.value, opportunity.execution_style.value]
    for leg in opportunity.legs:
        values.extend((leg.venue, leg.market_id, leg.side.value))
    return tuple(values)


def _opportunity_family_from_kind(kind: OpportunityKind) -> str:
    if kind in {OpportunityKind.CROSS_VENUE, OpportunityKind.STRUCTURAL_PARITY}:
        return "cross_parity"
    return kind.value


def _cross_parity_kind_rank(kind: OpportunityKind, *, prefer_parity: bool) -> int:
    # Note: this rank is used in a descending sort, so larger means preferred.
    if kind is OpportunityKind.STRUCTURAL_PARITY:
        return 1 if prefer_parity else 0
    if kind is OpportunityKind.CROSS_VENUE:
        return 0 if prefer_parity else 1
    return -1


def _cross_parity_alias(opportunity: ArbitrageOpportunity) -> tuple[str, ...] | None:
    if opportunity.kind not in {OpportunityKind.CROSS_VENUE, OpportunityKind.STRUCTURAL_PARITY}:
        return None

    # Alias key intentionally ignores kind, leg ordering, and side orientation
    # so equivalent cross/parity pair constructions cannot be double-counted.
    values: list[str] = [opportunity.execution_style.value, f"{opportunity.payout_per_contract:.9f}"]
    ordered_markets = sorted((leg.venue, leg.market_id) for leg in opportunity.legs)
    for venue, market_id in ordered_markets:
        values.extend((venue, market_id))
    return tuple(values)


def _cross_parity_alias_rank(opportunity: ArbitrageOpportunity) -> int:
    alias = _cross_parity_alias(opportunity)
    if alias is None:
        return -1
    return _cross_parity_kind_rank(
        opportunity.kind,
        prefer_parity=_alias_prefers_parity(alias),
    )


def _alias_prefers_parity(alias: tuple[str, ...]) -> bool:
    digest = hashlib.sha256("||".join(alias).encode("utf-8")).digest()
    return bool(digest[0] & 1)
