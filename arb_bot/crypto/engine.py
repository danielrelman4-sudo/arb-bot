"""Standalone crypto prediction trading engine for Kalshi.

Ties together the price feed, Monte Carlo model, market scanner,
edge detector, and sizing logic into a single async engine loop.

Can run independently of the cross-venue arbitrage engine or alongside it.

Usage::

    engine = CryptoEngine(settings)
    await engine.run(duration_minutes=60)
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from arb_bot.crypto.calibration import ModelCalibrator
from arb_bot.crypto.cycle_logger import CycleLogger, CycleSnapshot
from arb_bot.crypto.hawkes import HawkesIntensity
from arb_bot.crypto.ofi_calibrator import OFICalibrator
from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.edge_detector import CryptoEdge, EdgeDetector
from arb_bot.crypto.market_scanner import (
    CryptoMarket,
    CryptoMarketMeta,
    CryptoMarketQuote,
    MarketScanner,
    parse_ticker,
)
from arb_bot.crypto.price_feed import PriceFeed, PriceTick
from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate

LOGGER = logging.getLogger(__name__)


# ── Mapping from Kalshi series to Binance symbol ───────────────────
_KALSHI_TO_BINANCE = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
}


# ── Trade records ──────────────────────────────────────────────────

@dataclass
class CryptoPosition:
    """An open position in a crypto market."""

    ticker: str
    side: str              # "yes" or "no"
    contracts: int
    entry_price: float
    entry_time: float
    edge: CryptoEdge
    model_prob: float
    market_implied_prob: float


@dataclass
class CryptoTradeRecord:
    """Record of a completed trade for reporting."""

    ticker: str
    side: str
    contracts: int
    entry_price: float
    entry_time: float
    exit_price: float | None = None
    exit_time: float | None = None
    pnl: float = 0.0
    edge_at_entry: float = 0.0
    model_prob_at_entry: float = 0.0
    market_prob_at_entry: float = 0.0
    model_uncertainty: float = 0.0
    time_to_expiry_minutes: float = 0.0
    settled: bool = False


# ── Engine ─────────────────────────────────────────────────────────

class CryptoEngine:
    """Standalone async engine for crypto prediction trading on Kalshi.

    Parameters
    ----------
    settings:
        CryptoSettings instance.
    kalshi_adapter:
        Optional KalshiAdapter for live order placement. If None, the
        engine operates in paper-only mode.
    """

    def __init__(
        self,
        settings: CryptoSettings,
        kalshi_adapter: Any = None,
    ) -> None:
        self._settings = settings
        self._kalshi = kalshi_adapter

        # Subsystems
        self._price_feed = PriceFeed(
            ws_url=settings.price_feed_url,
            symbols=settings.price_feed_symbols,
            snapshot_url=settings.price_feed_snapshot_url,
            history_minutes=settings.price_history_minutes,
        )
        self._price_model = PriceModel(
            num_paths=settings.mc_num_paths,
            confidence_level=settings.confidence_level,
        )
        self._scanner = MarketScanner(
            symbols=settings.symbols,
            min_minutes_to_expiry=settings.min_minutes_to_expiry,
            max_minutes_to_expiry=settings.max_minutes_to_expiry,
            min_book_depth=settings.min_book_depth_contracts,
        )
        self._edge_detector = EdgeDetector(
            min_edge_pct=settings.min_edge_pct,
            min_edge_pct_daily=settings.min_edge_pct_daily,
            min_edge_cents=settings.min_edge_cents,
            max_model_uncertainty=settings.max_model_uncertainty,
        )
        self._calibrator = ModelCalibrator()
        self._ofi_calibrator = OFICalibrator()
        self._hawkes = HawkesIntensity(
            mu=settings.mc_jump_intensity,
            alpha=settings.hawkes_alpha,
            beta=settings.hawkes_beta,
            return_threshold_sigma=settings.hawkes_return_threshold_sigma,
        )

        # State
        self._positions: Dict[str, CryptoPosition] = {}
        self._trades: List[CryptoTradeRecord] = []
        self._session_pnl: float = 0.0
        self._cycle_count: int = 0
        self._running = False
        self._bankroll: float = settings.bankroll

        # Cycle logger (optional, for offline calibration)
        self._cycle_logger: Optional[CycleLogger] = None

        # OFI calibration cache (S1/S3: avoid recalibrating every quote)
        from arb_bot.crypto.ofi_calibrator import OFICalibrationResult
        self._ofi_cal_cache: Optional[OFICalibrationResult] = None
        self._ofi_cal_cache_time: float = 0.0

    # ── Properties ────────────────────────────────────────────────

    @property
    def positions(self) -> Dict[str, CryptoPosition]:
        return dict(self._positions)

    @property
    def trades(self) -> List[CryptoTradeRecord]:
        return list(self._trades)

    @property
    def session_pnl(self) -> float:
        return self._session_pnl

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def bankroll(self) -> float:
        return self._bankroll

    @property
    def price_feed(self) -> PriceFeed:
        return self._price_feed

    @property
    def calibrator(self) -> ModelCalibrator:
        return self._calibrator

    @property
    def ofi_calibrator(self) -> OFICalibrator:
        return self._ofi_calibrator

    # ── Main loop ─────────────────────────────────────────────────

    async def run(self, duration_minutes: float = 0) -> None:
        """Run the engine loop.

        Parameters
        ----------
        duration_minutes:
            How long to run (0 = indefinitely until stopped).
        """
        self._running = True
        start = time.monotonic()

        LOGGER.info(
            "CryptoEngine: starting (paper=%s, symbols=%s, mc_paths=%d, "
            "min_edge=%.1f%%, bankroll=$%.0f)",
            self._settings.paper_mode,
            self._settings.symbols,
            self._settings.mc_num_paths,
            self._settings.min_edge_pct * 100,
            self._bankroll,
        )

        try:
            # Bootstrap price feed with historical data
            for symbol in self._settings.price_feed_symbols:
                await self._price_feed.load_historical(symbol)

            # Start WebSocket streaming
            await self._price_feed.start()

            while self._running:
                # Check duration limit
                if duration_minutes > 0:
                    elapsed = (time.monotonic() - start) / 60.0
                    if elapsed >= duration_minutes:
                        LOGGER.info(
                            "CryptoEngine: duration limit reached (%.1f min)",
                            elapsed,
                        )
                        break

                await self._run_cycle()
                await asyncio.sleep(self._settings.scan_interval_seconds)

        except asyncio.CancelledError:
            LOGGER.info("CryptoEngine: cancelled")
        finally:
            await self._price_feed.stop()
            self._running = False
            self._log_summary()

    def stop(self) -> None:
        """Signal the engine to stop after the current cycle."""
        self._running = False

    # ── Position helpers ────────────────────────────────────────────

    def _count_positions_for_underlying(self, underlying: str) -> int:
        """Count open positions for a given underlying (e.g., 'BTC')."""
        count = 0
        for pos in self._positions.values():
            if pos.edge.market.meta.underlying == underlying:
                count += 1
        return count

    def _apply_ofi_impact(self, ofi: float, alpha: float) -> float:
        """Apply power-law impact model: drift = alpha * sgn(ofi) * |ofi|^theta."""
        theta = self._settings.ofi_impact_exponent
        if ofi == 0.0 or alpha == 0.0:
            return 0.0
        return alpha * math.copysign(abs(ofi) ** theta, ofi)

    def _compute_effective_horizon(
        self,
        binance_sym: str,
        clock_horizon_minutes: float,
    ) -> tuple:
        """Compute volume-clock effective horizon.

        Projects expected volume over remaining time to expiry using the
        current volume rate, then compares against a baseline rate to
        produce an activity-adjusted horizon.  During high-volume periods
        (crashes), the effective horizon is *longer* (more variance);
        during low-volume periods (dead hours), it is *shorter* (less
        variance).

        Returns
        -------
        tuple of (effective_horizon_minutes, projected_volume, baseline_rate)
            effective_horizon_minutes:
                ``clock_horizon * clamped(short_rate / long_rate)`` with
                clamp bounds [0.25, 4.0].  Falls back to *clock_horizon*
                when volume data is unavailable.
            projected_volume:
                ``short_rate × clock_horizon`` (total expected volume) or
                ``None`` when data is unavailable.
            baseline_rate:
                Volume-per-minute from the long-window baseline, or
                ``None`` when data is unavailable.
        """
        short_rate = self._price_feed.get_volume_flow_rate(
            binance_sym,
            window_seconds=self._settings.volume_clock_short_window_seconds,
        )
        long_rate = self._price_feed.get_volume_flow_rate(
            binance_sym,
            window_seconds=self._settings.volume_clock_baseline_window_seconds,
        )

        if long_rate <= 0 or short_rate <= 0:
            return clock_horizon_minutes, None, None

        # Project total volume: current_rate * horizon
        projected_volume = short_rate * clock_horizon_minutes

        # Effective horizon = clock_horizon * (short_rate / long_rate)
        activity_ratio = short_rate / long_rate
        floor = self._settings.volume_clock_ratio_floor
        ceiling = self._settings.volume_clock_ratio_ceiling
        activity_ratio = max(floor, min(ceiling, activity_ratio))

        effective_horizon = clock_horizon_minutes * activity_ratio
        return effective_horizon, projected_volume, long_rate

    def _get_ofi_calibration(self) -> "OFICalibrationResult":
        """Return cached OFI calibration, recalibrating on interval.

        Avoids calling ``calibrate()`` (which does OLS) on every quote
        every cycle. Respects ``ofi_recalibrate_interval_hours``.
        """
        now = time.monotonic()
        interval_secs = self._settings.ofi_recalibrate_interval_hours * 3600.0
        if (
            self._ofi_cal_cache is None
            or (now - self._ofi_cal_cache_time) >= interval_secs
        ):
            self._ofi_cal_cache = self._ofi_calibrator.calibrate()
            self._ofi_cal_cache_time = now
        return self._ofi_cal_cache

    def _update_hawkes_from_returns(self) -> None:
        """Check recent 1-min returns for each symbol and feed large ones to Hawkes."""
        now = time.monotonic()
        for binance_sym in self._settings.price_feed_symbols:
            returns = self._price_feed.get_returns(
                binance_sym, interval_seconds=60, window_minutes=1,
            )
            if len(returns) < 2:
                continue
            # Use realised vol over a slightly longer window for stability
            vol_returns = self._price_feed.get_returns(
                binance_sym, interval_seconds=60, window_minutes=5,
            )
            if len(vol_returns) < 2:
                continue
            arr = np.array(vol_returns, dtype=np.float64)
            realized_vol = float(np.std(arr, ddof=1))
            if realized_vol <= 0:
                continue
            # Feed only the most recent return
            latest_return = returns[-1]
            self._hawkes.record_return(now, latest_return, realized_vol)

    # ── Cycle logger ─────────────────────────────────────────────

    def set_cycle_logger(self, logger: CycleLogger) -> None:
        """Set a cycle logger for per-cycle data recording."""
        self._cycle_logger = logger

    def _log_cycle(self, edges_count: int) -> None:
        """Log per-cycle data to CSV if a logger is configured."""
        if self._cycle_logger is None:
            return
        import time as _time
        now = _time.monotonic()
        for binance_sym in self._settings.price_feed_symbols:
            price = self._price_feed.get_current_price(binance_sym)
            if price is None:
                continue
            ofi = self._price_feed.get_ofi(
                binance_sym, self._settings.ofi_window_seconds,
            )
            sr = self._price_feed.get_volume_flow_rate(
                binance_sym, self._settings.activity_scaling_short_window_seconds,
            )
            lr = self._price_feed.get_volume_flow_rate(
                binance_sym, self._settings.activity_scaling_long_window_seconds,
            )
            ar = sr / lr if lr > 0 and sr > 0 else 1.0
            returns = self._price_feed.get_returns(binance_sym, 60, 5)
            vol = float(np.std(returns)) if len(returns) >= 2 else 0.0
            hi = (
                self._hawkes.intensity(now)
                if self._settings.hawkes_enabled
                else self._settings.mc_jump_intensity
            )

            # Volume clock data for offline calibration
            eff_horizon = 0.0
            proj_vol = 0.0
            if self._settings.volume_clock_enabled:
                eh, pv, _ = self._compute_effective_horizon(
                    binance_sym, 10.0,  # representative 10-min horizon
                )
                eff_horizon = eh
                proj_vol = pv or 0.0

            self._cycle_logger.log(CycleSnapshot(
                timestamp=_time.time(),
                cycle=self._cycle_count,
                symbol=binance_sym,
                price=price,
                ofi=ofi,
                volume_rate_short=sr,
                volume_rate_long=lr,
                activity_ratio=ar,
                realized_vol=vol,
                hawkes_intensity=hi,
                num_edges=edges_count,
                num_positions=len(self._positions),
                session_pnl=self._session_pnl,
                bankroll=self._bankroll,
                effective_horizon=eff_horizon,
                projected_volume=proj_vol,
            ))
        self._cycle_logger.flush()

    # ── Cycle ─────────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
        """Single scan → model → detect → size → execute cycle."""
        self._cycle_count += 1
        cycle_start = time.monotonic()

        # 0. Update Hawkes intensity from recent returns
        self._update_hawkes_from_returns()

        # 1. Discover markets (from Kalshi if adapter available, else skip)
        market_quotes = await self._fetch_market_quotes()
        if not market_quotes:
            LOGGER.debug("CryptoEngine: cycle %d — no market quotes", self._cycle_count)
            return

        # 2. Generate MC paths for each underlying/horizon combination
        model_probs = self._compute_model_probabilities(market_quotes)

        # 3. Detect edges
        edges = self._edge_detector.detect_edges(market_quotes, model_probs)

        # 4. Size and execute edges
        trades_opened = 0
        for edge in edges:
            if len(self._positions) >= self._settings.max_concurrent_positions:
                break
            if edge.market.ticker in self._positions:
                continue  # Already have a position
            underlying = edge.market.meta.underlying
            if self._count_positions_for_underlying(underlying) >= self._settings.max_positions_per_underlying:
                continue

            contracts = self._compute_position_size(edge)
            if contracts <= 0:
                continue

            self._execute_paper_trade(edge, contracts)
            trades_opened += 1

        # 5. Check and settle expired positions
        self._settle_expired_positions()

        # 6. Log cycle data for offline calibration
        self._log_cycle(len(edges))

        elapsed_ms = (time.monotonic() - cycle_start) * 1000
        LOGGER.info(
            "CryptoEngine: cycle %d — %d quotes, %d edges, %d trades, "
            "%d positions, PnL=$%.2f (%.0fms)",
            self._cycle_count,
            len(market_quotes),
            len(edges),
            trades_opened,
            len(self._positions),
            self._session_pnl,
            elapsed_ms,
        )

    # ── Market data ───────────────────────────────────────────────

    async def _fetch_market_quotes(self) -> list[CryptoMarketQuote]:
        """Fetch crypto market quotes from Kalshi adapter or return empty."""
        if self._kalshi is None:
            return []

        # Use Kalshi adapter to discover crypto markets
        try:
            # Fetch markets for our crypto series
            all_tickers: list[str] = []
            for series in self._settings.symbols:
                tickers = await self._kalshi.fetch_series_tickers(series)
                all_tickers.extend(tickers)

            markets = self._scanner.parse_markets_from_tickers(all_tickers)
            filtered = self._scanner.filter_markets(markets)

            # Fetch quotes for filtered markets
            quotes: list[CryptoMarketQuote] = []
            for mkt in filtered:
                quote = await self._kalshi.fetch_single_ticker_quote(mkt.ticker)
                if quote is None:
                    continue

                tte = self._scanner.compute_time_to_expiry(mkt)
                implied = 0.5 * (
                    quote.get("yes_buy_price", 0.5)
                    + (1.0 - quote.get("no_buy_price", 0.5))
                )

                quotes.append(CryptoMarketQuote(
                    market=mkt,
                    yes_buy_price=quote.get("yes_buy_price", 0.5),
                    no_buy_price=quote.get("no_buy_price", 0.5),
                    yes_buy_size=quote.get("yes_buy_size", 0),
                    no_buy_size=quote.get("no_buy_size", 0),
                    yes_bid_price=quote.get("yes_bid_price"),
                    no_bid_price=quote.get("no_bid_price"),
                    time_to_expiry_minutes=tte,
                    implied_probability=implied,
                ))

            return quotes

        except Exception as exc:
            LOGGER.warning("CryptoEngine: market fetch failed: %s", exc)
            return []

    def inject_market_quotes(
        self, quotes: list[CryptoMarketQuote]
    ) -> None:
        """Inject quotes for testing — stores for next _run_cycle_with_quotes call."""
        self._injected_quotes = quotes

    async def run_cycle_with_quotes(
        self, quotes: list[CryptoMarketQuote]
    ) -> list[CryptoEdge]:
        """Run a single cycle with provided quotes. For testing."""
        self._cycle_count += 1

        # Update Hawkes intensity from recent returns
        self._update_hawkes_from_returns()

        model_probs = self._compute_model_probabilities(quotes)
        edges = self._edge_detector.detect_edges(quotes, model_probs)

        trades_opened = 0
        for edge in edges:
            if len(self._positions) >= self._settings.max_concurrent_positions:
                break
            if edge.market.ticker in self._positions:
                continue
            underlying = edge.market.meta.underlying
            if self._count_positions_for_underlying(underlying) >= self._settings.max_positions_per_underlying:
                continue

            contracts = self._compute_position_size(edge)
            if contracts <= 0:
                continue

            self._execute_paper_trade(edge, contracts)
            trades_opened += 1

        self._settle_expired_positions()
        self._log_cycle(len(edges))
        return edges

    # ── Model ─────────────────────────────────────────────────────

    def _compute_model_probabilities(
        self,
        quotes: list[CryptoMarketQuote],
    ) -> dict[str, ProbabilityEstimate]:
        """Compute MC probability for each quoted market."""
        result: dict[str, ProbabilityEstimate] = {}

        # Group quotes by underlying + horizon
        for mq in quotes:
            underlying = mq.market.meta.underlying
            binance_sym = _KALSHI_TO_BINANCE.get(underlying)
            if binance_sym is None:
                continue

            current_price = self._price_feed.get_current_price(binance_sym)
            if current_price is None:
                continue

            # Get vol from recent returns — select method
            vol_method = self._settings.mc_vol_method
            vol_window = self._settings.mc_vol_window_minutes

            if vol_method == "har":
                # HAR-RV: combine 1m, 5m, 30m timescales
                returns_1m = self._price_feed.get_returns(
                    binance_sym, interval_seconds=60, window_minutes=vol_window,
                )
                returns_5m = self._price_feed.get_returns(
                    binance_sym, interval_seconds=300, window_minutes=vol_window,
                )
                returns_30m = self._price_feed.get_returns(
                    binance_sym, interval_seconds=1800, window_minutes=vol_window,
                )
                vol = self._price_model.estimate_volatility_har(
                    returns_1m, returns_5m, returns_30m,
                )
            else:
                returns = self._price_feed.get_returns(
                    binance_sym,
                    interval_seconds=60,
                    window_minutes=vol_window,
                )
                vol = self._price_model.estimate_volatility(
                    returns, interval_seconds=60, method=vol_method,
                )

            if vol <= 0:
                # Use a reasonable default vol for crypto
                vol = 0.50  # 50% annualized — conservative default

            # Generate paths for this horizon
            horizon = mq.time_to_expiry_minutes
            if horizon <= 0:
                continue

            # Volume clock: replace clock-time horizon with effective
            # horizon based on projected volume.  This adjusts BOTH the
            # diffusion variance (sigma^2 * dt) AND the expected jump
            # count (lambda * dt) — jumps happen per unit of trading
            # activity, not per unit of wall clock.
            if self._settings.volume_clock_enabled and binance_sym:
                effective_horizon, _, _ = self._compute_effective_horizon(
                    binance_sym, horizon,
                )
                horizon = effective_horizon
            elif self._settings.activity_scaling_enabled and binance_sym:
                # Legacy activity scaling (backward compat)
                short_rate = self._price_feed.get_volume_flow_rate(
                    binance_sym,
                    window_seconds=self._settings.activity_scaling_short_window_seconds,
                )
                long_rate = self._price_feed.get_volume_flow_rate(
                    binance_sym,
                    window_seconds=self._settings.activity_scaling_long_window_seconds,
                )
                if long_rate > 0 and short_rate > 0:
                    activity_ratio = short_rate / long_rate
                    activity_ratio = max(0.25, min(4.0, activity_ratio))
                    vol *= math.sqrt(activity_ratio)

            # Compute OFI-based drift if enabled
            drift = 0.0
            if self._settings.ofi_enabled and binance_sym:
                ofi = self._price_feed.get_ofi(
                    binance_sym,
                    window_seconds=self._settings.ofi_window_seconds,
                )
                # Use cached calibration result, recalibrate on interval
                cal_result = self._get_ofi_calibration()
                alpha = cal_result.alpha if cal_result.alpha != 0 else self._settings.ofi_alpha
                drift = self._apply_ofi_impact(ofi, alpha)

            # Select path generator: jump diffusion or plain GBM
            if self._settings.use_jump_diffusion:
                # Use Hawkes dynamic intensity if enabled
                if self._settings.hawkes_enabled:
                    dynamic_intensity = self._hawkes.intensity(time.monotonic())
                else:
                    dynamic_intensity = self._settings.mc_jump_intensity
                paths = self._price_model.generate_paths_jump_diffusion(
                    current_price, vol, horizon, drift=drift,
                    jump_intensity=dynamic_intensity,
                    jump_mean=self._settings.mc_jump_mean,
                    jump_vol=self._settings.mc_jump_vol,
                )
            else:
                paths = self._price_model.generate_paths(
                    current_price, vol, horizon, drift=drift,
                )

            # Compute probability for this market's settlement condition
            direction = mq.market.meta.direction
            if direction not in self._settings.allowed_directions:
                continue
            strike = mq.market.meta.strike

            if direction == "above" and strike is not None:
                # Use control variate correction for above-strike markets
                prob = self._price_model.probability_above_with_control_variate(
                    paths, strike, current_price, vol, horizon, drift,
                )
            elif direction == "below" and strike is not None:
                prob = self._price_model.probability_below(paths, strike)
            elif direction == "up":
                ref_price = current_price  # fallback
                if mq.market.meta.interval_start_time is not None:
                    start_ts = mq.market.meta.interval_start_time.timestamp()
                    looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                    if looked_up is not None:
                        ref_price = looked_up
                prob = self._price_model.probability_up(paths, ref_price)
            elif direction == "down":
                # P(down) = 1 - P(up), using interval start price as reference
                ref_price = current_price  # fallback
                if mq.market.meta.interval_start_time is not None:
                    start_ts = mq.market.meta.interval_start_time.timestamp()
                    looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                    if looked_up is not None:
                        ref_price = looked_up
                up = self._price_model.probability_up(paths, ref_price)
                prob = ProbabilityEstimate(
                    probability=1.0 - up.probability,
                    ci_lower=1.0 - up.ci_upper,
                    ci_upper=1.0 - up.ci_lower,
                    uncertainty=up.uncertainty,
                    num_paths=up.num_paths,
                )
            else:
                continue

            # Scale uncertainty to reflect input uncertainty (vol estimate),
            # not just MC sampling error from Wilson CI
            unc_mult = self._settings.model_uncertainty_multiplier
            if unc_mult != 1.0:
                scaled_unc = prob.uncertainty * unc_mult
                prob = ProbabilityEstimate(
                    probability=prob.probability,
                    ci_lower=max(0.0, prob.probability - scaled_unc),
                    ci_upper=min(1.0, prob.probability + scaled_unc),
                    uncertainty=scaled_unc,
                    num_paths=prob.num_paths,
                )

            result[mq.market.ticker] = prob

        return result

    # ── Sizing ────────────────────────────────────────────────────

    def _compute_position_size(self, edge: CryptoEdge) -> int:
        """Compute contracts to trade using simplified Kelly sizing.

        Uses: f* = edge / cost, capped by kelly_fraction_cap and max_position.
        """
        if edge.side == "yes":
            cost = edge.yes_buy_price
        else:
            cost = edge.no_buy_price

        if cost <= 0 or cost >= 1.0:
            return 0

        # Simplified Kelly fraction
        kelly_f = edge.edge_cents / cost
        kelly_f = min(kelly_f, self._settings.kelly_fraction_cap)
        kelly_f = max(0.0, kelly_f)

        # Apply uncertainty haircut (Baker-McHale style)
        # k* = f / (1 + n * sigma_p^2)
        # where sigma_p is the standard error of the probability estimate
        # (Wilson CI half-width) and n is Fisher information scaling.
        if edge.model_uncertainty > 0:
            sigma_p = edge.model_uncertainty  # Wilson CI half-width
            if cost > 0 and cost < 1.0:
                n = 1.0 / (cost * (1.0 - cost))  # Fisher information scaling
                shrinkage = 1.0 / (1.0 + n * sigma_p * sigma_p)
            else:
                shrinkage = 1.0
            kelly_f *= shrinkage

        # Dollar amount to risk
        dollar_amount = kelly_f * self._bankroll
        dollar_amount = min(dollar_amount, self._settings.max_position_per_market)
        dollar_amount = min(dollar_amount, self._bankroll)

        # Convert to contracts
        contracts = int(dollar_amount / cost) if cost > 0 else 0
        return max(0, contracts)

    # ── Execution ─────────────────────────────────────────────────

    def _execute_paper_trade(self, edge: CryptoEdge, contracts: int) -> None:
        """Simulate a paper trade."""
        if edge.side == "yes":
            entry_price = edge.yes_buy_price
        else:
            entry_price = edge.no_buy_price

        # Apply paper slippage
        slippage = self._settings.paper_slippage_cents / 100.0
        entry_price = min(0.99, entry_price + slippage)

        capital_needed = entry_price * contracts
        if capital_needed > self._bankroll:
            contracts = int(self._bankroll / entry_price)
            if contracts <= 0:
                return
            capital_needed = entry_price * contracts

        self._bankroll -= capital_needed

        pos = CryptoPosition(
            ticker=edge.market.ticker,
            side=edge.side,
            contracts=contracts,
            entry_price=entry_price,
            entry_time=time.time(),
            edge=edge,
            model_prob=edge.model_prob.probability,
            market_implied_prob=edge.market_implied_prob,
        )
        self._positions[edge.market.ticker] = pos

        LOGGER.info(
            "CryptoEngine: PAPER %s %s %d@%.2f¢ (edge=%.1f%%, "
            "model=%.1f%%, market=%.1f%%, tte=%.1fm)",
            edge.side.upper(),
            edge.market.ticker,
            contracts,
            entry_price * 100,
            edge.edge_cents * 100,
            edge.model_prob.probability * 100,
            edge.market_implied_prob * 100,
            edge.time_to_expiry_minutes,
        )

    def _settle_expired_positions(self) -> None:
        """Settle positions whose markets have expired."""
        now = datetime.now(timezone.utc)
        to_settle: list[str] = []

        for ticker, pos in self._positions.items():
            if pos.edge.market.meta.expiry <= now:
                to_settle.append(ticker)

        for ticker in to_settle:
            pos = self._positions.pop(ticker)
            # In paper mode, simulate settlement outcome
            # The model probability at entry is our best estimate
            pnl = self._simulate_settlement(pos)
            self._session_pnl += pnl
            self._bankroll += pos.entry_price * pos.contracts + pnl

            record = CryptoTradeRecord(
                ticker=ticker,
                side=pos.side,
                contracts=pos.contracts,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                exit_time=time.time(),
                pnl=pnl,
                edge_at_entry=pos.edge.edge_cents,
                model_prob_at_entry=pos.model_prob,
                market_prob_at_entry=pos.market_implied_prob,
                model_uncertainty=pos.edge.model_uncertainty,
                time_to_expiry_minutes=pos.edge.time_to_expiry_minutes,
                settled=True,
            )
            self._trades.append(record)

            LOGGER.info(
                "CryptoEngine: SETTLED %s %s %d@%.2f¢ PnL=$%.4f",
                pos.side.upper(),
                ticker,
                pos.contracts,
                pos.entry_price * 100,
                pnl,
            )

    def settle_position_with_outcome(
        self, ticker: str, settled_yes: bool
    ) -> CryptoTradeRecord | None:
        """Manually settle a position with known outcome. For testing."""
        pos = self._positions.pop(ticker, None)
        if pos is None:
            return None

        if pos.side == "yes":
            pnl_per_contract = (1.0 if settled_yes else 0.0) - pos.entry_price
        else:
            pnl_per_contract = (1.0 if not settled_yes else 0.0) - pos.entry_price

        pnl = pnl_per_contract * pos.contracts
        self._session_pnl += pnl
        self._bankroll += pos.entry_price * pos.contracts + pnl

        # Record outcome for calibration
        self._calibrator.record_outcome(
            predicted_prob=pos.model_prob,
            outcome=settled_yes,
            timestamp=time.time(),
            ticker=ticker,
        )

        record = CryptoTradeRecord(
            ticker=ticker,
            side=pos.side,
            contracts=pos.contracts,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            pnl=pnl,
            edge_at_entry=pos.edge.edge_cents,
            model_prob_at_entry=pos.model_prob,
            market_prob_at_entry=pos.market_implied_prob,
            model_uncertainty=pos.edge.model_uncertainty,
            time_to_expiry_minutes=pos.edge.time_to_expiry_minutes,
            settled=True,
        )
        self._trades.append(record)
        return record

    def _simulate_settlement(self, pos: CryptoPosition) -> float:
        """Simulate settlement using model probability as true probability.

        In paper mode, we use the model's probability estimate to randomly
        settle the contract. This gives an unbiased estimate of expected PnL
        over many trades.
        """
        # Use numpy RNG for settlement
        rng = np.random.default_rng()
        settled_yes = rng.random() < pos.model_prob

        # Record outcome for calibration
        self._calibrator.record_outcome(
            predicted_prob=pos.model_prob,
            outcome=bool(settled_yes),
            timestamp=time.time(),
            ticker=pos.ticker,
        )

        if pos.side == "yes":
            pnl_per_contract = (1.0 if settled_yes else 0.0) - pos.entry_price
        else:
            pnl_per_contract = (1.0 if not settled_yes else 0.0) - pos.entry_price

        return pnl_per_contract * pos.contracts

    # ── Reporting ─────────────────────────────────────────────────

    def _log_summary(self) -> None:
        """Log session summary on shutdown."""
        total_trades = len(self._trades)
        if total_trades == 0:
            LOGGER.info("CryptoEngine: session ended — 0 trades")
            return

        wins = sum(1 for t in self._trades if t.pnl > 0)
        losses = sum(1 for t in self._trades if t.pnl < 0)
        avg_edge = sum(t.edge_at_entry for t in self._trades) / total_trades

        LOGGER.info(
            "CryptoEngine: session ended — %d trades, %d wins, %d losses, "
            "avg_edge=%.2f%%, net_pnl=$%.2f, final_bankroll=$%.2f",
            total_trades, wins, losses,
            avg_edge * 100,
            self._session_pnl,
            self._bankroll,
        )

    def export_trades_csv(self) -> str:
        """Export trades to CSV string."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "ticker", "side", "contracts", "entry_price",
            "entry_time", "exit_time", "pnl", "edge_at_entry",
            "model_prob_at_entry", "market_prob_at_entry",
            "model_uncertainty", "time_to_expiry_minutes", "settled",
        ])
        writer.writeheader()
        for t in self._trades:
            writer.writerow({
                "ticker": t.ticker,
                "side": t.side,
                "contracts": t.contracts,
                "entry_price": f"{t.entry_price:.4f}",
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "pnl": f"{t.pnl:.4f}",
                "edge_at_entry": f"{t.edge_at_entry:.4f}",
                "model_prob_at_entry": f"{t.model_prob_at_entry:.4f}",
                "market_prob_at_entry": f"{t.market_prob_at_entry:.4f}",
                "model_uncertainty": f"{t.model_uncertainty:.4f}",
                "time_to_expiry_minutes": f"{t.time_to_expiry_minutes:.1f}",
                "settled": t.settled,
            })
        return output.getvalue()
