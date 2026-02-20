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
from typing import Any, Dict, List, Optional, Tuple

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
from arb_bot.crypto.regime_detector import RegimeDetector, RegimeSnapshot, MarketRegime
from arb_bot.crypto.cycle_recorder import FilterResult
from arb_bot.crypto.staleness_detector import StalenessDetector
from arb_bot.crypto.vol_spread_model import VolSpreadModel

LOGGER = logging.getLogger(__name__)


# ── Mapping from Kalshi series to Binance symbol ───────────────────
_KALSHI_TO_BINANCE = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
}


# ── Momentum helpers (v18) ─────────────────────────────────────────


def _select_momentum_contract(
    quotes: list,
    spot_price: float,
    ofi_direction: int,
    settings,
) -> tuple | None:
    """Select best momentum contract: nearest OTM in price sweet spot.

    Returns (quote, side) or None if no suitable contract.
    """
    candidates = []
    for q in quotes:
        meta = q.market.meta
        if meta.strike is None:
            continue
        if meta.direction not in ("above", "below"):
            continue
        if q.time_to_expiry_minutes > settings.momentum_max_tte_minutes:
            continue
        if q.time_to_expiry_minutes <= 0:
            continue

        if ofi_direction > 0:  # Bullish
            if meta.direction == "above" and meta.strike > spot_price:
                side = "yes"
                buy_price = q.yes_buy_price
            elif meta.direction == "below" and meta.strike < spot_price:
                side = "no"
                buy_price = q.no_buy_price
            else:
                continue
        else:  # Bearish
            if meta.direction == "above" and meta.strike < spot_price:
                side = "no"
                buy_price = q.no_buy_price
            elif meta.direction == "below" and meta.strike > spot_price:
                side = "yes"
                buy_price = q.yes_buy_price
            else:
                continue

        if buy_price < settings.momentum_price_floor:
            continue
        if buy_price > settings.momentum_price_ceiling:
            continue

        strike_distance = abs(meta.strike - spot_price)
        candidates.append((q, side, buy_price, strike_distance))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[3])
    return (candidates[0][0], candidates[0][1])


def _compute_momentum_size(
    bankroll: float,
    buy_price: float,
    ofi_alignment: float,
    settings,
) -> int:
    """Compute momentum position size: fixed fraction scaled by OFI alignment."""
    dollar_amount = bankroll * settings.momentum_kelly_fraction * ofi_alignment
    dollar_amount = min(dollar_amount, settings.momentum_max_position)
    dollar_amount = min(dollar_amount, bankroll)
    if buy_price <= 0:
        return 0
    contracts = int(dollar_amount / buy_price)
    return min(contracts, settings.momentum_max_contracts)


def _update_ofi_streak(
    streaks: dict,
    last_dirs: dict,
    symbol: str,
    ofi_direction: int,
    ofi_alignment: float,
    min_alignment: float,
) -> int:
    """Update and return the OFI direction streak for a symbol.

    Streak increments when direction is consistent and alignment is above
    threshold. Resets to 0 on direction flip or alignment drop.
    """
    prev_dir = last_dirs.get(symbol)
    last_dirs[symbol] = ofi_direction

    if ofi_alignment < min_alignment:
        streaks[symbol] = 0
        return 0

    if prev_dir is None or ofi_direction != prev_dir:
        streaks[symbol] = 1 if prev_dir is None else 0
        return streaks[symbol]

    streaks[symbol] = streaks.get(symbol, 0) + 1
    return streaks[symbol]


def _check_ofi_acceleration(ofi_multiscale: dict) -> bool:
    """Check if OFI flow is accelerating (short-term stronger than medium-term).

    Compares |OFI at 30s| vs |OFI at 120s|. Returns True if short-term
    flow is stronger (momentum building) or if data is missing (permissive).
    """
    ofi_30 = ofi_multiscale.get(30)
    ofi_120 = ofi_multiscale.get(120)
    if ofi_30 is None or ofi_120 is None:
        return True  # Permissive when data missing
    return abs(ofi_30) > abs(ofi_120)


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
    strategy: str = "model"
    strategy_cell: str = ""  # "yes_15min", "yes_daily", "no_15min", "no_daily"


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
    actual_outcome: str = "unsettled"  # "yes" / "no" / "unsettled" / "simulated"
    strategy_cell: str = ""  # "yes_15min", "yes_daily", "no_15min", "no_daily"


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
            use_blending=not settings.garch_enabled,  # GARCH is the model — don't blend back toward market
            min_edge_pct_no_side=settings.min_edge_pct_no_side,
            min_model_market_divergence=settings.min_model_market_divergence,
            dynamic_edge_enabled=settings.dynamic_edge_threshold_enabled,
            dynamic_edge_k=settings.dynamic_edge_uncertainty_multiplier,
            market_disagreement_max=settings.market_disagreement_max,
        )
        self._calibrator = ModelCalibrator(
            min_samples_for_calibration=settings.calibration_min_samples,
            recalibrate_every=settings.calibration_recalibrate_every,
            method=settings.calibration_method,
            isotonic_min_samples=settings.calibration_isotonic_min_samples,
        )
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

        # Online recalibration: buffer of (predicted_prob, outcome_binary) per cell
        self._recal_buffer: Dict[str, List[tuple]] = {}  # cell → [(pred, outcome), ...]
        self._recal_curve: Dict[str, Any] = {}  # cell → fitted IsotonicRegression or None
        self._recal_trade_count: int = 0  # total settlements seen since last refit
        self._recal_window: int = settings.recalibration_window  # max buffer size
        self._recal_refit_interval: int = settings.recalibration_refit_interval  # refit every N

        # Cycle logger (optional, for offline calibration)
        self._cycle_logger: Optional[CycleLogger] = None

        # Cycle recorder (optional, for offline replay / Monte Carlo sim)
        self._cycle_recorder = None

        # OFI calibration cache (S1/S3: avoid recalibrating every quote)
        from arb_bot.crypto.ofi_calibrator import OFICalibrationResult
        self._ofi_cal_cache: Optional[OFICalibrationResult] = None
        self._ofi_cal_cache_time: float = 0.0

        # HTTP client for real settlement via Kalshi API
        self._http_client: Any = None  # httpx.AsyncClient when set
        self._pending_settlement: Dict[str, float] = {}  # ticker -> first_check_time

        # Momentum cooldowns (v18)
        self._momentum_cooldowns: Dict[str, float] = {}  # binance_sym -> last settle time
        self._ofi_direction_streak: Dict[str, int] = {}
        self._ofi_last_direction: Dict[str, int] = {}

        # Staleness detector (A1)
        if settings.staleness_enabled:
            self._staleness_detector: Optional[StalenessDetector] = StalenessDetector(
                spot_move_threshold=settings.staleness_spot_move_threshold,
                quote_change_threshold=settings.staleness_quote_change_threshold,
                lookback_seconds=settings.staleness_lookback_seconds,
                max_age_seconds=settings.staleness_max_age_seconds,
                edge_bonus=settings.staleness_edge_bonus,
            )
        else:
            self._staleness_detector = None

        # Cross-asset features (A5)
        self._cross_asset = None
        if settings.cross_asset_enabled:
            from arb_bot.crypto.cross_asset import CrossAssetFeatures
            self._cross_asset = CrossAssetFeatures(
                leader_symbol=settings.cross_asset_leader,
                ofi_weight=settings.cross_asset_ofi_weight,
                return_weight=settings.cross_asset_return_weight,
                vol_weight=settings.cross_asset_vol_weight,
                return_scale=settings.cross_asset_return_scale,
                max_drift=settings.cross_asset_max_drift,
            )

        # Funding rate tracker (A2)
        self._funding_tracker = None
        if settings.funding_rate_enabled:
            from arb_bot.crypto.funding_rate import FundingRateTracker
            # Map binance symbols to futures format (btcusdt -> BTCUSDT)
            futures_symbols = [s.upper() for s in settings.price_feed_symbols]
            self._funding_tracker = FundingRateTracker(
                symbols=futures_symbols,
                api_url=settings.funding_rate_api_url,
                poll_interval_seconds=settings.funding_rate_poll_interval_seconds,
                extreme_threshold=settings.funding_rate_extreme_threshold,
            )

        # VPIN calculators (B1)
        self._vpin_calculators: Dict[str, Any] = {}
        if settings.vpin_enabled:
            from arb_bot.crypto.vpin import VPINCalculator
            for sym in settings.price_feed_symbols:
                calc = VPINCalculator(
                    bucket_volume=settings.vpin_bucket_volume,
                    num_buckets=settings.vpin_num_buckets,
                    adaptive_history_size=settings.vpin_adaptive_history_size,
                )
                self._vpin_calculators[sym] = calc
                self._price_feed.register_vpin(sym, calc)

        # Student-t analytical model cache
        self._student_t_nu: Dict[str, float] = {}       # symbol -> fitted nu
        self._student_t_nu_stderr: Dict[str, float] = {} # symbol -> nu stderr
        self._student_t_last_fit_cycle: int = 0
        self._ab_student_t_probs: Dict[str, ProbabilityEstimate] = {}  # A/B storage

        # Empirical CDF model cache
        self._ab_empirical_probs: Dict[str, ProbabilityEstimate] = {}  # A/B storage
        self._empirical_returns_cache: Dict[str, list] = {}  # cached returns per symbol
        self._empirical_returns_cache_cycle: int = 0  # cache freshness

        # v42: MC GBM cache for ensemble (store per-ticker for current cycle)
        self._ab_mc_probs: Dict[str, ProbabilityEstimate] = {}  # ticker → MC prob

        # v42: Per-cell rolling stats for meta-features
        from collections import deque
        self._cell_outcomes: Dict[str, deque] = {}  # cell → deque of (timestamp, pnl, won)
        self._cell_outcome_window: int = 20  # last N trades per cell

        # v42: Funding rate persistence tracking
        self._funding_sign_persistence: Dict[str, int] = {}  # symbol → consecutive same-sign count
        self._funding_last_sign: Dict[str, int] = {}  # symbol → last sign (-1, 0, 1)

        # v45: GARCH vol-spread model
        self._garch_models: Dict[str, VolSpreadModel] = {}  # binance_sym → VolSpreadModel
        if settings.garch_enabled:
            from arb_bot.crypto.garch_vol import GarchForecaster
            for sym in settings.price_feed_symbols:
                forecaster = GarchForecaster(
                    min_obs=settings.garch_min_obs,
                    refit_interval=settings.garch_refit_interval_minutes,
                    lookback_obs=settings.garch_lookback_minutes,
                    interval_seconds=settings.garch_interval_seconds,
                )
                self._garch_models[sym] = VolSpreadModel(
                    garch_forecaster=forecaster,
                    spread_history_size=settings.garch_spread_history_size,
                    entry_z_threshold=settings.garch_vol_spread_entry_z,
                    min_moneyness_sigma=settings.garch_min_moneyness_sigma,
                    max_moneyness_sigma=settings.garch_max_moneyness_sigma,
                    uncertainty_base=settings.garch_uncertainty_base,
                )

        # Confidence scorer (B2)
        self._confidence_scorer = None
        if settings.confidence_scoring_enabled:
            from arb_bot.crypto.confidence_scorer import ConfidenceScorer
            self._confidence_scorer = ConfidenceScorer(
                min_score=settings.confidence_min_score,
                min_agreement=settings.confidence_min_agreement,
                staleness_weight=settings.confidence_staleness_weight,
                vpin_weight=settings.confidence_vpin_weight,
                ofi_weight=settings.confidence_ofi_weight,
                funding_weight=settings.confidence_funding_weight,
                vol_regime_weight=settings.confidence_vol_regime_weight,
                cross_asset_weight=settings.confidence_cross_asset_weight,
                model_edge_weight=settings.confidence_model_edge_weight,
            )

        # Feature store (C1)
        self._feature_store = None
        if settings.feature_store_enabled:
            from arb_bot.crypto.feature_store import FeatureStore
            self._feature_store = FeatureStore(
                path=settings.feature_store_path,
                min_samples_for_classifier=settings.feature_store_min_samples,
            )

        # Classifier (C2) — per-cell classifiers or global fallback
        self._classifiers: Dict[str, Any] = {}  # cell_name → BinaryClassifier
        self._classifier_last_train: float = 0.0
        if settings.classifier_enabled:
            from arb_bot.crypto.classifier import BinaryClassifier, HAS_XGBOOST
            if HAS_XGBOOST:
                # Try loading per-cell classifiers first
                _cell_paths = {
                    "yes_15min": settings.classifier_model_path_yes_15min,
                    "yes_daily": settings.classifier_model_path_yes_daily,
                    "no_15min": settings.classifier_model_path_no_15min,
                    "no_daily": settings.classifier_model_path_no_daily,
                }
                for cell_name, path in _cell_paths.items():
                    if path:
                        clf = BinaryClassifier(
                            max_depth=settings.classifier_max_depth,
                            n_estimators=settings.classifier_n_estimators,
                            learning_rate=settings.classifier_learning_rate,
                            min_child_weight=settings.classifier_min_child_weight,
                            subsample=settings.classifier_subsample,
                            use_isotonic_calibration=settings.classifier_use_isotonic_calibration,
                            model_path=path,
                        )
                        if clf.load_model():
                            self._classifiers[cell_name] = clf
                            LOGGER.info(
                                "CryptoEngine: loaded per-cell classifier for %s from %s",
                                cell_name, path,
                            )

                # Fallback: if no per-cell classifiers, try global model
                if not self._classifiers and settings.classifier_model_path:
                    global_clf = BinaryClassifier(
                        max_depth=settings.classifier_max_depth,
                        n_estimators=settings.classifier_n_estimators,
                        learning_rate=settings.classifier_learning_rate,
                        min_child_weight=settings.classifier_min_child_weight,
                        subsample=settings.classifier_subsample,
                        use_isotonic_calibration=settings.classifier_use_isotonic_calibration,
                        model_path=settings.classifier_model_path,
                    )
                    if global_clf.load_model():
                        self._classifiers["_global"] = global_clf
                        LOGGER.info(
                            "CryptoEngine: loaded global classifier from %s",
                            settings.classifier_model_path,
                        )

                if self._classifiers:
                    LOGGER.info(
                        "CryptoEngine: %d classifier(s) loaded: %s",
                        len(self._classifiers),
                        list(self._classifiers.keys()),
                    )
            else:
                LOGGER.warning("CryptoEngine: classifier_enabled but xgboost not installed")

        # Regime detector
        self._regime_detector: Optional[RegimeDetector] = None
        self._current_regime: Optional[MarketRegime] = None
        if settings.regime_detection_enabled:
            self._regime_detector = RegimeDetector(
                ofi_trend_threshold=settings.regime_ofi_trend_threshold,
                vol_expansion_threshold=settings.regime_vol_expansion_threshold,
                autocorr_window=settings.regime_autocorr_window,
                min_returns=settings.regime_min_returns,
                vpin_spike_threshold=settings.regime_vpin_spike_threshold,
            )

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

    def set_http_client(self, client: Any) -> None:
        """Set httpx client for real Kalshi settlement checks."""
        self._http_client = client

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

    def _compute_trend_drift(self, binance_sym: str) -> float:
        """Compute annualized drift from exponentially-weighted recent returns.

        Uses the last N minutes of 1-min log returns with EWM weighting
        to capture short-term momentum/trend.
        """
        window = self._settings.trend_drift_window_minutes
        returns = self._price_feed.get_returns(
            binance_sym, interval_seconds=60, window_minutes=window,
        )
        if len(returns) < 3:
            return 0.0

        arr = np.array(returns, dtype=np.float64)

        # Exponentially-weighted mean with half-life in minutes
        half_life = self._settings.trend_drift_decay
        alpha_ewm = 1.0 - np.exp(-np.log(2.0) / max(half_life, 0.1))

        # Manual EWM computation (no pandas dependency)
        n = len(arr)
        weights = np.array([(1.0 - alpha_ewm) ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()
        ewm_mean = float(np.dot(weights, arr))

        # Annualize: per-minute drift -> annualized
        # Crypto trades ~365.25 * 24 * 60 = 525,960 minutes per year
        annualized_drift = ewm_mean * 525960.0

        # Clamp to prevent extreme extrapolation
        max_drift = self._settings.trend_drift_max_annualized
        annualized_drift = max(-max_drift, min(max_drift, annualized_drift))

        return annualized_drift

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

    # ── Confidence scoring ─────────────────────────────────────────

    def _build_confidence_components(self, edge: CryptoEdge) -> "ConfidenceComponents":
        """Build confidence signal components for a detected edge.

        Each signal is aligned with the trade direction:
        - For YES trades: positive values mean signals support upward probability
        - For NO trades: signals are flipped so positive still means "supports trade"
        """
        from arb_bot.crypto.confidence_scorer import ConfidenceComponents

        underlying = edge.market.meta.underlying
        binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
        is_yes = edge.side == "yes"
        direction_sign = 1.0 if is_yes else -1.0

        # 1. Staleness signal: stale quotes in trade direction = positive
        staleness = 0.0
        if self._staleness_detector is not None:
            staleness = edge.staleness_score  # Already 0-1

        # 2. VPIN signal: signed VPIN aligned with trade direction
        vpin_signal = 0.0
        if self._settings.vpin_enabled and binance_sym in self._vpin_calculators:
            calc = self._vpin_calculators[binance_sym]
            signed_vpin = calc.get_signed_vpin()
            if signed_vpin is not None:
                # Positive signed VPIN = buy pressure. For YES trades (bullish), align.
                # For NO trades (bearish), flip.
                vpin_signal = signed_vpin * direction_sign

        # 3. OFI signal: order flow imbalance direction
        ofi_signal = 0.0
        if self._settings.ofi_enabled and binance_sym:
            ofi = self._price_feed.get_ofi(
                binance_sym, self._settings.ofi_window_seconds,
            )
            # Normalize OFI to [-1, 1] range using tanh
            normalized_ofi = math.tanh(ofi / 1000.0)  # Scale factor for typical OFI magnitudes
            ofi_signal = normalized_ofi * direction_sign

        # 4. Funding signal: funding rate drift aligned with trade
        funding_signal = 0.0
        if self._funding_tracker is not None:
            funding_sym = binance_sym.upper()
            signal = self._funding_tracker.get_funding_signal(funding_sym)
            if signal is not None and signal.drift_adjustment != 0.0:
                # Positive drift = expected upward move
                funding_signal = max(-1.0, min(1.0, signal.drift_adjustment * 100)) * direction_sign

        # 5. Vol regime signal: lower vol = more predictable = positive
        vol_regime = 0.0
        if binance_sym:
            returns = self._price_feed.get_returns(binance_sym, 60, 5)
            if len(returns) >= 2:
                vol = float(np.std(returns))
                # Low vol (< 0.003 per min) = favorable, high vol (> 0.01) = unfavorable
                if vol < 0.003:
                    vol_regime = 0.5
                elif vol < 0.006:
                    vol_regime = 0.2
                elif vol > 0.01:
                    vol_regime = -0.5
                else:
                    vol_regime = 0.0

        # 6. Cross-asset signal: leader momentum aligned with trade
        cross_asset = 0.0
        if self._cross_asset is not None and binance_sym != self._cross_asset.leader_symbol:
            cross_signal = self._cross_asset.compute_features(self._price_feed, binance_sym)
            if cross_signal is not None:
                cross_asset = max(-1.0, min(1.0, cross_signal.drift_adjustment * 10)) * direction_sign

        # 7. Model edge signal: edge strength normalized to [0, 1]
        model_edge = 0.0
        if edge.edge_cents > 0:
            # Map edge: 0.05 (minimum) -> 0.2, 0.15 -> 0.6, 0.25+ -> 1.0
            model_edge = min(1.0, edge.edge_cents / 0.25)

        return ConfidenceComponents(
            staleness_signal=staleness,
            vpin_signal=vpin_signal,
            ofi_signal=ofi_signal,
            funding_signal=funding_signal,
            vol_regime_signal=vol_regime,
            cross_asset_signal=cross_asset,
            model_edge_signal=model_edge,
        )

    # ── Classifier (C2) ────────────────────────────────────────────

    def _maybe_retrain_classifier(self) -> None:
        """Check if the classifier should be retrained based on interval and sample count.

        For per-cell classifiers, online retraining is skipped — use the
        offline ``train_classifier.py --cell`` script instead.  Only the
        global fallback (``_global`` key) supports online retraining.
        """
        if not self._classifiers or self._feature_store is None:
            return

        # Only retrain global classifier online
        global_clf = self._classifiers.get("_global")
        if global_clf is None:
            return

        now = time.time()
        interval = self._settings.classifier_retrain_interval_hours * 3600

        # Check if enough time has passed since last training
        if global_clf.is_trained and (now - self._classifier_last_train) < interval:
            return

        # Check if we have enough samples
        if not self._feature_store.has_enough_samples():
            return

        # Load training data and train
        from arb_bot.crypto.feature_store import FEATURE_COLUMNS
        X, y = self._feature_store.load_training_data()
        if len(y) < self._settings.classifier_min_training_samples:
            return

        LOGGER.info("CryptoEngine: retraining global classifier with %d samples", len(y))
        report = global_clf.train(X, y, feature_names=FEATURE_COLUMNS)
        self._classifier_last_train = now

        LOGGER.info(
            "CryptoEngine: global classifier retrained -- accuracy=%.3f brier=%.4f",
            report.accuracy, report.brier_score,
        )

    def _compute_classifier_probability(
        self,
        edge_features: np.ndarray,
        strategy_cell: str = "",
    ) -> Optional[ProbabilityEstimate]:
        """Compute probability from the trained classifier.

        Looks up per-cell classifier first, falls back to global.
        Returns None if classifier is not available or not trained.
        """
        if not self._classifiers:
            return None

        # Look up per-cell classifier, then global fallback
        classifier = self._classifiers.get(strategy_cell) or self._classifiers.get("_global")
        if classifier is None or not classifier.is_trained:
            return None

        from arb_bot.crypto.classifier import ProbabilityEstimate as ClassifierEstimate
        result = classifier.predict(edge_features)
        if not result.is_classifier:
            return None

        # Convert classifier's ProbabilityEstimate to price_model's ProbabilityEstimate
        return ProbabilityEstimate(
            probability=result.probability,
            ci_lower=max(0.0, result.probability - result.uncertainty),
            ci_upper=min(1.0, result.probability + result.uncertainty),
            uncertainty=result.uncertainty,
            num_paths=0,  # Not MC-based
        )

    def _classifier_veto(self, edge: CryptoEdge) -> Tuple[bool, float, Optional["FeatureVector"]]:
        """Check if the classifier vetoes this trade.

        Uses per-cell classifier if available, falls back to global.
        Uses per-cell veto threshold if configured.

        Returns (vetoed, p_win, feature_vector).
        - vetoed: True if classifier says don't trade (p_win < threshold)
        - p_win: classifier's estimated P(win), or -1 if not available
        - feature_vector: built FeatureVector (reuse for feature store recording)
        """
        if not self._classifiers:
            return False, -1.0, None

        # Look up per-cell classifier, then global fallback
        cell_name = edge.strategy_cell or ""
        classifier = self._classifiers.get(cell_name) or self._classifiers.get("_global")
        if classifier is None or not classifier.is_trained:
            return False, -1.0, None

        from arb_bot.crypto.feature_store import FEATURE_COLUMNS, _REGIME_ENCODE

        # Build feature vector
        fv = self._build_feature_vector(edge)

        # Convert to numpy array matching training schema
        n_base = len(FEATURE_COLUMNS)
        features = np.zeros(n_base + 2)  # +2 for side_numeric, is_daily
        for j, col in enumerate(FEATURE_COLUMNS):
            val = getattr(fv, col, 0.0)
            if col == "regime":
                features[j] = float(_REGIME_ENCODE.get(str(val), 0))
            else:
                try:
                    features[j] = float(val)
                except (ValueError, TypeError):
                    features[j] = 0.0

        # Extra derived features (must match training)
        features[n_base + 0] = 1.0 if edge.side == "yes" else 0.0  # side_numeric
        features[n_base + 1] = 1.0 if "daily" in (edge.strategy_cell or "") else 0.0  # is_daily

        # Predict
        result = classifier.predict(features)
        p_win = result.probability if result.is_classifier else -1.0

        # Per-cell veto threshold (falls back to global)
        _threshold_map = {
            "yes_15min": self._settings.classifier_veto_threshold_yes_15min,
            "yes_daily": self._settings.classifier_veto_threshold_yes_daily,
            "no_15min": self._settings.classifier_veto_threshold_no_15min,
            "no_daily": self._settings.classifier_veto_threshold_no_daily,
        }
        threshold = _threshold_map.get(cell_name, self._settings.classifier_veto_threshold)
        vetoed = p_win >= 0 and p_win < threshold

        # Log with classifier source info
        clf_source = cell_name if cell_name in self._classifiers else "_global"
        if vetoed:
            LOGGER.info(
                "CryptoEngine: classifier veto %s — P(win)=%.2f < %.2f "
                "(cell=%s, clf=%s)",
                edge.market.ticker, p_win, threshold,
                cell_name or "unknown", clf_source,
            )
        else:
            LOGGER.debug(
                "CryptoEngine: classifier pass %s — P(win)=%.2f "
                "(cell=%s, clf=%s)",
                edge.market.ticker, p_win,
                cell_name or "unknown", clf_source,
            )

        return vetoed, p_win, fv

    # ── Cycle logger ─────────────────────────────────────────────

    def set_cycle_logger(self, logger: CycleLogger) -> None:
        """Set a cycle logger for per-cycle data recording."""
        self._cycle_logger = logger

    def set_cycle_recorder(self, recorder) -> None:
        """Set a cycle recorder for full-state offline replay."""
        self._cycle_recorder = recorder

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
        _rec_cycle_start = time.time()

        # 0. Update Hawkes intensity from recent returns
        self._update_hawkes_from_returns()

        # 1. Discover markets (from Kalshi if adapter available, else skip)
        market_quotes = await self._fetch_market_quotes()
        if not market_quotes:
            LOGGER.debug("CryptoEngine: cycle %d — no market quotes", self._cycle_count)
            return

        # 1b. Record quote snapshots for staleness detection
        if self._staleness_detector is not None:
            for mq in market_quotes:
                self._staleness_detector.record_quote_snapshot(
                    mq.market.ticker,
                    mq.yes_buy_price,
                    mq.no_buy_price,
                )

        # 1c. Maybe retrain classifier (C2)
        self._maybe_retrain_classifier()

        # 1d. VPIN gate: per-symbol tiered gate with momentum zone (v18/v20)
        _momentum_zone = False
        _vpin_halted_symbols: set = set()   # Symbols above ceiling — fully halted
        _vpin_momentum_symbols: set = set() # Symbols in momentum zone
        _vpin_normal_symbols: set = set()   # Symbols below momentum floor — model-path ok
        _vpin_stale_symbols: set = set()    # Symbols with stale VPIN (treated as normal)
        if self._settings.vpin_halt_enabled and self._vpin_calculators:
            for sym, calc in self._vpin_calculators.items():
                vpin_val = calc.get_vpin()
                if vpin_val is None:
                    _vpin_normal_symbols.add(sym)
                    if hasattr(calc, "is_stale") and calc.is_stale:
                        _vpin_stale_symbols.add(sym)
                    continue
                # Get effective thresholds (adaptive or static)
                mom_floor, halt_ceil, is_adaptive = self._get_vpin_thresholds(sym)
                if self._settings.momentum_enabled:
                    if vpin_val > halt_ceil:
                        _vpin_halted_symbols.add(sym)
                    elif vpin_val >= mom_floor:
                        _vpin_momentum_symbols.add(sym)
                    else:
                        _vpin_normal_symbols.add(sym)
                else:
                    if vpin_val > halt_ceil:
                        _vpin_halted_symbols.add(sym)
                    else:
                        _vpin_normal_symbols.add(sym)

            # Log per-symbol VPIN status
            if _vpin_stale_symbols:
                stale_details = ", ".join(
                    f"{s}(stale {self._vpin_calculators[s].seconds_since_last_trade:.0f}s)"
                    if hasattr(self._vpin_calculators[s], "seconds_since_last_trade")
                    else s
                    for s in sorted(_vpin_stale_symbols)
                )
                LOGGER.info("CryptoEngine: VPIN stale (treated as normal): %s", stale_details)
            if _vpin_halted_symbols:
                halted_details = []
                for s in sorted(_vpin_halted_symbols):
                    v = self._vpin_calculators[s].get_vpin()
                    if v is not None:
                        _, hc, ada = self._get_vpin_thresholds(s)
                        tag = "adaptive" if ada else "static"
                        halted_details.append(f"{s}={v:.3f}(halt@{hc:.3f}/{tag})")
                if halted_details:
                    LOGGER.info("CryptoEngine: VPIN halt symbols: %s", ", ".join(halted_details))
            if _vpin_momentum_symbols:
                mom_details = []
                for s in sorted(_vpin_momentum_symbols):
                    v = self._vpin_calculators[s].get_vpin()
                    if v is not None:
                        mf, hc, ada = self._get_vpin_thresholds(s)
                        tag = "adaptive" if ada else "static"
                        mom_details.append(f"{s}={v:.3f}(zone@{mf:.3f}-{hc:.3f}/{tag})")
                if mom_details:
                    LOGGER.info("CryptoEngine: VPIN momentum symbols: %s", ", ".join(mom_details))
                _momentum_zone = True

            # Only skip entire cycle if ALL symbols are halted
            if _vpin_halted_symbols and not _vpin_momentum_symbols and not _vpin_normal_symbols:
                LOGGER.info("CryptoEngine: VPIN full halt — all symbols above ceiling, skipping cycle")
                return

        # 2. Generate MC paths for each underlying/horizon combination
        model_probs = self._compute_model_probabilities(market_quotes)

        # 2b. Classify market regime
        self._update_regime()

        # 2c. Momentum path (v18/v20): try momentum trades for momentum-zone symbols
        if _momentum_zone and self._settings.momentum_enabled:
            await self._try_momentum_trades(market_quotes, momentum_symbols=_vpin_momentum_symbols)

        # If no symbols are in normal zone, skip model-path trades for 15-min markets.
        # Daily (above/below TTE>30m) markets are still processed — VPIN measures
        # short-term microstructure toxicity, not structural model-vs-market edges.
        _has_daily_quotes = any(
            mq.market.meta.direction in ("above", "below") and mq.time_to_expiry_minutes > 30
            for mq in market_quotes
        )
        if not _vpin_normal_symbols and (_vpin_halted_symbols or _vpin_momentum_symbols) and not _has_daily_quotes:
            await self._settle_expired_positions()
            cycle_elapsed = time.monotonic() - cycle_start
            if cycle_elapsed > 1.0:
                LOGGER.info("CryptoEngine: no normal-VPIN symbols, cycle %d took %.1fs",
                            self._cycle_count, cycle_elapsed)
            return

        # ── Cycle recorder Hook 1: market snapshots ──
        _rec_cycle_id = None
        _filter_results: Dict[str, FilterResult] = {}
        if self._cycle_recorder is not None:
            _rec_cycle_id = self._cycle_recorder.begin_cycle(self._cycle_count, _rec_cycle_start)
            _rec_spots: Dict[str, float] = {}
            _rec_vpins: Dict[str, tuple] = {}
            _rec_ofis: Dict[str, Dict[int, float]] = {}
            _rec_returns: Dict[str, list] = {}
            for sym in self._settings.price_feed_symbols:
                _rec_spots[sym] = self._price_feed.get_current_price(sym) or 0.0
                if self._settings.vpin_enabled and sym in self._vpin_calculators:
                    calc = self._vpin_calculators[sym]
                    _rec_vpins[sym] = (calc.get_vpin() or 0.0, calc.get_signed_vpin() or 0.0, calc.get_vpin_trend() or 0.0)
                if self._settings.ofi_enabled:
                    _rec_ofis[sym] = self._price_feed.get_ofi_multiscale(sym, [30, 60, 120, 300])
                _rec_returns[sym] = list(self._price_feed.get_returns(sym, interval_seconds=60, window_minutes=self._settings.mc_vol_window_minutes))
            self._cycle_recorder.record_market_snapshots(
                _rec_cycle_id, market_quotes, _rec_spots, _rec_vpins, _rec_ofis, _rec_returns,
            )

        # 3. Detect edges
        edges = self._edge_detector.detect_edges(market_quotes, model_probs)

        # 3-vpin. Filter edges for symbols in halted or momentum zone (model-path only)
        # Daily (above/below) markets with TTE > 30m are EXEMPT from VPIN filtering:
        # VPIN measures short-term order toxicity, not whether a strike will be
        # reached over the next hour+.  Daily edges are structural (model vs market),
        # not microstructure-driven.
        if _vpin_halted_symbols or _vpin_momentum_symbols:
            blocked_syms = _vpin_halted_symbols | _vpin_momentum_symbols
            pre_count = len(edges)
            edges = [
                e for e in edges
                if _KALSHI_TO_BINANCE.get(e.market.meta.underlying, "") not in blocked_syms
                or (e.market.meta.direction in ("above", "below")
                    and e.time_to_expiry_minutes > 30)
            ]
            if len(edges) < pre_count:
                LOGGER.info(
                    "CryptoEngine: VPIN symbol filter removed %d/%d model-path edges (halted/momentum symbols)",
                    pre_count - len(edges), pre_count,
                )

        # ── Cycle recorder Hook 2: snapshot raw edges ──
        all_raw_edges = list(edges)
        if _rec_cycle_id is not None:
            _filter_results = {e.market.ticker: FilterResult() for e in edges}

        # 3-cell. Per-cell model adjustments (blending weight, prob haircut)
        edges = self._apply_cell_model_adjustments(edges)

        # 3-regime. Regime min edge filter
        if self._settings.regime_min_edge_enabled and self._current_regime is not None:
            regime = self._current_regime.regime
            _regime_min_edges = {
                "mean_reverting": self._settings.regime_min_edge_mean_reverting,
                "trending_up": self._settings.regime_min_edge_trending,
                "trending_down": self._settings.regime_min_edge_trending,
                "high_vol": self._settings.regime_min_edge_high_vol,
            }
            regime_min = _regime_min_edges.get(regime, 0.0)
            if regime_min > 0:
                pre_filter_tickers = {e.market.ticker for e in edges}
                pre_count = len(edges)
                edges = [e for e in edges if e.edge_cents >= regime_min]
                if len(edges) < pre_count:
                    LOGGER.info(
                        "CryptoEngine: regime min_edge filter (%s, min=%.2f) removed %d/%d edges",
                        regime, regime_min, pre_count - len(edges), pre_count,
                    )
                # ── Cycle recorder Hook 3: regime filter tracking ──
                if _rec_cycle_id is not None:
                    post_filter_tickers = {e.market.ticker for e in edges}
                    for ticker in pre_filter_tickers - post_filter_tickers:
                        if ticker in _filter_results:
                            _filter_results[ticker].passed_regime_min_edge = False
                            if _filter_results[ticker].reject_reason is None:
                                _filter_results[ticker].reject_reason = "regime_min_edge"
                    for ticker in post_filter_tickers:
                        if ticker in _filter_results:
                            _filter_results[ticker].passed_regime_min_edge = True

        # 3-zscore. Z-score reachability post-edge filter
        if self._settings.zscore_filter_enabled:
            pre_zscore_tickers = {e.market.ticker for e in edges}
            zscore_filtered = []
            for edge in edges:
                underlying = edge.market.meta.underlying
                binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                if not binance_sym:
                    zscore_filtered.append(edge)
                    continue
                spot = self._price_feed.get_current_price(binance_sym)
                if spot is None:
                    zscore_filtered.append(edge)
                    continue
                z_strike = edge.market.meta.strike
                direction = edge.market.meta.direction
                if direction in ("up", "down"):
                    z_strike = None
                    if edge.market.meta.interval_start_time is not None:
                        start_ts = edge.market.meta.interval_start_time.timestamp()
                        z_strike = self._price_feed.get_price_at_time(binance_sym, start_ts)
                if z_strike is None or z_strike <= 0:
                    zscore_filtered.append(edge)
                    continue
                local_returns = self._price_feed.get_returns(
                    binance_sym, interval_seconds=60,
                    window_minutes=self._settings.zscore_vol_window_minutes,
                )
                local_vol = self._price_model.estimate_volatility(
                    local_returns, interval_seconds=60,
                )
                if local_vol <= 0:
                    local_vol = 0.50
                z = self._compute_zscore(spot, z_strike, local_vol, edge.time_to_expiry_minutes)
                # ── Cycle recorder: capture zscore value ──
                if _rec_cycle_id is not None and edge.market.ticker in _filter_results:
                    _filter_results[edge.market.ticker].zscore_value = z
                if z > self._settings.zscore_max:
                    LOGGER.info(
                        "CryptoEngine: Z-score reject edge %s: Z=%.2f > %.2f",
                        edge.market.ticker, z, self._settings.zscore_max,
                    )
                    continue
                zscore_filtered.append(edge)
            edges = zscore_filtered
            # ── Cycle recorder Hook 3: zscore filter tracking ──
            if _rec_cycle_id is not None:
                post_zscore_tickers = {e.market.ticker for e in edges}
                for ticker in pre_zscore_tickers - post_zscore_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_zscore = False
                        if _filter_results[ticker].reject_reason is None:
                            _filter_results[ticker].reject_reason = "zscore"
                for ticker in post_zscore_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_zscore = True

        # 3-ofi. Counter-trend OFI filter: skip trades opposing the trend in trending regimes
        # Daily (above/below) markets with TTE > 30m are EXEMPT: the micro regime
        # (last few minutes) doesn't predict whether price stays above a strike
        # over the next hour+.
        if (self._settings.regime_skip_counter_trend
            and self._current_regime is not None
            and self._current_regime.is_trending
            and self._current_regime.confidence >= self._settings.regime_skip_counter_trend_min_conf):
            pre_ct_tickers = {e.market.ticker for e in edges}
            trend_dir = self._current_regime.trend_direction  # 1=up, -1=down
            ofi_filtered = []
            for edge in edges:
                # Exempt daily markets from counter-trend filter
                if (edge.market.meta.direction in ("above", "below")
                        and edge.time_to_expiry_minutes > 30):
                    ofi_filtered.append(edge)
                    continue
                direction = edge.market.meta.direction
                if direction in ("above", "up"):
                    trade_dir = 1 if edge.side == "yes" else -1
                else:  # below, down
                    trade_dir = -1 if edge.side == "yes" else 1
                if trade_dir * trend_dir < 0:
                    LOGGER.info(
                        "CryptoEngine: counter-trend skip %s side=%s (trend=%s, conf=%.2f)",
                        edge.market.ticker, edge.side, self._current_regime.regime,
                        self._current_regime.confidence,
                    )
                    continue
                ofi_filtered.append(edge)
            edges = ofi_filtered
            # ── Cycle recorder Hook 3: counter-trend filter tracking ──
            if _rec_cycle_id is not None:
                post_ct_tickers = {e.market.ticker for e in edges}
                for ticker in pre_ct_tickers - post_ct_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_counter_trend = False
                        if _filter_results[ticker].reject_reason is None:
                            _filter_results[ticker].reject_reason = "counter_trend"
                for ticker in post_ct_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_counter_trend = True

        # ── Cycle recorder: mark survivors ──
        if _rec_cycle_id is not None:
            for e in edges:
                if e.market.ticker in _filter_results:
                    _filter_results[e.market.ticker].survived_all = True

        # 3a. Compute staleness scores for detected edges
        if self._staleness_detector is not None:
            for edge in edges:
                underlying = edge.market.meta.underlying
                binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                if binance_sym:
                    current_spot = self._price_feed.get_current_price(binance_sym)
                    lookback_ts = time.time() - self._settings.staleness_lookback_seconds
                    spot_at_lookback = self._price_feed.get_price_at_time(binance_sym, lookback_ts)
                    if current_spot is not None:
                        stale_result = self._staleness_detector.compute_staleness(
                            ticker=edge.market.ticker,
                            current_spot=current_spot,
                            spot_at_lookback=spot_at_lookback,
                            current_yes_ask=edge.yes_buy_price,
                            current_no_ask=edge.no_buy_price,
                        )
                        object.__setattr__(edge, "staleness_score", stale_result.staleness_score)

        # 3b. Confidence scoring filter (B2)
        if self._confidence_scorer is not None:
            scored_edges = []
            for edge in edges:
                components = self._build_confidence_components(edge)
                result = self._confidence_scorer.score(components)
                if self._confidence_scorer.passes(result):
                    scored_edges.append(edge)
                    LOGGER.info(
                        "CryptoEngine: CONFIDENCE PASS %s score=%.2f agree=%d/%d [%s]",
                        edge.market.ticker, result.score,
                        result.signal_agreement, result.total_signals,
                        ", ".join(result.reasons[:3]),
                    )
                else:
                    LOGGER.debug(
                        "CryptoEngine: confidence reject %s score=%.2f agree=%d/%d",
                        edge.market.ticker, result.score,
                        result.signal_agreement, result.total_signals,
                    )
            edges = scored_edges

        # ── Cycle recorder Hook 3b: record edges with filter results ──
        _rec_edge_id_map: Dict[str, int] = {}
        if self._cycle_recorder is not None and _rec_cycle_id is not None:
            _rec_edge_id_map = self._cycle_recorder.record_edges(
                _rec_cycle_id, all_raw_edges, _filter_results,
            )

        # ── Set temporary attrs for trade recording in _execute_paper_trade ──
        self._rec_cycle_id_current = _rec_cycle_id
        self._rec_edge_id_map = _rec_edge_id_map
        self._rec_filter_results_current = _filter_results

        # 4. Size and execute edges (with per-cell filtering + sizing)
        from arb_bot.crypto.strategy_cell import StrategyCell, get_cell_config
        trades_opened = 0
        for edge in edges:
            if len(self._positions) >= self._settings.max_concurrent_positions:
                break
            if edge.market.ticker in self._positions:
                continue  # Already have a position
            underlying = edge.market.meta.underlying
            if self._count_positions_for_underlying(underlying) >= self._settings.max_positions_per_underlying:
                continue

            # Per-cell filtering and sizing
            cell = StrategyCell(edge.strategy_cell) if edge.strategy_cell else None
            cell_cfg = get_cell_config(cell, self._settings) if cell else None

            # Per-cell edge threshold
            if cell_cfg is not None and edge.edge_cents < cell_cfg.min_edge_pct:
                LOGGER.info(
                    "CryptoEngine: cell reject %s — edge %.1f%% < %s floor %.1f%%",
                    edge.market.ticker, edge.edge_cents * 100,
                    cell.value, cell_cfg.min_edge_pct * 100,
                )
                continue

            # Per-cell signal gates
            if cell is not None and cell_cfg is not None:
                if not self._passes_cell_signal_gates(edge, cell, cell_cfg):
                    continue

            # Quiet hours gate
            if self._quiet_hours_gate(edge):
                continue

            # Classifier veto gate
            vetoed, p_win, fv = self._classifier_veto(edge)
            if vetoed:
                continue

            contracts = self._compute_position_size(edge, cell_cfg=cell_cfg)
            if contracts <= 0:
                continue

            self._execute_paper_trade(edge, contracts, prebuilt_fv=fv)
            trades_opened += 1
            # Per-cycle entry throttle (v44 Fix G)
            if trades_opened >= self._settings.max_new_trades_per_cycle:
                break

        # 5. Check and settle expired positions
        await self._settle_expired_positions()

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

        # ── Cycle recorder Hook 5: end cycle ──
        if self._cycle_recorder is not None and _rec_cycle_id is not None:
            _rec_elapsed_ms = (time.time() - _rec_cycle_start) * 1000
            self._cycle_recorder.end_cycle(
                _rec_cycle_id,
                elapsed_ms=_rec_elapsed_ms,
                regime=self._current_regime.regime if self._current_regime else None,
                regime_confidence=self._current_regime.confidence if self._current_regime else None,
                regime_is_transitioning=(
                    self._current_regime.is_transitioning
                    if self._current_regime and hasattr(self._current_regime, 'is_transitioning')
                    else None
                ),
                num_quotes=len(market_quotes),
                num_edges_raw=len(all_raw_edges),
                num_edges_final=len(edges),
                num_trades=trades_opened,
                session_pnl=self._session_pnl,
                bankroll=self._bankroll,
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
        _rec_cycle_start = time.time()

        # Update Hawkes intensity from recent returns
        self._update_hawkes_from_returns()

        # Record quote snapshots for staleness detection
        if self._staleness_detector is not None:
            for mq in quotes:
                self._staleness_detector.record_quote_snapshot(
                    mq.market.ticker,
                    mq.yes_buy_price,
                    mq.no_buy_price,
                )

        # VPIN gate: per-symbol tiered gate with momentum zone (v18/v20)
        _momentum_zone = False
        _vpin_halted_symbols: set = set()
        _vpin_momentum_symbols: set = set()
        _vpin_normal_symbols: set = set()
        _vpin_stale_symbols: set = set()
        if self._settings.vpin_halt_enabled and self._vpin_calculators:
            for sym, calc in self._vpin_calculators.items():
                vpin_val = calc.get_vpin()
                if vpin_val is None:
                    _vpin_normal_symbols.add(sym)
                    if hasattr(calc, "is_stale") and calc.is_stale:
                        _vpin_stale_symbols.add(sym)
                    continue
                # Get effective thresholds (adaptive or static)
                mom_floor, halt_ceil, is_adaptive = self._get_vpin_thresholds(sym)
                if self._settings.momentum_enabled:
                    if vpin_val > halt_ceil:
                        _vpin_halted_symbols.add(sym)
                    elif vpin_val >= mom_floor:
                        _vpin_momentum_symbols.add(sym)
                    else:
                        _vpin_normal_symbols.add(sym)
                else:
                    if vpin_val > halt_ceil:
                        _vpin_halted_symbols.add(sym)
                    else:
                        _vpin_normal_symbols.add(sym)

            if _vpin_stale_symbols:
                stale_details = ", ".join(
                    f"{s}(stale {self._vpin_calculators[s].seconds_since_last_trade:.0f}s)"
                    if hasattr(self._vpin_calculators[s], "seconds_since_last_trade")
                    else s
                    for s in sorted(_vpin_stale_symbols)
                )
                LOGGER.info("CryptoEngine: VPIN stale (treated as normal): %s", stale_details)
            if _vpin_halted_symbols:
                halted_details = []
                for s in sorted(_vpin_halted_symbols):
                    v = self._vpin_calculators[s].get_vpin()
                    if v is not None:
                        _, hc, ada = self._get_vpin_thresholds(s)
                        tag = "adaptive" if ada else "static"
                        halted_details.append(f"{s}={v:.3f}(halt@{hc:.3f}/{tag})")
                if halted_details:
                    LOGGER.info("CryptoEngine: VPIN halt symbols: %s", ", ".join(halted_details))
            if _vpin_momentum_symbols:
                mom_details = []
                for s in sorted(_vpin_momentum_symbols):
                    v = self._vpin_calculators[s].get_vpin()
                    if v is not None:
                        mf, hc, ada = self._get_vpin_thresholds(s)
                        tag = "adaptive" if ada else "static"
                        mom_details.append(f"{s}={v:.3f}(zone@{mf:.3f}-{hc:.3f}/{tag})")
                if mom_details:
                    LOGGER.info("CryptoEngine: VPIN momentum symbols: %s", ", ".join(mom_details))
                _momentum_zone = True

            if _vpin_halted_symbols and not _vpin_momentum_symbols and not _vpin_normal_symbols:
                LOGGER.info("CryptoEngine: VPIN full halt — all symbols above ceiling, skipping cycle")
                return []

        model_probs = self._compute_model_probabilities(quotes)

        # Classify market regime
        self._update_regime()

        # Momentum path (v18/v20): try momentum trades for momentum-zone symbols
        if _momentum_zone and self._settings.momentum_enabled:
            await self._try_momentum_trades(quotes, momentum_symbols=_vpin_momentum_symbols)

        # If no symbols are in normal zone, skip model-path trades for 15-min markets.
        # Daily (above/below TTE>30m) markets are still processed.
        _has_daily_quotes = any(
            mq.market.meta.direction in ("above", "below") and mq.time_to_expiry_minutes > 30
            for mq in quotes
        )
        if not _vpin_normal_symbols and (_vpin_halted_symbols or _vpin_momentum_symbols) and not _has_daily_quotes:
            await self._settle_expired_positions()
            return []

        # ── Cycle recorder Hook 1: market snapshots ──
        _rec_cycle_id = None
        _filter_results: Dict[str, FilterResult] = {}
        if self._cycle_recorder is not None:
            _rec_cycle_id = self._cycle_recorder.begin_cycle(self._cycle_count, _rec_cycle_start)
            _rec_spots: Dict[str, float] = {}
            _rec_vpins: Dict[str, tuple] = {}
            _rec_ofis: Dict[str, Dict[int, float]] = {}
            _rec_returns: Dict[str, list] = {}
            for sym in self._settings.price_feed_symbols:
                _rec_spots[sym] = self._price_feed.get_current_price(sym) or 0.0
                if self._settings.vpin_enabled and sym in self._vpin_calculators:
                    calc = self._vpin_calculators[sym]
                    _rec_vpins[sym] = (calc.get_vpin() or 0.0, calc.get_signed_vpin() or 0.0, calc.get_vpin_trend() or 0.0)
                if self._settings.ofi_enabled:
                    _rec_ofis[sym] = self._price_feed.get_ofi_multiscale(sym, [30, 60, 120, 300])
                _rec_returns[sym] = list(self._price_feed.get_returns(sym, interval_seconds=60, window_minutes=self._settings.mc_vol_window_minutes))
            self._cycle_recorder.record_market_snapshots(
                _rec_cycle_id, quotes, _rec_spots, _rec_vpins, _rec_ofis, _rec_returns,
            )

        edges = self._edge_detector.detect_edges(quotes, model_probs)

        # Filter edges for symbols in halted or momentum zone (model-path only)
        # Daily (above/below) markets with TTE > 30m are EXEMPT from VPIN filtering.
        if _vpin_halted_symbols or _vpin_momentum_symbols:
            blocked_syms = _vpin_halted_symbols | _vpin_momentum_symbols
            pre_count = len(edges)
            edges = [
                e for e in edges
                if _KALSHI_TO_BINANCE.get(e.market.meta.underlying, "") not in blocked_syms
                or (e.market.meta.direction in ("above", "below")
                    and e.time_to_expiry_minutes > 30)
            ]
            if len(edges) < pre_count:
                LOGGER.info(
                    "CryptoEngine: VPIN symbol filter removed %d/%d model-path edges (halted/momentum symbols)",
                    pre_count - len(edges), pre_count,
                )

        # ── Cycle recorder Hook 2: snapshot raw edges ──
        all_raw_edges = list(edges)
        if _rec_cycle_id is not None:
            _filter_results = {e.market.ticker: FilterResult() for e in edges}

        # Per-cell model adjustments (blending weight, prob haircut)
        edges = self._apply_cell_model_adjustments(edges)

        # Regime min edge filter
        if self._settings.regime_min_edge_enabled and self._current_regime is not None:
            regime = self._current_regime.regime
            _regime_min_edges = {
                "mean_reverting": self._settings.regime_min_edge_mean_reverting,
                "trending_up": self._settings.regime_min_edge_trending,
                "trending_down": self._settings.regime_min_edge_trending,
                "high_vol": self._settings.regime_min_edge_high_vol,
            }
            regime_min = _regime_min_edges.get(regime, 0.0)
            if regime_min > 0:
                pre_filter_tickers = {e.market.ticker for e in edges}
                pre_count = len(edges)
                edges = [e for e in edges if e.edge_cents >= regime_min]
                if len(edges) < pre_count:
                    LOGGER.info(
                        "CryptoEngine: regime min_edge filter (%s, min=%.2f) removed %d/%d edges",
                        regime, regime_min, pre_count - len(edges), pre_count,
                    )
                # ── Cycle recorder Hook 3: regime filter tracking ──
                if _rec_cycle_id is not None:
                    post_filter_tickers = {e.market.ticker for e in edges}
                    for ticker in pre_filter_tickers - post_filter_tickers:
                        if ticker in _filter_results:
                            _filter_results[ticker].passed_regime_min_edge = False
                            if _filter_results[ticker].reject_reason is None:
                                _filter_results[ticker].reject_reason = "regime_min_edge"
                    for ticker in post_filter_tickers:
                        if ticker in _filter_results:
                            _filter_results[ticker].passed_regime_min_edge = True

        # Z-score reachability post-edge filter
        if self._settings.zscore_filter_enabled:
            pre_zscore_tickers = {e.market.ticker for e in edges}
            zscore_filtered = []
            for edge in edges:
                underlying = edge.market.meta.underlying
                binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                if not binance_sym:
                    zscore_filtered.append(edge)
                    continue
                spot = self._price_feed.get_current_price(binance_sym)
                if spot is None:
                    zscore_filtered.append(edge)
                    continue
                z_strike = edge.market.meta.strike
                direction = edge.market.meta.direction
                if direction in ("up", "down"):
                    z_strike = None
                    if edge.market.meta.interval_start_time is not None:
                        start_ts = edge.market.meta.interval_start_time.timestamp()
                        z_strike = self._price_feed.get_price_at_time(binance_sym, start_ts)
                if z_strike is None or z_strike <= 0:
                    zscore_filtered.append(edge)
                    continue
                local_returns = self._price_feed.get_returns(
                    binance_sym, interval_seconds=60,
                    window_minutes=self._settings.zscore_vol_window_minutes,
                )
                local_vol = self._price_model.estimate_volatility(
                    local_returns, interval_seconds=60,
                )
                if local_vol <= 0:
                    local_vol = 0.50
                z = self._compute_zscore(spot, z_strike, local_vol, edge.time_to_expiry_minutes)
                # ── Cycle recorder: capture zscore value ──
                if _rec_cycle_id is not None and edge.market.ticker in _filter_results:
                    _filter_results[edge.market.ticker].zscore_value = z
                if z > self._settings.zscore_max:
                    LOGGER.info(
                        "CryptoEngine: Z-score reject edge %s: Z=%.2f > %.2f",
                        edge.market.ticker, z, self._settings.zscore_max,
                    )
                    continue
                zscore_filtered.append(edge)
            edges = zscore_filtered
            # ── Cycle recorder Hook 3: zscore filter tracking ──
            if _rec_cycle_id is not None:
                post_zscore_tickers = {e.market.ticker for e in edges}
                for ticker in pre_zscore_tickers - post_zscore_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_zscore = False
                        if _filter_results[ticker].reject_reason is None:
                            _filter_results[ticker].reject_reason = "zscore"
                for ticker in post_zscore_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_zscore = True

        # Counter-trend OFI filter: skip trades opposing the trend in trending regimes
        # Daily (above/below) markets with TTE > 30m are EXEMPT.
        if (self._settings.regime_skip_counter_trend
            and self._current_regime is not None
            and self._current_regime.is_trending
            and self._current_regime.confidence >= self._settings.regime_skip_counter_trend_min_conf):
            pre_ct_tickers = {e.market.ticker for e in edges}
            trend_dir = self._current_regime.trend_direction  # 1=up, -1=down
            ofi_filtered = []
            for edge in edges:
                # Exempt daily markets from counter-trend filter
                if (edge.market.meta.direction in ("above", "below")
                        and edge.time_to_expiry_minutes > 30):
                    ofi_filtered.append(edge)
                    continue
                direction = edge.market.meta.direction
                if direction in ("above", "up"):
                    trade_dir = 1 if edge.side == "yes" else -1
                else:  # below, down
                    trade_dir = -1 if edge.side == "yes" else 1
                if trade_dir * trend_dir < 0:
                    LOGGER.info(
                        "CryptoEngine: counter-trend skip %s side=%s (trend=%s, conf=%.2f)",
                        edge.market.ticker, edge.side, self._current_regime.regime,
                        self._current_regime.confidence,
                    )
                    continue
                ofi_filtered.append(edge)
            edges = ofi_filtered
            # ── Cycle recorder Hook 3: counter-trend filter tracking ──
            if _rec_cycle_id is not None:
                post_ct_tickers = {e.market.ticker for e in edges}
                for ticker in pre_ct_tickers - post_ct_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_counter_trend = False
                        if _filter_results[ticker].reject_reason is None:
                            _filter_results[ticker].reject_reason = "counter_trend"
                for ticker in post_ct_tickers:
                    if ticker in _filter_results:
                        _filter_results[ticker].passed_counter_trend = True

        # ── Cycle recorder: mark survivors ──
        if _rec_cycle_id is not None:
            for e in edges:
                if e.market.ticker in _filter_results:
                    _filter_results[e.market.ticker].survived_all = True

        # Compute staleness scores for detected edges
        if self._staleness_detector is not None:
            for edge in edges:
                underlying = edge.market.meta.underlying
                binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                if binance_sym:
                    current_spot = self._price_feed.get_current_price(binance_sym)
                    lookback_ts = time.time() - self._settings.staleness_lookback_seconds
                    spot_at_lookback = self._price_feed.get_price_at_time(binance_sym, lookback_ts)
                    if current_spot is not None:
                        stale_result = self._staleness_detector.compute_staleness(
                            ticker=edge.market.ticker,
                            current_spot=current_spot,
                            spot_at_lookback=spot_at_lookback,
                            current_yes_ask=edge.yes_buy_price,
                            current_no_ask=edge.no_buy_price,
                        )
                        object.__setattr__(edge, "staleness_score", stale_result.staleness_score)

        # Confidence scoring filter (B2)
        if self._confidence_scorer is not None:
            scored_edges = []
            for edge in edges:
                components = self._build_confidence_components(edge)
                result = self._confidence_scorer.score(components)
                if self._confidence_scorer.passes(result):
                    scored_edges.append(edge)
                    LOGGER.info(
                        "CryptoEngine: CONFIDENCE PASS %s score=%.2f agree=%d/%d [%s]",
                        edge.market.ticker, result.score,
                        result.signal_agreement, result.total_signals,
                        ", ".join(result.reasons[:3]),
                    )
                else:
                    LOGGER.debug(
                        "CryptoEngine: confidence reject %s score=%.2f agree=%d/%d",
                        edge.market.ticker, result.score,
                        result.signal_agreement, result.total_signals,
                    )
            edges = scored_edges

        # ── Cycle recorder Hook 3b: record edges with filter results ──
        _rec_edge_id_map: Dict[str, int] = {}
        if self._cycle_recorder is not None and _rec_cycle_id is not None:
            _rec_edge_id_map = self._cycle_recorder.record_edges(
                _rec_cycle_id, all_raw_edges, _filter_results,
            )

        # ── Set temporary attrs for trade recording in _execute_paper_trade ──
        self._rec_cycle_id_current = _rec_cycle_id
        self._rec_edge_id_map = _rec_edge_id_map
        self._rec_filter_results_current = _filter_results

        from arb_bot.crypto.strategy_cell import StrategyCell, get_cell_config as _get_cell_config
        trades_opened = 0
        for edge in edges:
            if len(self._positions) >= self._settings.max_concurrent_positions:
                break
            if edge.market.ticker in self._positions:
                continue
            underlying = edge.market.meta.underlying
            if self._count_positions_for_underlying(underlying) >= self._settings.max_positions_per_underlying:
                continue

            # Per-cell filtering and sizing
            cell = StrategyCell(edge.strategy_cell) if edge.strategy_cell else None
            cell_cfg = _get_cell_config(cell, self._settings) if cell else None

            # Per-cell edge threshold
            if cell_cfg is not None and edge.edge_cents < cell_cfg.min_edge_pct:
                LOGGER.info(
                    "CryptoEngine: cell reject %s — edge %.1f%% < %s floor %.1f%%",
                    edge.market.ticker, edge.edge_cents * 100,
                    cell.value, cell_cfg.min_edge_pct * 100,
                )
                continue

            # Per-cell signal gates
            if cell is not None and cell_cfg is not None:
                if not self._passes_cell_signal_gates(edge, cell, cell_cfg):
                    continue

            # Quiet hours gate
            if self._quiet_hours_gate(edge):
                continue

            # Classifier veto gate
            vetoed, p_win, fv = self._classifier_veto(edge)
            if vetoed:
                continue

            contracts = self._compute_position_size(edge, cell_cfg=cell_cfg)
            if contracts <= 0:
                continue

            self._execute_paper_trade(edge, contracts, prebuilt_fv=fv)
            trades_opened += 1
            # Per-cycle entry throttle (v44 Fix G)
            if trades_opened >= self._settings.max_new_trades_per_cycle:
                break

        await self._settle_expired_positions()
        self._log_cycle(len(edges))

        # ── Cycle recorder Hook 5: end cycle ──
        if self._cycle_recorder is not None and _rec_cycle_id is not None:
            _rec_elapsed_ms = (time.time() - _rec_cycle_start) * 1000
            self._cycle_recorder.end_cycle(
                _rec_cycle_id,
                elapsed_ms=_rec_elapsed_ms,
                regime=self._current_regime.regime if self._current_regime else None,
                regime_confidence=self._current_regime.confidence if self._current_regime else None,
                regime_is_transitioning=(
                    self._current_regime.is_transitioning
                    if self._current_regime and hasattr(self._current_regime, 'is_transitioning')
                    else None
                ),
                num_quotes=len(quotes),
                num_edges_raw=len(all_raw_edges),
                num_edges_final=len(edges),
                num_trades=trades_opened,
                session_pnl=self._session_pnl,
                bankroll=self._bankroll,
            )

        return edges

    # ── Adaptive VPIN thresholds ────────────────────────────────

    def _get_vpin_thresholds(self, sym: str) -> tuple:
        """Get effective (momentum_floor, halt_ceiling) for a symbol.

        When adaptive thresholds are enabled and enough history has
        accumulated, use percentile-based thresholds from the rolling
        VPIN distribution.  Otherwise fall back to static config values.

        Returns (momentum_floor, halt_ceiling, is_adaptive).
        """
        static_floor = self._settings.momentum_vpin_floor
        static_ceiling = self._settings.momentum_vpin_ceiling
        static_halt = self._settings.vpin_halt_threshold

        if not self._settings.vpin_adaptive_enabled:
            if self._settings.momentum_enabled:
                return (static_floor, static_ceiling, False)
            return (static_halt, static_halt, False)

        calc = self._vpin_calculators.get(sym)
        if calc is None or not hasattr(calc, "get_adaptive_momentum_thresholds"):
            if self._settings.momentum_enabled:
                return (static_floor, static_ceiling, False)
            return (static_halt, static_halt, False)

        momentum_t, halt_t = calc.get_adaptive_momentum_thresholds(
            halt_percentile=self._settings.vpin_adaptive_halt_percentile,
            momentum_percentile=self._settings.vpin_adaptive_momentum_percentile,
            halt_floor=self._settings.vpin_adaptive_halt_floor,
            halt_ceiling=self._settings.vpin_adaptive_halt_ceiling,
            momentum_floor=self._settings.vpin_adaptive_momentum_floor,
            momentum_ceiling=self._settings.vpin_adaptive_momentum_ceiling,
            min_history=self._settings.vpin_adaptive_min_history,
        )

        if momentum_t is None or halt_t is None:
            # Not enough history yet — fall back to static
            if self._settings.momentum_enabled:
                return (static_floor, static_ceiling, False)
            return (static_halt, static_halt, False)

        return (momentum_t, halt_t, True)

    # ── Regime detection ─────────────────────────────────────────

    def _update_regime(self) -> None:
        """Classify market regime for each underlying and aggregate.

        Uses OFI, returns, vol, and VPIN signals already available
        from the price feed and VPIN calculators.
        """
        if self._regime_detector is None:
            return

        snapshots: Dict[str, RegimeSnapshot] = {}

        for binance_sym in self._settings.price_feed_symbols:
            # OFI at multiple timescales
            ofi_multiscale: Dict[int, float] = {}
            if self._settings.ofi_enabled:
                ofi_multiscale = self._price_feed.get_ofi_multiscale(
                    binance_sym, [30, 60, 120, 300],
                )

            # Recent 1-minute returns
            returns_1m = self._price_feed.get_returns(
                binance_sym, interval_seconds=60, window_minutes=30,
            )

            # Short vol (5-min returns over 15 min) and long vol (1-min over 120 min)
            short_returns = self._price_feed.get_returns(
                binance_sym, interval_seconds=60, window_minutes=15,
            )
            long_returns = self._price_feed.get_returns(
                binance_sym, interval_seconds=60, window_minutes=120,
            )
            vol_short = float(np.std(short_returns)) if len(short_returns) >= 2 else 0.0
            vol_long = float(np.std(long_returns)) if len(long_returns) >= 2 else 0.0

            # VPIN
            vpin_val: Optional[float] = None
            signed_vpin_val: Optional[float] = None
            if self._settings.vpin_enabled and binance_sym in self._vpin_calculators:
                calc = self._vpin_calculators[binance_sym]
                vpin_val = calc.get_vpin()
                signed_vpin_val = calc.get_signed_vpin()

            snap = self._regime_detector.classify(
                symbol=binance_sym,
                ofi_multiscale=ofi_multiscale,
                returns_1m=returns_1m,
                vol_short=vol_short,
                vol_long=vol_long,
                vpin=vpin_val,
                signed_vpin=signed_vpin_val,
            )
            snapshots[binance_sym] = snap

        self._current_regime = self._regime_detector.classify_market(snapshots)

        LOGGER.debug(
            "CryptoEngine: regime=%s conf=%.2f [%s]",
            self._current_regime.regime,
            self._current_regime.confidence,
            ", ".join(
                f"{sym}={snap.regime}"
                for sym, snap in self._current_regime.per_symbol.items()
            ),
        )

    # ── Mean-reversion helper ────────────────────────────────────

    def _compute_mean_reversion(self, binance_sym: str) -> float:
        """OU-like mean-reversion correction: pull drift against recent price move.

        If price just went up, this returns negative drift (expect pullback).
        If price just went down, this returns positive drift (expect bounce).
        """
        lookback = int(self._settings.mean_reversion_lookback_minutes)
        returns = self._price_feed.get_returns(
            binance_sym, interval_seconds=60, window_minutes=lookback,
        )
        if not returns:
            return 0.0
        recent_return = sum(returns)  # cumulative log return over lookback
        kappa = self._settings.mean_reversion_kappa
        return -kappa * recent_return

    # ── Z-score reachability ─────────────────────────────────────────

    @staticmethod
    def _compute_zscore(
        current_price: float,
        strike: float,
        vol: float,
        tte_minutes: float,
    ) -> float:
        """Compute moneyness Z-score: distance to strike in local sigma units.

        Z = |ln(price/strike)| / (σ_local × √(TTE_minutes / minutes_per_year))

        High Z means the strike is statistically unreachable in remaining time.
        """
        if strike is None or strike <= 0 or current_price <= 0 or vol <= 0:
            return 0.0

        if tte_minutes <= 0:
            return float('inf')  # No time left = unreachable

        minutes_per_year = 365.25 * 24 * 60
        vol_horizon = vol * math.sqrt(tte_minutes / minutes_per_year)

        if vol_horizon <= 0:
            return float('inf')

        distance = abs(math.log(current_price / strike))
        return distance / vol_horizon

    # ── Student-t fitting ──────────────────────────────────────────

    def _fit_student_t_params(self, binance_sym: str) -> tuple:
        """Fit Student-t nu from rolling returns. Returns (nu, nu_stderr)."""
        # Check cache freshness
        if (self._student_t_last_fit_cycle > 0
            and (self._cycle_count - self._student_t_last_fit_cycle)
                < self._settings.student_t_refit_every_cycles
            and binance_sym in self._student_t_nu):
            return (self._student_t_nu[binance_sym],
                    self._student_t_nu_stderr.get(binance_sym, 0.0))

        returns = self._price_feed.get_returns(
            binance_sym,
            interval_seconds=60,
            window_minutes=self._settings.student_t_fit_window_minutes,
        )

        nu, nu_stderr = self._price_model.fit_nu_from_returns(
            returns,
            min_samples=self._settings.student_t_fit_min_samples,
            nu_floor=self._settings.student_t_nu_floor,
            nu_ceiling=self._settings.student_t_nu_ceiling,
        )

        self._student_t_nu[binance_sym] = nu
        self._student_t_nu_stderr[binance_sym] = nu_stderr
        self._student_t_last_fit_cycle = self._cycle_count

        LOGGER.info(
            "CryptoEngine: Student-t fit %s: nu=%.2f stderr=%.2f from %d returns",
            binance_sym, nu, nu_stderr, len(returns),
        )
        return (nu, nu_stderr)

    def _get_empirical_returns(self, binance_sym: str) -> list[float]:
        """Get cached empirical returns for bootstrap resampling."""
        # Refresh cache periodically (reuse student_t_refit_every_cycles)
        if (self._empirical_returns_cache_cycle > 0
            and (self._cycle_count - self._empirical_returns_cache_cycle)
                < self._settings.student_t_refit_every_cycles
            and binance_sym in self._empirical_returns_cache):
            return self._empirical_returns_cache[binance_sym]

        returns = self._price_feed.get_returns(
            binance_sym,
            interval_seconds=self._settings.empirical_return_interval_seconds,
            window_minutes=self._settings.empirical_window_minutes,
        )

        self._empirical_returns_cache[binance_sym] = returns
        self._empirical_returns_cache_cycle = self._cycle_count

        LOGGER.info(
            "CryptoEngine: Empirical returns %s: %d returns (window=%dm, interval=%ds)",
            binance_sym, len(returns),
            self._settings.empirical_window_minutes,
            self._settings.empirical_return_interval_seconds,
        )
        return returns

    def _get_regime_adjusted_empirical_returns(self, binance_sym: str) -> list:
        """Get empirical returns with regime-conditional window and scaling."""
        # Determine adjusted window based on regime
        window = self._settings.empirical_window_minutes
        if self._current_regime is not None:
            regime = self._current_regime.regime
            if regime in ("trending_up", "trending_down"):
                window = self._settings.regime_empirical_window_trending
            elif regime == "high_vol":
                window = self._settings.regime_empirical_window_high_vol

        # If window differs from default, fetch directly (bypass cache)
        if window != self._settings.empirical_window_minutes:
            returns = self._price_feed.get_returns(
                binance_sym,
                interval_seconds=self._settings.empirical_return_interval_seconds,
                window_minutes=window,
            )
        else:
            returns = self._get_empirical_returns(binance_sym)

        # Scale returns in high_vol regime
        if (self._current_regime is not None
            and self._current_regime.regime == "high_vol"
            and self._settings.regime_vol_boost_high_vol != 1.0):
            boost = self._settings.regime_vol_boost_high_vol
            returns = [r * boost for r in returns]

        return returns

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

            # ── Variance ratio vol correction (v43 — 1A) ──────────
            # For daily contracts, adjust vol using variance ratio to
            # account for mean-reversion that makes √T scaling inaccurate.
            is_daily_contract = (
                mq.market.meta.direction in ("above", "below")
                and horizon > 20
            )
            if (
                is_daily_contract
                and self._settings.variance_ratio_enabled
                and binance_sym
            ):
                vr_returns = self._price_feed.get_returns(
                    binance_sym, interval_seconds=60,
                    window_minutes=self._settings.mc_vol_window_minutes,
                )
                if len(vr_returns) >= self._settings.variance_ratio_min_samples:
                    vr_q = max(2, round(horizon))
                    vr = self._price_model.variance_ratio(vr_returns, vr_q)
                    vol_before = vol
                    vol = self._price_model.vol_with_variance_ratio_correction(
                        vol, horizon, vr,
                    )
                    LOGGER.debug(
                        "Vol correction: raw=%.1f%% VR(%d)=%.2f adjusted=%.1f%%",
                        vol_before * 100, vr_q, vr, vol * 100,
                    )

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
                # Multi-timescale OFI (weighted combination of multiple windows)
                if self._settings.multiscale_ofi_enabled:
                    ms_windows = [int(w) for w in self._settings.multiscale_ofi_windows.split(",")]
                    ms_weights = [float(w) for w in self._settings.multiscale_ofi_weights.split(",")]
                    ofi_multi = self._price_feed.get_ofi_multiscale(binance_sym, ms_windows)
                    ofi = sum(ofi_multi.get(w, 0.0) * wt for w, wt in zip(ms_windows, ms_weights))
                else:
                    ofi = self._price_feed.get_ofi(
                        binance_sym,
                        window_seconds=self._settings.ofi_window_seconds,
                    )
                # Use cached calibration result, recalibrate on interval
                cal_result = self._get_ofi_calibration()
                alpha = cal_result.alpha if cal_result.alpha != 0 else self._settings.ofi_alpha
                drift = self._apply_ofi_impact(ofi, alpha)

            # Add trend-based drift if enabled (regime-conditional or legacy)
            if binance_sym:
                if self._settings.regime_conditional_drift:
                    # Only apply trend drift during trending regimes
                    if (self._current_regime is not None
                        and self._current_regime.is_trending):
                        drift += self._compute_trend_drift(binance_sym)
                elif self._settings.trend_drift_enabled:
                    drift += self._compute_trend_drift(binance_sym)

            # Funding rate drift (A2)
            if self._funding_tracker is not None:
                funding_sym = binance_sym.upper()  # FundingRateTracker uses uppercase
                signal = self._funding_tracker.get_funding_signal(funding_sym)
                if signal is not None:
                    drift += signal.drift_adjustment * self._settings.funding_rate_drift_weight

            # VPIN signal (B1)
            if self._settings.vpin_enabled and binance_sym in self._vpin_calculators:
                vpin_calc = self._vpin_calculators[binance_sym]

                # Auto-calibrate bucket size on first use
                if vpin_calc.bucket_volume <= 0:
                    # Per-symbol minimum bucket volumes — ensures each bucket
                    # contains ~30 trades for meaningful VPIN.
                    _min_bv = {"btcusdt": 0.10, "ethusdt": 1.0, "solusdt": 50.0}
                    min_bv = _min_bv.get(binance_sym, 1.0)
                    # Try trade-size calibration first
                    buy_sells = self._price_feed._buy_sells.get(binance_sym, [])
                    trade_sizes = [vol for _, vol, _ in buy_sells]
                    if len(trade_sizes) >= 10:
                        vpin_calc.calibrate_from_trades(
                            trade_sizes=trade_sizes,
                            trades_per_bucket=30,
                            min_bucket_volume=min_bv,
                        )
                    else:
                        total_vol = self._price_feed.get_total_volume(
                            binance_sym,
                            window_seconds=self._settings.vpin_auto_calibrate_window_minutes * 60,
                        )
                        vpin_calc.auto_calibrate_bucket_size(
                            total_volume=total_vol,
                            window_minutes=float(self._settings.vpin_auto_calibrate_window_minutes),
                            min_bucket_volume=min_bv,
                        )

                vpin = vpin_calc.get_vpin()
                signed_vpin = vpin_calc.get_signed_vpin()

                if vpin is not None and signed_vpin is not None:
                    # Directional drift from signed VPIN
                    drift += signed_vpin * self._settings.vpin_drift_weight

                    # Vol boost when VPIN is extreme (imminent volatility)
                    if vpin > self._settings.vpin_extreme_threshold:
                        vol *= self._settings.vpin_vol_boost_factor

            # Cross-asset features (A5)
            if self._cross_asset is not None and binance_sym != self._cross_asset.leader_symbol:
                cross_signal = self._cross_asset.compute_features(self._price_feed, binance_sym)
                if cross_signal is not None:
                    drift += cross_signal.drift_adjustment
                    vol *= cross_signal.vol_adjustment

            # ── Drift safety: mean reversion + total clamp ──────────────
            # Mean-reversion correction (pull against recent move)
            if self._settings.mean_reversion_enabled:
                drift += self._compute_mean_reversion(binance_sym)

            # Clamp total drift to prevent extreme model predictions
            max_drift = self._settings.max_total_drift
            drift = max(-max_drift, min(max_drift, drift))

            # ── v43 Daily model routing ──────────────────────────────
            # For daily contracts (above/below, TTE > 20min), use the
            # dedicated daily pricing model with regime-aware routing
            # instead of the standard MC/empirical/student-t pipeline.
            direction = mq.market.meta.direction
            if direction not in self._settings.allowed_directions:
                continue
            strike = mq.market.meta.strike

            # ── v45: GARCH vol-spread model ──────────────────────────
            # When enabled, compute GARCH-based probability from the
            # divergence between GARCH-forecasted vol and market-implied vol.
            # The result feeds into the standard post-processing pipeline
            # (z-score clamp, calibration, uncertainty scaling, prob cap).
            # Works for any contract type that has a strike; compute_signal
            # returns None gracefully when it can't produce a signal.
            garch_prob: Optional[ProbabilityEstimate] = None
            if (
                self._settings.garch_enabled
                and binance_sym in self._garch_models
                and strike is not None
            ):
                garch_model = self._garch_models[binance_sym]
                garch_returns_1m = self._price_feed.get_returns(
                    binance_sym, interval_seconds=60,
                    window_minutes=self._settings.garch_lookback_minutes,
                )
                if len(garch_returns_1m) >= self._settings.garch_min_obs:
                    garch_returns_arr = np.array(garch_returns_1m, dtype=np.float64)

                    # Get market price from best bid/ask midpoint
                    garch_market_price = None
                    if mq.yes_bid_price is not None and mq.yes_buy_price is not None:
                        garch_market_price = (mq.yes_buy_price + mq.yes_bid_price) / 2.0
                    elif mq.yes_buy_price is not None:
                        garch_market_price = mq.yes_buy_price
                    elif mq.yes_bid_price is not None:
                        garch_market_price = mq.yes_bid_price

                    if garch_market_price is not None and 0.01 < garch_market_price < 0.99:
                        signal = garch_model.compute_signal(
                            garch_returns_arr, current_price, strike, horizon,
                            garch_market_price, direction, drift=drift,
                        )
                        if signal is not None:
                            garch_est = garch_model.compute_garch_probability(
                                current_price, strike, signal.garch_vol,
                                horizon, drift=drift, direction=direction,
                            )

                            # Blend GARCH probability with market price
                            gw = self._settings.garch_probability_weight
                            mw = self._settings.garch_market_weight
                            blended_p = gw * garch_est.probability + mw * garch_market_price
                            blended_p = max(0.01, min(0.99, blended_p))

                            garch_prob = ProbabilityEstimate(
                                probability=blended_p,
                                ci_lower=max(0.0, blended_p - garch_est.uncertainty),
                                ci_upper=min(1.0, blended_p + garch_est.uncertainty),
                                uncertainty=garch_est.uncertainty,
                                num_paths=0,
                            )

                            LOGGER.info(
                                "GARCH %s: garch_vol=%.1f%% iv=%.1f%% "
                                "spread=%.4f z=%.2f signal=%s "
                                "garch_p=%.3f market=%.3f blended=%.3f conf=%.3f",
                                mq.market.ticker,
                                signal.garch_vol * 100,
                                signal.implied_vol * 100,
                                signal.vol_spread,
                                signal.vol_spread_zscore,
                                signal.signal_direction,
                                garch_est.probability,
                                garch_market_price,
                                blended_p,
                                signal.confidence,
                            )

            # When GARCH is enabled and produced a probability, skip the
            # old pipeline entirely — it's proven not to work.  If GARCH
            # couldn't produce a signal for this contract, skip it too
            # rather than falling back to a broken model.
            if self._settings.garch_enabled:
                if garch_prob is not None:
                    # Jump straight to post-processing with the GARCH result.
                    prob = garch_prob
                    # Apply prob cap (v43 Fix A) — still useful as a safety rail
                    cap = self._settings.model_prob_cap
                    floor = self._settings.model_prob_floor
                    if cap < 1.0 or floor > 0.0:
                        raw_p = prob.probability
                        capped_p = max(floor, min(cap, raw_p))
                        if capped_p != raw_p:
                            prob = ProbabilityEstimate(
                                probability=capped_p,
                                ci_lower=max(0.0, capped_p - prob.uncertainty),
                                ci_upper=min(1.0, capped_p + prob.uncertainty),
                                uncertainty=prob.uncertainty,
                                num_paths=0,
                            )
                    result[mq.market.ticker] = prob
                    continue
                else:
                    # GARCH enabled but couldn't produce signal — skip, don't
                    # fall back to the old models.
                    continue

            if is_daily_contract and self._settings.daily_model_enabled:
                # OFI weight reversal for daily contracts (2C)
                if self._settings.ofi_enabled and self._settings.multiscale_ofi_enabled and binance_sym:
                    ms_windows = [int(w) for w in self._settings.multiscale_ofi_windows.split(",")]
                    daily_weights = [float(w) for w in self._settings.daily_ofi_weights.split(",")]
                    ofi_multi = self._price_feed.get_ofi_multiscale(binance_sym, ms_windows)
                    # Re-weight OFI using daily-specific weights (longer timescales matter more)
                    cal_result = self._get_ofi_calibration()
                    alpha = cal_result.alpha if cal_result.alpha != 0 else self._settings.ofi_alpha
                    daily_ofi = sum(ofi_multi.get(w, 0.0) * wt for w, wt in zip(ms_windows, daily_weights))
                    ofi_drift_correction = self._apply_ofi_impact(daily_ofi, alpha) - (drift if self._settings.ofi_enabled else 0.0)
                    # Only apply the difference (since drift already has 15min-weighted OFI)
                    drift += ofi_drift_correction * 0.5  # partial correction

                if strike is not None and direction in ("above", "below"):
                    daily_prob = self._compute_daily_probability(
                        current_price=current_price,
                        strike=strike,
                        direction=direction,
                        horizon_minutes=horizon,
                        vol=vol,
                        drift=drift,
                        binance_sym=binance_sym,
                    )
                    if daily_prob is not None:
                        result[mq.market.ticker] = daily_prob
                        continue
                    # Fall through to standard pipeline if daily model returns None

            # ── Probability computation: MC or Student-t ──────────
            use_mc = self._settings.probability_model in ("mc_gbm", "ab_test", "ab_empirical")
            use_student_t = self._settings.probability_model in ("student_t", "ab_test")
            use_empirical = self._settings.probability_model in ("empirical", "ab_empirical")

            mc_prob = None
            st_prob = None
            emp_prob = None

            # MC path (existing behavior)
            if use_mc:
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
                    mc_prob = self._price_model.probability_above_with_control_variate(
                        paths, strike, current_price, vol, horizon, drift,
                    )
                elif direction == "below" and strike is not None:
                    mc_prob = self._price_model.probability_below(paths, strike)
                elif direction == "up":
                    ref_price = current_price
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                        if looked_up is not None:
                            ref_price = looked_up
                    mc_prob = self._price_model.probability_up(paths, ref_price)
                elif direction == "down":
                    ref_price = current_price
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                        if looked_up is not None:
                            ref_price = looked_up
                    up = self._price_model.probability_up(paths, ref_price)
                    mc_prob = ProbabilityEstimate(
                        probability=1.0 - up.probability,
                        ci_lower=1.0 - up.ci_upper,
                        ci_upper=1.0 - up.ci_lower,
                        uncertainty=up.uncertainty,
                        num_paths=up.num_paths,
                    )
                else:
                    continue
            else:
                # Student-t only mode: still need direction/strike filtering
                direction = mq.market.meta.direction
                if direction not in self._settings.allowed_directions:
                    continue
                strike = mq.market.meta.strike

            # Student-t analytical path
            if use_student_t:
                nu, nu_stderr = self._fit_student_t_params(binance_sym)
                # Get returns for vol stderr (reuse 1-minute returns)
                vol_returns = self._price_feed.get_returns(
                    binance_sym, interval_seconds=60,
                    window_minutes=self._settings.mc_vol_window_minutes,
                )
                vol_stderr = self._price_model.estimate_vol_stderr(vol_returns)

                if direction == "above" and strike is not None:
                    st_prob = self._price_model.probability_above_student_t(
                        current_price, strike, vol, horizon, drift=drift,
                        nu=nu, nu_stderr=nu_stderr, vol_stderr=vol_stderr,
                        min_uncertainty=self._settings.student_t_min_uncertainty,
                    )
                elif direction == "below" and strike is not None:
                    above = self._price_model.probability_above_student_t(
                        current_price, strike, vol, horizon, drift=drift,
                        nu=nu, nu_stderr=nu_stderr, vol_stderr=vol_stderr,
                        min_uncertainty=self._settings.student_t_min_uncertainty,
                    )
                    st_prob = ProbabilityEstimate(
                        probability=1.0 - above.probability,
                        ci_lower=1.0 - above.ci_upper,
                        ci_upper=1.0 - above.ci_lower,
                        uncertainty=above.uncertainty,
                        num_paths=0,
                    )
                elif direction == "up":
                    ref_price = current_price
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                        if looked_up is not None:
                            ref_price = looked_up
                    st_prob = self._price_model.probability_above_student_t(
                        current_price, ref_price, vol, horizon, drift=drift,
                        nu=nu, nu_stderr=nu_stderr, vol_stderr=vol_stderr,
                        min_uncertainty=self._settings.student_t_min_uncertainty,
                    )
                elif direction == "down":
                    ref_price = current_price
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                        if looked_up is not None:
                            ref_price = looked_up
                    up = self._price_model.probability_above_student_t(
                        current_price, ref_price, vol, horizon, drift=drift,
                        nu=nu, nu_stderr=nu_stderr, vol_stderr=vol_stderr,
                        min_uncertainty=self._settings.student_t_min_uncertainty,
                    )
                    st_prob = ProbabilityEstimate(
                        probability=1.0 - up.probability,
                        ci_lower=1.0 - up.ci_upper,
                        ci_upper=1.0 - up.ci_lower,
                        uncertainty=up.uncertainty,
                        num_paths=0,
                    )

            # Empirical CDF bootstrap path
            if use_empirical:
                emp_returns = self._get_regime_adjusted_empirical_returns(binance_sym)
                interval_sec = self._settings.empirical_return_interval_seconds
                horizon_steps = max(1, round(horizon * 60 / interval_sec))
                if len(emp_returns) < self._settings.empirical_min_samples:
                    LOGGER.info(
                        "CryptoEngine: empirical fallback for %s: "
                        "only %d returns (need %d), regime=%s",
                        mq.market.ticker, len(emp_returns),
                        self._settings.empirical_min_samples,
                        getattr(self._current_regime, 'regime', 'none')
                        if self._current_regime else 'none',
                    )

                if direction == "above" and strike is not None:
                    emp_prob = self._price_model.probability_above_empirical(
                        emp_returns, current_price, strike, horizon_steps,
                        bootstrap_paths=self._settings.empirical_bootstrap_paths,
                        min_samples=self._settings.empirical_min_samples,
                        min_uncertainty=self._settings.empirical_min_uncertainty,
                    )
                elif direction == "below" and strike is not None:
                    above = self._price_model.probability_above_empirical(
                        emp_returns, current_price, strike, horizon_steps,
                        bootstrap_paths=self._settings.empirical_bootstrap_paths,
                        min_samples=self._settings.empirical_min_samples,
                        min_uncertainty=self._settings.empirical_min_uncertainty,
                    )
                    emp_prob = ProbabilityEstimate(
                        probability=1.0 - above.probability,
                        ci_lower=1.0 - above.ci_upper,
                        ci_upper=1.0 - above.ci_lower,
                        uncertainty=above.uncertainty,
                        num_paths=above.num_paths,
                    )
                elif direction == "up":
                    ref_price = current_price
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                        if looked_up is not None:
                            ref_price = looked_up
                    emp_prob = self._price_model.probability_above_empirical(
                        emp_returns, current_price, ref_price, horizon_steps,
                        bootstrap_paths=self._settings.empirical_bootstrap_paths,
                        min_samples=self._settings.empirical_min_samples,
                        min_uncertainty=self._settings.empirical_min_uncertainty,
                    )
                elif direction == "down":
                    ref_price = current_price
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                        if looked_up is not None:
                            ref_price = looked_up
                    up = self._price_model.probability_above_empirical(
                        emp_returns, current_price, ref_price, horizon_steps,
                        bootstrap_paths=self._settings.empirical_bootstrap_paths,
                        min_samples=self._settings.empirical_min_samples,
                        min_uncertainty=self._settings.empirical_min_uncertainty,
                    )
                    emp_prob = ProbabilityEstimate(
                        probability=1.0 - up.probability,
                        ci_lower=1.0 - up.ci_upper,
                        ci_upper=1.0 - up.ci_lower,
                        uncertainty=up.uncertainty,
                        num_paths=up.num_paths,
                    )

            # Always store all model probs for feature store + ensemble
            if emp_prob is not None:
                self._ab_empirical_probs[mq.market.ticker] = emp_prob
            if mc_prob is not None:
                self._ab_mc_probs[mq.market.ticker] = mc_prob
            if st_prob is not None:
                self._ab_student_t_probs[mq.market.ticker] = st_prob

            # Select which probability to use for trading
            if self._settings.probability_model == "student_t":
                prob = st_prob
            elif self._settings.probability_model == "empirical":
                prob = emp_prob
            elif self._settings.probability_model == "ab_test":
                prob = mc_prob  # Trade on MC, but log Student-t
            elif self._settings.probability_model == "ab_empirical":
                prob = mc_prob  # Trade on MC, but log empirical
            else:
                prob = mc_prob

            if prob is None:
                continue

            # ── Z-score reachability clamp ──────────────────────────
            # If the strike is statistically unreachable (Z >> threshold),
            # override model probability so the edge detector naturally
            # takes the opposite side instead of blocking the trade.
            if self._settings.zscore_filter_enabled:
                # Determine effective strike for Z-score
                z_strike = strike  # above/below use meta.strike
                if direction in ("up", "down"):
                    z_strike = None
                    if mq.market.meta.interval_start_time is not None:
                        start_ts = mq.market.meta.interval_start_time.timestamp()
                        z_strike = self._price_feed.get_price_at_time(
                            binance_sym, start_ts,
                        )

                if z_strike is not None and z_strike > 0:
                    # Compute local vol for current regime (short window)
                    local_returns = self._price_feed.get_returns(
                        binance_sym,
                        interval_seconds=60,
                        window_minutes=self._settings.zscore_vol_window_minutes,
                    )
                    local_vol = self._price_model.estimate_volatility(
                        local_returns, interval_seconds=60,
                    )
                    if local_vol <= 0:
                        local_vol = 0.50  # fallback

                    z = self._compute_zscore(
                        current_price, z_strike, local_vol,
                        mq.time_to_expiry_minutes,
                    )

                    if z > self._settings.zscore_max:
                        # Strike is unreachable — override probability
                        # to reflect which side price is currently on.
                        #
                        # For "above"/"up": if price < strike → P(cross) ≈ 0
                        #                   if price > strike → P(cross) ≈ 1
                        # For "below"/"down": if price > strike → P(below) ≈ 0
                        #                     if price < strike → P(below) ≈ 1
                        if direction in ("above", "up"):
                            clamped_p = 0.01 if current_price < z_strike else 0.99
                        else:  # below, down
                            clamped_p = 0.01 if current_price > z_strike else 0.99

                        LOGGER.debug(
                            "Z-score clamp on %s: Z=%.2f > %.2f, "
                            "prob %.3f → %.3f (price=%.1f, strike=%.1f)",
                            mq.market.ticker, z, self._settings.zscore_max,
                            prob.probability, clamped_p,
                            current_price, z_strike,
                        )
                        prob = ProbabilityEstimate(
                            probability=clamped_p,
                            ci_lower=max(0.0, clamped_p - 0.02),
                            ci_upper=min(1.0, clamped_p + 0.02),
                            uncertainty=0.02,  # very confident
                            num_paths=prob.num_paths,
                        )

            # Apply Platt calibration if available
            if self._settings.calibration_enabled and self._calibrator.is_calibrated:
                calibrated_p = self._calibrator.calibrate(prob.probability)
                prob = ProbabilityEstimate(
                    probability=calibrated_p,
                    ci_lower=prob.ci_lower,
                    ci_upper=prob.ci_upper,
                    uncertainty=prob.uncertainty,
                    num_paths=prob.num_paths,
                )

            # Scale uncertainty
            if self._settings.probability_model == "student_t":
                unc_mult = self._settings.student_t_uncertainty_multiplier
            elif self._settings.probability_model == "empirical":
                unc_mult = self._settings.empirical_uncertainty_multiplier
            else:
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

            # ── Model probability cap (v43 Fix A) ──────────────────
            # Prevent empirical model from saturating at 0.0/1.0 which
            # causes Kelly to massively oversize on false confidence.
            cap = self._settings.model_prob_cap
            floor = self._settings.model_prob_floor
            if cap < 1.0 or floor > 0.0:
                raw_p = prob.probability
                capped_p = max(floor, min(cap, raw_p))
                if capped_p != raw_p:
                    LOGGER.debug(
                        "Prob cap on %s: %.3f → %.3f",
                        mq.market.ticker, raw_p, capped_p,
                    )
                    prob = ProbabilityEstimate(
                        probability=capped_p,
                        ci_lower=max(0.0, capped_p - prob.uncertainty),
                        ci_upper=min(1.0, capped_p + prob.uncertainty),
                        uncertainty=prob.uncertainty,
                        num_paths=prob.num_paths,
                    )

            result[mq.market.ticker] = prob

        return result

    # ── Strategy cell logic ──────────────────────────────────────

    def _apply_cell_model_adjustments(
        self, edges: list[CryptoEdge],
    ) -> list[CryptoEdge]:
        """Apply per-cell model adjustments: vol dampening, blending, haircut.

        Three-layer per-cell model correction:
        1. Vol dampening: Recompute empirical bootstrap with dampened returns
           (fixes root cause of YES overconfidence — IID bootstrap ignores
           mean-reversion in 1-min returns).
        2. Re-blend with per-cell model weight (adjusts market vs model deference).
        3. Probability haircut (residual correction).

        Classifies each edge into a strategy cell, applies corrections,
        and recalculates edge.  Edges that flip negative are dropped.

        Modifies edges in-place via ``object.__setattr__`` (frozen dataclass).
        """
        from arb_bot.crypto.strategy_cell import StrategyCell, classify_cell, get_cell_config
        from arb_bot.crypto.edge_detector import blend_probabilities

        adjusted: list[CryptoEdge] = []
        for edge in edges:
            is_daily = (
                edge.market.meta.direction in ("above", "below")
                and edge.time_to_expiry_minutes > 30
            )
            cell = classify_cell(edge.side, is_daily)
            cell_cfg = get_cell_config(cell, self._settings)

            # Tag edge with cell label (for feature store + logging)
            object.__setattr__(edge, "strategy_cell", cell.value)

            # ── Layer 0: Per-cell probability model override ──
            # If cell specifies a different probability model than global,
            # recompute the probability using the cell's preferred model.
            cell_prob_model = cell_cfg.probability_model
            global_prob_model = self._settings.probability_model
            needs_model_switch = (
                cell_prob_model
                and cell_prob_model != global_prob_model
            )

            model_prob = edge.model_prob.probability
            model_unc = edge.model_uncertainty

            if needs_model_switch:
                recomputed = None
                if cell_prob_model == "mc_gbm":
                    recomputed = self._recompute_with_mc_gbm(
                        edge, cell_cfg.uncertainty_multiplier,
                    )
                elif cell_prob_model == "empirical":
                    # Recompute with empirical bootstrap (use vol_dampening=1.0
                    # here; actual dampening applied in Layer 1 if needed)
                    recomputed = self._recompute_with_vol_dampening(
                        edge, 1.0, cell_cfg.uncertainty_multiplier,
                    )

                if recomputed is not None:
                    model_prob = recomputed.probability
                    model_unc = recomputed.uncertainty
                    LOGGER.info(
                        "CryptoEngine: cell probability model override %s "
                        "(%s→%s): prob %.1f%%→%.1f%%",
                        edge.market.ticker, global_prob_model, cell_prob_model,
                        edge.model_prob.probability * 100, model_prob * 100,
                    )

            # Determine if any further correction is needed
            needs_vol_dampening = (
                abs(cell_cfg.vol_dampening - 1.0) > 0.001
                and not needs_model_switch  # Already recomputed above
            )
            needs_reblend = (
                abs(cell_cfg.model_weight - 0.7) > 0.001
                or abs(cell_cfg.prob_haircut - 1.0) > 0.001
            )

            if not needs_vol_dampening and not needs_reblend and not needs_model_switch:
                adjusted.append(edge)
                continue

            # ── Layer 1: Vol dampening (recompute probability) ──
            if needs_vol_dampening:
                dampened = self._recompute_with_vol_dampening(
                    edge, cell_cfg.vol_dampening,
                    cell_cfg.uncertainty_multiplier,
                )
                if dampened is not None:
                    model_prob = dampened.probability
                    model_unc = dampened.uncertainty
                    LOGGER.info(
                        "CryptoEngine: vol dampening %s (%.2f): "
                        "prob %.1f%%→%.1f%%, unc %.3f→%.3f",
                        edge.market.ticker, cell_cfg.vol_dampening,
                        edge.model_prob.probability * 100, model_prob * 100,
                        edge.model_uncertainty, model_unc,
                    )

            # ── Layer 2: Re-blend with per-cell weight ──
            new_blended = blend_probabilities(
                model_prob=model_prob,
                market_prob=edge.market_implied_prob,
                model_uncertainty=model_unc,
                base_model_weight=cell_cfg.model_weight,
            )

            # ── Layer 3: Probability haircut ──
            new_blended *= cell_cfg.prob_haircut
            new_blended = max(0.01, min(0.99, new_blended))

            # Recalculate effective edge
            if edge.side == "yes":
                new_edge = new_blended - edge.yes_buy_price
            else:
                new_edge = (1.0 - new_blended) - edge.no_buy_price

            # Drop if edge flipped negative
            if new_edge <= 0:
                LOGGER.info(
                    "CryptoEngine: cell model correction killed edge on %s "
                    "(%s: blended %.1f%%→%.1f%%, edge %.1f%%→%.1f%%, "
                    "dampening=%.2f)",
                    edge.market.ticker, cell.value,
                    edge.blended_probability * 100, new_blended * 100,
                    edge.edge_cents * 100, new_edge * 100,
                    cell_cfg.vol_dampening,
                )
                continue

            # Update edge fields in-place
            object.__setattr__(edge, "blended_probability", new_blended)
            object.__setattr__(edge, "edge_cents", new_edge)
            object.__setattr__(
                edge, "edge", new_blended - edge.market_implied_prob,
            )
            adjusted.append(edge)

        return adjusted

    def _recompute_with_vol_dampening(
        self,
        edge: CryptoEdge,
        vol_dampening: float,
        uncertainty_multiplier: float,
    ) -> Optional[ProbabilityEstimate]:
        """Recompute empirical probability with vol-dampened returns.

        Re-runs the bootstrap with each resampled return scaled by
        ``vol_dampening`` to model mean-reversion the IID bootstrap misses.

        Returns ``None`` if recomputation isn't possible (missing data).
        """
        meta = edge.market.meta
        underlying = meta.underlying
        direction = meta.direction
        strike = meta.strike

        # Map Kalshi underlying to Binance symbol
        binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
        if not binance_sym:
            return None

        # Get current price
        current_price = self._price_feed.get_current_price(binance_sym)
        if current_price is None or current_price <= 0:
            return None

        # Get returns (regime-adjusted)
        emp_returns = self._get_regime_adjusted_empirical_returns(binance_sym)
        if len(emp_returns) < self._settings.empirical_min_samples:
            return None

        # Compute horizon steps
        horizon = edge.time_to_expiry_minutes
        interval_sec = self._settings.empirical_return_interval_seconds
        horizon_steps = max(1, round(horizon * 60 / interval_sec))

        # Determine effective strike for up/down contracts
        effective_strike = strike
        if direction in ("up", "down"):
            effective_strike = current_price
            if meta.interval_start_time is not None:
                start_ts = meta.interval_start_time.timestamp()
                looked_up = self._price_feed.get_price_at_time(
                    binance_sym, start_ts,
                )
                if looked_up is not None:
                    effective_strike = looked_up

        if effective_strike is None or effective_strike <= 0:
            return None

        # Run vol-dampened empirical bootstrap
        raw_est = self._price_model.probability_above_empirical(
            emp_returns, current_price, effective_strike, horizon_steps,
            bootstrap_paths=self._settings.empirical_bootstrap_paths,
            min_samples=self._settings.empirical_min_samples,
            min_uncertainty=self._settings.empirical_min_uncertainty,
            vol_dampening=vol_dampening,
        )

        # For below/down, invert
        if direction in ("below", "down"):
            raw_est = ProbabilityEstimate(
                probability=1.0 - raw_est.probability,
                ci_lower=1.0 - raw_est.ci_upper,
                ci_upper=1.0 - raw_est.ci_lower,
                uncertainty=raw_est.uncertainty,
                num_paths=raw_est.num_paths,
            )

        # Apply per-cell uncertainty multiplier
        scaled_unc = raw_est.uncertainty * uncertainty_multiplier
        return ProbabilityEstimate(
            probability=raw_est.probability,
            ci_lower=max(0.0, raw_est.probability - scaled_unc),
            ci_upper=min(1.0, raw_est.probability + scaled_unc),
            uncertainty=scaled_unc,
            num_paths=raw_est.num_paths,
        )

    def _recompute_with_mc_gbm(
        self,
        edge: CryptoEdge,
        uncertainty_multiplier: float,
    ) -> Optional[ProbabilityEstimate]:
        """Recompute probability using MC GBM paths instead of empirical bootstrap.

        Used when a cell's probability_model overrides the global setting to
        ``mc_gbm``.  Generates fresh GBM paths (with jump diffusion if enabled)
        and computes P(above/below/up/down) from the terminal distribution.

        Returns ``None`` if recomputation isn't possible (missing data).
        """
        meta = edge.market.meta
        underlying = meta.underlying
        direction = meta.direction
        strike = meta.strike

        # Map Kalshi underlying to Binance symbol
        binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
        if not binance_sym:
            return None

        # Get current price
        current_price = self._price_feed.get_current_price(binance_sym)
        if current_price is None or current_price <= 0:
            return None

        # Get volatility
        vol_returns = self._price_feed.get_returns(
            binance_sym,
            window_minutes=self._settings.mc_vol_window_minutes,
            interval_seconds=60,
        )
        if len(vol_returns) < 5:
            return None

        vol = self._price_model.estimate_volatility(
            vol_returns,
            interval_seconds=60,
            method=self._settings.mc_vol_method,
        )
        if vol <= 0:
            return None

        horizon = edge.time_to_expiry_minutes

        # Generate paths (jump-diffusion or plain GBM)
        drift = 0.0  # Use zero drift for simplicity in per-cell override
        if self._settings.use_jump_diffusion:
            if self._settings.hawkes_enabled:
                import time as _time
                dynamic_intensity = self._hawkes.intensity(_time.monotonic())
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

        # Compute probability based on direction
        if direction == "above" and strike is not None:
            raw_est = self._price_model.probability_above_with_control_variate(
                paths, strike, current_price, vol, horizon, drift,
            )
        elif direction == "below" and strike is not None:
            raw_est = self._price_model.probability_below(paths, strike)
        elif direction == "up":
            ref_price = current_price
            if meta.interval_start_time is not None:
                start_ts = meta.interval_start_time.timestamp()
                looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                if looked_up is not None:
                    ref_price = looked_up
            raw_est = self._price_model.probability_up(paths, ref_price)
        elif direction == "down":
            ref_price = current_price
            if meta.interval_start_time is not None:
                start_ts = meta.interval_start_time.timestamp()
                looked_up = self._price_feed.get_price_at_time(binance_sym, start_ts)
                if looked_up is not None:
                    ref_price = looked_up
            up = self._price_model.probability_up(paths, ref_price)
            raw_est = ProbabilityEstimate(
                probability=1.0 - up.probability,
                ci_lower=1.0 - up.ci_upper,
                ci_upper=1.0 - up.ci_lower,
                uncertainty=up.uncertainty,
                num_paths=up.num_paths,
            )
        else:
            return None

        # Apply per-cell uncertainty multiplier
        scaled_unc = raw_est.uncertainty * uncertainty_multiplier
        return ProbabilityEstimate(
            probability=raw_est.probability,
            ci_lower=max(0.0, raw_est.probability - scaled_unc),
            ci_upper=min(1.0, raw_est.probability + scaled_unc),
            uncertainty=scaled_unc,
            num_paths=raw_est.num_paths,
        )

    def _compute_daily_probability(
        self,
        current_price: float,
        strike: float,
        direction: str,
        horizon_minutes: float,
        vol: float,
        drift: float,
        binance_sym: str,
    ) -> Optional[ProbabilityEstimate]:
        """Dedicated daily contract pricing with regime-aware model routing.

        Uses Merton jump-diffusion, OU mean-reversion, and GBM with
        proper drift, blended according to the current market regime
        and strike distance (moneyness).

        Parameters
        ----------
        current_price:
            Current spot price.
        strike:
            Settlement strike price.
        direction:
            Contract direction ("above" or "below").
        horizon_minutes:
            Time to expiry in minutes.
        vol:
            Annualized volatility (already VR-corrected if enabled).
        drift:
            Total assembled drift from all sources.
        binance_sym:
            Binance symbol for price feed lookups.

        Returns ProbabilityEstimate or None if inputs are degenerate.
        """
        from arb_bot.crypto.regime_detector import MEAN_REVERTING, HIGH_VOL

        if current_price <= 0 or strike <= 0 or vol <= 0 or horizon_minutes <= 0:
            return None

        settings = self._settings
        dt = horizon_minutes / (365.25 * 24.0 * 60.0)
        sqrt_dt = math.sqrt(dt)

        # ── Moneyness in vol units ────────────────────────────────
        moneyness = abs(math.log(current_price / strike)) / (vol * sqrt_dt) if vol * sqrt_dt > 0 else 0.0

        # ── Jump intensity from Hawkes (or default) ──────────────
        if settings.hawkes_enabled:
            lambda_daily = self._hawkes.intensity(time.monotonic())
        else:
            lambda_daily = settings.merton_default_intensity

        # ── Component model probabilities ─────────────────────────

        # 1. Merton jump-diffusion (always computed — handles tails)
        merton_est = self._price_model.probability_above_merton_series(
            current_price, strike, vol, horizon_minutes,
            drift=drift,
            lambda_daily=lambda_daily,
            jump_mean=settings.merton_jump_mean,
            jump_vol=settings.merton_jump_vol,
            n_terms=settings.merton_n_terms,
        )

        # 2. OU mean-reversion model
        ou_est = None
        returns_1min = self._price_feed.get_returns(
            binance_sym, interval_seconds=60,
            window_minutes=settings.mc_vol_window_minutes,
        )
        ou_params = self._price_model.estimate_ou_params(returns_1min, interval_seconds=60)
        if ou_params is not None:
            theta, mu_ou, sigma_ou = ou_params
            # Adjust mu with drift signal
            mu_adjusted = mu_ou + drift * dt
            ou_est = self._price_model.probability_above_ou(
                current_price, strike, horizon_minutes,
                theta=theta, mu=mu_adjusted, sigma=sigma_ou,
            )

        # 3. GBM analytical (uses real drift, not drift=0)
        gbm_prob = self._price_model.probability_above_analytical(
            current_price, strike, vol, horizon_minutes, drift=drift,
        )
        gbm_est = ProbabilityEstimate(
            probability=gbm_prob,
            ci_lower=max(0.0, gbm_prob - 0.03),
            ci_upper=min(1.0, gbm_prob + 0.03),
            uncertainty=0.03,
            num_paths=0,
        )

        # ── Model routing by regime × moneyness ──────────────────
        regime = self._current_regime
        deep_threshold = settings.daily_moneyness_deep_threshold
        atm_threshold = settings.daily_moneyness_atm_threshold

        if moneyness > deep_threshold:
            # Deep OTM/ITM: Merton dominates (tails matter most)
            w_merton = settings.daily_merton_weight_deep
            w_ou = (1.0 - w_merton) * 0.5
            w_gbm = (1.0 - w_merton) * 0.5
        elif regime is not None and regime.regime == MEAN_REVERTING:
            # Mean-reverting: OU dominates
            w_ou = settings.daily_ou_weight_mean_reverting
            w_merton = (1.0 - w_ou) * 0.5
            w_gbm = (1.0 - w_ou) * 0.5
        elif regime is not None and regime.is_trending:
            # Trending: GBM dominates (drift matters)
            w_gbm = settings.daily_gbm_weight_trending
            w_merton = (1.0 - w_gbm) * 0.5
            w_ou = (1.0 - w_gbm) * 0.5
        elif moneyness < atm_threshold:
            # Near ATM, no strong regime: equal weight
            w_merton = 1.0 / 3.0
            w_ou = 1.0 / 3.0
            w_gbm = 1.0 / 3.0
        else:
            # Default: slight Merton bias
            w_merton = 0.4
            w_ou = 0.3
            w_gbm = 0.3

        # If OU not available, redistribute its weight
        if ou_est is None:
            w_gbm += w_ou * 0.5
            w_merton += w_ou * 0.5
            w_ou = 0.0

        # ── Soft regime transition blending ───────────────────────
        if regime is not None and regime.is_transitioning:
            tau = settings.daily_regime_transition_tau_minutes
            # Blend toward equal weights during transitions
            blend = 0.5  # half-way blend during transition
            eq = 1.0 / 3.0
            w_merton = w_merton * (1.0 - blend) + eq * blend
            w_ou = w_ou * (1.0 - blend) + eq * blend
            w_gbm = w_gbm * (1.0 - blend) + eq * blend

        # Normalize weights
        total_w = w_merton + w_ou + w_gbm
        if total_w > 0:
            w_merton /= total_w
            w_ou /= total_w
            w_gbm /= total_w

        # ── Weighted blend ────────────────────────────────────────
        p_above = (
            w_merton * merton_est.probability
            + w_gbm * gbm_est.probability
            + (w_ou * ou_est.probability if ou_est is not None else 0.0)
        )

        # Uncertainty from model disagreement + component uncertainties
        probs_available = [merton_est.probability, gbm_est.probability]
        if ou_est is not None:
            probs_available.append(ou_est.probability)
        model_spread = max(probs_available) - min(probs_available)
        blended_unc = max(0.02, model_spread * 0.3 + 0.01)

        # ── Probability floors/ceilings (1C) ─────────────────────
        if settings.probability_floor_enabled:
            floor, ceiling = self._price_model.compute_probability_bounds(
                lambda_daily=lambda_daily,
                horizon_minutes=horizon_minutes,
                volatility=vol,
                moneyness_sigma=moneyness,
                jump_vol=settings.merton_jump_vol,
                floor_min=settings.probability_floor_min,
                ceiling_max=settings.probability_ceiling_max,
            )
            p_above = self._price_model.clamp_probability(p_above, floor, ceiling)

        # ── Direction handling ────────────────────────────────────
        if direction == "above":
            final_prob = p_above
        elif direction == "below":
            final_prob = 1.0 - p_above
        else:
            return None

        final_prob = max(0.0, min(1.0, final_prob))

        LOGGER.info(
            "DailyModel %s: P=%.3f (merton=%.3f×%.2f, gbm=%.3f×%.2f, ou=%s×%.2f) "
            "m=%.1fσ regime=%s λ=%.1f vol=%.1f%% drift=%.4f",
            direction, final_prob,
            merton_est.probability, w_merton,
            gbm_est.probability, w_gbm,
            f"{ou_est.probability:.3f}" if ou_est else "N/A", w_ou,
            moneyness,
            regime.regime if regime else "none",
            lambda_daily, vol * 100, drift,
        )

        return ProbabilityEstimate(
            probability=final_prob,
            ci_lower=max(0.0, final_prob - blended_unc),
            ci_upper=min(1.0, final_prob + blended_unc),
            uncertainty=blended_unc,
            num_paths=0,
        )

    def _compute_ensemble_probability(
        self,
        edge: CryptoEdge,
        is_yes: bool,
    ) -> tuple[float, float]:
        """Compute weighted ensemble probability from all 3 models.

        Returns (ensemble_prob, model_disagreement).
        model_disagreement = max - min across available model probabilities.

        Falls back to the primary model if fewer than 2 models are available.
        """
        ticker = edge.market.ticker

        # Collect available model probabilities
        probs: Dict[str, float] = {}
        if ticker in self._ab_mc_probs:
            probs["mc_gbm"] = self._ab_mc_probs[ticker].probability
        if ticker in self._ab_empirical_probs:
            probs["empirical"] = self._ab_empirical_probs[ticker].probability
        if ticker in self._ab_student_t_probs:
            probs["student_t"] = self._ab_student_t_probs[ticker].probability

        # Model disagreement (even if ensemble is disabled — always capture)
        if len(probs) >= 2:
            disagreement = max(probs.values()) - min(probs.values())
        else:
            disagreement = 0.0

        # If ensemble disabled or fewer than 2 models, return primary model
        if not self._settings.ensemble_enabled or len(probs) < 2:
            return edge.model_prob.probability, disagreement

        # Select weights based on YES/NO cell
        if is_yes:
            weights = {
                "empirical": self._settings.ensemble_weight_empirical_yes,
                "student_t": self._settings.ensemble_weight_student_t_yes,
                "mc_gbm": self._settings.ensemble_weight_mc_gbm_yes,
            }
        else:
            weights = {
                "empirical": self._settings.ensemble_weight_empirical_no,
                "student_t": self._settings.ensemble_weight_student_t_no,
                "mc_gbm": self._settings.ensemble_weight_mc_gbm_no,
            }

        # Weighted average (normalize weights for available models only)
        total_weight = sum(weights.get(k, 0.0) for k in probs)
        if total_weight <= 0:
            return edge.model_prob.probability, disagreement

        ensemble_prob = sum(
            probs[k] * weights.get(k, 0.0) for k in probs
        ) / total_weight

        ensemble_prob = max(0.01, min(0.99, ensemble_prob))

        LOGGER.debug(
            "CryptoEngine: ensemble %s: %s → %.1f%% (disagreement %.1f%%)",
            ticker,
            {k: f"{v*100:.1f}%" for k, v in probs.items()},
            ensemble_prob * 100,
            disagreement * 100,
        )

        return ensemble_prob, disagreement

    def _compute_iv_crosscheck(
        self,
        edge: CryptoEdge,
        model_vol: float,
    ) -> tuple[float, float | None]:
        """IV cross-check: compare model vol against market-implied vol.

        Returns (edge_multiplier, iv_rv_ratio).
        edge_multiplier: 1.0 = no change, <1.0 = dampen, >1.0 = boost.
        iv_rv_ratio: IV/RV ratio for feature store, or None if IV unavailable.
        """
        if not self._settings.iv_crosscheck_enabled or model_vol <= 0:
            return 1.0, None

        # Extract IV from current Kalshi quotes for this contract
        # We use the current edge's market price as the binary option price
        market_price = edge.yes_buy_price if edge.side == "yes" else edge.no_buy_price
        if market_price is None or market_price <= 0:
            return 1.0, None

        meta = edge.market.meta
        iv = self._price_model.extract_implied_vol(
            market_price=market_price,
            strike=meta.strike or 0,
            spot=self._price_feed.get_current_price(
                _KALSHI_TO_BINANCE.get(meta.underlying, "")
            ) or 0,
            tte_minutes=edge.time_to_expiry_minutes,
            direction="above" if meta.direction in ("above", "up") else "below",
        )

        if iv is None:
            return 1.0, None

        iv_rv_ratio = iv / model_vol

        # Apply dampen/boost based on ratio
        multiplier = 1.0
        if iv_rv_ratio > self._settings.iv_crosscheck_dampen_threshold:
            multiplier = self._settings.iv_crosscheck_dampen_factor
            LOGGER.info(
                "CryptoEngine: IV cross-check DAMPEN %s: IV=%.1f%% vs model=%.1f%% "
                "(ratio=%.2f > %.2f), edge×%.2f",
                edge.market.ticker, iv * 100, model_vol * 100,
                iv_rv_ratio, self._settings.iv_crosscheck_dampen_threshold,
                multiplier,
            )
        elif iv_rv_ratio < self._settings.iv_crosscheck_boost_threshold:
            multiplier = self._settings.iv_crosscheck_boost_factor
            LOGGER.info(
                "CryptoEngine: IV cross-check BOOST %s: IV=%.1f%% vs model=%.1f%% "
                "(ratio=%.2f < %.2f), edge×%.2f",
                edge.market.ticker, iv * 100, model_vol * 100,
                iv_rv_ratio, self._settings.iv_crosscheck_boost_threshold,
                multiplier,
            )

        return multiplier, iv_rv_ratio

    def _update_funding_persistence(self, binance_sym: str) -> int:
        """Track how many consecutive intervals funding rate stays same-sign.

        Returns the current persistence count.
        """
        if self._funding_tracker is None:
            return 0

        funding_sym = binance_sym.upper()
        rate = self._funding_tracker.get_current_rate(funding_sym)
        if rate is None:
            return self._funding_sign_persistence.get(funding_sym, 0)

        # Determine current sign
        if rate > 0.0001:
            current_sign = 1
        elif rate < -0.0001:
            current_sign = -1
        else:
            current_sign = 0

        last_sign = self._funding_last_sign.get(funding_sym, 0)
        if current_sign == last_sign and current_sign != 0:
            self._funding_sign_persistence[funding_sym] = (
                self._funding_sign_persistence.get(funding_sym, 0) + 1
            )
        else:
            self._funding_sign_persistence[funding_sym] = 1 if current_sign != 0 else 0

        self._funding_last_sign[funding_sym] = current_sign
        return self._funding_sign_persistence.get(funding_sym, 0)

    def _get_cell_rolling_stats(
        self, cell_name: str,
    ) -> tuple[float, float, int]:
        """Get rolling win rate, PnL, and 24h trade count for a cell.

        Returns (win_rate, rolling_pnl, trades_24h).
        """
        outcomes = self._cell_outcomes.get(cell_name)
        if not outcomes or len(outcomes) == 0:
            return 0.0, 0.0, 0

        now = time.time()
        wins = sum(1 for _, _, won in outcomes if won)
        win_rate = wins / len(outcomes) if len(outcomes) > 0 else 0.0
        rolling_pnl = sum(pnl for _, pnl, _ in outcomes)
        trades_24h = sum(1 for ts, _, _ in outcomes if now - ts < 86400)

        return win_rate, rolling_pnl, trades_24h

    def _record_cell_outcome(
        self, cell_name: str, pnl: float, won: bool,
    ) -> None:
        """Record a settled trade outcome for per-cell rolling stats."""
        from collections import deque
        if cell_name not in self._cell_outcomes:
            self._cell_outcomes[cell_name] = deque(maxlen=self._cell_outcome_window)
        self._cell_outcomes[cell_name].append((time.time(), pnl, won))

    def _passes_cell_signal_gates(
        self,
        edge: CryptoEdge,
        cell: "StrategyCell",
        cell_cfg: "CellConfig",
    ) -> bool:
        """Check per-cell signal requirements.

        Returns True if the edge passes all gates for its cell.
        """
        underlying = edge.market.meta.underlying
        binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")

        # Gate 1: OFI alignment (e.g. YES/15min)
        if cell_cfg.require_ofi_alignment and binance_sym:
            ofi = self._price_feed.get_ofi_multiscale(binance_sym, [30, 60, 120])
            weighted_ofi = (
                0.5 * ofi.get(30, 0.0)
                + 0.3 * ofi.get(60, 0.0)
                + 0.2 * ofi.get(120, 0.0)
            )
            # For YES above/up: need positive OFI (buying pressure)
            # For YES below/down: need negative OFI (selling pressure)
            direction = edge.market.meta.direction
            if direction in ("above", "up"):
                if weighted_ofi < cell_cfg.ofi_alignment_min:
                    LOGGER.info(
                        "CryptoEngine: cell OFI reject %s (%s) — OFI=%.2f < %.2f",
                        edge.market.ticker, cell.value,
                        weighted_ofi, cell_cfg.ofi_alignment_min,
                    )
                    return False
            elif direction in ("below", "down"):
                if weighted_ofi > -cell_cfg.ofi_alignment_min:
                    LOGGER.info(
                        "CryptoEngine: cell OFI reject %s (%s) — OFI=%.2f > -%.2f",
                        edge.market.ticker, cell.value,
                        weighted_ofi, cell_cfg.ofi_alignment_min,
                    )
                    return False

        # Gate 2: Price past strike (e.g. YES/15min)
        if cell_cfg.require_price_past_strike and binance_sym:
            spot = self._price_feed.get_current_price(binance_sym)
            strike = edge.market.meta.strike
            if spot is not None and strike is not None:
                direction = edge.market.meta.direction
                # "above" YES: spot should be > strike
                if direction in ("above", "up") and spot < strike:
                    LOGGER.info(
                        "CryptoEngine: cell price-gate reject %s (%s) — "
                        "spot %.2f < strike %.2f",
                        edge.market.ticker, cell.value, spot, strike,
                    )
                    return False
                # "below" YES: spot should be < strike
                if direction in ("below", "down") and spot > strike:
                    LOGGER.info(
                        "CryptoEngine: cell price-gate reject %s (%s) — "
                        "spot %.2f > strike %.2f",
                        edge.market.ticker, cell.value, spot, strike,
                    )
                    return False

        # Gate 3: Trend confirmation (e.g. YES/daily)
        if cell_cfg.require_trend_confirmation and binance_sym:
            window_sec = int(cell_cfg.trend_window_minutes * 60)
            returns = self._price_feed.get_returns(binance_sym, window_sec, 5)
            if len(returns) >= 2:
                cumulative_return = sum(returns)
                direction = edge.market.meta.direction
                # "above" YES: need positive trend
                if direction in ("above", "up") and cumulative_return <= 0:
                    LOGGER.info(
                        "CryptoEngine: cell trend reject %s (%s) — "
                        "return=%.4f not positive",
                        edge.market.ticker, cell.value, cumulative_return,
                    )
                    return False
                # "below" YES: need negative trend
                if direction in ("below", "down") and cumulative_return >= 0:
                    LOGGER.info(
                        "CryptoEngine: cell trend reject %s (%s) — "
                        "return=%.4f not negative",
                        edge.market.ticker, cell.value, cumulative_return,
                    )
                    return False

        return True

    # ── Online recalibration ─────────────────────────────────────

    def _recal_record(self, predicted_prob: float, outcome_binary: float, cell: str) -> None:
        """Record a (predicted, outcome) pair for online recalibration."""
        if not self._settings.recalibration_enabled:
            return

        if cell not in self._recal_buffer:
            self._recal_buffer[cell] = []

        self._recal_buffer[cell].append((predicted_prob, outcome_binary))

        # Trim to window size
        if len(self._recal_buffer[cell]) > self._recal_window:
            self._recal_buffer[cell] = self._recal_buffer[cell][-self._recal_window:]

        self._recal_trade_count += 1

        # Refit periodically
        if self._recal_trade_count % self._recal_refit_interval == 0:
            self._recal_refit(cell)

    def _recal_refit(self, cell: str) -> None:
        """Refit isotonic regression for a cell's recalibration curve."""
        buf = self._recal_buffer.get(cell, [])
        if len(buf) < self._settings.recalibration_min_samples:
            return

        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            # sklearn not available — use simple piecewise linear fallback
            self._recal_refit_simple(cell)
            return

        preds = [p for p, _ in buf]
        outcomes = [o for _, o in buf]

        ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        ir.fit(preds, outcomes)
        self._recal_curve[cell] = ir

        LOGGER.info(
            "CryptoEngine: recalibration refit for %s (%d samples)",
            cell, len(buf),
        )

    def _recal_refit_simple(self, cell: str) -> None:
        """Simple fallback recalibration without sklearn.

        Bins predictions into 5 buckets and computes realized rate per bucket.
        Stores as a dict: {bucket_midpoint: realized_rate}.
        """
        buf = self._recal_buffer.get(cell, [])
        if len(buf) < self._settings.recalibration_min_samples:
            return

        # Bin into 5 buckets: [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
        bins = {0.1: [0, 0], 0.3: [0, 0], 0.5: [0, 0], 0.7: [0, 0], 0.9: [0, 0]}
        for pred, outcome in buf:
            if pred < 0.2:
                mid = 0.1
            elif pred < 0.4:
                mid = 0.3
            elif pred < 0.6:
                mid = 0.5
            elif pred < 0.8:
                mid = 0.7
            else:
                mid = 0.9
            bins[mid][0] += 1  # count
            bins[mid][1] += outcome  # sum of outcomes

        # Build lookup: midpoint → realized rate
        lookup = {}
        for mid, (count, wins) in bins.items():
            if count >= 3:
                lookup[mid] = wins / count

        if lookup:
            self._recal_curve[cell] = lookup
            LOGGER.info(
                "CryptoEngine: recalibration refit (simple) for %s (%d samples, %d bins)",
                cell, len(buf), len(lookup),
            )

    def _recal_apply(self, raw_prob: float, cell: str) -> float:
        """Apply recalibration curve to a raw probability estimate.

        Returns the recalibrated probability, or raw_prob if no curve is fitted.
        """
        if not self._settings.recalibration_enabled:
            return raw_prob

        curve = self._recal_curve.get(cell)
        if curve is None:
            return raw_prob

        # sklearn IsotonicRegression
        if hasattr(curve, 'predict'):
            try:
                result = float(curve.predict([raw_prob])[0])
                return max(0.01, min(0.99, result))
            except Exception:
                return raw_prob

        # Simple dict lookup fallback
        if isinstance(curve, dict):
            # Find nearest bin midpoint
            if raw_prob < 0.2:
                mid = 0.1
            elif raw_prob < 0.4:
                mid = 0.3
            elif raw_prob < 0.6:
                mid = 0.5
            elif raw_prob < 0.8:
                mid = 0.7
            else:
                mid = 0.9

            if mid in curve:
                return curve[mid]

        return raw_prob

    # ── Time-of-day gate ────────────────────────────────────────

    def _quiet_hours_gate(self, edge: CryptoEdge) -> bool:
        """Return True if the trade should be blocked by quiet hours gate.

        During quiet hours, only allow trades with edge >= quiet_hours_min_edge.
        """
        if not self._settings.quiet_hours_utc:
            return False  # No quiet hours configured

        import datetime
        utc_hour = datetime.datetime.utcnow().hour
        try:
            quiet_hours = {int(h.strip()) for h in self._settings.quiet_hours_utc.split(",") if h.strip()}
        except ValueError:
            return False

        if utc_hour not in quiet_hours:
            return False  # Not in quiet hours

        # During quiet hours, require higher edge
        if abs(edge.edge) < self._settings.quiet_hours_min_edge:
            LOGGER.info(
                "CryptoEngine: quiet hours gate blocked %s — edge %.1f%% < min %.1f%% (UTC hour %d)",
                edge.market.ticker, edge.edge * 100, self._settings.quiet_hours_min_edge * 100, utc_hour,
            )
            return True

        return False

    # ── Sizing ────────────────────────────────────────────────────

    def _compute_position_size(self, edge: CryptoEdge, cell_cfg: "CellConfig | None" = None) -> int:
        """Compute contracts to trade using simplified Kelly sizing.

        Uses: f* = edge / cost, capped by kelly_fraction_cap and max_position.
        Applies regime-conditional multipliers and transition caution.
        """
        if edge.side == "yes":
            cost = edge.yes_buy_price
        else:
            cost = edge.no_buy_price

        if cost <= 0 or cost >= 1.0:
            return 0

        # 0. Online recalibration: adjust edge for sizing based on realized
        #    calibration. If the model is overconfident, this shrinks the edge.
        edge_for_sizing = edge.edge_cents
        cell_name = edge.strategy_cell or ""
        if cell_name and self._settings.recalibration_enabled:
            recal_prob = self._recal_apply(edge.blended_probability, cell_name)
            if recal_prob != edge.blended_probability:
                # Recompute edge: recalibrated model prob vs market
                if edge.side == "yes":
                    recal_edge = recal_prob - edge.market_implied_prob
                else:
                    recal_edge = (1.0 - recal_prob) - edge.market_implied_prob
                edge_for_sizing = max(0.0, recal_edge)
                if edge_for_sizing < edge.edge_cents:
                    LOGGER.debug(
                        "CryptoEngine: recal sizing adjustment %s: edge %.1f%%→%.1f%% (prob %.1f%%→%.1f%%)",
                        edge.market.id if hasattr(edge.market, 'id') else '',
                        edge.edge_cents * 100, edge_for_sizing * 100,
                        edge.blended_probability * 100, recal_prob * 100,
                    )

        # 1. Raw Kelly fraction (with optional edge cap for sizing)
        if self._settings.kelly_edge_cap > 0:
            # Cap effective edge used for sizing — model can detect large edges
            # but sizing treats them as at most edge_cap (e.g., 10¢ = 10%)
            edge_for_sizing = min(edge_for_sizing, self._settings.kelly_edge_cap)
        kelly_f = edge_for_sizing / cost

        # 2. Compute effective Kelly cap (may be boosted for mean_reverting)
        effective_kelly_cap = self._settings.kelly_fraction_cap
        if (self._current_regime is not None
            and self._current_regime.regime == "mean_reverting"
            and self._current_regime.confidence > 0.7
            and self._settings.regime_kelly_cap_boost_mean_reverting != 1.0):
            effective_kelly_cap *= self._settings.regime_kelly_cap_boost_mean_reverting

        # 3. Cap Kelly fraction
        kelly_f = min(kelly_f, effective_kelly_cap)
        kelly_f = max(0.0, kelly_f)

        # 4. Regime Kelly multiplier (Tier 1)
        if self._settings.regime_sizing_enabled and self._current_regime is not None:
            regime = self._current_regime.regime
            _regime_multipliers = {
                "mean_reverting": self._settings.regime_kelly_mean_reverting,
                "trending_up": self._settings.regime_kelly_trending_up,
                "trending_down": self._settings.regime_kelly_trending_down,
                "high_vol": self._settings.regime_kelly_high_vol,
            }
            kelly_f *= _regime_multipliers.get(regime, 1.0)

        # 5. Uncertainty haircut (Baker-McHale style)
        if edge.model_uncertainty > 0:
            sigma_p = edge.model_uncertainty
            if cost > 0 and cost < 1.0:
                n = 1.0 / (cost * (1.0 - cost))
                shrinkage = 1.0 / (1.0 + n * sigma_p * sigma_p)
            else:
                shrinkage = 1.0
            kelly_f *= shrinkage

        # 6. Transition caution zone (Tier 3)
        if (self._current_regime is not None
            and hasattr(self._current_regime, 'is_transitioning')
            and self._current_regime.is_transitioning
            and self._settings.regime_transition_sizing_multiplier < 1.0):
            kelly_f *= self._settings.regime_transition_sizing_multiplier
            LOGGER.debug(
                "CryptoEngine: transition caution — kelly_f *= %.2f",
                self._settings.regime_transition_sizing_multiplier,
            )

        # 7. Per-cell Kelly multiplier (from strategy cell config)
        if cell_cfg is not None:
            kelly_f *= cell_cfg.kelly_multiplier

        # 8. Dollar amount and contract conversion
        dollar_amount = kelly_f * self._bankroll
        max_pos = (
            cell_cfg.max_position if cell_cfg is not None
            else self._settings.max_position_per_market
        )
        dollar_amount = min(dollar_amount, max_pos)
        dollar_amount = min(dollar_amount, self._bankroll)

        contracts = int(dollar_amount / cost) if cost > 0 else 0
        return max(0, contracts)

    # ── Feature store ──────────────────────────────────────────────

    def _build_feature_vector(self, edge: CryptoEdge) -> "FeatureVector":
        """Build a FeatureVector from a detected edge for the feature store."""
        from arb_bot.crypto.feature_store import FeatureVector

        underlying = edge.market.meta.underlying
        binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
        current_price = self._price_feed.get_current_price(binance_sym) if binance_sym else None

        # Strike distance
        strike_dist = 0.0
        if edge.market.meta.strike is not None and current_price:
            strike_dist = (edge.market.meta.strike - current_price) / current_price

        # Spread
        spread = abs(edge.yes_buy_price - edge.no_buy_price) if edge.yes_buy_price and edge.no_buy_price else 0.0

        # VPIN features
        vpin_val = 0.0
        signed_vpin_val = 0.0
        vpin_trend_val = 0.0
        if self._settings.vpin_enabled and binance_sym in self._vpin_calculators:
            calc = self._vpin_calculators[binance_sym]
            vpin_val = calc.get_vpin() or 0.0
            signed_vpin_val = calc.get_signed_vpin() or 0.0
            vpin_trend_val = calc.get_vpin_trend() or 0.0

        # OFI at multiple timescales
        ofi_30 = ofi_60 = ofi_120 = ofi_300 = 0.0
        if self._settings.ofi_enabled and binance_sym:
            ofi_multi = self._price_feed.get_ofi_multiscale(binance_sym, [30, 60, 120, 300])
            ofi_30 = ofi_multi.get(30, 0.0)
            ofi_60 = ofi_multi.get(60, 0.0)
            ofi_120 = ofi_multi.get(120, 0.0)
            ofi_300 = ofi_multi.get(300, 0.0)

        # Aggressor ratio and volume acceleration
        aggressor = 0.0
        vol_accel = 0.0
        if binance_sym:
            aggressor = self._price_feed.get_aggressor_ratio(binance_sym, 300)
            vol_accel = self._price_feed.get_volume_acceleration(binance_sym, 60, 300)

        # Funding rate
        fr = fr_avg = fr_change = 0.0
        if self._funding_tracker is not None:
            funding_sym = binance_sym.upper()
            fr = self._funding_tracker.get_current_rate(funding_sym) or 0.0
            fr_avg = self._funding_tracker.get_rolling_avg(funding_sym) or 0.0
            fr_change = self._funding_tracker.get_rate_of_change(funding_sym) or 0.0

        # Volatility
        rv_1m = rv_5m = 0.0
        vol_ratio = 0.0
        if binance_sym:
            returns_1m = self._price_feed.get_returns(binance_sym, 60, 5)
            returns_5m = self._price_feed.get_returns(binance_sym, 300, 30)
            if len(returns_1m) >= 2:
                rv_1m = float(np.std(returns_1m))
            if len(returns_5m) >= 2:
                rv_5m = float(np.std(returns_5m))
            if rv_5m > 0:
                vol_ratio = rv_1m / rv_5m

        # Cross-asset
        l_ofi = l_ret = l_vol = 0.0
        if self._cross_asset is not None and binance_sym != self._cross_asset.leader_symbol:
            cross_sig = self._cross_asset.compute_features(self._price_feed, binance_sym)
            if cross_sig is not None:
                l_ofi = cross_sig.leader_ofi
                l_ret = cross_sig.leader_return_5m
                l_vol = cross_sig.leader_vol_ratio

        # Student-t A/B comparison
        st_prob_val = 0.0
        st_nu_val = 0.0
        if edge.market.ticker in self._ab_student_t_probs:
            st_est = self._ab_student_t_probs[edge.market.ticker]
            st_prob_val = st_est.probability
        binance_upper = binance_sym.upper() if binance_sym else ""
        if binance_upper in self._student_t_nu:
            st_nu_val = self._student_t_nu[binance_upper]

        # Empirical CDF A/B comparison
        emp_prob_val = 0.0
        if edge.market.ticker in self._ab_empirical_probs:
            emp_est = self._ab_empirical_probs[edge.market.ticker]
            emp_prob_val = emp_est.probability

        # Regime features
        regime_label = ""
        regime_conf = 0.0
        regime_trend = 0.0
        regime_vol = 0.0
        regime_mr = 0.0
        regime_ofi_align = 0.0
        regime_is_trans = 0
        if self._current_regime is not None and binance_sym:
            snap = self._current_regime.per_symbol.get(binance_sym)
            if snap is not None:
                regime_label = snap.regime
                regime_conf = snap.confidence
                regime_trend = snap.trend_score
                regime_vol = snap.vol_score
                regime_mr = snap.mean_reversion_score
                regime_ofi_align = snap.ofi_alignment
                regime_is_trans = 1 if getattr(snap, "is_transitioning", False) else 0
            else:
                # Use aggregate regime if no per-symbol snapshot
                regime_label = self._current_regime.regime
                regime_conf = self._current_regime.confidence

        # Regime-conditional decision features
        regime_kelly_mult = 1.0
        if self._settings.regime_sizing_enabled and regime_label:
            kelly_map = {
                "mean_reverting": self._settings.regime_kelly_mean_reverting,
                "trending_up": self._settings.regime_kelly_trending_up,
                "trending_down": self._settings.regime_kelly_trending_down,
                "high_vol": self._settings.regime_kelly_high_vol,
            }
            regime_kelly_mult = kelly_map.get(regime_label, 1.0)

        regime_min_edge_val = 0.0
        if self._settings.regime_min_edge_enabled and regime_label:
            min_edges = {
                "mean_reverting": self._settings.regime_min_edge_mean_reverting,
                "trending_up": self._settings.regime_min_edge_trending,
                "trending_down": self._settings.regime_min_edge_trending,
                "high_vol": self._settings.regime_min_edge_high_vol,
            }
            regime_min_edge_val = min_edges.get(regime_label, 0.0)

        vpin_at_entry_val = vpin_val

        # ── v42 Tier 1: Temporal features ──────────────────────────
        import datetime as _dt
        import math as _math
        now_utc = _dt.datetime.utcnow()
        hour_frac = now_utc.hour + now_utc.minute / 60.0
        hour_sin = _math.sin(2.0 * _math.pi * hour_frac / 24.0)
        hour_cos = _math.cos(2.0 * _math.pi * hour_frac / 24.0)
        day_of_week = float(now_utc.weekday())  # Monday=0

        # Nonlinear TTE: captures gamma acceleration near expiry
        tte = edge.time_to_expiry_minutes
        inv_sqrt_tte = 1.0 / _math.sqrt(max(tte, 0.1))

        # Model disagreement (ensemble)
        is_yes = (edge.side == "yes")
        _, model_disagreement = self._compute_ensemble_probability(edge, is_yes)

        # Hawkes intensity
        hawkes_val = 0.0
        if self._settings.hawkes_enabled:
            hawkes_val = self._hawkes.intensity(time.monotonic())

        # Volume flow rate
        vol_flow = 0.0
        if binance_sym:
            vol_flow = self._price_feed.get_volume_flow_rate(binance_sym, window_seconds=300)

        # ── v42 Tier 2: Meta-features ──────────────────────────────
        cell_name = getattr(edge, "strategy_cell", "")
        cell_wr, cell_pnl, cell_24h = self._get_cell_rolling_stats(cell_name)

        # Cross-asset OFI divergence
        cross_div = 0.0
        if self._cross_asset is not None and binance_sym and binance_sym != self._cross_asset.leader_symbol:
            cross_sig = self._cross_asset.compute_features(self._price_feed, binance_sym)
            if cross_sig is not None:
                cross_div = cross_sig.cross_ofi_divergence

        # Realized vol at contract horizon (15min and 1h)
        rv_15m = rv_1h = 0.0
        if binance_sym:
            returns_15m = self._price_feed.get_returns(binance_sym, 60, 15)
            if len(returns_15m) >= 2:
                rv_15m = float(np.std(returns_15m)) * _math.sqrt(15)  # scale to 15-min vol
            returns_1h = self._price_feed.get_returns(binance_sym, 300, 60)
            if len(returns_1h) >= 2:
                rv_1h = float(np.std(returns_1h)) * _math.sqrt(12)  # scale to 1-hour vol

        # ── v42 Tier 3: IV ratio + funding persistence ─────────────
        iv_rv = 0.0
        # Get model vol for IV cross-check
        model_vol = 0.0
        if binance_sym:
            vol_returns = self._price_feed.get_returns(
                binance_sym, interval_seconds=60,
                window_minutes=self._settings.mc_vol_window_minutes,
            )
            if len(vol_returns) >= 5:
                model_vol = self._price_model.estimate_volatility(
                    vol_returns, interval_seconds=60,
                    method=self._settings.mc_vol_method,
                )
        _, iv_ratio_result = self._compute_iv_crosscheck(edge, model_vol)
        if iv_ratio_result is not None:
            iv_rv = iv_ratio_result

        funding_persist = self._update_funding_persistence(binance_sym) if binance_sym else 0

        return FeatureVector(
            ticker=edge.market.ticker,
            timestamp=time.time(),
            strike_distance_pct=strike_dist,
            time_to_expiry_minutes=edge.time_to_expiry_minutes,
            implied_probability=edge.market_implied_prob,
            spread_cents=spread,
            book_depth_yes=0,  # Not available from edge
            book_depth_no=0,
            model_probability=edge.model_prob.probability,
            model_uncertainty=edge.model_uncertainty,
            edge_cents=edge.edge_cents,
            blended_probability=edge.blended_probability,
            staleness_score=edge.staleness_score,
            vpin=vpin_val,
            signed_vpin=signed_vpin_val,
            vpin_trend=vpin_trend_val,
            ofi_30s=ofi_30,
            ofi_60s=ofi_60,
            ofi_120s=ofi_120,
            ofi_300s=ofi_300,
            aggressor_ratio=aggressor,
            volume_acceleration=vol_accel,
            funding_rate=fr,
            funding_rate_8h_avg=fr_avg,
            funding_rate_change=fr_change,
            student_t_probability=st_prob_val,
            student_t_nu=st_nu_val,
            empirical_probability=emp_prob_val,
            realized_vol_1m=rv_1m,
            realized_vol_5m=rv_5m,
            vol_ratio=vol_ratio,
            leader_ofi=l_ofi,
            leader_return_5m=l_ret,
            leader_vol_ratio=l_vol,
            regime=regime_label,
            regime_confidence=regime_conf,
            regime_trend_score=regime_trend,
            regime_vol_score=regime_vol,
            regime_mean_reversion_score=regime_mr,
            regime_ofi_alignment=regime_ofi_align,
            regime_is_transitioning=regime_is_trans,
            regime_kelly_multiplier=regime_kelly_mult,
            regime_min_edge_applied=regime_min_edge_val,
            vpin_at_entry=vpin_at_entry_val,
            # v42 Tier 1
            hour_utc_sin=hour_sin,
            hour_utc_cos=hour_cos,
            day_of_week=day_of_week,
            inv_sqrt_tte=inv_sqrt_tte,
            model_disagreement=model_disagreement,
            hawkes_intensity=hawkes_val,
            volume_flow_rate=vol_flow,
            # v42 Tier 2
            cell_rolling_win_rate=cell_wr,
            cell_rolling_pnl=cell_pnl,
            cell_trades_last_24h=cell_24h,
            cross_ofi_divergence=cross_div,
            realized_vol_15m=rv_15m,
            realized_vol_1h=rv_1h,
            # v42 Tier 3
            iv_rv_ratio=iv_rv,
            funding_rate_persistence=funding_persist,
            # Trade identifiers
            side=edge.side,
            entry_price=edge.yes_buy_price if edge.side == "yes" else edge.no_buy_price,
            strategy_cell=edge.strategy_cell,
        )

    # ── Execution ─────────────────────────────────────────────────

    def _execute_paper_trade(
        self,
        edge: CryptoEdge,
        contracts: int,
        prebuilt_fv: Optional["FeatureVector"] = None,
    ) -> None:
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
            strategy_cell=edge.strategy_cell,
        )
        self._positions[edge.market.ticker] = pos

        # Record feature vector for training (C1)
        if self._feature_store is not None:
            fv = prebuilt_fv or self._build_feature_vector(edge)
            fv.entry_price = entry_price  # Use actual slippage-adjusted price
            self._feature_store.record_entry(fv)

        LOGGER.info(
            "CryptoEngine: PAPER %s %s %d@%.2f¢ (edge=%.1f%%, "
            "model=%.1f%%, market=%.1f%%, tte=%.1fm, cell=%s)",
            edge.side.upper(),
            edge.market.ticker,
            contracts,
            entry_price * 100,
            edge.edge_cents * 100,
            edge.model_prob.probability * 100,
            edge.market_implied_prob * 100,
            edge.time_to_expiry_minutes,
            edge.strategy_cell or "unknown",
        )

        # ── Cycle recorder Hook 4: record trade ──
        if (self._cycle_recorder is not None
                and hasattr(self, '_rec_cycle_id_current')
                and self._rec_cycle_id_current is not None):
            edge_id = self._rec_edge_id_map.get(edge.market.ticker)
            if edge_id is not None:
                self._cycle_recorder.record_trade(
                    self._rec_cycle_id_current, edge_id, edge.market.ticker,
                    edge.side, contracts, entry_price,
                )
            if (hasattr(self, '_rec_filter_results_current')
                    and edge.market.ticker in self._rec_filter_results_current):
                fr = self._rec_filter_results_current[edge.market.ticker]
                fr.contracts_sized = contracts
                fr.was_traded = True

    # ── Momentum trading (v18) ───────────────────────────────────

    async def _try_momentum_trades(self, market_quotes: list, momentum_symbols: set | None = None) -> None:
        """Attempt momentum trades in VPIN momentum zone.

        Uses per-symbol regime classification: each symbol is independently
        checked for high_vol regime and OFI trigger conditions.

        Parameters
        ----------
        momentum_symbols : set or None
            If provided, only consider these symbols for momentum trades.
            Symbols not in this set are skipped (they may be halted or normal).
        """
        if self._current_regime is None:
            LOGGER.info("CryptoEngine: momentum skip — no regime classified")
            return

        momentum_count = sum(
            1 for p in self._positions.values()
            if getattr(p, 'strategy', 'model') == 'momentum'
        )
        if momentum_count >= self._settings.momentum_max_concurrent:
            LOGGER.debug("CryptoEngine: momentum skip — %d/%d concurrent",
                          momentum_count, self._settings.momentum_max_concurrent)
            return

        symbols = self._settings.price_feed_symbols
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        for binance_sym in symbols:
            # Per-symbol VPIN filter: only trade symbols in momentum zone
            if momentum_symbols is not None and binance_sym not in momentum_symbols:
                continue
            # Per-symbol regime — used for OFI alignment info, NOT as a hard gate.
            # VPIN zone (0.85-0.95) is the volatility signal; requiring regime=="high_vol"
            # was redundant and too restrictive (blocked valid momentum opportunities).
            sym_snap = self._current_regime.per_symbol.get(binance_sym) if self._current_regime.per_symbol else None
            if sym_snap is None:
                LOGGER.info("CryptoEngine: momentum skip %s — no per-symbol regime", binance_sym)
                continue
            LOGGER.debug("CryptoEngine: momentum %s — regime=%s vol_score=%.2f",
                         binance_sym, sym_snap.regime, sym_snap.vol_score)

            # Cooldown check
            last_settled = self._momentum_cooldowns.get(binance_sym)
            if last_settled is not None:
                if (time.time() - last_settled) < self._settings.momentum_cooldown_seconds:
                    continue

            # Get OFI data
            ofi_multi = self._price_feed.get_ofi_multiscale(binance_sym, [30, 60, 120, 300])
            if not ofi_multi:
                continue

            # Per-symbol OFI alignment
            ofi_alignment = sym_snap.ofi_alignment

            # Compute weighted OFI magnitude + direction
            weights = {30: 4.0, 60: 3.0, 120: 2.0, 300: 1.0}
            weighted_sum = 0.0
            weight_total = 0.0
            for window, ofi_val in ofi_multi.items():
                w = weights.get(window, 1.0)
                weighted_sum += ofi_val * w
                weight_total += w
            ofi_direction_raw = weighted_sum / weight_total if weight_total > 0 else 0.0
            ofi_magnitude = abs(ofi_direction_raw)
            ofi_direction = 1 if ofi_direction_raw > 0 else -1

            # OFI confirmation streak: require N consecutive aligned cycles
            streak = _update_ofi_streak(
                self._ofi_direction_streak,
                self._ofi_last_direction,
                binance_sym,
                ofi_direction=ofi_direction,
                ofi_alignment=ofi_alignment,
                min_alignment=self._settings.momentum_ofi_alignment_min,
            )
            if streak < self._settings.momentum_min_ofi_streak:
                LOGGER.debug(
                    "CryptoEngine: momentum skip %s — OFI streak %d < %d",
                    binance_sym, streak, self._settings.momentum_min_ofi_streak,
                )
                continue

            # OFI acceleration filter: skip if flow is fading (short < medium)
            if self._settings.momentum_require_ofi_acceleration:
                if not _check_ofi_acceleration(ofi_multi):
                    LOGGER.debug(
                        "CryptoEngine: momentum skip %s — OFI decelerating",
                        binance_sym,
                    )
                    continue

            # Trigger checks
            if ofi_alignment < self._settings.momentum_ofi_alignment_min:
                continue
            if ofi_magnitude < self._settings.momentum_ofi_magnitude_min:
                continue

            # Get spot price
            spot = self._price_feed.get_current_price(binance_sym)
            if spot is None or spot <= 0:
                continue

            # Filter quotes to this underlying
            underlying = binance_sym.replace("usdt", "").upper()
            sym_quotes = [q for q in market_quotes if q.market.meta.underlying == underlying]

            result = _select_momentum_contract(sym_quotes, spot, ofi_direction, self._settings)
            if result is None:
                LOGGER.debug("CryptoEngine: momentum — no contract for %s (dir=%+d)", binance_sym, ofi_direction)
                continue

            quote, side = result

            if quote.market.ticker in self._positions:
                continue

            buy_price = quote.yes_buy_price if side == "yes" else quote.no_buy_price
            contracts = _compute_momentum_size(self._bankroll, buy_price, ofi_alignment, self._settings)
            if contracts <= 0:
                continue

            self._execute_momentum_trade(quote, side, contracts, ofi_alignment, ofi_magnitude)

            LOGGER.info(
                "CryptoEngine: MOMENTUM %s %s %d@%.2f¢ (OFI dir=%+d, align=%.2f, mag=%.0f, tte=%.1fm)",
                side.upper(), quote.market.ticker, contracts, buy_price * 100,
                ofi_direction, ofi_alignment, ofi_magnitude, quote.time_to_expiry_minutes,
            )

            momentum_count += 1
            if momentum_count >= self._settings.momentum_max_concurrent:
                break

    def _execute_momentum_trade(self, quote, side: str, contracts: int,
                                 ofi_alignment: float, ofi_magnitude: float) -> None:
        """Execute a momentum paper trade."""
        buy_price = quote.yes_buy_price if side == "yes" else quote.no_buy_price
        slippage = self._settings.paper_slippage_cents / 100.0
        entry_price = min(0.99, buy_price + slippage)

        capital_needed = entry_price * contracts
        if capital_needed > self._bankroll:
            contracts = int(self._bankroll / entry_price)
            if contracts <= 0:
                return
            capital_needed = entry_price * contracts

        self._bankroll -= capital_needed

        # Synthetic edge for position tracking
        synthetic_edge = CryptoEdge(
            market=quote.market,
            model_prob=ProbabilityEstimate(probability=0.0, uncertainty=1.0, ci_low=0.0, ci_high=1.0),
            market_implied_prob=quote.implied_probability,
            edge=0.0,
            edge_cents=0.0,
            side=side,
            recommended_price=buy_price,
            model_uncertainty=1.0,
            time_to_expiry_minutes=quote.time_to_expiry_minutes,
            yes_buy_price=quote.yes_buy_price,
            no_buy_price=quote.no_buy_price,
        )

        pos = CryptoPosition(
            ticker=quote.market.ticker,
            side=side,
            contracts=contracts,
            entry_price=entry_price,
            entry_time=time.time(),
            edge=synthetic_edge,
            model_prob=0.0,
            market_implied_prob=quote.implied_probability,
            strategy="momentum",
        )
        self._positions[quote.market.ticker] = pos

        if self._feature_store is not None:
            fv = self._build_feature_vector(synthetic_edge)
            fv.entry_price = entry_price
            fv.strategy = "momentum"
            self._feature_store.record_entry(fv)

        # Cycle recorder hook
        if (self._cycle_recorder is not None
                and hasattr(self, '_rec_cycle_id_current')
                and self._rec_cycle_id_current is not None):
            self._cycle_recorder.record_trade(
                self._rec_cycle_id_current, None, quote.market.ticker,
                side, contracts, entry_price,
            )

    async def _settle_expired_positions(self) -> None:
        """Settle positions whose markets have expired.

        Uses real Kalshi API outcomes when an HTTP client is available.
        Falls back to simulated settlement when no client is set.
        """
        now = datetime.now(timezone.utc)
        to_check: list[str] = []

        for ticker, pos in self._positions.items():
            if pos.edge.market.meta.expiry <= now:
                to_check.append(ticker)

        for ticker in to_check:
            pos = self._positions[ticker]

            if self._http_client is None:
                # No HTTP client — fall back to simulated settlement
                self._positions.pop(ticker)
                # Record momentum cooldown (v18)
                if getattr(pos, 'strategy', 'model') == 'momentum':
                    underlying = pos.edge.market.meta.underlying
                    binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                    if binance_sym:
                        self._momentum_cooldowns[binance_sym] = time.time()
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
                    actual_outcome="simulated",
                    strategy_cell=getattr(pos, 'strategy_cell', ''),
                )
                self._trades.append(record)
                LOGGER.info(
                    "CryptoEngine: SETTLED (simulated) %s %s %d@%.2f¢ PnL=$%.4f",
                    pos.side.upper(), ticker, pos.contracts,
                    pos.entry_price * 100, pnl,
                )
                # ── Cycle recorder: settlement hook (simulated) ──
                if self._cycle_recorder is not None:
                    self._cycle_recorder.record_settlement(
                        ticker, "simulated", pnl, time.time(),
                    )
                continue

            # Try real settlement from Kalshi API
            outcome = await self._fetch_settlement_outcome(ticker)

            if outcome is not None:
                # Real outcome available
                self._positions.pop(ticker)
                self._pending_settlement.pop(ticker, None)
                # Record momentum cooldown (v18)
                if getattr(pos, 'strategy', 'model') == 'momentum':
                    underlying = pos.edge.market.meta.underlying
                    binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                    if binance_sym:
                        self._momentum_cooldowns[binance_sym] = time.time()

                settled_yes = outcome
                if pos.side == "yes":
                    pnl_per_contract = (1.0 if settled_yes else 0.0) - pos.entry_price
                else:
                    pnl_per_contract = (1.0 if not settled_yes else 0.0) - pos.entry_price

                pnl = pnl_per_contract * pos.contracts
                self._session_pnl += pnl
                self._bankroll += pos.entry_price * pos.contracts + pnl

                self._calibrator.record_outcome(
                    predicted_prob=pos.model_prob,
                    outcome=settled_yes,
                    timestamp=time.time(),
                    ticker=ticker,
                    market_implied_prob=pos.market_implied_prob,
                )

                # Record outcome in feature store (C1)
                if self._feature_store is not None:
                    self._feature_store.record_outcome(ticker, settled_yes)

                # v42: Record cell outcome for rolling meta-features
                cell_name = getattr(pos, 'strategy_cell', '')
                won = (
                    (settled_yes and pos.side == "yes")
                    or (not settled_yes and pos.side == "no")
                )
                if cell_name:
                    self._record_cell_outcome(cell_name, pnl, won)

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
                    actual_outcome="yes" if settled_yes else "no",
                    strategy_cell=getattr(pos, 'strategy_cell', ''),
                )
                self._trades.append(record)
                LOGGER.info(
                    "CryptoEngine: SETTLED (real) %s %s %d@%.2f¢ PnL=$%.4f outcome=%s",
                    pos.side.upper(), ticker, pos.contracts,
                    pos.entry_price * 100, pnl,
                    "YES" if settled_yes else "NO",
                )

                # ── Online recalibration: feed settlement outcome ──
                _recal_cell = getattr(pos, 'strategy_cell', '')
                if _recal_cell:
                    # For YES side: outcome_binary = 1 if settled_yes else 0
                    # For NO side: outcome_binary = 0 if settled_yes else 1
                    if pos.side == "yes":
                        _recal_outcome = 1.0 if settled_yes else 0.0
                    else:
                        _recal_outcome = 0.0 if settled_yes else 1.0
                    self._recal_record(pos.model_prob, _recal_outcome, _recal_cell)

                # ── Cycle recorder: settlement hook (real) ──
                if self._cycle_recorder is not None:
                    self._cycle_recorder.record_settlement(
                        ticker, "yes" if settled_yes else "no", pnl, time.time(),
                    )
            else:
                # Not yet settled — check grace period
                first_check = self._pending_settlement.get(ticker)
                if first_check is None:
                    self._pending_settlement[ticker] = time.time()
                    LOGGER.debug("Settlement pending for %s (first check)", ticker)
                else:
                    elapsed_minutes = (time.time() - first_check) / 60.0
                    grace = self._settings.settlement_grace_minutes
                    if elapsed_minutes > grace:
                        # Grace period exceeded — mark unsettled
                        self._positions.pop(ticker)
                        self._pending_settlement.pop(ticker, None)
                        # Record momentum cooldown (v18)
                        if getattr(pos, 'strategy', 'model') == 'momentum':
                            underlying = pos.edge.market.meta.underlying
                            binance_sym = _KALSHI_TO_BINANCE.get(underlying, "")
                            if binance_sym:
                                self._momentum_cooldowns[binance_sym] = time.time()
                        self._bankroll += pos.entry_price * pos.contracts  # return capital

                        record = CryptoTradeRecord(
                            ticker=ticker,
                            side=pos.side,
                            contracts=pos.contracts,
                            entry_price=pos.entry_price,
                            entry_time=pos.entry_time,
                            exit_time=time.time(),
                            pnl=0.0,
                            edge_at_entry=pos.edge.edge_cents,
                            model_prob_at_entry=pos.model_prob,
                            market_prob_at_entry=pos.market_implied_prob,
                            model_uncertainty=pos.edge.model_uncertainty,
                            time_to_expiry_minutes=pos.edge.time_to_expiry_minutes,
                            settled=False,
                            actual_outcome="unsettled",
                            strategy_cell=getattr(pos, 'strategy_cell', ''),
                        )
                        self._trades.append(record)
                        LOGGER.warning(
                            "CryptoEngine: UNSETTLED %s after %.1f min grace",
                            ticker, elapsed_minutes,
                        )

    async def _fetch_settlement_outcome(self, ticker: str) -> Optional[bool]:
        """Fetch real settlement outcome from Kalshi public API.

        Returns True if settled YES, False if settled NO, None if not yet settled.
        """
        if self._http_client is None:
            return None

        try:
            url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
            resp = await self._http_client.get(url, timeout=5.0)
            if resp.status_code != 200:
                LOGGER.warning("Settlement check %s: HTTP %d", ticker, resp.status_code)
                return None
            data = resp.json()
            market = data.get("market", data)
            result = market.get("result", "")
            if result == "yes":
                return True
            elif result == "no":
                return False
            return None  # Not yet settled
        except Exception as exc:
            LOGGER.warning("Settlement check %s failed: %s", ticker, exc)
            return None

    def settle_position_with_outcome(
        self, ticker: str, settled_yes: bool
    ) -> CryptoTradeRecord | None:
        """Manually settle a position with known outcome. For testing and real settlement."""
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
            market_implied_prob=pos.market_implied_prob,
        )

        # Record outcome in feature store (C1)
        if self._feature_store is not None:
            self._feature_store.record_outcome(ticker, settled_yes)

        # v42: Record cell outcome for rolling meta-features
        cell_name = getattr(pos, 'strategy_cell', '')
        won = (
            (settled_yes and pos.side == "yes")
            or (not settled_yes and pos.side == "no")
        )
        if cell_name:
            self._record_cell_outcome(cell_name, pnl, won)

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
            actual_outcome="yes" if settled_yes else "no",
            strategy_cell=getattr(pos, 'strategy_cell', ''),
        )
        self._trades.append(record)
        return record

    def _simulate_settlement(self, pos: CryptoPosition) -> float:
        """Simulate settlement using model probability (legacy fallback).

        Only used when no HTTP client is available for real settlement.
        """
        rng = np.random.default_rng()
        settled_yes = rng.random() < pos.model_prob

        self._calibrator.record_outcome(
            predicted_prob=pos.model_prob,
            outcome=bool(settled_yes),
            timestamp=time.time(),
            ticker=pos.ticker,
            market_implied_prob=pos.market_implied_prob,
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
            "actual_outcome",
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
                "actual_outcome": t.actual_outcome,
            })
        return output.getvalue()
