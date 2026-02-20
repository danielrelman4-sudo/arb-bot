"""Paper test script for crypto prediction engine.

Fetches live BTC/SOL prices from Binance REST API and crypto market
quotes from Kalshi's public API, runs MC model to find edge.

Usage::

    python3 -m arb_bot.crypto.paper_test --duration-minutes 5
    python3 -m arb_bot.crypto.paper_test --symbols BTC SOL --mc-paths 2000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import httpx
import numpy as np

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.cycle_logger import CycleLogger
from arb_bot.crypto.edge_detector import CryptoEdge, EdgeDetector, compute_implied_probability
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.market_scanner import (
    CryptoMarket,
    CryptoMarketMeta,
    CryptoMarketQuote,
)
from arb_bot.crypto.price_feed import PriceTick
from arb_bot.crypto.price_model import PriceModel

LOGGER = logging.getLogger(__name__)

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# OKX REST API endpoints (primary data source — no geo-blocking, high volume)
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
OKX_TRADES_URL = "https://www.okx.com/api/v5/market/trades"

# Map underlying to Kalshi series tickers
_SERIES_MAP = {
    "BTC": ["KXBTC15M", "KXBTCD"],
    "ETH": ["KXETH15M", "KXETHD"],
    "SOL": ["KXSOL15M", "KXSOLD"],
}

# Binance-style symbols used internally as keys (engine, price feed, etc.)
_BINANCE_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

# OKX instrument IDs
_OKX_MAP = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "SOL": "SOL-USDT"}

# Reverse: Binance symbol -> OKX instrument ID
_BINANCE_TO_OKX = {v: _OKX_MAP[k] for k, v in _BINANCE_MAP.items()}


# ── OKX REST price feed ────────────────────────────────────────────

async def fetch_okx_prices(
    symbols: list[str],
    client: httpx.AsyncClient,
) -> dict[str, float]:
    """Fetch current prices from OKX REST API.

    *symbols* are Binance-style uppercase symbols (e.g. "BTCUSDT").
    Returns a dict mapping lowercase symbols to prices.
    """
    prices: dict[str, float] = {}
    for sym in symbols:
        okx_inst = _BINANCE_TO_OKX.get(sym)
        if not okx_inst:
            continue
        try:
            resp = await client.get(
                OKX_TICKER_URL,
                params={"instId": okx_inst},
                timeout=5.0,
            )
            if resp.status_code == 200:
                body = resp.json()
                data_list = body.get("data", [])
                if data_list:
                    prices[sym.lower()] = float(data_list[0]["last"])
        except Exception as exc:
            LOGGER.debug("OKX price fetch %s failed: %s", okx_inst, exc)
    return prices


async def fetch_okx_klines(
    symbol: str,
    minutes: int,
    client: httpx.AsyncClient,
) -> list[tuple[float, float]]:
    """Fetch 1-minute candles (timestamp, close) from OKX.

    *symbol* is a Binance-style symbol (e.g. "BTCUSDT").
    OKX returns max 100 candles per request, so we paginate if needed.
    """
    okx_inst = _BINANCE_TO_OKX.get(symbol)
    if not okx_inst:
        return []

    all_candles: list[tuple[float, float]] = []
    after: str | None = None  # OKX uses "after" for pagination (older data)

    try:
        remaining = min(minutes, 300)  # Cap at 5 hours of 1m candles
        while remaining > 0:
            batch = min(remaining, 100)
            params: dict[str, str] = {
                "instId": okx_inst,
                "bar": "1m",
                "limit": str(batch),
            }
            if after:
                params["after"] = after
            resp = await client.get(OKX_CANDLES_URL, params=params, timeout=10.0)
            if resp.status_code != 200:
                break
            body = resp.json()
            data_list = body.get("data", [])
            if not data_list:
                break
            for candle in data_list:
                # OKX format: [ts_ms, open, high, low, close, vol, volCcy, ...]
                ts = float(candle[0]) / 1000.0
                close = float(candle[4])
                all_candles.append((ts, close))
            # OKX returns newest first; "after" = oldest timestamp for next page
            after = data_list[-1][0]
            remaining -= len(data_list)
            if len(data_list) < batch:
                break  # No more data
    except Exception as exc:
        LOGGER.warning("OKX klines %s failed: %s", okx_inst, exc)

    # OKX returns newest-first; reverse for chronological order
    all_candles.reverse()
    return all_candles


async def fetch_okx_trades(
    okx_symbol: str,
    limit: int,
    client: httpx.AsyncClient,
) -> list[dict]:
    """Fetch recent trades from OKX (high volume, no geo-blocking)."""
    try:
        resp = await client.get(
            OKX_TRADES_URL,
            params={
                "instId": okx_symbol,
                "limit": str(min(limit, 100)),  # OKX max is 100 per request
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            return body.get("data", [])
    except Exception as exc:
        LOGGER.warning("OKX trades %s failed: %s", okx_symbol, exc)
    return []


# Legacy aliases for backward compatibility
fetch_binance_prices = fetch_okx_prices
fetch_binance_klines = fetch_okx_klines


# ── Kalshi market parsing ──────────────────────────────────────────

def _infer_underlying(ticker: str) -> str:
    """Infer underlying from ticker prefix."""
    t = ticker.upper()
    if "BTC" in t:
        return "BTC"
    elif "ETH" in t:
        return "ETH"
    elif "SOL" in t:
        return "SOL"
    return "UNKNOWN"


def _infer_interval(series: str) -> str:
    """Infer interval from series ticker."""
    s = series.upper()
    if "15M" in s:
        return "15min"
    elif "1H" in s:
        return "1hour"
    elif s.endswith("D"):
        return "daily"
    return "unknown"


def _infer_direction_and_strike(
    ticker: str, subtitle: str, series: str,
) -> tuple[str, float | None]:
    """Infer direction and strike from ticker/subtitle."""
    t = ticker.upper()
    # Daily above/below: has -T or -B followed by number
    m = re.search(r"-T([\d.]+)$", t)
    if m:
        return "above", float(m.group(1))
    m = re.search(r"-B([\d.]+)$", t)
    if m:
        return "below", float(m.group(1))

    # 15M up/down: series ends in 15M, suffix is -00 or similar
    if "15M" in series.upper() or "1H" in series.upper():
        # These are up/down markets — check subtitle or default to up
        sub_lower = subtitle.lower()
        if "down" in sub_lower or "decrease" in sub_lower:
            return "down", None
        return "up", None

    return "up", None


def kalshi_raw_to_quote(
    raw: dict,
    now: datetime,
    min_book_volume: int = 0,
) -> CryptoMarketQuote | None:
    """Convert a raw Kalshi market dict to CryptoMarketQuote.

    Instead of parsing the ticker format (which varies), we use
    the close_time, series_ticker, subtitle, and price fields directly.
    """
    ticker = raw.get("ticker", "")
    series = raw.get("series_ticker", "")
    subtitle = raw.get("subtitle", "")
    status = raw.get("status", "")

    if status not in ("active", "open"):
        return None

    # Parse close time
    close_str = raw.get("close_time", "")
    if not close_str:
        return None
    try:
        close_str = close_str.replace("Z", "+00:00")
        close_time = datetime.fromisoformat(close_str)
    except (ValueError, TypeError):
        return None

    tte_minutes = (close_time - now).total_seconds() / 60.0
    if tte_minutes <= 0:
        return None

    # Infer metadata
    underlying = _infer_underlying(ticker)
    interval = _infer_interval(series)
    direction, strike = _infer_direction_and_strike(ticker, subtitle, series)

    # Compute interval_start_time for up/down contracts
    interval_start_time = None
    if direction in ("up", "down"):
        # Use open_time from API if available, otherwise derive from close_time
        open_str = raw.get("open_time", "")
        if open_str:
            try:
                open_str_clean = open_str.replace("Z", "+00:00")
                interval_start_time = datetime.fromisoformat(open_str_clean)
            except (ValueError, TypeError):
                pass
        if interval_start_time is None:
            if interval == "15min":
                interval_start_time = close_time - timedelta(minutes=15)
            elif interval == "1hour":
                interval_start_time = close_time - timedelta(hours=1)

    meta = CryptoMarketMeta(
        underlying=underlying,
        interval=interval,
        expiry=close_time,
        strike=strike,
        direction=direction,
        series_ticker=series,
        interval_index=None,
        interval_start_time=interval_start_time,
    )
    market = CryptoMarket(ticker=ticker, meta=meta)

    # Extract prices (Kalshi returns in cents)
    yes_bid = (raw.get("yes_bid") or 0) / 100.0
    yes_ask = (raw.get("yes_ask") or 0) / 100.0
    no_bid = (raw.get("no_bid") or 0) / 100.0
    no_ask = (raw.get("no_ask") or 0) / 100.0

    yes_buy = yes_ask if yes_ask > 0 else 0.50
    no_buy = no_ask if no_ask > 0 else 0.50

    # Skip markets with no real book (yes_ask=1¢ or no_ask=1¢)
    if yes_buy <= 0.02 or no_buy <= 0.02:
        return None

    # ── Book quality filters ──
    # A tradeable market needs BOTH sides to have real bids.
    # If yes_bid=0, there are no YES buyers — the YES ask is just the
    # platform minimum (50¢). Similarly if no_bid=0.
    # Even if one side has bids, we can't reliably compute implied
    # probability from a one-sided book, and we certainly can't trade
    # against phantom liquidity.
    has_yes_book = yes_bid > 0.01  # Real YES bids exist
    has_no_book = no_bid > 0.01   # Real NO bids exist

    if not has_yes_book or not has_no_book:
        # One-sided or empty book — skip entirely
        return None

    # Also skip if the spread is impossibly wide (> 40¢ on either side)
    yes_spread = yes_ask - yes_bid if yes_bid > 0 else 1.0
    no_spread = no_ask - no_bid if no_bid > 0 else 1.0
    if yes_spread > 0.40 or no_spread > 0.40:
        return None

    # Volume filter: skip low-volume markets
    volume = raw.get("volume", 0)
    if min_book_volume > 0 and isinstance(volume, (int, float)) and volume < min_book_volume:
        return None

    # Use bid/ask midpoints for more accurate implied probability
    yes_mid = (yes_bid + yes_ask) / 2.0
    no_mid = (no_bid + no_ask) / 2.0
    implied = compute_implied_probability(yes_mid, no_mid)

    return CryptoMarketQuote(
        market=market,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=raw.get("volume", 0),
        no_buy_size=raw.get("volume", 0),
        yes_bid_price=yes_bid if yes_bid > 0 else None,
        no_bid_price=no_bid if no_bid > 0 else None,
        time_to_expiry_minutes=tte_minutes,
        implied_probability=implied,
    )


# ── Kalshi market fetching ─────────────────────────────────────────

async def fetch_kalshi_crypto_markets(
    underlyings: list[str],
    client: httpx.AsyncClient,
) -> list[dict]:
    """Fetch crypto markets from Kalshi public API."""
    all_markets: list[dict] = []

    for underlying in underlyings:
        series_list = _SERIES_MAP.get(underlying, [])
        for series in series_list:
            try:
                resp = await client.get(
                    f"{KALSHI_API}/markets",
                    params={
                        "series_ticker": series,
                        "status": "open",
                        "limit": 200,
                    },
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    markets = data.get("markets", [])
                    # Inject series_ticker since API might not return it
                    for m in markets:
                        m.setdefault("series_ticker", series)
                    all_markets.extend(markets)
                    LOGGER.info("Kalshi: %d markets for %s", len(markets), series)
                else:
                    LOGGER.warning("Kalshi %s: HTTP %d", series, resp.status_code)
            except Exception as exc:
                LOGGER.warning("Kalshi %s: %s", series, exc)

    return all_markets


# ── Main test loop ─────────────────────────────────────────────────

async def run_paper_test(
    underlyings: list[str],
    duration_minutes: float,
    mc_paths: int,
    min_edge: float,
    scan_interval: float,
    max_tte: int,
    probability_model: str = "empirical",
    daily_model: bool = False,
    garch_enabled: bool = False,
) -> None:
    """Run the crypto engine paper test."""

    binance_symbols = [_BINANCE_MAP[u] for u in underlyings if u in _BINANCE_MAP]

    # Feature store path (unique per run)
    fs_path = f"arb_bot/output/feature_store_v11_{int(time.time())}.csv"

    settings = CryptoSettings(
        enabled=True,
        paper_mode=True,
        symbols=[f"KX{u}" for u in underlyings],
        price_feed_symbols=[s.lower() for s in binance_symbols],
        mc_num_paths=mc_paths,
        price_history_minutes=1440 if garch_enabled else 60,  # GARCH needs 24h of 1-min data at startup
        min_edge_pct=min_edge,
        min_edge_pct_daily=min_edge,  # Same as min_edge; per-cell logic applies stricter thresholds
        min_edge_cents=min_edge,
        dynamic_edge_threshold_enabled=False,  # Per-cell logic applies real thresholds; this extra layer is redundant
        max_model_uncertainty=0.30,  # Relaxed from 0.25 — allow more uncertain edges for data collection
        model_uncertainty_multiplier=3.0,
        bankroll=500.0,
        max_position_per_market=50.0,
        max_concurrent_positions=20,  # Relaxed from 10 — allow more simultaneous trades for data
        max_positions_per_underlying=5,  # Relaxed from 3 — allow more trades per underlying
        max_new_trades_per_cycle=2,  # v44 Fix G: max 2 new entries per scan cycle (prevents correlated triple wipeout)
        kelly_fraction_cap=0.06,
        kelly_edge_cap=0.10,                  # v41: Cap edge at 10% for Kelly sizing (prevents overconfident large positions)
        scan_interval_seconds=scan_interval,
        paper_slippage_cents=0.5,
        confidence_level=0.95,
        mc_vol_window_minutes=30,
        min_minutes_to_expiry=1,
        max_minutes_to_expiry=max_tte,
        min_book_depth_contracts=1,
        allowed_directions=["above", "below", "up", "down"],
        # ── v41: Time-of-day gate ─────────────────────────────────────
        quiet_hours_utc="0,1,2",              # UTC 00-02 (local 16-18): 29% WR in v40, require higher edge
        quiet_hours_min_edge=0.08,            # 8% min edge during quiet hours (vs 4% normal)
        # ── v41: Online recalibration ─────────────────────────────────
        recalibration_enabled=True,
        recalibration_window=50,              # Rolling buffer of last 50 trades per cell
        recalibration_refit_interval=10,      # Refit isotonic curve every 10 settlements
        recalibration_min_samples=15,         # Need 15 samples before applying recal curve
        # Fix 1: Trend drift
        trend_drift_enabled=True,
        trend_drift_window_minutes=15,
        trend_drift_decay=5.0,
        trend_drift_max_annualized=5.0,
        # Fix 2: Raised thresholds (min_edge_pct/min_edge_cents from --min-edge arg)
        # min_edge_pct_daily lowered from 0.15 → 0.12 (per-cell logic applies stricter thresholds)
        # Fix 3: Isotonic calibration (upgraded from Platt)
        calibration_enabled=True,
        calibration_min_samples=20,
        calibration_recalibrate_every=10,
        calibration_method="isotonic",
        calibration_isotonic_min_samples=20,
        # Fix 4: NO-side threshold equalized with YES (was 0.20)
        min_edge_pct_no_side=min_edge,  # Relaxed to match --min-edge for max data collection
        # Fix 5: Model-market divergence filter
        min_model_market_divergence=0.02,  # Relaxed from 0.06 for max data collection
        # Fix 6: Book volume filter (0 = no filter; daily strikes have low volume early)
        min_book_volume=0,
        # OFI microstructure drift
        ofi_enabled=True,
        ofi_window_seconds=600,
        ofi_alpha=0.0,  # starts neutral, calibrated at runtime
        ofi_recalibrate_interval_hours=4.0,
        # Multi-timescale OFI (Phase A3)
        multiscale_ofi_enabled=True,
        multiscale_ofi_windows="30,60,120,300",
        multiscale_ofi_weights="0.4,0.3,0.2,0.1",
        # Volume clock — calibrated from 30-day historical data
        volume_clock_enabled=True,
        volume_clock_short_window_seconds=300,
        volume_clock_baseline_window_seconds=14400,  # 4hr
        volume_clock_ratio_floor=0.25,
        volume_clock_ratio_ceiling=2.5,
        activity_scaling_enabled=False,  # Superseded by volume clock
        activity_scaling_short_window_seconds=300,
        activity_scaling_long_window_seconds=3600,
        # Jump diffusion + Hawkes self-exciting intensity
        use_jump_diffusion=True,
        hawkes_enabled=True,
        mc_jump_intensity=3.0,
        mc_jump_mean=0.0,
        mc_jump_vol=0.02,
        # ── Phase A: New signal features ──────────────────────────
        # A1: Staleness detection
        staleness_enabled=True,
        staleness_spot_move_threshold=0.003,
        staleness_quote_change_threshold=0.005,
        staleness_lookback_seconds=120,
        staleness_max_age_seconds=300,
        staleness_edge_bonus=0.02,
        # A2: Funding rate
        funding_rate_enabled=True,
        funding_rate_poll_interval_seconds=300,
        funding_rate_extreme_threshold=0.0005,
        funding_rate_drift_weight=0.5,
        # A5: Cross-asset features
        cross_asset_enabled=True,
        cross_asset_leader="btcusdt",
        cross_asset_ofi_weight=0.3,
        cross_asset_vol_weight=0.2,
        cross_asset_return_weight=0.2,
        cross_asset_return_scale=20.0,  # v9: was 105,192x annualization
        cross_asset_max_drift=2.0,     # v9: clamp cross-asset drift
        # ── v9: Drift safety ────────────────────────────────────
        max_total_drift=2.0,           # clamp sum of all 5 drift sources
        mean_reversion_enabled=True,   # OU-like pull against recent moves
        mean_reversion_kappa=50.0,     # reversion strength
        mean_reversion_lookback_minutes=5.0,
        # ── Phase B: Core signals ─────────────────────────────────
        # B1: VPIN
        vpin_enabled=True,
        vpin_bucket_volume=0.0,  # Auto-calibrate
        vpin_num_buckets=50,
        vpin_auto_calibrate_window_minutes=60,
        vpin_extreme_threshold=0.7,
        vpin_drift_weight=0.4,
        vpin_vol_boost_factor=1.5,
        # B2: Confidence scoring — disabled for now, just observe
        confidence_scoring_enabled=False,
        confidence_min_score=0.65,
        confidence_min_agreement=3,
        # ── Phase C: Training pipeline ────────────────────────────
        # C1: Feature store — ENABLED to collect training data
        feature_store_enabled=True,
        feature_store_path=fs_path,
        feature_store_min_samples=200,
        # C2: Classifier — ENABLED with trained model
        classifier_enabled=True,
        classifier_model_path="arb_bot/output/classifier_model_model.json",  # Global fallback
        classifier_veto_threshold=0.4,  # Default reject trades where P(win) < 40%
        classifier_min_training_samples=50,  # Lower for small dataset
        classifier_retrain_interval_hours=1.0,  # Retrain hourly as new data comes in
        # Per-cell classifiers (trained via: python3 -m arb_bot.crypto.train_classifier --cell <cell>)
        classifier_model_path_yes_15min="arb_bot/output/classifier_model_model_yes_15min.json",
        classifier_model_path_yes_daily="arb_bot/output/classifier_model_model_yes_daily.json",
        classifier_model_path_no_15min="arb_bot/output/classifier_model_model_no_15min.json",
        classifier_model_path_no_daily="arb_bot/output/classifier_model_model_no_daily.json",
        # Per-cell veto thresholds
        classifier_veto_threshold_yes_15min=0.45,
        classifier_veto_threshold_yes_daily=0.40,
        classifier_veto_threshold_no_15min=0.40,
        classifier_veto_threshold_no_daily=0.40,
        # ── v11: Student-t A/B test ────────────────────────────────
        probability_model=probability_model,
        # ── Empirical CDF model ──────────────────────────────────────
        empirical_window_minutes=120,
        empirical_min_samples=30,
        empirical_bootstrap_paths=2000,
        empirical_min_uncertainty=0.02,
        empirical_uncertainty_multiplier=1.5,
        empirical_return_interval_seconds=60,
        # ── Regime-conditional improvements ──────────────────────────
        # Tier 1: Regime Kelly multiplier
        regime_sizing_enabled=True,
        regime_kelly_mean_reverting=0.7,   # v43 Fix C: was 1.5 — boosted sizing in overconfident regime (7/9 MR trades lost)
        regime_kelly_trending_up=0.4,
        regime_kelly_trending_down=0.5,
        regime_kelly_high_vol=0.3,  # Was 0.0; zero blocks ALL trades in high-vol (common in crypto)
        # Tier 1: Regime min edge thresholds — DISABLED: per-cell logic handles thresholds
        regime_min_edge_enabled=False,
        regime_min_edge_mean_reverting=0.12,
        regime_min_edge_trending=0.12,
        regime_min_edge_high_vol=0.12,
        # Tier 1: VPIN halt gate — raised threshold for data collection
        vpin_halt_enabled=True,
        vpin_halt_threshold=0.95,  # Was 0.85; only halt on extreme VPIN for data collection
        # Tier 2: Counter-trend filter — DISABLED for data collection
        regime_skip_counter_trend=False,  # Let classifier learn which trends matter
        regime_skip_counter_trend_min_conf=0.6,
        # Tier 2: Vol regime adjustment
        regime_vol_boost_high_vol=1.5,
        regime_empirical_window_high_vol=30,
        regime_empirical_window_trending=60,
        # Tier 2: Mean-reverting size boost
        regime_kelly_cap_boost_mean_reverting=1.0,  # v43 Fix C: was 1.5 — no cap boost in mean-reverting
        # Tier 3: Conditional trend drift
        regime_conditional_drift=True,
        # Tier 3: Transition caution zone
        regime_transition_sizing_multiplier=0.3,
        # ── Cycle recorder ──────────────────────────────────────────
        cycle_recorder_enabled=True,
        cycle_recorder_db_dir="arb_bot/output/recordings",
        # ── v18: Micro-momentum lane ────────────────────────────────
        momentum_enabled=True,
        momentum_vpin_floor=0.85,
        momentum_vpin_ceiling=0.95,
        momentum_ofi_alignment_min=0.6,
        momentum_ofi_magnitude_min=200.0,
        momentum_max_tte_minutes=15.0,
        momentum_price_floor=0.15,
        momentum_price_ceiling=0.40,
        momentum_kelly_fraction=0.03,
        momentum_max_position=25.0,
        momentum_max_concurrent=2,
        momentum_cooldown_seconds=120.0,
        # v19: OFI streak + acceleration filters
        momentum_min_ofi_streak=3,
        momentum_require_ofi_acceleration=True,
        momentum_max_contracts=100,
        # ── Strategy cell overrides (horizon-aware, classifier-gated) ────
        # 15min cells: model has good short-term data, trust it more.
        # Daily cells: model extrapolates beyond data window, defer to
        # market + require higher edge.  Classifier veto gate on all.
        # YES/15min — empirical bootstrap: captures actual return distribution
        cell_yes_15min_probability_model="empirical",
        cell_yes_15min_min_edge_pct=min_edge,        # 4% — model reliable here
        cell_yes_15min_vol_dampening=1.0,
        cell_yes_15min_prob_haircut=1.0,
        cell_yes_15min_require_ofi=False,
        cell_yes_15min_require_price_past_strike=False,
        cell_yes_15min_model_weight=0.65,            # Trust model for short horizon
        cell_yes_15min_uncertainty_mult=1.5,
        cell_yes_15min_kelly_multiplier=0.6,
        cell_yes_15min_max_position=25.0,
        cell_yes_15min_empirical_window=15,   # v43 Fix D: was 30 — 30m lookback too long for 15m contracts
        # YES/daily — empirical with horizon-aware vol scaling
        cell_yes_daily_probability_model="empirical",
        cell_yes_daily_min_edge_pct=0.08,            # 8% — higher bar for daily (model extrapolating)
        cell_yes_daily_vol_dampening=1.0,
        cell_yes_daily_prob_haircut=1.0,
        cell_yes_daily_require_trend=False,
        cell_yes_daily_model_weight=0.45,            # Defer to market for long horizon
        cell_yes_daily_uncertainty_mult=1.5,
        cell_yes_daily_kelly_multiplier=0.6,
        cell_yes_daily_max_position=25.0,
        # NO/15min — mc_gbm: Gaussian tails underestimate extremes → edge for "won't reach X"
        cell_no_15min_probability_model="mc_gbm",
        cell_no_15min_min_edge_pct=1.0,              # DISABLED — 25% WR in v40, no edge (100% edge = impossible)
        cell_no_15min_model_weight=0.65,             # Trust model for short horizon
        cell_no_15min_uncertainty_mult=1.5,
        cell_no_15min_kelly_multiplier=0.6,
        cell_no_15min_max_position=25.0,
        cell_no_15min_empirical_window=15,    # v43 Fix D: was 30 — match YES/15min lookback
        # NO/daily — mc_gbm: thin tails = proven alpha for "price won't move that far"
        cell_no_daily_probability_model="mc_gbm",
        cell_no_daily_min_edge_pct=0.08,             # 8% — higher bar for daily (model extrapolating)
        cell_no_daily_model_weight=0.45,             # Defer to market for long horizon
        cell_no_daily_uncertainty_mult=1.5,
        cell_no_daily_kelly_multiplier=0.6,
        cell_no_daily_max_position=25.0,
        # ── v42: Ensemble probability ──────────────────────────────
        ensemble_enabled=True,
        ensemble_weight_empirical_yes=0.60,
        ensemble_weight_student_t_yes=0.25,
        ensemble_weight_mc_gbm_yes=0.15,
        ensemble_weight_empirical_no=0.20,
        ensemble_weight_student_t_no=0.30,
        ensemble_weight_mc_gbm_no=0.50,
        # ── v42: IV cross-check (start collecting data, dampen OFF for now) ──
        iv_crosscheck_enabled=False,  # Start disabled — collecting iv_rv_ratio feature first
        iv_crosscheck_dampen_threshold=1.2,
        iv_crosscheck_dampen_factor=0.7,
        iv_crosscheck_boost_threshold=0.8,
        iv_crosscheck_boost_factor=1.1,
        # ── v43: Post-mortem fixes ────────────────────────────────
        model_prob_cap=0.95,               # Fix A: safety rail only — v44 Fix F handles saturation via uncertainty floor
        model_prob_floor=0.10,             # Fix A: prevent P=0.0 extremes
        market_disagreement_max=0.30,      # Fix B: skip when model vs market > 30pp
        # ── v43: Daily pricing model ──────────────────────────────
        daily_model_enabled=daily_model,
        # ── v45: GARCH vol-spread model ──────────────────────────
        garch_enabled=garch_enabled,
        garch_lookback_minutes=1440,          # 24h of 1-min data
        garch_min_obs=120,                    # Min 2h before producing signals
        garch_refit_interval_minutes=60,      # Re-estimate hourly
        garch_vol_spread_entry_z=1.5,         # Min |z| for directional signal
        garch_spread_history_size=500,        # Rolling window for z-scoring
        garch_probability_weight=0.6,         # 60% GARCH, 40% market
        garch_market_weight=0.4,
        garch_min_moneyness_sigma=0.3,        # Filter ATM (no edge)
        garch_max_moneyness_sigma=2.5,        # Filter deep OTM (no liquidity)
        garch_uncertainty_base=0.03,          # Base uncertainty
        garch_interval_seconds=60,            # 1-min intervals
        variance_ratio_enabled=True,
        variance_ratio_min_samples=50,
        merton_enabled=True,
        merton_jump_mean=0.0,
        merton_jump_vol=0.02,
        merton_default_intensity=3.0,
        merton_n_terms=15,
        probability_floor_enabled=True,
        probability_floor_min=0.005,
        probability_ceiling_max=0.995,
        daily_ou_weight_mean_reverting=0.7,
        daily_gbm_weight_trending=0.7,
        daily_merton_weight_deep=0.8,
        daily_regime_transition_tau_minutes=5.0,
        daily_moneyness_deep_threshold=3.0,
        daily_moneyness_atm_threshold=1.0,
        daily_ofi_weights="0.1,0.2,0.3,0.4",
    )

    engine = CryptoEngine(settings)

    print(f"\n{'='*70}")
    model_label = {"mc_gbm": "MC-GBM", "student_t": "Student-t", "ab_test": "MC+Student-t A/B", "empirical": "Empirical CDF", "ab_empirical": "MC+Empirical A/B"}
    print(f"  Crypto Prediction Engine — Paper Test v11 ({model_label.get(probability_model, probability_model)})")
    print(f"  v11: Student-t analytical model A/B test,")
    print(f"        OKX trades, all predictive alpha features")
    print(f"{'='*70}")
    print(f"  Underlyings:    {', '.join(underlyings)}")
    print(f"  Binance feeds:  {', '.join(binance_symbols)}")
    print(f"  MC paths:       {mc_paths}")
    print(f"  Min edge:       {min_edge*100:.1f}% (daily: {settings.min_edge_pct_daily*100:.1f}%)")
    print(f"  Duration:       {duration_minutes} min")
    print(f"  Max TTE:        {max_tte} min")
    print(f"  Scan interval:  {scan_interval}s")
    print(f"  Prob model:     {probability_model}")
    _cell_models = {c: getattr(settings, f"cell_{c}_probability_model", "")
                    for c in ["yes_15min", "yes_daily", "no_15min", "no_daily"]}
    _overrides = {c: m for c, m in _cell_models.items() if m}
    if _overrides:
        print(f"  Cell models:    {_overrides}")
    print(f"  Calibration:    {settings.calibration_method} (min_samples={settings.calibration_min_samples})")
    print(f"  Staleness:      {'ON' if settings.staleness_enabled else 'OFF'}")
    print(f"  Multi-OFI:      {'ON' if settings.multiscale_ofi_enabled else 'OFF'} (windows={settings.multiscale_ofi_windows})")
    print(f"  Funding rate:   {'ON' if settings.funding_rate_enabled else 'OFF'} (weight={settings.funding_rate_drift_weight})")
    print(f"  Cross-asset:    {'ON' if settings.cross_asset_enabled else 'OFF'} (leader={settings.cross_asset_leader}, scale={settings.cross_asset_return_scale}, max={settings.cross_asset_max_drift})")
    print(f"  VPIN:           {'ON' if settings.vpin_enabled else 'OFF'} (drift_wt={settings.vpin_drift_weight})")
    print(f"  Drift safety:   max_total={settings.max_total_drift}, mean_rev={'ON' if settings.mean_reversion_enabled else 'OFF'} (κ={settings.mean_reversion_kappa}, lookback={settings.mean_reversion_lookback_minutes}m)")
    print(f"  Edge thresholds: YES={settings.min_edge_pct*100:.0f}%, NO={settings.min_edge_pct_no_side*100:.0f}%")
    print(f"  Dynamic edge:   {'ON' if settings.dynamic_edge_threshold_enabled else 'OFF'} (k={settings.dynamic_edge_uncertainty_multiplier})")
    print(f"  Z-score filter: {'ON' if settings.zscore_filter_enabled else 'OFF'} (max={settings.zscore_max}, vol_window={settings.zscore_vol_window_minutes}m)")
    print(f"  Regime detect:  {'ON' if settings.regime_detection_enabled else 'OFF'} (ofi_thresh={settings.regime_ofi_trend_threshold}, vpin_spike={settings.regime_vpin_spike_threshold})")
    print(f"  Regime sizing:  {'ON' if settings.regime_sizing_enabled else 'OFF'} (mr={settings.regime_kelly_mean_reverting}, tu={settings.regime_kelly_trending_up}, td={settings.regime_kelly_trending_down}, hv={settings.regime_kelly_high_vol})")
    print(f"  Regime edge:    {'ON' if settings.regime_min_edge_enabled else 'OFF'} (mr={settings.regime_min_edge_mean_reverting}, tr={settings.regime_min_edge_trending}, hv={settings.regime_min_edge_high_vol})")
    print(f"  VPIN halt:      {'ON' if settings.vpin_halt_enabled else 'OFF'} (threshold={settings.vpin_halt_threshold})")
    print(f"  Counter-trend:  {'SKIP' if settings.regime_skip_counter_trend else 'OFF'} (min_conf={settings.regime_skip_counter_trend_min_conf})")
    print(f"  Cond. drift:    {'ON' if settings.regime_conditional_drift else 'OFF'}")
    print(f"  Transition:     sizing×{settings.regime_transition_sizing_multiplier}")
    print(f"  Confidence:     {'ON' if settings.confidence_scoring_enabled else 'OFF (observing)'}")
    print(f"  Feature store:  {'ON' if settings.feature_store_enabled else 'OFF'} → {fs_path}")
    _clf_cells = [c for c in ["yes_15min", "yes_daily", "no_15min", "no_daily"]
                  if getattr(settings, f"classifier_model_path_{c}", "")]
    print(f"  Classifier:     {'ON' if settings.classifier_enabled else 'OFF (collecting data)'}"
          f" (per-cell: {', '.join(_clf_cells) if _clf_cells else 'none'})")
    print(f"  Volume clock:   ON (short=300s, baseline=4hr)")
    print(f"  Hawkes jumps:   ON (α=5.0, β=0.00115, threshold=3.0σ)")
    if settings.cycle_recorder_enabled:
        rec_db_path = os.path.join(settings.cycle_recorder_db_dir, f"session_{int(time.time())}.db")
        print(f"  Recorder:       ON → {rec_db_path}")
    else:
        rec_db_path = None
        print(f"  Recorder:       OFF")
    # v41 improvements
    print(f"  Kelly edge cap: {'%.0f%%' % (settings.kelly_edge_cap * 100) if settings.kelly_edge_cap > 0 else 'OFF'}")
    print(f"  Quiet hours:    {settings.quiet_hours_utc if settings.quiet_hours_utc else 'OFF'}"
          f"{' (min_edge=' + str(settings.quiet_hours_min_edge) + ')' if settings.quiet_hours_utc else ''}")
    print(f"  Recalibration:  {'ON' if settings.recalibration_enabled else 'OFF'}"
          f" (window={settings.recalibration_window}, refit_every={settings.recalibration_refit_interval},"
          f" min_samples={settings.recalibration_min_samples})")
    print(f"  Momentum:       {'ON' if settings.momentum_enabled else 'OFF'}"
          f"  VPIN zone: {settings.momentum_vpin_floor:.2f}–{settings.momentum_vpin_ceiling:.2f}"
          f"  OFI align>{settings.momentum_ofi_alignment_min:.1f} mag>{settings.momentum_ofi_magnitude_min:.0f}")
    # v42 improvements
    print(f"  Ensemble:       {'ON' if settings.ensemble_enabled else 'OFF'}"
          f" (YES: emp={settings.ensemble_weight_empirical_yes:.0%}/st={settings.ensemble_weight_student_t_yes:.0%}/mc={settings.ensemble_weight_mc_gbm_yes:.0%},"
          f" NO: emp={settings.ensemble_weight_empirical_no:.0%}/st={settings.ensemble_weight_student_t_no:.0%}/mc={settings.ensemble_weight_mc_gbm_no:.0%})")
    print(f"  IV cross-check: {'ON' if settings.iv_crosscheck_enabled else 'OFF (collecting data)'}"
          f" (dampen>{settings.iv_crosscheck_dampen_threshold:.1f}x, boost<{settings.iv_crosscheck_boost_threshold:.1f}x)")
    print(f"  New features:   15 v42 columns (temporal, hawkes, ensemble, meta, IV, funding)")
    print(f"  Daily model:    {'ON (v43 Merton/OU/GBM hybrid)' if settings.daily_model_enabled else 'OFF'}")
    print(f"  VR correction:  {'ON' if settings.variance_ratio_enabled else 'OFF'} (min_samples={settings.variance_ratio_min_samples})")
    print(f"  Prob bounds:    {'ON' if settings.probability_floor_enabled else 'OFF'} (floor={settings.probability_floor_min}, ceil={settings.probability_ceiling_max})")
    print(f"  Prob cap:       [{settings.model_prob_floor:.2f}, {settings.model_prob_cap:.2f}]  (v43 Fix A)")
    print(f"  Mkt disagree:   max={settings.market_disagreement_max:.0%}  (v43 Fix B)")
    print(f"  GARCH model:    {'ON (v45 vol-spread)' if settings.garch_enabled else 'OFF'}"
          f"{' (lookback=' + str(settings.garch_lookback_minutes) + 'm, refit=' + str(settings.garch_refit_interval_minutes) + 'm, z_entry=' + str(settings.garch_vol_spread_entry_z) + ')' if settings.garch_enabled else ''}")
    print(f"{'='*70}\n")

    # Wire up cycle recorder if enabled
    cycle_recorder = None
    if settings.cycle_recorder_enabled and rec_db_path:
        os.makedirs(settings.cycle_recorder_db_dir, exist_ok=True)
        from arb_bot.crypto.cycle_recorder import CycleRecorder
        cycle_recorder = CycleRecorder(rec_db_path, settings)
        cycle_recorder.open()
        engine.set_cycle_recorder(cycle_recorder)

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
        # Wire up HTTP client for real Kalshi settlement
        engine.set_http_client(client)

        # Start funding rate tracker polling
        if engine._funding_tracker is not None:
            await engine._funding_tracker.start(client)

        # 1. Bootstrap with OKX klines
        print("Loading historical prices from OKX REST API...")
        for sym in binance_symbols:
            klines = await fetch_okx_klines(sym, 60, client)
            if klines:
                for ts, close in klines:
                    engine.price_feed.inject_tick(
                        PriceTick(symbol=sym.lower(), price=close, timestamp=ts, volume=0)
                    )
                print(f"   {sym}: {len(klines)} 1m candles, latest=${klines[-1][1]:,.2f}")
            else:
                print(f"   {sym}: no data (API may be unavailable)")

        # 2. Fetch current prices
        print("\nFetching current prices from OKX...")
        prices = await fetch_okx_prices(binance_symbols, client)
        for sym, price in prices.items():
            engine.price_feed.inject_tick(
                PriceTick(symbol=sym, price=price, timestamp=time.time(), volume=0)
            )
            print(f"   {sym.upper()}: ${price:,.2f}")

        if not prices:
            print("   WARNING: No OKX prices available. Using historical data only.")

        # 2b. Bootstrap OFI data from OKX trades
        print("\nBootstrapping OFI data from OKX trades...")
        for underlying in underlyings:
            okx_sym = _OKX_MAP.get(underlying)
            binance_sym = _BINANCE_MAP.get(underlying)
            if not okx_sym or not binance_sym:
                continue
            trades = await fetch_okx_trades(okx_sym, 100, client)
            if trades:
                buy_count = 0
                sell_count = 0
                for trade in trades:
                    try:
                        price = float(trade["px"])
                        qty = float(trade["sz"])
                        ts = float(trade["ts"]) / 1000.0
                        side = trade.get("side", "")
                    except (KeyError, ValueError, TypeError):
                        continue
                    is_buyer_maker = side == "sell"
                    tick = PriceTick(
                        symbol=binance_sym.lower(), price=price, timestamp=ts,
                        volume=qty, is_buyer_maker=is_buyer_maker,
                    )
                    engine.price_feed.inject_tick(tick)
                    if side == "buy":
                        buy_count += 1
                    elif side == "sell":
                        sell_count += 1
                ofi = engine.price_feed.get_ofi(binance_sym.lower(), window_seconds=600)
                print(
                    f"   {underlying} ({okx_sym}): {len(trades)} trades loaded, "
                    f"{buy_count} buys / {sell_count} sells, "
                    f"OFI={ofi:+.3f}"
                )

        # 2c. Auto-calibrate VPIN bucket sizes from bootstrap trade data.
        # Key insight: bucket volume must be large enough to contain many
        # individual trades (~30) so the buy/sell ratio is statistically
        # meaningful.  Tiny buckets → each bucket filled by 1 trade →
        # VPIN = 1.0 permanently.  Use per-symbol minimum floors based on
        # typical trade sizes on OKX spot.
        _VPIN_MIN_BUCKET = {
            "btcusdt": 0.10,    # ~$9,700 per bucket at $97k/BTC
            "ethusdt": 1.0,     # ~$2,700 per bucket at $2,700/ETH
            "solusdt": 50.0,    # ~$7,500 per bucket at $150/SOL
        }
        if settings.vpin_enabled:
            print("\nCalibrating VPIN bucket sizes...")
            for sym in [s.lower() for s in binance_symbols]:
                if sym in engine._vpin_calculators:
                    calc = engine._vpin_calculators[sym]
                    if calc.bucket_volume <= 0:
                        min_bv = _VPIN_MIN_BUCKET.get(sym, 1.0)
                        # Try trade-size-based calibration first (more reliable)
                        buy_sells = engine.price_feed._buy_sells.get(sym, [])
                        trade_sizes = [vol for _, vol, _ in buy_sells]
                        if len(trade_sizes) >= 10:
                            bv = calc.calibrate_from_trades(
                                trade_sizes=trade_sizes,
                                trades_per_bucket=30,
                                min_bucket_volume=min_bv,
                            )
                        else:
                            # Fallback to volume-rate method
                            total_vol = engine.price_feed.get_total_volume(
                                sym, window_seconds=settings.vpin_auto_calibrate_window_minutes * 60,
                            )
                            bv = calc.auto_calibrate_bucket_size(
                                total_volume=total_vol,
                                window_minutes=float(settings.vpin_auto_calibrate_window_minutes),
                                min_bucket_volume=min_bv,
                            )
                        LOGGER.info("VPIN: calibrated bucket_volume=%.4f for %s (min=%.4f)", bv, sym, min_bv)
                        print(f"   {sym.upper()}: bucket_volume={bv:.4f} (min_floor={min_bv:.4f})")

            # Re-inject bootstrap aggTrades so VPIN can process them
            for sym in [s.lower() for s in binance_symbols]:
                if sym in engine._vpin_calculators:
                    calc = engine._vpin_calculators[sym]
                    # Get classified trades from price feed buy/sell history
                    buy_sells = engine.price_feed._buy_sells.get(sym, [])
                    replayed = 0
                    for ts, vol, is_buy in buy_sells:
                        price = engine.price_feed.get_current_price(sym) or 0.0
                        calc.process_trade(price, vol, is_buy, ts)
                        replayed += 1
                    if replayed:
                        vpin_val = calc.get_vpin()
                        print(f"   {sym.upper()}: replayed {replayed} trades, VPIN={vpin_val}")

        # 3. Show vol estimates
        print("\nVolatility estimates:")
        for sym in [s.lower() for s in binance_symbols]:
            returns = engine.price_feed.get_returns(sym, interval_seconds=60)
            if len(returns) >= 2:
                model = PriceModel(num_paths=100)
                vol = model.estimate_volatility(returns, interval_seconds=60)
                print(f"   {sym.upper()}: {vol*100:.1f}% annualized ({len(returns)} returns)")
            else:
                print(f"   {sym.upper()}: insufficient data for vol estimate")

        # 4. Start OKX trades WebSocket for continuous OFI/VPIN data
        agg_trades_task = None
        _ws_restart_count = 0
        if settings.agg_trades_ws_enabled:
            agg_trades_task = asyncio.create_task(
                engine.price_feed.connect_agg_trades_ws()
            )
            print("\nStarted OKX trades WebSocket stream for continuous OFI/VPIN data")

        # 5. Set up per-cycle CSV logger
        log_path = f"arb_bot/output/paper_v3_{int(time.time())}.csv"
        logger = CycleLogger(log_path)
        engine.set_cycle_logger(logger)
        print(f"\nCycle data will be logged to: {log_path}")

        # 6. Main loop
        start_time = time.monotonic()
        cycle = 0

        while True:
            elapsed = (time.monotonic() - start_time) / 60.0
            if elapsed >= duration_minutes:
                break

            cycle += 1
            now = datetime.now(timezone.utc)
            print(f"\n{'─'*50}")
            print(f"Cycle {cycle} ({elapsed:.1f}m / {duration_minutes}m)")
            print(f"{'─'*50}")

            # Health check: auto-restart dead WebSocket task
            if agg_trades_task is not None and agg_trades_task.done():
                exc = agg_trades_task.exception() if not agg_trades_task.cancelled() else None
                _ws_restart_count += 1
                LOGGER.warning(
                    "OKX trades WS task died (restart #%d): %s",
                    _ws_restart_count, exc,
                )
                print(f"   ⚠ OKX trades WS died (restart #{_ws_restart_count}): {exc}")
                agg_trades_task = asyncio.create_task(
                    engine.price_feed.connect_agg_trades_ws()
                )

            # Refresh prices from OKX REST
            new_prices = await fetch_okx_prices(binance_symbols, client)
            for sym, price in new_prices.items():
                engine.price_feed.inject_tick(
                    PriceTick(symbol=sym, price=price, timestamp=time.time(), volume=0)
                )

            # OFI/VPIN data comes via OKX trades WebSocket (no per-cycle REST poll)

            for sym in [s.lower() for s in binance_symbols]:
                p = engine.price_feed.get_current_price(sym)
                ofi = engine.price_feed.get_ofi(sym, window_seconds=300)
                vpin_str = ""
                if settings.vpin_enabled and sym in engine._vpin_calculators:
                    calc = engine._vpin_calculators[sym]
                    v = calc.get_vpin()
                    if v is not None:
                        vpin_str = f"  VPIN={v:.3f}"
                    elif hasattr(calc, "is_stale") and calc.is_stale:
                        age = calc.seconds_since_last_trade
                        vpin_str = f"  VPIN=stale({age:.0f}s)"
                    else:
                        vpin_str = "  VPIN=None"
                if p:
                    print(f"   {sym.upper()}: ${p:,.2f}  OFI={ofi:+.3f}{vpin_str}")

            # Fetch Kalshi markets
            raw_markets = await fetch_kalshi_crypto_markets(underlyings, client)

            # Parse into quotes
            quotes: list[CryptoMarketQuote] = []
            skipped = 0
            skipped_tte = 0
            skipped_daily = 0
            for raw in raw_markets:
                q = kalshi_raw_to_quote(raw, now, min_book_volume=settings.min_book_volume)
                if q is None:
                    skipped += 1
                    continue
                # v43 Fix E: Exclude daily contracts by ticker when targeting 15min only.
                # Daily tickers: KXBTCD-*, KXETHD-*, KXSOLD-* — these leak through
                # TTE filter near their expiry hour.
                if max_tte < 30 and q.market.meta.interval == "daily":
                    skipped_daily += 1
                    skipped += 1
                    continue
                if q.time_to_expiry_minutes <= max_tte:
                    quotes.append(q)
                else:
                    skipped_tte += 1
                    skipped += 1

            daily_str = f", daily={skipped_daily}" if skipped_daily else ""
            print(f"   Markets: {len(quotes)} tradeable, {skipped} filtered (TTE={skipped_tte}{daily_str})")

            if quotes:
                # Group by interval type
                by_interval: dict[str, int] = {}
                for q in quotes:
                    by_interval[q.market.meta.interval] = by_interval.get(q.market.meta.interval, 0) + 1
                print(f"   Breakdown: {by_interval}")

                # Show market examples (closest to expiry)
                for q in sorted(quotes, key=lambda x: x.time_to_expiry_minutes)[:8]:
                    strike_str = f"${q.market.meta.strike:,.0f}" if q.market.meta.strike else "N/A"
                    bid_str = f"Ybid={q.yes_bid_price:.2f}" if q.yes_bid_price else "Ybid=   -"
                    nbid_str = f"Nbid={q.no_bid_price:.2f}" if q.no_bid_price else "Nbid=   -"
                    print(
                        f"     {q.market.ticker[:35]:35s} "
                        f"{q.market.meta.direction:6s} "
                        f"strike={strike_str:>10s} "
                        f"YES={q.yes_buy_price:.2f} NO={q.no_buy_price:.2f} "
                        f"{bid_str} {nbid_str} "
                        f"impl={q.implied_probability:.0%} "
                        f"TTE={q.time_to_expiry_minutes:.0f}m"
                    )

                # Run MC model
                edges = await engine.run_cycle_with_quotes(quotes)

                if edges:
                    print(f"\n   EDGES DETECTED: {len(edges)}")
                    for e in edges[:8]:
                        # Show Student-t probability if available (A/B mode)
                        st_str = ""
                        if hasattr(engine, '_ab_student_t_probs'):
                            st_est = engine._ab_student_t_probs.get(e.market.ticker)
                            if st_est is not None:
                                st_str = f" st={st_est.probability:.0%}"
                        # Show empirical probability if available (A/B mode)
                        emp_str = ""
                        if hasattr(engine, '_ab_empirical_probs'):
                            emp_est = engine._ab_empirical_probs.get(e.market.ticker)
                            if emp_est is not None:
                                emp_str = f" emp={emp_est.probability:.0%}"
                        print(
                            f"     {e.market.ticker[:35]:35s} → {e.side.upper():3s} "
                            f"edge={e.edge_cents*100:+.1f}% "
                            f"mc={e.model_prob.probability:.0%}{st_str}{emp_str} "
                            f"market={e.market_implied_prob:.0%} "
                            f"unc={e.model_uncertainty:.3f} spread={e.spread_cost*100:.1f}\u00a2"
                        )
                else:
                    print("   No edges above threshold")
            else:
                print("   No markets to evaluate")

            # Position/PnL summary
            if engine.positions:
                print(f"\n   Open positions: {len(engine.positions)}")
                for ticker, pos in engine.positions.items():
                    print(f"     {ticker}: {pos.side.upper()} {pos.contracts}@{pos.entry_price:.2f}")

            if engine.trades:
                wins = sum(1 for t in engine.trades if t.pnl > 0)
                print(f"   Trades: {len(engine.trades)} (wins={wins}) PnL=${engine.session_pnl:.2f}")

            print(f"   Bankroll: ${engine.bankroll:.2f}")

            await asyncio.sleep(scan_interval)

    # Clean up funding rate tracker
    if engine._funding_tracker is not None:
        await engine._funding_tracker.stop()

    # Clean up aggTrades WebSocket
    if agg_trades_task is not None:
        agg_trades_task.cancel()
        try:
            await agg_trades_task
        except asyncio.CancelledError:
            pass

    # Settle remaining open positions via real Kalshi API
    open_positions = list(engine.positions.keys())
    if open_positions:
        print(f"\n  Settling {len(open_positions)} open positions via Kalshi API...")
        settled_count = 0
        unsettled_count = 0
        for ticker in open_positions:
            pos = engine.positions.get(ticker)
            if pos is None:
                continue
            # Try real settlement from Kalshi
            try:
                outcome = await engine._fetch_settlement_outcome(ticker)
            except Exception as exc:
                LOGGER.warning("Settlement fetch failed for %s: %s", ticker, exc)
                outcome = None

            if outcome is not None:
                record = engine.settle_position_with_outcome(ticker, outcome)
                if record:
                    outcome_str = "YES" if outcome else "NO"
                    settled_count += 1
                    print(
                        f"    {ticker}: settled {outcome_str} (real) → "
                        f"PnL=${record.pnl:+.2f} ({record.side.upper()} "
                        f"{record.contracts}@{record.entry_price:.2f})"
                    )
            else:
                unsettled_count += 1
                print(f"    {ticker}: not yet settled (skipping — no simulated fallback)")
        if unsettled_count:
            print(f"  ⚠ {unsettled_count} positions remain unsettled (markets still open)")

    # Flush remaining feature store entries
    if hasattr(engine, '_feature_store') and engine._feature_store is not None:
        engine._feature_store.flush()

    # Close cycle logger and recorder
    logger.close()
    if cycle_recorder is not None:
        cycle_recorder.close()
        print(f"  Cycle recorder saved: {rec_db_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Cycles:         {cycle}")
    print(f"  Trades:         {len(engine.trades)}")
    if engine.trades:
        wins = sum(1 for t in engine.trades if t.pnl > 0)
        losses = sum(1 for t in engine.trades if t.pnl < 0)
        avg_edge = sum(t.edge_at_entry for t in engine.trades) / len(engine.trades)
        total_pnl = sum(t.pnl for t in engine.trades)
        avg_contracts = sum(t.contracts for t in engine.trades) / len(engine.trades)
        print(f"  Wins:           {wins} ({wins/len(engine.trades)*100:.0f}%)")
        print(f"  Losses:         {losses}")
        print(f"  Avg edge:       {avg_edge*100:.2f}%")
        print(f"  Avg contracts:  {avg_contracts:.0f}")
        print(f"  Total PnL:      ${total_pnl:+.2f}")
        print()
        print(f"  Per-trade breakdown:")
        for t in engine.trades:
            outcome_tag = f"[{t.actual_outcome}]" if t.actual_outcome != "unsettled" else ""
            print(
                f"    {t.ticker[:35]:35s} {t.side.upper():3s} "
                f"{t.contracts:3d}@{t.entry_price:.2f} "
                f"PnL=${t.pnl:+.2f} "
                f"edge={t.edge_at_entry*100:.1f}% "
                f"model={t.model_prob_at_entry:.0%} market={t.market_prob_at_entry:.0%} "
                f"{outcome_tag}"
            )
        # Per-cell breakdown
        cell_stats: dict[str, dict] = {}
        for t in engine.trades:
            cell = getattr(t, 'strategy_cell', '') or 'unknown'
            if cell not in cell_stats:
                cell_stats[cell] = {"trades": 0, "wins": 0, "pnl": 0.0}
            cell_stats[cell]["trades"] += 1
            if t.pnl > 0:
                cell_stats[cell]["wins"] += 1
            cell_stats[cell]["pnl"] += t.pnl
        if cell_stats:
            print(f"\n  Per-cell breakdown:")
            for cell_name in sorted(cell_stats.keys()):
                s = cell_stats[cell_name]
                wr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
                print(f"    {cell_name:15s}  {s['trades']} trades  {s['wins']} wins ({wr:.0f}%)  PnL=${s['pnl']:+.2f}")
    print(f"\n  Net PnL:        ${engine.session_pnl:+.2f}")
    print(f"  Final bankroll: ${engine.bankroll:.2f}")
    print(f"  Cycle data:     {log_path}")
    print(f"{'='*70}")

    # Brier score comparison: model vs market
    cal = engine.calibrator
    if cal.num_records >= 1:
        real_count = sum(
            1 for t in engine.trades
            if t.actual_outcome in ("yes", "no")
        )
        sim_count = sum(
            1 for t in engine.trades
            if t.actual_outcome == "simulated"
        )
        model_brier = cal.compute_brier_score()
        market_brier = cal.compute_market_brier_score()
        bss = cal.compute_brier_skill_score()

        print(f"\n{'='*70}")
        print(f"  BRIER SCORE COMPARISON ({cal.num_records} settled trades)")
        print(f"  Real outcomes:  {real_count}  |  Simulated: {sim_count}")
        print(f"{'='*70}")
        print(f"  Model Brier:    {model_brier:.4f}")
        print(f"  Market Brier:   {market_brier:.4f}")
        if bss >= 0:
            print(f"  Brier Skill:    {bss:+.4f}  (model beats market)")
        else:
            print(f"  Brier Skill:    {bss:+.4f}  (market beats model)")

        # Calibration curve
        curve = cal.compute_calibration_curve(num_bins=5)
        if curve:
            print(f"\n  Calibration curve:")
            for b in curve:
                print(
                    f"    [{b.bin_lower:.1f}-{b.bin_upper:.1f}]  "
                    f"predicted={b.mean_predicted:.3f}  "
                    f"realized={b.mean_realized:.3f}  "
                    f"count={b.count}"
                )

        print(f"{'='*70}")
    else:
        print("\n  (No settled trades for Brier score comparison)")

    # Export
    csv_data = engine.export_trades_csv()
    if engine.trades:
        fname = f"crypto_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = f"/Users/danielelman/Documents/New project/arb_bot/.claude/worktrees/focused-mclean/arb_bot/output/{fname}"
        try:
            with open(path, "w") as f:
                f.write(csv_data)
            print(f"\n  Trades exported to: {path}")
        except Exception as exc:
            print(f"\n  CSV export failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto engine paper test")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"],
                        help="Underlyings to trade (default: BTC ETH SOL)")
    parser.add_argument("--duration-minutes", type=float, default=5,
                        help="How long to run (default: 5)")
    parser.add_argument("--mc-paths", type=int, default=1000,
                        help="Monte Carlo paths (default: 1000)")
    parser.add_argument("--min-edge", type=float, default=0.04,
                        help="Min edge fraction (default: 0.04 = 4%%)")
    parser.add_argument("--scan-interval", type=float, default=15.0,
                        help="Seconds between scans (default: 15)")
    parser.add_argument("--max-tte", type=int, default=600,
                        help="Max time-to-expiry in minutes (default: 600)")
    parser.add_argument("--model", choices=["mc_gbm", "student_t", "ab_test", "empirical", "ab_empirical"],
                        default="empirical",
                        help="Probability model (default: empirical)")
    parser.add_argument("--daily-model", action="store_true",
                        help="Enable v43 daily pricing model (Merton/OU/GBM hybrid)")
    parser.add_argument("--garch", action="store_true",
                        help="Enable v45 GARCH vol-spread model (replaces bootstrap/MC)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    asyncio.run(run_paper_test(
        underlyings=[u.upper() for u in args.symbols],
        duration_minutes=args.duration_minutes,
        mc_paths=args.mc_paths,
        min_edge=args.min_edge,
        scan_interval=args.scan_interval,
        max_tte=args.max_tte,
        probability_model=args.model,
        daily_model=args.daily_model,
        garch_enabled=args.garch,
    ))


if __name__ == "__main__":
    main()
