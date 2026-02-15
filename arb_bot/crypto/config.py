"""Configuration for the crypto prediction trading module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _as_csv(value: str | None) -> List[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class CryptoSettings:
    """Settings for the crypto prediction trading module.

    All env vars are prefixed with ``ARB_CRYPTO_``.
    """

    enabled: bool = False

    # ── Kalshi crypto series to trade ──────────────────────────────
    symbols: List[str] = field(default_factory=lambda: ["KXBTC", "KXETH"])

    # ── Price feed (Binance) ───────────────────────────────────────
    price_feed_url: str = "wss://stream.binance.com:9443/ws"
    price_feed_symbols: List[str] = field(
        default_factory=lambda: ["btcusdt", "ethusdt"],
    )
    price_feed_snapshot_url: str = "https://api.binance.com/api/v3/klines"
    price_history_minutes: int = 60

    # ── Monte Carlo model ──────────────────────────────────────────
    mc_num_paths: int = 1000
    mc_vol_window_minutes: int = 30
    mc_vol_method: str = "realized"  # "realized" | "ewma" | "har"
    mc_drift_mode: str = "zero"  # "zero" | "recent_trend"

    # ── Jump diffusion ──────────────────────────────────────────────
    use_jump_diffusion: bool = True
    mc_jump_intensity: float = 3.0   # Expected jumps per day
    mc_jump_mean: float = 0.0        # Mean log jump size
    mc_jump_vol: float = 0.02        # Stdev of log jump size

    # ── Edge detection ─────────────────────────────────────────────
    min_edge_pct: float = 0.05
    min_edge_pct_daily: float = 0.06
    min_edge_cents: float = 0.02
    max_model_uncertainty: float = 0.15
    model_uncertainty_multiplier: float = 3.0
    confidence_level: float = 0.95

    # ── Market filtering ───────────────────────────────────────────
    allowed_directions: List[str] = field(
        default_factory=lambda: ["above", "below"],
    )
    min_minutes_to_expiry: int = 2
    max_minutes_to_expiry: int = 14
    min_book_depth_contracts: int = 5

    # ── Sizing ─────────────────────────────────────────────────────
    bankroll: float = 500.0
    max_position_per_market: float = 50.0
    max_concurrent_positions: int = 10
    max_positions_per_underlying: int = 3
    kelly_fraction_cap: float = 0.10

    # ── Execution ──────────────────────────────────────────────────
    use_maker: bool = True
    maker_timeout_seconds: float = 5.0
    taker_fallback: bool = True

    # ── Cycle timing ───────────────────────────────────────────────
    scan_interval_seconds: float = 5.0

    # ── Paper mode ─────────────────────────────────────────────────
    paper_mode: bool = True
    paper_slippage_cents: float = 0.5


def load_crypto_settings() -> CryptoSettings:
    """Build ``CryptoSettings`` from ``ARB_CRYPTO_*`` environment variables."""
    return CryptoSettings(
        enabled=_as_bool(os.getenv("ARB_CRYPTO_ENABLED"), False),
        symbols=_as_csv(os.getenv("ARB_CRYPTO_SYMBOLS")) or ["KXBTC", "KXETH"],
        price_feed_url=os.getenv(
            "ARB_CRYPTO_PRICE_FEED_URL",
            "wss://stream.binance.com:9443/ws",
        ),
        price_feed_symbols=_as_csv(os.getenv("ARB_CRYPTO_PRICE_FEED_SYMBOLS"))
        or ["btcusdt", "ethusdt"],
        price_feed_snapshot_url=os.getenv(
            "ARB_CRYPTO_PRICE_FEED_SNAPSHOT_URL",
            "https://api.binance.com/api/v3/klines",
        ),
        price_history_minutes=_as_int(os.getenv("ARB_CRYPTO_PRICE_HISTORY_MINUTES"), 60),
        mc_num_paths=_as_int(os.getenv("ARB_CRYPTO_MC_NUM_PATHS"), 1000),
        mc_vol_window_minutes=_as_int(os.getenv("ARB_CRYPTO_MC_VOL_WINDOW_MINUTES"), 30),
        mc_vol_method=os.getenv("ARB_CRYPTO_MC_VOL_METHOD", "realized"),
        mc_drift_mode=os.getenv("ARB_CRYPTO_MC_DRIFT_MODE", "zero"),
        use_jump_diffusion=_as_bool(os.getenv("ARB_CRYPTO_USE_JUMP_DIFFUSION"), True),
        mc_jump_intensity=_as_float(os.getenv("ARB_CRYPTO_MC_JUMP_INTENSITY"), 3.0),
        mc_jump_mean=_as_float(os.getenv("ARB_CRYPTO_MC_JUMP_MEAN"), 0.0),
        mc_jump_vol=_as_float(os.getenv("ARB_CRYPTO_MC_JUMP_VOL"), 0.02),
        min_edge_pct=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_PCT"), 0.05),
        min_edge_pct_daily=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_PCT_DAILY"), 0.06),
        min_edge_cents=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_CENTS"), 0.02),
        max_model_uncertainty=_as_float(os.getenv("ARB_CRYPTO_MAX_MODEL_UNCERTAINTY"), 0.15),
        model_uncertainty_multiplier=_as_float(os.getenv("ARB_CRYPTO_MODEL_UNCERTAINTY_MULTIPLIER"), 3.0),
        confidence_level=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_LEVEL"), 0.95),
        allowed_directions=_as_csv(os.getenv("ARB_CRYPTO_ALLOWED_DIRECTIONS")) or ["above", "below"],
        min_minutes_to_expiry=_as_int(os.getenv("ARB_CRYPTO_MIN_MINUTES_TO_EXPIRY"), 2),
        max_minutes_to_expiry=_as_int(os.getenv("ARB_CRYPTO_MAX_MINUTES_TO_EXPIRY"), 14),
        min_book_depth_contracts=_as_int(os.getenv("ARB_CRYPTO_MIN_BOOK_DEPTH_CONTRACTS"), 5),
        bankroll=_as_float(os.getenv("ARB_CRYPTO_BANKROLL"), 500.0),
        max_position_per_market=_as_float(os.getenv("ARB_CRYPTO_MAX_POSITION_PER_MARKET"), 50.0),
        max_concurrent_positions=_as_int(os.getenv("ARB_CRYPTO_MAX_CONCURRENT_POSITIONS"), 10),
        max_positions_per_underlying=_as_int(os.getenv("ARB_CRYPTO_MAX_POSITIONS_PER_UNDERLYING"), 3),
        kelly_fraction_cap=_as_float(os.getenv("ARB_CRYPTO_KELLY_FRACTION_CAP"), 0.10),
        use_maker=_as_bool(os.getenv("ARB_CRYPTO_USE_MAKER"), True),
        maker_timeout_seconds=_as_float(os.getenv("ARB_CRYPTO_MAKER_TIMEOUT_SECONDS"), 5.0),
        taker_fallback=_as_bool(os.getenv("ARB_CRYPTO_TAKER_FALLBACK"), True),
        scan_interval_seconds=_as_float(os.getenv("ARB_CRYPTO_SCAN_INTERVAL_SECONDS"), 5.0),
        paper_mode=_as_bool(os.getenv("ARB_CRYPTO_PAPER_MODE"), True),
        paper_slippage_cents=_as_float(os.getenv("ARB_CRYPTO_PAPER_SLIPPAGE_CENTS"), 0.5),
    )
