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

    # ── Price feed (OKX) ────────────────────────────────────────────
    price_feed_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    price_feed_symbols: List[str] = field(
        default_factory=lambda: ["btcusdt", "ethusdt"],
    )
    price_feed_snapshot_url: str = "https://www.okx.com/api/v5/market/candles"
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

    # ── Hawkes self-exciting jumps ────────────────────────────────
    hawkes_enabled: bool = True
    hawkes_alpha: float = 5.0        # Excitation amplitude
    hawkes_beta: float = 0.00115     # Decay rate (ln(2)/600 ~ 10-min half-life)
    hawkes_return_threshold_sigma: float = 3.0  # Sigma threshold to trigger shock
    # Calibrated from 30-day historical data: avg 1-min vol = 0.435% (BTC),
    # 0.645% (SOL).  At 3.0σ the trigger is ~1.3% (BTC) / ~1.9% (SOL),
    # giving ~4-8 shocks/day (fat-tail adjusted).  Previous 4.0σ fired
    # only ~once per 11 days — far too rare to be useful.

    # ── Probability model selection ──────────────────────────────────
    probability_model: str = "mc_gbm"  # "mc_gbm" | "student_t" | "empirical" | "ab_test" | "ab_empirical"
    student_t_nu_default: float = 5.0        # Default degrees of freedom
    student_t_nu_floor: float = 2.5          # Minimum nu (variance must exist)
    student_t_nu_ceiling: float = 30.0       # Maximum nu (converges to normal)
    student_t_fit_window_minutes: int = 120  # Window for fitting nu from returns
    student_t_fit_min_samples: int = 30      # Minimum returns to fit nu
    student_t_refit_every_cycles: int = 20   # Re-fit nu every N scan cycles
    student_t_min_uncertainty: float = 0.02  # Floor on analytical uncertainty
    student_t_uncertainty_multiplier: float = 1.5  # Separate from MC's 3.0

    # ── Empirical CDF model ──────────────────────────────────────────
    empirical_window_minutes: int = 120           # Lookback window for returns
    empirical_min_samples: int = 30               # Minimum returns required
    empirical_bootstrap_paths: int = 2000         # Number of bootstrap resamples
    empirical_min_uncertainty: float = 0.02       # Floor on uncertainty
    empirical_uncertainty_multiplier: float = 1.5 # Separate from MC's 3.0
    empirical_return_interval_seconds: int = 60   # Return bucketing interval

    # ── Edge detection ─────────────────────────────────────────────
    min_edge_pct: float = 0.12
    min_edge_pct_daily: float = 0.12  # Was 0.15; per-cell logic applies stricter thresholds
    min_edge_cents: float = 0.05
    min_edge_pct_no_side: float = 0.12
    dynamic_edge_threshold_enabled: bool = True
    dynamic_edge_uncertainty_multiplier: float = 1.0  # Was 2.0; k=1 already lifts floor by full uncertainty
    # ── Z-score reachability filter ────────────────────────────────
    zscore_filter_enabled: bool = True
    zscore_max: float = 2.0
    zscore_vol_window_minutes: int = 15
    min_model_market_divergence: float = 0.06  # Was 0.12; after blending compresses, 0.12 needs 17%+ true divergence
    max_model_uncertainty: float = 0.25  # Was 0.15; with 1.5x Student-t multiplier, raw 0.10 = 0.15 — too tight
    # Calibrated from paper run v4: trades with <5% edge went 1/6 (17%
    # win rate).  Trades with >6% edge went 4/5 (80%).  Raising the
    # floor from 5% → 6% cuts marginal losers.
    model_uncertainty_multiplier: float = 3.0
    confidence_level: float = 0.95

    # ── Calibration ──────────────────────────────────────────────
    calibration_enabled: bool = True
    calibration_min_samples: int = 30
    calibration_recalibrate_every: int = 20
    calibration_method: str = "isotonic"  # "platt" | "isotonic"
    calibration_isotonic_min_samples: int = 20

    # ── Market filtering ───────────────────────────────────────────
    allowed_directions: List[str] = field(
        default_factory=lambda: ["above", "below"],
    )
    min_minutes_to_expiry: int = 2
    max_minutes_to_expiry: int = 14
    min_book_depth_contracts: int = 5
    min_book_volume: int = 10

    # ── Staleness detection ──────────────────────────────────────
    staleness_enabled: bool = True
    staleness_spot_move_threshold: float = 0.003
    staleness_quote_change_threshold: float = 0.005
    staleness_lookback_seconds: int = 120
    staleness_max_age_seconds: int = 300
    staleness_edge_bonus: float = 0.02

    # ── Sizing ─────────────────────────────────────────────────────
    bankroll: float = 500.0
    max_position_per_market: float = 50.0
    max_concurrent_positions: int = 10
    max_positions_per_underlying: int = 3
    kelly_fraction_cap: float = 0.10
    kelly_edge_cap: float = 0.0  # 0 = disabled; e.g., 0.10 caps edge at 10¢ for sizing

    # ── Time-of-day gate ──────────────────────────────────────────
    quiet_hours_utc: str = ""              # Comma-separated UTC hours, e.g., "0,1,2"
    quiet_hours_min_edge: float = 0.08     # Min edge required during quiet hours

    # ── Online recalibration ──────────────────────────────────────
    recalibration_enabled: bool = False
    recalibration_window: int = 50         # Max trades in recal buffer per cell
    recalibration_refit_interval: int = 10 # Refit isotonic curve every N settlements
    recalibration_min_samples: int = 15    # Min samples before applying recal curve

    # ── Execution ──────────────────────────────────────────────────
    use_maker: bool = True
    maker_timeout_seconds: float = 5.0
    taker_fallback: bool = True

    # ── OFI (Order Flow Imbalance) drift ────────────────────────────
    ofi_enabled: bool = True
    ofi_window_seconds: int = 300
    ofi_alpha: float = 0.0   # Starting alpha (0 = neutral, calibrated at runtime)
    ofi_recalibrate_interval_hours: float = 4.0
    ofi_impact_exponent: float = 0.5  # Power-law exponent (0.5 = square root law)

    # ── Multi-timescale features ─────────────────────────────────────
    multiscale_ofi_enabled: bool = True
    multiscale_ofi_windows: str = "30,60,120,300"  # comma-separated ints
    multiscale_ofi_weights: str = "0.4,0.3,0.2,0.1"  # comma-separated floats

    # ── AggTrades WebSocket ────────────────────────────────────────────
    agg_trades_ws_enabled: bool = True

    # ── Activity scaling ─────────────────────────────────────────────
    activity_scaling_enabled: bool = True
    activity_scaling_short_window_seconds: int = 300    # "current" volume window
    activity_scaling_long_window_seconds: int = 3600    # "average" volume window

    # ── Trend drift ──────────────────────────────────────────────
    trend_drift_enabled: bool = False
    trend_drift_window_minutes: int = 15
    trend_drift_decay: float = 5.0
    trend_drift_max_annualized: float = 5.0

    # ── Drift safety ────────────────────────────────────────────
    max_total_drift: float = 2.0  # Maximum annualized drift magnitude from all sources
    mean_reversion_enabled: bool = True
    mean_reversion_kappa: float = 50.0  # Annualized OU reversion speed
    mean_reversion_lookback_minutes: float = 5.0

    # ── Volume clock ────────────────────────────────────────────
    volume_clock_enabled: bool = False  # Default off for backward compat
    volume_clock_short_window_seconds: int = 300    # Current volume rate window
    volume_clock_baseline_window_seconds: int = 14400  # Baseline ("normal") volume window
    # Calibrated from 30-day historical data: BTC has 11× diurnal volume
    # range (peak $4,697/min at 14 UTC vs trough $426/min at 10 UTC).
    # A 1-hour baseline runs out of data in thin periods.  4 hours
    # smooths the ratio to [0.40, 1.55], preventing extreme scaling.
    volume_clock_ratio_floor: float = 0.25    # Minimum activity ratio
    volume_clock_ratio_ceiling: float = 2.5   # Maximum activity ratio
    # With 4-hr baseline, the smoothed ratio ranges ~0.40-1.55 under
    # normal conditions.  Ceiling of 2.5 gives headroom for genuine
    # volume spikes (news events, crashes) while blocking noise.

    # ── Cross-asset features ──────────────────────────────────────
    cross_asset_enabled: bool = True
    cross_asset_leader: str = "btcusdt"
    cross_asset_ofi_weight: float = 0.3
    cross_asset_vol_weight: float = 0.2
    cross_asset_return_weight: float = 0.2
    cross_asset_return_scale: float = 20.0
    cross_asset_max_drift: float = 2.0

    # ── Funding rate ────────────────────────────────────────────────
    funding_rate_enabled: bool = True
    funding_rate_poll_interval_seconds: int = 300
    funding_rate_extreme_threshold: float = 0.0005
    funding_rate_drift_weight: float = 0.5
    funding_rate_api_url: str = "https://www.okx.com/api/v5/public/funding-rate"
    funding_rate_history_url: str = "https://www.okx.com/api/v5/public/funding-rate-history"

    # ── VPIN (Volume-Synchronized Probability of Informed Trading) ──
    vpin_enabled: bool = True
    vpin_bucket_volume: float = 0.0  # 0 = auto-calibrate
    vpin_num_buckets: int = 50
    vpin_auto_calibrate_window_minutes: int = 60
    vpin_extreme_threshold: float = 0.7
    vpin_drift_weight: float = 0.4
    vpin_vol_boost_factor: float = 1.5

    # ── Adaptive VPIN thresholds ───────────────────────────────────
    # When enabled, the halt/momentum thresholds are derived from a rolling
    # percentile of recent VPIN readings instead of the hardcoded values.
    # Falls back to static thresholds until enough VPIN history accumulates.
    vpin_adaptive_enabled: bool = True
    vpin_adaptive_history_size: int = 500        # Rolling window of VPIN snapshots
    vpin_adaptive_min_history: int = 30           # Min readings before adaptive kicks in
    vpin_adaptive_halt_percentile: float = 90.0   # Percentile for halt threshold
    vpin_adaptive_momentum_percentile: float = 75.0  # Percentile for momentum floor
    vpin_adaptive_halt_floor: float = 0.70        # Adaptive halt never below this
    vpin_adaptive_halt_ceiling: float = 0.98      # Adaptive halt never above this
    vpin_adaptive_momentum_floor: float = 0.50    # Adaptive momentum floor minimum
    vpin_adaptive_momentum_ceiling: float = 0.95  # Adaptive momentum floor maximum

    # ── Confidence scoring ──────────────────────────────────────────
    confidence_scoring_enabled: bool = False  # Start disabled, enable after testing
    confidence_min_score: float = 0.65
    confidence_min_agreement: int = 3
    confidence_staleness_weight: float = 0.25
    confidence_vpin_weight: float = 0.20
    confidence_ofi_weight: float = 0.15
    confidence_funding_weight: float = 0.10
    confidence_vol_regime_weight: float = 0.10
    confidence_cross_asset_weight: float = 0.10
    confidence_model_edge_weight: float = 0.10

    # ── Regime detection ──────────────────────────────────────────────
    regime_detection_enabled: bool = True
    regime_ofi_trend_threshold: float = 0.3
    regime_vol_expansion_threshold: float = 2.0
    regime_autocorr_window: int = 15
    regime_min_returns: int = 10
    regime_vpin_spike_threshold: float = 0.85

    # ── Regime-conditional improvements ────────────────────────────────

    # Tier 1: Regime Kelly multiplier
    regime_sizing_enabled: bool = False
    regime_kelly_mean_reverting: float = 1.5
    regime_kelly_trending_up: float = 0.4
    regime_kelly_trending_down: float = 0.5
    regime_kelly_high_vol: float = 0.0

    # Tier 1: Regime min edge threshold
    regime_min_edge_enabled: bool = False
    regime_min_edge_mean_reverting: float = 0.10
    regime_min_edge_trending: float = 0.20
    regime_min_edge_high_vol: float = 0.30

    # Tier 1: VPIN halt gate
    vpin_halt_enabled: bool = True
    vpin_halt_threshold: float = 0.85

    # Tier 2: Counter-trend filter
    regime_skip_counter_trend: bool = True
    regime_skip_counter_trend_min_conf: float = 0.6

    # Tier 2: Vol regime adjustment
    regime_vol_boost_high_vol: float = 1.5
    regime_empirical_window_high_vol: int = 30
    regime_empirical_window_trending: int = 60

    # Tier 2: Mean-reverting size boost
    regime_kelly_cap_boost_mean_reverting: float = 1.5

    # Tier 3: Conditional trend drift (only in trending regimes)
    regime_conditional_drift: bool = True

    # Tier 3: Transition caution zone
    regime_transition_sizing_multiplier: float = 0.3

    # ── Momentum strategy (v18) ──────────────────────────────────────
    momentum_enabled: bool = False
    momentum_vpin_floor: float = 0.85       # Lower bound of momentum zone
    momentum_vpin_ceiling: float = 0.95     # Full halt above this
    momentum_ofi_alignment_min: float = 0.6 # Min cross-timescale OFI agreement
    momentum_ofi_magnitude_min: float = 200.0  # Min |weighted OFI|
    momentum_max_tte_minutes: float = 15.0  # Only short-dated contracts
    momentum_price_floor: float = 0.15      # Min buy price (avoid dead money)
    momentum_price_ceiling: float = 0.40    # Max buy price (avoid priced-in)
    momentum_kelly_fraction: float = 0.03   # Fixed fraction of bankroll
    momentum_max_position: float = 25.0     # Max dollar per momentum trade
    momentum_max_concurrent: int = 2        # Max open momentum positions
    momentum_cooldown_seconds: float = 120.0  # Per-symbol cooldown
    momentum_min_ofi_streak: int = 3           # Consecutive aligned OFI cycles before entry
    momentum_require_ofi_acceleration: bool = True  # Skip if |OFI_30s| <= |OFI_120s|
    momentum_max_contracts: int = 100          # Hard cap on contracts per momentum trade

    # ── Strategy cell parameters (model path) ──────────────────────
    #
    # Per-cell vol dampening addresses the ROOT CAUSE of YES overconfidence:
    # the IID bootstrap ignores mean-reversion in 1-minute returns, inflating
    # tail probabilities.  By shrinking each resampled return toward zero,
    # we model the autocorrelation the bootstrap misses.
    #
    # With vol dampening fixing the model, we can relax the downstream
    # corrections (model_weight, haircut, min_edge) to achievable levels.

    # YES/15min — microstructure-confirmed
    cell_yes_15min_model_weight: float = 0.60        # Moderate market deference
    cell_yes_15min_prob_haircut: float = 0.95         # Light haircut (vol dampening does heavy lifting)
    cell_yes_15min_vol_dampening: float = 0.85        # MODEL: shrink returns 15% for mean-reversion
    cell_yes_15min_uncertainty_mult: float = 2.0      # Higher uncertainty for YES (model misspec)
    cell_yes_15min_empirical_window: int = 30         # Shorter lookback for short contracts
    cell_yes_15min_min_edge_pct: float = 0.10         # Achievable edge bar
    cell_yes_15min_kelly_multiplier: float = 0.5      # Half-size (less proven)
    cell_yes_15min_max_position: float = 25.0
    cell_yes_15min_require_ofi: bool = True           # Require OFI alignment
    cell_yes_15min_ofi_min: float = 0.3               # OFI magnitude floor
    cell_yes_15min_require_price_past_strike: bool = True  # Price already past strike

    # YES/daily — momentum-confirmed directional
    cell_yes_daily_model_weight: float = 0.55         # Moderate market deference (was 0.4 — too harsh)
    cell_yes_daily_prob_haircut: float = 0.92          # Light haircut (was 0.85 — too harsh)
    cell_yes_daily_vol_dampening: float = 0.75         # MODEL: strongest shrinkage (39pp overconfident)
    cell_yes_daily_uncertainty_mult: float = 2.5       # Highest uncertainty (longest horizon, most error)
    cell_yes_daily_empirical_window: int = 0           # Use global (2hr ok for daily)
    cell_yes_daily_min_edge_pct: float = 0.12          # Achievable (was 0.18 — impossible)
    cell_yes_daily_kelly_multiplier: float = 0.5       # Half-size (still cautious)
    cell_yes_daily_max_position: float = 25.0
    cell_yes_daily_require_trend: bool = True          # Require recent price trend toward strike
    cell_yes_daily_trend_window_minutes: float = 10.0

    # NO/15min — short-term vol fade
    cell_no_15min_model_weight: float = 0.7            # Keep current (no correction needed)
    cell_no_15min_prob_haircut: float = 1.0             # No haircut
    cell_no_15min_vol_dampening: float = 1.0            # MODEL: no dampening (wider tails help NO)
    cell_no_15min_uncertainty_mult: float = 1.5         # Standard uncertainty
    cell_no_15min_empirical_window: int = 30            # Shorter lookback for short contracts
    cell_no_15min_min_edge_pct: float = 0.10            # Lower bar (vol fade = tighter edge ok)
    cell_no_15min_kelly_multiplier: float = 0.7
    cell_no_15min_max_position: float = 30.0

    # NO/daily — big-move fade (proven alpha)
    cell_no_daily_model_weight: float = 0.75            # Slight boost (model underconfident here)
    cell_no_daily_prob_haircut: float = 1.0             # No haircut (model works for NO)
    cell_no_daily_vol_dampening: float = 1.0            # MODEL: no dampening (wider tails help NO)
    cell_no_daily_uncertainty_mult: float = 1.5         # Standard uncertainty
    cell_no_daily_empirical_window: int = 0             # Use global
    cell_no_daily_min_edge_pct: float = 0.12            # Keep current — this cell works
    cell_no_daily_kelly_multiplier: float = 1.0         # Full size — proven alpha
    cell_no_daily_max_position: float = 50.0

    # Per-cell probability model overrides (empty = use global probability_model)
    # YES cells use empirical (captures return distribution for "will price reach X?")
    # NO cells use mc_gbm (Gaussian tails underestimate extremes → edge for "won't reach X")
    cell_yes_15min_probability_model: str = ""
    cell_yes_daily_probability_model: str = ""
    cell_no_15min_probability_model: str = ""
    cell_no_daily_probability_model: str = ""

    # Per-cell classifier model paths (empty = use global classifier_model_path)
    classifier_model_path_yes_15min: str = ""
    classifier_model_path_yes_daily: str = ""
    classifier_model_path_no_15min: str = ""
    classifier_model_path_no_daily: str = ""

    # Per-cell classifier veto thresholds
    classifier_veto_threshold_yes_15min: float = 0.4
    classifier_veto_threshold_yes_daily: float = 0.4
    classifier_veto_threshold_no_15min: float = 0.4
    classifier_veto_threshold_no_daily: float = 0.4

    # ── Cycle recorder ────────────────────────────────────────────────
    cycle_recorder_enabled: bool = False
    cycle_recorder_db_dir: str = "recordings"

    # ── Feature store ──────────────────────────────────────────────────
    feature_store_enabled: bool = False
    feature_store_path: str = "feature_store.csv"
    feature_store_min_samples: int = 200

    # ── Classifier ─────────────────────────────────────────────────────
    classifier_enabled: bool = False
    classifier_model_path: str = ""
    classifier_min_training_samples: int = 200
    classifier_retrain_interval_hours: float = 24.0
    classifier_use_isotonic_calibration: bool = True
    classifier_max_depth: int = 4
    classifier_n_estimators: int = 100
    classifier_learning_rate: float = 0.1
    classifier_min_child_weight: int = 5
    classifier_subsample: float = 0.8
    classifier_veto_threshold: float = 0.4  # Reject trades where P(win) < this

    # ── Cycle timing ───────────────────────────────────────────────
    scan_interval_seconds: float = 5.0

    # ── Paper mode ─────────────────────────────────────────────────
    paper_mode: bool = True
    paper_slippage_cents: float = 0.5

    # ── Settlement ────────────────────────────────────────────────
    settlement_grace_minutes: float = 10.0  # How long to wait for Kalshi settlement data


def load_crypto_settings() -> CryptoSettings:
    """Build ``CryptoSettings`` from ``ARB_CRYPTO_*`` environment variables."""
    return CryptoSettings(
        enabled=_as_bool(os.getenv("ARB_CRYPTO_ENABLED"), False),
        symbols=_as_csv(os.getenv("ARB_CRYPTO_SYMBOLS")) or ["KXBTC", "KXETH"],
        price_feed_url=os.getenv(
            "ARB_CRYPTO_PRICE_FEED_URL",
            "wss://ws.okx.com:8443/ws/v5/public",
        ),
        price_feed_symbols=_as_csv(os.getenv("ARB_CRYPTO_PRICE_FEED_SYMBOLS"))
        or ["btcusdt", "ethusdt"],
        price_feed_snapshot_url=os.getenv(
            "ARB_CRYPTO_PRICE_FEED_SNAPSHOT_URL",
            "https://www.okx.com/api/v5/market/candles",
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
        hawkes_enabled=_as_bool(os.getenv("ARB_CRYPTO_HAWKES_ENABLED"), True),
        hawkes_alpha=_as_float(os.getenv("ARB_CRYPTO_HAWKES_ALPHA"), 5.0),
        hawkes_beta=_as_float(os.getenv("ARB_CRYPTO_HAWKES_BETA"), 0.00115),
        hawkes_return_threshold_sigma=_as_float(os.getenv("ARB_CRYPTO_HAWKES_RETURN_THRESHOLD_SIGMA"), 3.0),
        probability_model=os.getenv("ARB_CRYPTO_PROBABILITY_MODEL", "mc_gbm"),
        student_t_nu_default=_as_float(os.getenv("ARB_CRYPTO_STUDENT_T_NU_DEFAULT"), 5.0),
        student_t_nu_floor=_as_float(os.getenv("ARB_CRYPTO_STUDENT_T_NU_FLOOR"), 2.5),
        student_t_nu_ceiling=_as_float(os.getenv("ARB_CRYPTO_STUDENT_T_NU_CEILING"), 30.0),
        student_t_fit_window_minutes=_as_int(os.getenv("ARB_CRYPTO_STUDENT_T_FIT_WINDOW_MINUTES"), 120),
        student_t_fit_min_samples=_as_int(os.getenv("ARB_CRYPTO_STUDENT_T_FIT_MIN_SAMPLES"), 30),
        student_t_refit_every_cycles=_as_int(os.getenv("ARB_CRYPTO_STUDENT_T_REFIT_EVERY_CYCLES"), 20),
        student_t_min_uncertainty=_as_float(os.getenv("ARB_CRYPTO_STUDENT_T_MIN_UNCERTAINTY"), 0.02),
        student_t_uncertainty_multiplier=_as_float(os.getenv("ARB_CRYPTO_STUDENT_T_UNCERTAINTY_MULTIPLIER"), 1.5),
        empirical_window_minutes=_as_int(os.getenv("ARB_CRYPTO_EMPIRICAL_WINDOW_MINUTES"), 120),
        empirical_min_samples=_as_int(os.getenv("ARB_CRYPTO_EMPIRICAL_MIN_SAMPLES"), 30),
        empirical_bootstrap_paths=_as_int(os.getenv("ARB_CRYPTO_EMPIRICAL_BOOTSTRAP_PATHS"), 2000),
        empirical_min_uncertainty=_as_float(os.getenv("ARB_CRYPTO_EMPIRICAL_MIN_UNCERTAINTY"), 0.02),
        empirical_uncertainty_multiplier=_as_float(os.getenv("ARB_CRYPTO_EMPIRICAL_UNCERTAINTY_MULTIPLIER"), 1.5),
        empirical_return_interval_seconds=_as_int(os.getenv("ARB_CRYPTO_EMPIRICAL_RETURN_INTERVAL_SECONDS"), 60),
        min_edge_pct=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_PCT"), 0.12),
        min_edge_pct_daily=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_PCT_DAILY"), 0.12),
        min_edge_cents=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_CENTS"), 0.05),
        min_edge_pct_no_side=_as_float(os.getenv("ARB_CRYPTO_MIN_EDGE_PCT_NO_SIDE"), 0.12),
        dynamic_edge_threshold_enabled=_as_bool(os.getenv("ARB_CRYPTO_DYNAMIC_EDGE_THRESHOLD_ENABLED"), True),
        dynamic_edge_uncertainty_multiplier=_as_float(os.getenv("ARB_CRYPTO_DYNAMIC_EDGE_UNCERTAINTY_MULTIPLIER"), 1.0),
        zscore_filter_enabled=_as_bool(os.getenv("ARB_CRYPTO_ZSCORE_FILTER_ENABLED"), True),
        zscore_max=_as_float(os.getenv("ARB_CRYPTO_ZSCORE_MAX"), 2.0),
        zscore_vol_window_minutes=_as_int(os.getenv("ARB_CRYPTO_ZSCORE_VOL_WINDOW_MINUTES"), 15),
        min_model_market_divergence=_as_float(os.getenv("ARB_CRYPTO_MIN_MODEL_MARKET_DIVERGENCE"), 0.06),
        max_model_uncertainty=_as_float(os.getenv("ARB_CRYPTO_MAX_MODEL_UNCERTAINTY"), 0.25),
        model_uncertainty_multiplier=_as_float(os.getenv("ARB_CRYPTO_MODEL_UNCERTAINTY_MULTIPLIER"), 3.0),
        confidence_level=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_LEVEL"), 0.95),
        calibration_enabled=_as_bool(os.getenv("ARB_CRYPTO_CALIBRATION_ENABLED"), True),
        calibration_min_samples=_as_int(os.getenv("ARB_CRYPTO_CALIBRATION_MIN_SAMPLES"), 30),
        calibration_recalibrate_every=_as_int(os.getenv("ARB_CRYPTO_CALIBRATION_RECALIBRATE_EVERY"), 20),
        calibration_method=os.environ.get("ARB_CRYPTO_CALIBRATION_METHOD", "isotonic"),
        calibration_isotonic_min_samples=int(os.environ.get("ARB_CRYPTO_CALIBRATION_ISOTONIC_MIN_SAMPLES", "20")),
        allowed_directions=_as_csv(os.getenv("ARB_CRYPTO_ALLOWED_DIRECTIONS")) or ["above", "below"],
        min_minutes_to_expiry=_as_int(os.getenv("ARB_CRYPTO_MIN_MINUTES_TO_EXPIRY"), 2),
        max_minutes_to_expiry=_as_int(os.getenv("ARB_CRYPTO_MAX_MINUTES_TO_EXPIRY"), 14),
        min_book_depth_contracts=_as_int(os.getenv("ARB_CRYPTO_MIN_BOOK_DEPTH_CONTRACTS"), 5),
        min_book_volume=_as_int(os.getenv("ARB_CRYPTO_MIN_BOOK_VOLUME"), 10),
        staleness_enabled=_as_bool(os.getenv("ARB_CRYPTO_STALENESS_ENABLED"), True),
        staleness_spot_move_threshold=_as_float(os.getenv("ARB_CRYPTO_STALENESS_SPOT_MOVE_THRESHOLD"), 0.003),
        staleness_quote_change_threshold=_as_float(os.getenv("ARB_CRYPTO_STALENESS_QUOTE_CHANGE_THRESHOLD"), 0.005),
        staleness_lookback_seconds=_as_int(os.getenv("ARB_CRYPTO_STALENESS_LOOKBACK_SECONDS"), 120),
        staleness_max_age_seconds=_as_int(os.getenv("ARB_CRYPTO_STALENESS_MAX_AGE_SECONDS"), 300),
        staleness_edge_bonus=_as_float(os.getenv("ARB_CRYPTO_STALENESS_EDGE_BONUS"), 0.02),
        bankroll=_as_float(os.getenv("ARB_CRYPTO_BANKROLL"), 500.0),
        max_position_per_market=_as_float(os.getenv("ARB_CRYPTO_MAX_POSITION_PER_MARKET"), 50.0),
        max_concurrent_positions=_as_int(os.getenv("ARB_CRYPTO_MAX_CONCURRENT_POSITIONS"), 10),
        max_positions_per_underlying=_as_int(os.getenv("ARB_CRYPTO_MAX_POSITIONS_PER_UNDERLYING"), 3),
        kelly_fraction_cap=_as_float(os.getenv("ARB_CRYPTO_KELLY_FRACTION_CAP"), 0.10),
        kelly_edge_cap=_as_float(os.getenv("ARB_CRYPTO_KELLY_EDGE_CAP"), 0.0),
        quiet_hours_utc=os.getenv("ARB_CRYPTO_QUIET_HOURS_UTC", ""),
        quiet_hours_min_edge=_as_float(os.getenv("ARB_CRYPTO_QUIET_HOURS_MIN_EDGE"), 0.08),
        recalibration_enabled=_as_bool(os.getenv("ARB_CRYPTO_RECALIBRATION_ENABLED"), False),
        recalibration_window=_as_int(os.getenv("ARB_CRYPTO_RECALIBRATION_WINDOW"), 50),
        recalibration_refit_interval=_as_int(os.getenv("ARB_CRYPTO_RECALIBRATION_REFIT_INTERVAL"), 10),
        recalibration_min_samples=_as_int(os.getenv("ARB_CRYPTO_RECALIBRATION_MIN_SAMPLES"), 15),
        use_maker=_as_bool(os.getenv("ARB_CRYPTO_USE_MAKER"), True),
        maker_timeout_seconds=_as_float(os.getenv("ARB_CRYPTO_MAKER_TIMEOUT_SECONDS"), 5.0),
        taker_fallback=_as_bool(os.getenv("ARB_CRYPTO_TAKER_FALLBACK"), True),
        activity_scaling_enabled=_as_bool(os.getenv("ARB_CRYPTO_ACTIVITY_SCALING_ENABLED"), True),
        activity_scaling_short_window_seconds=_as_int(os.getenv("ARB_CRYPTO_ACTIVITY_SCALING_SHORT_WINDOW"), 300),
        activity_scaling_long_window_seconds=_as_int(os.getenv("ARB_CRYPTO_ACTIVITY_SCALING_LONG_WINDOW"), 3600),
        ofi_enabled=_as_bool(os.getenv("ARB_CRYPTO_OFI_ENABLED"), True),
        ofi_window_seconds=_as_int(os.getenv("ARB_CRYPTO_OFI_WINDOW_SECONDS"), 300),
        ofi_alpha=_as_float(os.getenv("ARB_CRYPTO_OFI_ALPHA"), 0.0),
        ofi_recalibrate_interval_hours=_as_float(os.getenv("ARB_CRYPTO_OFI_RECALIBRATE_HOURS"), 4.0),
        ofi_impact_exponent=_as_float(os.getenv("ARB_CRYPTO_OFI_IMPACT_EXPONENT"), 0.5),
        multiscale_ofi_enabled=_as_bool(os.getenv("ARB_CRYPTO_MULTISCALE_OFI_ENABLED"), True),
        multiscale_ofi_windows=os.environ.get("ARB_CRYPTO_MULTISCALE_OFI_WINDOWS", "30,60,120,300"),
        multiscale_ofi_weights=os.environ.get("ARB_CRYPTO_MULTISCALE_OFI_WEIGHTS", "0.4,0.3,0.2,0.1"),
        agg_trades_ws_enabled=_as_bool(os.getenv("ARB_CRYPTO_AGG_TRADES_WS_ENABLED"), True),
        volume_clock_enabled=_as_bool(os.getenv("ARB_CRYPTO_VOLUME_CLOCK_ENABLED"), False),
        volume_clock_short_window_seconds=_as_int(os.getenv("ARB_CRYPTO_VOLUME_CLOCK_SHORT_WINDOW"), 300),
        volume_clock_baseline_window_seconds=_as_int(os.getenv("ARB_CRYPTO_VOLUME_CLOCK_BASELINE_WINDOW"), 14400),
        volume_clock_ratio_floor=_as_float(os.getenv("ARB_CRYPTO_VOLUME_CLOCK_RATIO_FLOOR"), 0.25),
        volume_clock_ratio_ceiling=_as_float(os.getenv("ARB_CRYPTO_VOLUME_CLOCK_RATIO_CEILING"), 2.5),
        trend_drift_enabled=_as_bool(os.getenv("ARB_CRYPTO_TREND_DRIFT_ENABLED"), False),
        trend_drift_window_minutes=_as_int(os.getenv("ARB_CRYPTO_TREND_DRIFT_WINDOW_MINUTES"), 15),
        trend_drift_decay=_as_float(os.getenv("ARB_CRYPTO_TREND_DRIFT_DECAY"), 5.0),
        trend_drift_max_annualized=_as_float(os.getenv("ARB_CRYPTO_TREND_DRIFT_MAX_ANNUALIZED"), 5.0),
        max_total_drift=_as_float(os.getenv("ARB_CRYPTO_MAX_TOTAL_DRIFT"), 2.0),
        mean_reversion_enabled=_as_bool(os.getenv("ARB_CRYPTO_MEAN_REVERSION_ENABLED"), True),
        mean_reversion_kappa=_as_float(os.getenv("ARB_CRYPTO_MEAN_REVERSION_KAPPA"), 50.0),
        mean_reversion_lookback_minutes=_as_float(os.getenv("ARB_CRYPTO_MEAN_REVERSION_LOOKBACK_MINUTES"), 5.0),
        cross_asset_enabled=_as_bool(os.getenv("ARB_CRYPTO_CROSS_ASSET_ENABLED"), True),
        cross_asset_leader=os.getenv("ARB_CRYPTO_CROSS_ASSET_LEADER", "btcusdt"),
        cross_asset_ofi_weight=_as_float(os.getenv("ARB_CRYPTO_CROSS_ASSET_OFI_WEIGHT"), 0.3),
        cross_asset_vol_weight=_as_float(os.getenv("ARB_CRYPTO_CROSS_ASSET_VOL_WEIGHT"), 0.2),
        cross_asset_return_weight=_as_float(os.getenv("ARB_CRYPTO_CROSS_ASSET_RETURN_WEIGHT"), 0.2),
        cross_asset_return_scale=_as_float(os.getenv("ARB_CRYPTO_CROSS_ASSET_RETURN_SCALE"), 20.0),
        cross_asset_max_drift=_as_float(os.getenv("ARB_CRYPTO_CROSS_ASSET_MAX_DRIFT"), 2.0),
        funding_rate_enabled=_as_bool(os.getenv("ARB_CRYPTO_FUNDING_RATE_ENABLED"), True),
        funding_rate_poll_interval_seconds=_as_int(os.getenv("ARB_CRYPTO_FUNDING_RATE_POLL_INTERVAL"), 300),
        funding_rate_extreme_threshold=_as_float(os.getenv("ARB_CRYPTO_FUNDING_RATE_EXTREME_THRESHOLD"), 0.0005),
        funding_rate_drift_weight=_as_float(os.getenv("ARB_CRYPTO_FUNDING_RATE_DRIFT_WEIGHT"), 0.5),
        funding_rate_api_url=os.getenv("ARB_CRYPTO_FUNDING_RATE_API_URL", "https://www.okx.com/api/v5/public/funding-rate"),
        funding_rate_history_url=os.getenv("ARB_CRYPTO_FUNDING_RATE_HISTORY_URL", "https://www.okx.com/api/v5/public/funding-rate-history"),
        vpin_enabled=_as_bool(os.getenv("ARB_CRYPTO_VPIN_ENABLED"), True),
        vpin_bucket_volume=_as_float(os.getenv("ARB_CRYPTO_VPIN_BUCKET_VOLUME"), 0.0),
        vpin_num_buckets=_as_int(os.getenv("ARB_CRYPTO_VPIN_NUM_BUCKETS"), 50),
        vpin_auto_calibrate_window_minutes=_as_int(os.getenv("ARB_CRYPTO_VPIN_AUTO_CALIBRATE_WINDOW"), 60),
        vpin_extreme_threshold=_as_float(os.getenv("ARB_CRYPTO_VPIN_EXTREME_THRESHOLD"), 0.7),
        vpin_drift_weight=_as_float(os.getenv("ARB_CRYPTO_VPIN_DRIFT_WEIGHT"), 0.4),
        vpin_vol_boost_factor=_as_float(os.getenv("ARB_CRYPTO_VPIN_VOL_BOOST_FACTOR"), 1.5),
        vpin_adaptive_enabled=_as_bool(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_ENABLED"), True),
        vpin_adaptive_history_size=_as_int(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_HISTORY_SIZE"), 500),
        vpin_adaptive_min_history=_as_int(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_MIN_HISTORY"), 30),
        vpin_adaptive_halt_percentile=_as_float(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_HALT_PERCENTILE"), 90.0),
        vpin_adaptive_momentum_percentile=_as_float(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_MOMENTUM_PERCENTILE"), 75.0),
        vpin_adaptive_halt_floor=_as_float(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_HALT_FLOOR"), 0.70),
        vpin_adaptive_halt_ceiling=_as_float(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_HALT_CEILING"), 0.98),
        vpin_adaptive_momentum_floor=_as_float(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_MOMENTUM_FLOOR"), 0.50),
        vpin_adaptive_momentum_ceiling=_as_float(os.getenv("ARB_CRYPTO_VPIN_ADAPTIVE_MOMENTUM_CEILING"), 0.95),
        confidence_scoring_enabled=_as_bool(os.getenv("ARB_CRYPTO_CONFIDENCE_SCORING_ENABLED"), False),
        confidence_min_score=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_MIN_SCORE"), 0.65),
        confidence_min_agreement=_as_int(os.getenv("ARB_CRYPTO_CONFIDENCE_MIN_AGREEMENT"), 3),
        confidence_staleness_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_STALENESS_WEIGHT"), 0.25),
        confidence_vpin_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_VPIN_WEIGHT"), 0.20),
        confidence_ofi_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_OFI_WEIGHT"), 0.15),
        confidence_funding_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_FUNDING_WEIGHT"), 0.10),
        confidence_vol_regime_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_VOL_REGIME_WEIGHT"), 0.10),
        confidence_cross_asset_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_CROSS_ASSET_WEIGHT"), 0.10),
        confidence_model_edge_weight=_as_float(os.getenv("ARB_CRYPTO_CONFIDENCE_MODEL_EDGE_WEIGHT"), 0.10),
        regime_detection_enabled=_as_bool(os.getenv("ARB_CRYPTO_REGIME_DETECTION_ENABLED"), True),
        regime_ofi_trend_threshold=_as_float(os.getenv("ARB_CRYPTO_REGIME_OFI_TREND_THRESHOLD"), 0.3),
        regime_vol_expansion_threshold=_as_float(os.getenv("ARB_CRYPTO_REGIME_VOL_EXPANSION_THRESHOLD"), 2.0),
        regime_autocorr_window=_as_int(os.getenv("ARB_CRYPTO_REGIME_AUTOCORR_WINDOW"), 15),
        regime_min_returns=_as_int(os.getenv("ARB_CRYPTO_REGIME_MIN_RETURNS"), 10),
        regime_vpin_spike_threshold=_as_float(os.getenv("ARB_CRYPTO_REGIME_VPIN_SPIKE_THRESHOLD"), 0.85),
        # Regime-conditional improvements
        regime_sizing_enabled=_as_bool(os.getenv("ARB_CRYPTO_REGIME_SIZING_ENABLED"), False),
        regime_kelly_mean_reverting=_as_float(os.getenv("ARB_CRYPTO_REGIME_KELLY_MEAN_REVERTING"), 1.0),
        regime_kelly_trending_up=_as_float(os.getenv("ARB_CRYPTO_REGIME_KELLY_TRENDING_UP"), 0.4),
        regime_kelly_trending_down=_as_float(os.getenv("ARB_CRYPTO_REGIME_KELLY_TRENDING_DOWN"), 0.5),
        regime_kelly_high_vol=_as_float(os.getenv("ARB_CRYPTO_REGIME_KELLY_HIGH_VOL"), 0.0),
        regime_min_edge_enabled=_as_bool(os.getenv("ARB_CRYPTO_REGIME_MIN_EDGE_ENABLED"), False),
        regime_min_edge_mean_reverting=_as_float(os.getenv("ARB_CRYPTO_REGIME_MIN_EDGE_MEAN_REVERTING"), 0.10),
        regime_min_edge_trending=_as_float(os.getenv("ARB_CRYPTO_REGIME_MIN_EDGE_TRENDING"), 0.20),
        regime_min_edge_high_vol=_as_float(os.getenv("ARB_CRYPTO_REGIME_MIN_EDGE_HIGH_VOL"), 0.30),
        vpin_halt_enabled=_as_bool(os.getenv("ARB_CRYPTO_VPIN_HALT_ENABLED"), True),
        vpin_halt_threshold=_as_float(os.getenv("ARB_CRYPTO_VPIN_HALT_THRESHOLD"), 0.85),
        regime_skip_counter_trend=_as_bool(os.getenv("ARB_CRYPTO_REGIME_SKIP_COUNTER_TREND"), True),
        regime_skip_counter_trend_min_conf=_as_float(os.getenv("ARB_CRYPTO_REGIME_SKIP_COUNTER_TREND_MIN_CONF"), 0.6),
        regime_vol_boost_high_vol=_as_float(os.getenv("ARB_CRYPTO_REGIME_VOL_BOOST_HIGH_VOL"), 1.5),
        regime_empirical_window_high_vol=_as_int(os.getenv("ARB_CRYPTO_REGIME_EMPIRICAL_WINDOW_HIGH_VOL"), 30),
        regime_empirical_window_trending=_as_int(os.getenv("ARB_CRYPTO_REGIME_EMPIRICAL_WINDOW_TRENDING"), 60),
        regime_kelly_cap_boost_mean_reverting=_as_float(os.getenv("ARB_CRYPTO_REGIME_KELLY_CAP_BOOST_MEAN_REVERTING"), 1.25),
        regime_conditional_drift=_as_bool(os.getenv("ARB_CRYPTO_REGIME_CONDITIONAL_DRIFT"), True),
        regime_transition_sizing_multiplier=_as_float(os.getenv("ARB_CRYPTO_REGIME_TRANSITION_SIZING_MULTIPLIER"), 0.3),
        momentum_enabled=_as_bool(os.getenv("ARB_CRYPTO_MOMENTUM_ENABLED"), False),
        momentum_vpin_floor=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_VPIN_FLOOR"), 0.85),
        momentum_vpin_ceiling=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_VPIN_CEILING"), 0.95),
        momentum_ofi_alignment_min=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_OFI_ALIGNMENT_MIN"), 0.6),
        momentum_ofi_magnitude_min=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_OFI_MAGNITUDE_MIN"), 200.0),
        momentum_max_tte_minutes=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_TTE_MINUTES"), 15.0),
        momentum_price_floor=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_PRICE_FLOOR"), 0.15),
        momentum_price_ceiling=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_PRICE_CEILING"), 0.40),
        momentum_kelly_fraction=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_KELLY_FRACTION"), 0.03),
        momentum_max_position=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_POSITION"), 25.0),
        momentum_max_concurrent=_as_int(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_CONCURRENT"), 2),
        momentum_cooldown_seconds=_as_float(os.getenv("ARB_CRYPTO_MOMENTUM_COOLDOWN_SECONDS"), 120.0),
        momentum_min_ofi_streak=_as_int(os.getenv("ARB_CRYPTO_MOMENTUM_MIN_OFI_STREAK"), 3),
        momentum_require_ofi_acceleration=_as_bool(os.getenv("ARB_CRYPTO_MOMENTUM_REQUIRE_OFI_ACCELERATION"), True),
        momentum_max_contracts=_as_int(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_CONTRACTS"), 100),
        # Strategy cell parameters
        cell_yes_15min_model_weight=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_MODEL_WEIGHT"), 0.60),
        cell_yes_15min_prob_haircut=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_PROB_HAIRCUT"), 0.95),
        cell_yes_15min_vol_dampening=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_VOL_DAMPENING"), 0.85),
        cell_yes_15min_uncertainty_mult=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_UNCERTAINTY_MULT"), 2.0),
        cell_yes_15min_empirical_window=_as_int(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_EMPIRICAL_WINDOW"), 30),
        cell_yes_15min_min_edge_pct=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_MIN_EDGE_PCT"), 0.10),
        cell_yes_15min_kelly_multiplier=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_KELLY_MULTIPLIER"), 0.5),
        cell_yes_15min_max_position=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_MAX_POSITION"), 25.0),
        cell_yes_15min_require_ofi=_as_bool(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_REQUIRE_OFI"), True),
        cell_yes_15min_ofi_min=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_OFI_MIN"), 0.3),
        cell_yes_15min_require_price_past_strike=_as_bool(os.getenv("ARB_CRYPTO_CELL_YES_15MIN_REQUIRE_PRICE_PAST_STRIKE"), True),
        cell_yes_daily_model_weight=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_MODEL_WEIGHT"), 0.55),
        cell_yes_daily_prob_haircut=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_PROB_HAIRCUT"), 0.92),
        cell_yes_daily_vol_dampening=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_VOL_DAMPENING"), 0.75),
        cell_yes_daily_uncertainty_mult=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_UNCERTAINTY_MULT"), 2.5),
        cell_yes_daily_empirical_window=_as_int(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_EMPIRICAL_WINDOW"), 0),
        cell_yes_daily_min_edge_pct=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_MIN_EDGE_PCT"), 0.12),
        cell_yes_daily_kelly_multiplier=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_KELLY_MULTIPLIER"), 0.5),
        cell_yes_daily_max_position=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_MAX_POSITION"), 25.0),
        cell_yes_daily_require_trend=_as_bool(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_REQUIRE_TREND"), True),
        cell_yes_daily_trend_window_minutes=_as_float(os.getenv("ARB_CRYPTO_CELL_YES_DAILY_TREND_WINDOW_MINUTES"), 10.0),
        cell_no_15min_model_weight=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_MODEL_WEIGHT"), 0.7),
        cell_no_15min_prob_haircut=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_PROB_HAIRCUT"), 1.0),
        cell_no_15min_vol_dampening=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_VOL_DAMPENING"), 1.0),
        cell_no_15min_uncertainty_mult=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_UNCERTAINTY_MULT"), 1.5),
        cell_no_15min_empirical_window=_as_int(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_EMPIRICAL_WINDOW"), 30),
        cell_no_15min_min_edge_pct=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_MIN_EDGE_PCT"), 0.10),
        cell_no_15min_kelly_multiplier=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_KELLY_MULTIPLIER"), 0.7),
        cell_no_15min_max_position=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_15MIN_MAX_POSITION"), 30.0),
        cell_no_daily_model_weight=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_MODEL_WEIGHT"), 0.75),
        cell_no_daily_prob_haircut=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_PROB_HAIRCUT"), 1.0),
        cell_no_daily_vol_dampening=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_VOL_DAMPENING"), 1.0),
        cell_no_daily_uncertainty_mult=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_UNCERTAINTY_MULT"), 1.5),
        cell_no_daily_empirical_window=_as_int(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_EMPIRICAL_WINDOW"), 0),
        cell_no_daily_min_edge_pct=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_MIN_EDGE_PCT"), 0.12),
        cell_no_daily_kelly_multiplier=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_KELLY_MULTIPLIER"), 1.0),
        cell_no_daily_max_position=_as_float(os.getenv("ARB_CRYPTO_CELL_NO_DAILY_MAX_POSITION"), 50.0),
        # Per-cell probability model overrides
        cell_yes_15min_probability_model=os.getenv("ARB_CRYPTO_CELL_YES_15MIN_PROBABILITY_MODEL", ""),
        cell_yes_daily_probability_model=os.getenv("ARB_CRYPTO_CELL_YES_DAILY_PROBABILITY_MODEL", ""),
        cell_no_15min_probability_model=os.getenv("ARB_CRYPTO_CELL_NO_15MIN_PROBABILITY_MODEL", ""),
        cell_no_daily_probability_model=os.getenv("ARB_CRYPTO_CELL_NO_DAILY_PROBABILITY_MODEL", ""),
        # Per-cell classifier model paths
        classifier_model_path_yes_15min=os.getenv("ARB_CRYPTO_CLASSIFIER_MODEL_PATH_YES_15MIN", ""),
        classifier_model_path_yes_daily=os.getenv("ARB_CRYPTO_CLASSIFIER_MODEL_PATH_YES_DAILY", ""),
        classifier_model_path_no_15min=os.getenv("ARB_CRYPTO_CLASSIFIER_MODEL_PATH_NO_15MIN", ""),
        classifier_model_path_no_daily=os.getenv("ARB_CRYPTO_CLASSIFIER_MODEL_PATH_NO_DAILY", ""),
        # Per-cell classifier veto thresholds
        classifier_veto_threshold_yes_15min=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_VETO_THRESHOLD_YES_15MIN"), 0.4),
        classifier_veto_threshold_yes_daily=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_VETO_THRESHOLD_YES_DAILY"), 0.4),
        classifier_veto_threshold_no_15min=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_VETO_THRESHOLD_NO_15MIN"), 0.4),
        classifier_veto_threshold_no_daily=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_VETO_THRESHOLD_NO_DAILY"), 0.4),
        cycle_recorder_enabled=_as_bool(os.getenv("ARB_CRYPTO_CYCLE_RECORDER_ENABLED"), False),
        cycle_recorder_db_dir=os.getenv("ARB_CRYPTO_CYCLE_RECORDER_DB_DIR", "recordings"),
        feature_store_enabled=_as_bool(os.getenv("ARB_CRYPTO_FEATURE_STORE_ENABLED"), False),
        feature_store_path=os.getenv("ARB_CRYPTO_FEATURE_STORE_PATH", "feature_store.csv"),
        feature_store_min_samples=_as_int(os.getenv("ARB_CRYPTO_FEATURE_STORE_MIN_SAMPLES"), 200),
        classifier_enabled=_as_bool(os.getenv("ARB_CRYPTO_CLASSIFIER_ENABLED"), False),
        classifier_model_path=os.getenv("ARB_CRYPTO_CLASSIFIER_MODEL_PATH", ""),
        classifier_min_training_samples=_as_int(os.getenv("ARB_CRYPTO_CLASSIFIER_MIN_TRAINING_SAMPLES"), 200),
        classifier_retrain_interval_hours=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_RETRAIN_INTERVAL_HOURS"), 24.0),
        classifier_use_isotonic_calibration=_as_bool(os.getenv("ARB_CRYPTO_CLASSIFIER_USE_ISOTONIC"), True),
        classifier_max_depth=_as_int(os.getenv("ARB_CRYPTO_CLASSIFIER_MAX_DEPTH"), 4),
        classifier_n_estimators=_as_int(os.getenv("ARB_CRYPTO_CLASSIFIER_N_ESTIMATORS"), 100),
        classifier_learning_rate=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_LEARNING_RATE"), 0.1),
        classifier_min_child_weight=_as_int(os.getenv("ARB_CRYPTO_CLASSIFIER_MIN_CHILD_WEIGHT"), 5),
        classifier_subsample=_as_float(os.getenv("ARB_CRYPTO_CLASSIFIER_SUBSAMPLE"), 0.8),
        scan_interval_seconds=_as_float(os.getenv("ARB_CRYPTO_SCAN_INTERVAL_SECONDS"), 5.0),
        paper_mode=_as_bool(os.getenv("ARB_CRYPTO_PAPER_MODE"), True),
        paper_slippage_cents=_as_float(os.getenv("ARB_CRYPTO_PAPER_SLIPPAGE_CENTS"), 0.5),
        settlement_grace_minutes=_as_float(os.getenv("ARB_CRYPTO_SETTLEMENT_GRACE_MINUTES"), 10.0),
    )
