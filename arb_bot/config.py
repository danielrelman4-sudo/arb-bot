from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in bare environments
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


def _as_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _as_optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    return _as_bool(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _as_csv(value: str | None) -> List[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _as_env_map(value: str | None) -> Dict[str, float]:
    """Parses `venue=value,venue2=value2` into a float map."""
    items: Dict[str, float] = {}
    for chunk in _as_csv(value):
        if "=" not in chunk:
            continue
        key, amount = chunk.split("=", 1)
        key = key.strip().lower()
        if not key:
            continue
        items[key] = float(amount)
    return items


def _normalize_kalshi_host_url(value: str | None, default: str) -> str:
    candidate = (value or default).strip()
    if not candidate:
        return default
    parsed = urlparse(candidate)
    if (parsed.hostname or "").lower() != "api.kalshi.com":
        return candidate
    return candidate.replace("//api.kalshi.com", "//api.elections.kalshi.com", 1)


def _derive_kalshi_stream_priority_tickers(
    cross_map_path: str | None,
    structural_rules_path: str | None,
    hard_cap: int = 800,
    cross_quota: int = 0,
    bucket_quota: int = 0,
) -> List[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        ticker = value.strip()
        if not ticker or ticker in seen:
            return
        seen.add(ticker)
        ordered.append(ticker)

    cross_tickers = _dedupe_in_order(_read_kalshi_tickers_from_cross_map(cross_map_path))
    if cross_quota > 0:
        for ticker in cross_tickers[:cross_quota]:
            _add(ticker)
    else:
        for ticker in cross_tickers:
            _add(ticker)

    _, kalshi_bucket_groups = _read_structural_bucket_groups_by_venue(structural_rules_path)
    if bucket_quota != 0:
        added_for_buckets = 0
        # Favor smaller complete groups so more groups are fully represented.
        for group in sorted(kalshi_bucket_groups, key=len):
            unique_group = [ticker for ticker in group if ticker and ticker not in seen]
            if not unique_group:
                continue
            if bucket_quota > 0 and added_for_buckets + len(unique_group) > bucket_quota:
                continue
            for ticker in unique_group:
                _add(ticker)
            added_for_buckets += len(unique_group)

    for ticker in _read_kalshi_tickers_from_structural_rules(structural_rules_path):
        _add(ticker)
    for ticker in cross_tickers:
        _add(ticker)

    if hard_cap > 0:
        return ordered[:hard_cap]
    return ordered


def _read_kalshi_tickers_from_cross_map(path: str | None) -> list[str]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    tickers: list[str] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                for key in ("kalshi_market_id", "kalshi_ticker"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        tickers.append(value)
                        break
    except Exception:
        return []
    return tickers


def _dedupe_in_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _read_structural_bucket_groups_by_venue(path: str | None) -> tuple[list[list[str]], list[list[str]]]:
    if not path:
        return [], []
    json_path = Path(path)
    if not json_path.exists():
        return [], []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    if not isinstance(payload, dict):
        return [], []

    polymarket_groups: list[list[str]] = []
    kalshi_groups: list[list[str]] = []

    for bucket in payload.get("mutually_exclusive_buckets", []):
        if not isinstance(bucket, dict):
            continue
        pm_group: list[str] = []
        k_group: list[str] = []
        for leg in bucket.get("legs", []):
            if not isinstance(leg, dict):
                continue
            venue = str(leg.get("venue") or "").strip().lower()
            market_id = str(leg.get("market_id") or "").strip()
            if not market_id:
                continue
            if venue == "polymarket":
                pm_group.append(market_id)
            elif venue == "kalshi":
                k_group.append(market_id)
        if pm_group:
            polymarket_groups.append(_dedupe_in_order(pm_group))
        if k_group:
            kalshi_groups.append(_dedupe_in_order(k_group))

    return polymarket_groups, kalshi_groups


def _read_polymarket_market_ids_from_cross_map(path: str | None) -> list[str]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    market_ids: list[str] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                for key in ("polymarket_market_id", "polymarket_condition_id"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        market_ids.append(value)
                        break
    except Exception:
        return []
    return market_ids


def _read_kalshi_tickers_from_structural_rules(path: str | None) -> list[str]:
    if not path:
        return []
    json_path = Path(path)
    if not json_path.exists():
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    tickers: list[str] = []

    def _from_leg(leg: Any) -> None:
        if not isinstance(leg, dict):
            return
        venue = str(leg.get("venue") or "").strip().lower()
        if venue != "kalshi":
            return
        market_id = str(leg.get("market_id") or "").strip()
        if market_id:
            tickers.append(market_id)

    for bucket in payload.get("mutually_exclusive_buckets", []):
        if not isinstance(bucket, dict):
            continue
        for leg in bucket.get("legs", []):
            _from_leg(leg)

    for tree in payload.get("event_trees", []):
        if not isinstance(tree, dict):
            continue
        _from_leg(tree.get("parent"))
        for leg in tree.get("children", []):
            _from_leg(leg)

    for parity in payload.get("cross_market_parity_checks", []):
        if not isinstance(parity, dict):
            continue
        _from_leg(parity.get("left"))
        _from_leg(parity.get("right"))

    return tickers


def _read_polymarket_market_ids_from_structural_rules(path: str | None) -> list[str]:
    if not path:
        return []
    json_path = Path(path)
    if not json_path.exists():
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    market_ids: list[str] = []

    def _from_leg(leg: Any) -> None:
        if not isinstance(leg, dict):
            return
        venue = str(leg.get("venue") or "").strip().lower()
        if venue != "polymarket":
            return
        market_id = str(leg.get("market_id") or "").strip()
        if market_id:
            market_ids.append(market_id)

    for bucket in payload.get("mutually_exclusive_buckets", []):
        if not isinstance(bucket, dict):
            continue
        for leg in bucket.get("legs", []):
            _from_leg(leg)

    for tree in payload.get("event_trees", []):
        if not isinstance(tree, dict):
            continue
        _from_leg(tree.get("parent"))
        for leg in tree.get("children", []):
            _from_leg(leg)

    for parity in payload.get("cross_market_parity_checks", []):
        if not isinstance(parity, dict):
            continue
        _from_leg(parity.get("left"))
        _from_leg(parity.get("right"))

    return market_ids


def _derive_polymarket_priority_market_ids(
    cross_map_path: str | None,
    structural_rules_path: str | None,
    hard_cap: int = 5000,
    cross_quota: int = 0,
    bucket_quota: int = 0,
) -> List[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        market_id = value.strip()
        if not market_id or market_id in seen:
            return
        seen.add(market_id)
        ordered.append(market_id)

    cross_market_ids = _dedupe_in_order(_read_polymarket_market_ids_from_cross_map(cross_map_path))
    if cross_quota > 0:
        for market_id in cross_market_ids[:cross_quota]:
            _add(market_id)
    else:
        for market_id in cross_market_ids:
            _add(market_id)

    polymarket_bucket_groups, _ = _read_structural_bucket_groups_by_venue(structural_rules_path)
    if bucket_quota != 0:
        added_for_buckets = 0
        for group in sorted(polymarket_bucket_groups, key=len):
            unique_group = [market_id for market_id in group if market_id and market_id not in seen]
            if not unique_group:
                continue
            if bucket_quota > 0 and added_for_buckets + len(unique_group) > bucket_quota:
                continue
            for market_id in unique_group:
                _add(market_id)
            added_for_buckets += len(unique_group)

    for market_id in _read_polymarket_market_ids_from_structural_rules(structural_rules_path):
        _add(market_id)
    for market_id in cross_market_ids:
        _add(market_id)

    if hard_cap > 0:
        return ordered[:hard_cap]
    return ordered


@dataclass(frozen=True)
class StrategySettings:
    min_net_edge_per_contract: float = 0.01
    min_expected_profit_usd: float = 1.0
    enable_cross_venue: bool = True
    cross_venue_min_match_score: float = 0.62
    cross_venue_mapping_path: str | None = None
    cross_venue_mapping_required: bool = False
    # B1: Max edge sanity gate — reject cross-venue edges above this
    # threshold as likely mapping errors (e.g., run vs win mismatch).
    cross_venue_max_edge_sanity: float = 0.30
    enable_fuzzy_cross_venue_fallback: bool = True
    enable_maker_estimates: bool = True
    discovery_mode: bool = False
    discovery_min_net_edge_per_contract: float = -0.01
    discovery_min_expected_profit_usd: float = 0.0
    near_arb_total_cost_threshold: float = 1.01
    enable_structural_arb: bool = True
    structural_rules_path: str | None = None
    enable_constraint_pricing: bool = True
    constraint_max_bruteforce_markets: int = 14
    cross_venue_equivalence_min_match_score: float = 0.9
    assume_structural_buckets_exhaustive: bool = True
    enable_time_regime_switching: bool = True
    far_regime_days_threshold: float = 30.0
    near_regime_days_threshold: float = 7.0
    far_regime_edge_multiplier: float = 0.9
    far_regime_expected_profit_multiplier: float = 0.9
    far_regime_fill_probability_delta: float = -0.03
    far_regime_realized_profit_multiplier: float = 0.9
    near_regime_edge_multiplier: float = 1.15
    near_regime_expected_profit_multiplier: float = 1.1
    near_regime_fill_probability_delta: float = 0.03
    near_regime_realized_profit_multiplier: float = 1.1
    enable_bucket_quality_model: bool = True
    bucket_quality_history_glob: str | None = "arb_bot/output/paper_*.csv"
    bucket_quality_history_max_files: int = 50
    bucket_quality_min_observations: int = 8
    bucket_quality_max_active_buckets: int = 0
    bucket_quality_explore_fraction: float = 0.15
    bucket_quality_prior_mean_realized_profit: float = 0.002
    bucket_quality_prior_strength: float = 12.0
    bucket_quality_min_score: float = -0.02
    bucket_quality_leg_count_penalty: float = 0.00025
    bucket_quality_live_update_interval: int = 25
    strict_mapping_temporal_join_seconds: float = 600.0
    cross_lane_min_covered_pairs: int = 0
    parity_lane_min_covered_rules: int = 0
    # Maximum number of legs in a structural bucket. Buckets with more legs
    # are filtered at detection time because multi-leg fill probability
    # decays exponentially, making them untradeable.
    max_bucket_legs: int = 0  # 0 = unlimited (no filter)
    # Maximum consecutive execution failures per bucket before disabling it
    # for the session. Prevents bleeding on illiquid buckets. 0 = unlimited.
    max_consecutive_bucket_failures: int = 3


@dataclass(frozen=True)
class LaneTuningSettings:
    enabled: bool = True
    min_net_edge_per_contract: float | None = None
    min_expected_profit_usd: float | None = None
    min_fill_probability: float | None = None
    min_expected_realized_profit_usd: float | None = None
    use_execution_aware_kelly: bool | None = None
    kelly_fraction_multiplier: float | None = None
    kelly_fraction_floor: float | None = None
    kelly_fraction_cap: float | None = None


@dataclass(frozen=True)
class OpportunityLaneSettings:
    intra_venue: LaneTuningSettings = field(default_factory=LaneTuningSettings)
    cross_venue: LaneTuningSettings = field(default_factory=LaneTuningSettings)
    structural_bucket: LaneTuningSettings = field(default_factory=LaneTuningSettings)
    structural_event_tree: LaneTuningSettings = field(default_factory=LaneTuningSettings)
    structural_parity: LaneTuningSettings = field(default_factory=LaneTuningSettings)


@dataclass(frozen=True)
class SizingSettings:
    max_dollars_per_trade: float = 40.0
    max_contracts_per_trade: int = 100
    max_bankroll_fraction_per_trade: float = 0.08
    max_liquidity_fraction_per_trade: float = 0.5
    enable_execution_aware_kelly: bool = True
    kelly_fraction_cap: float = 1.0
    kelly_failure_loss_floor_fraction: float = 0.0005
    enable_kelly_confidence_shrinkage: bool = True
    kelly_confidence_window: int = 200
    kelly_confidence_sensitivity: float = 1.0
    kelly_confidence_floor: float = 0.25
    enable_lane_kelly_autotune: bool = True
    lane_kelly_autotune_step: float = 0.02
    lane_kelly_autotune_decay_step: float = 0.01
    lane_kelly_autotune_max_floor: float = 0.2
    lane_kelly_autotune_min_detected: int = 12
    lane_kelly_autotune_kelly_zero_ratio: float = 0.5
    cluster_budget_fraction: float = 0.35
    max_open_positions_per_cluster: int = 3
    cluster_exposure_penalty_lambda: float = 0.75


@dataclass(frozen=True)
class RiskSettings:
    max_exposure_per_venue_usd: float = 200.0
    max_open_markets_per_venue: int = 10
    non_intra_open_market_reserve_per_venue: int = 2
    market_cooldown_seconds: int = 900
    market_cooldown_scope: str = "market"
    opportunity_cooldown_seconds: int = 0
    sequential_legs: bool = True
    leg_quote_drift_tolerance: float = 0.03
    leg_max_time_window_seconds: float = 10.0
    order_poll_interval_seconds: float = 1.0
    order_poll_timeout_seconds: float = 15.0
    cancel_on_poll_timeout: bool = True
    # Phase 1A: Operational risk controls
    kill_switch_file: str = ".kill_switch"
    kill_switch_env_var: str = "ARB_KILL_SWITCH"
    venue_kill_switch_env_prefix: str = "ARB_KILL_"
    daily_loss_cap_usd: float = 0.0  # 0 = disabled
    max_consecutive_failures: int = 0  # 0 = disabled
    canary_mode: bool = False
    canary_max_dollars_per_trade: float = 10.0
    canary_max_contracts_per_trade: int = 25


@dataclass(frozen=True)
class UniverseRankingSettings:
    enabled: bool = True
    hotset_size: int = 0
    enable_cold_scan_fallback: bool = True
    volume_weight: float = 1.0
    liquidity_weight: float = 1.0
    spread_weight: float = 1.0
    staleness_weight: float = 0.05


@dataclass(frozen=True)
class FillModelSettings:
    enabled: bool = True
    min_fill_probability: float = 0.30
    queue_depth_factor: float = 1.0
    stale_quote_half_life_seconds: float = 120.0
    spread_penalty_weight: float = 1.0
    transform_source_penalty: float = 0.05
    partial_fill_penalty_per_contract: float = 0.0003
    min_fill_quality_score: float = -0.5
    min_expected_realized_profit_usd: float = 0.0
    # Phase 8A: Correlated multi-leg fill model
    same_venue_fill_correlation: float = 0.7  # 0.0 = independent, 1.0 = single fill
    enable_lane_fill_autotune: bool = True
    lane_fill_autotune_step: float = 0.02
    lane_fill_autotune_decay_step: float = 0.01
    lane_fill_autotune_min_fill_probability: float = 0.15
    lane_fill_autotune_min_detected: int = 12
    lane_fill_autotune_fill_skip_ratio: float = 0.5
    lane_fill_autotune_max_relaxation: float = 0.2


@dataclass(frozen=True)
class KalshiSettings:
    enabled: bool = True
    api_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    enable_stream: bool = True
    stream_reconnect_delay_seconds: float = 2.0
    stream_ping_interval_seconds: float = 20.0
    stream_subscription_batch_size: int = 200
    stream_subscription_retry_attempts: int = 3
    stream_subscription_ack_timeout_seconds: float = 2.5
    stream_max_tickers_per_socket: int = 250
    stream_priority_tickers: List[str] = field(default_factory=list)
    stream_pinned_tickers: List[str] = field(default_factory=list)
    stream_priority_refresh_limit: int = 300
    stream_allow_rest_topup: bool = False
    stream_bootstrap_scan_pages: int = 3
    stream_bootstrap_enrich_limit: int = 0
    stream_subscribe_all: bool = False
    market_tickers: List[str] = field(default_factory=list)
    market_limit: int = 40
    use_orderbook_quotes: bool = True
    max_orderbook_concurrency: int = 10
    market_scan_pages: int = 1
    market_page_size: int = 100
    request_pause_seconds: float = 0.1
    min_liquidity: float = 1.0
    exclude_ticker_prefixes: List[str] = field(default_factory=list)
    include_event_tickers: List[str] = field(default_factory=list)
    require_nondegenerate_quotes: bool = True
    maker_tick_size: float = 0.01
    maker_aggressiveness_ticks: int = 2
    taker_fee_per_contract: float = 0.0
    key_id: str | None = None
    private_key_path: str | None = None
    private_key_pem: str | None = None
    events_429_circuit_threshold: int = 6
    events_429_circuit_cooldown_seconds: float = 180.0


@dataclass(frozen=True)
class PolymarketSettings:
    enabled: bool = False
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    ws_base_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws"
    market_limit: int = 25
    market_scan_pages: int = 1
    market_page_size: int = 100
    priority_backfill_scan_pages: int = 120
    min_liquidity: float = 1.0
    min_pairs_before_event_fallback: int = 10
    quote_depth_contracts: float = 20.0
    max_pair_staleness_seconds: float = 2.0
    max_orderbook_concurrency: int = 8
    book_retry_attempts: int = 3
    book_retry_base_delay_seconds: float = 0.2
    book_retry_max_delay_seconds: float = 2.0
    require_strict_yes_no_labels: bool = True
    maker_tick_size: float = 0.001
    maker_aggressiveness_ticks: int = 2
    enable_stream: bool = True
    ws_custom_feature_enabled: bool = False
    taker_fee_per_contract: float = 0.0
    market_ids: List[str] = field(default_factory=list)
    priority_market_ids: List[str] = field(default_factory=list)
    pinned_market_ids: List[str] = field(default_factory=list)

    # Needed for live order posting through py-clob-client
    chain_id: int = 137
    private_key: str | None = None
    funder: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None


@dataclass(frozen=True)
class ForecastExSettings:
    """Settings for ForecastEx / Interactive Brokers TWS adapter.

    ForecastEx event contracts are binary ($1 payout) options routed via
    the IBKR TWS API.  Connection requires a running TWS or IB Gateway
    instance.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7496  # 7496=TWS live, 7497=TWS paper, 4001=Gateway live, 4002=Gateway paper
    client_id: int = 1
    account_id: str = ""
    market_limit: int = 50
    enable_stream: bool = True
    contract_type: str = "forecast"  # "forecast" for ForecastEx ($1), "cme" for CME event ($100)
    default_tif: str = "GTC"  # DAY, GTC, or IOC
    # Fee: $0.01 per YES/NO pair = $0.005 per individual contract
    fee_per_contract: float = 0.005
    # Payout per contract ($1.00 for ForecastEx, $100.00 for CME)
    payout_per_contract: float = 1.0
    # Priority tickers to always fetch
    priority_symbols: List[str] = field(default_factory=list)
    # Symbols to exclude from discovery
    exclude_symbols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppSettings:
    live_mode: bool
    run_once: bool
    poll_interval_seconds: int
    dry_run: bool
    paper_strict_simulation: bool
    paper_position_lifetime_seconds: int
    paper_dynamic_lifetime_enabled: bool
    paper_dynamic_lifetime_resolution_fraction: float
    paper_dynamic_lifetime_min_seconds: int
    paper_dynamic_lifetime_max_seconds: int
    stream_mode: bool
    stream_recompute_cooldown_ms: int
    default_bankroll_usd: float
    bankroll_by_venue: Dict[str, float]
    log_level: str

    strategy: StrategySettings
    lanes: OpportunityLaneSettings
    sizing: SizingSettings
    risk: RiskSettings
    universe: UniverseRankingSettings
    fill_model: FillModelSettings
    kalshi: KalshiSettings
    polymarket: PolymarketSettings
    forecastex: ForecastExSettings = field(default_factory=ForecastExSettings)
    stream_poll_decision_clock: bool = True
    stream_full_discovery_interval_seconds: int = 600
    stream_low_coverage_trigger_cycles: int = 3
    stream_low_coverage_min_cross_pairs: int = 1
    stream_low_coverage_min_parity_rules: int = 1
    stream_low_coverage_recovery_cycles: int = 3
    stream_low_coverage_enable_full_scan_fallback: bool = True
    stream_poll_only_on_stale: bool = True
    stream_stale_degrade_seconds: float = 0.0
    stream_recovery_attempt_seconds: float = 30.0
    paper_checkpoint_flush_rows: int = 1
    paper_checkpoint_fsync: bool = False
    # Phase 8C: Rolling settlement — settle oldest positions early when they
    # exceed min_hold time, freeing capital for new opportunities.
    paper_rolling_settlement_enabled: bool = False
    paper_rolling_settlement_min_hold_seconds: int = 120
    # B2: Monte Carlo execution simulation for paper runs.
    paper_monte_carlo_enabled: bool = True
    paper_monte_carlo_legging_loss_fraction: float = 0.03
    # Spread-based legging loss: cost in cents to unwind one filled leg by
    # crossing the spread. When > 0, replaces the flat legging_loss_fraction
    # model with a per-contract spread-crossing cost.
    paper_monte_carlo_legging_unwind_spread_cents: float = 0.0
    paper_monte_carlo_adverse_selection_probability: float = 0.15
    paper_monte_carlo_adverse_selection_edge_loss: float = 0.5
    paper_monte_carlo_slippage_std_cents: float = 0.5
    paper_monte_carlo_slippage_max_cents: float = 2.0
    paper_monte_carlo_expected_latency_seconds: float = 1.5
    paper_monte_carlo_edge_decay_half_life_seconds: float = 30.0
    paper_monte_carlo_resolution_success_rate: float = 0.95
    paper_monte_carlo_seed: int | None = None
    control_socket_port: int = 9120



def load_settings() -> AppSettings:
    load_dotenv(override=False)

    live_mode = _as_bool(os.getenv("ARB_LIVE_MODE"), default=False)
    run_once = _as_bool(os.getenv("ARB_RUN_ONCE"), default=False)

    bankroll_by_venue = _as_env_map(os.getenv("ARB_BANKROLL_BY_VENUE"))

    kalshi_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if kalshi_key_path:
        kalshi_key_path = str(Path(kalshi_key_path).expanduser())

    cross_map_path = os.getenv("ARB_CROSS_VENUE_MAPPING_PATH")
    if cross_map_path:
        cross_map_path = str(Path(cross_map_path).expanduser())

    structural_rules_path = os.getenv("ARB_STRUCTURAL_RULES_PATH")
    if structural_rules_path:
        structural_rules_path = str(Path(structural_rules_path).expanduser())
    kalshi_stream_priority_hard_cap = _as_int(
        os.getenv("KALSHI_STREAM_PRIORITY_TICKERS_HARD_CAP"),
        800,
    )
    kalshi_stream_priority_cross_quota = _as_int(
        os.getenv("KALSHI_STREAM_PRIORITY_CROSS_QUOTA"),
        max(0, kalshi_stream_priority_hard_cap // 2),
    )
    kalshi_stream_priority_bucket_quota = _as_int(
        os.getenv("KALSHI_STREAM_PRIORITY_BUCKET_QUOTA"),
        max(0, kalshi_stream_priority_hard_cap // 3),
    )
    kalshi_stream_priority_tickers = _derive_kalshi_stream_priority_tickers(
        cross_map_path=cross_map_path,
        structural_rules_path=structural_rules_path,
        hard_cap=kalshi_stream_priority_hard_cap,
        cross_quota=kalshi_stream_priority_cross_quota,
        bucket_quota=kalshi_stream_priority_bucket_quota,
    )
    polymarket_priority_market_ids_hard_cap = _as_int(
        os.getenv("POLYMARKET_PRIORITY_MARKET_IDS_HARD_CAP"),
        5000,
    )
    polymarket_priority_cross_quota = _as_int(
        os.getenv("POLYMARKET_PRIORITY_CROSS_QUOTA"),
        max(0, polymarket_priority_market_ids_hard_cap // 2),
    )
    polymarket_priority_bucket_quota = _as_int(
        os.getenv("POLYMARKET_PRIORITY_BUCKET_QUOTA"),
        max(0, polymarket_priority_market_ids_hard_cap // 3),
    )
    polymarket_priority_market_ids = _derive_polymarket_priority_market_ids(
        cross_map_path=cross_map_path,
        structural_rules_path=structural_rules_path,
        hard_cap=polymarket_priority_market_ids_hard_cap,
        cross_quota=polymarket_priority_cross_quota,
        bucket_quota=polymarket_priority_bucket_quota,
    )

    # Phase 8B: Pinned tickers — always fetched every REST refresh cycle.
    pin_structural = _as_bool(os.getenv("ARB_PIN_STRUCTURAL_TICKERS"), True)
    if pin_structural:
        kalshi_pinned_tickers = _dedupe_in_order(
            _read_kalshi_tickers_from_structural_rules(structural_rules_path)
        )
        polymarket_pinned_market_ids = _dedupe_in_order(
            _read_polymarket_market_ids_from_structural_rules(structural_rules_path)
        )
    else:
        kalshi_pinned_tickers = []
        polymarket_pinned_market_ids = []

    def _lane(prefix: str) -> LaneTuningSettings:
        return LaneTuningSettings(
            enabled=_as_bool(os.getenv(f"ARB_LANE_{prefix}_ENABLED"), True),
            min_net_edge_per_contract=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_MIN_NET_EDGE_PER_CONTRACT")
            ),
            min_expected_profit_usd=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_MIN_EXPECTED_PROFIT_USD")
            ),
            min_fill_probability=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_MIN_FILL_PROBABILITY")
            ),
            min_expected_realized_profit_usd=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_MIN_REALIZED_PROFIT_USD")
            ),
            use_execution_aware_kelly=_as_optional_bool(
                os.getenv(f"ARB_LANE_{prefix}_USE_EXECUTION_AWARE_KELLY")
            ),
            kelly_fraction_multiplier=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_KELLY_FRACTION_MULTIPLIER")
            ),
            kelly_fraction_floor=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_KELLY_FRACTION_FLOOR")
            ),
            kelly_fraction_cap=_as_optional_float(
                os.getenv(f"ARB_LANE_{prefix}_KELLY_FRACTION_CAP")
            ),
        )

    kalshi_api_base_url = _normalize_kalshi_host_url(
        os.getenv("KALSHI_API_BASE_URL"),
        "https://api.elections.kalshi.com/trade-api/v2",
    )
    kalshi_ws_url = _normalize_kalshi_host_url(
        os.getenv("KALSHI_WS_URL"),
        "wss://api.elections.kalshi.com/trade-api/ws/v2",
    )

    return AppSettings(
        live_mode=live_mode,
        run_once=run_once,
        poll_interval_seconds=_as_int(os.getenv("ARB_POLL_INTERVAL_SECONDS"), 60),
        dry_run=not live_mode,
        paper_strict_simulation=_as_bool(os.getenv("ARB_PAPER_STRICT_SIMULATION"), True),
        paper_position_lifetime_seconds=_as_int(os.getenv("ARB_PAPER_POSITION_LIFETIME_SECONDS"), 900),
        paper_dynamic_lifetime_enabled=_as_bool(os.getenv("ARB_PAPER_DYNAMIC_LIFETIME_ENABLED"), False),
        paper_dynamic_lifetime_resolution_fraction=_as_float(
            os.getenv("ARB_PAPER_DYNAMIC_LIFETIME_RESOLUTION_FRACTION"),
            0.02,
        ),
        paper_dynamic_lifetime_min_seconds=_as_int(
            os.getenv("ARB_PAPER_DYNAMIC_LIFETIME_MIN_SECONDS"),
            60,
        ),
        paper_dynamic_lifetime_max_seconds=_as_int(
            os.getenv("ARB_PAPER_DYNAMIC_LIFETIME_MAX_SECONDS"),
            900,
        ),
        paper_checkpoint_flush_rows=_as_int(
            os.getenv("ARB_PAPER_CHECKPOINT_FLUSH_ROWS"),
            1,
        ),
        paper_checkpoint_fsync=_as_bool(
            os.getenv("ARB_PAPER_CHECKPOINT_FSYNC"),
            False,
        ),
        paper_rolling_settlement_enabled=_as_bool(
            os.getenv("ARB_PAPER_ROLLING_SETTLEMENT_ENABLED"),
            False,
        ),
        paper_rolling_settlement_min_hold_seconds=_as_int(
            os.getenv("ARB_PAPER_ROLLING_SETTLEMENT_MIN_HOLD_SECONDS"),
            120,
        ),
        paper_monte_carlo_enabled=_as_bool(
            os.getenv("ARB_PAPER_MONTE_CARLO_ENABLED"),
            True,
        ),
        paper_monte_carlo_legging_loss_fraction=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_LEGGING_LOSS_FRACTION"),
            0.03,
        ),
        paper_monte_carlo_legging_unwind_spread_cents=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_LEGGING_UNWIND_SPREAD_CENTS"),
            0.0,
        ),
        paper_monte_carlo_adverse_selection_probability=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_ADVERSE_SELECTION_PROBABILITY"),
            0.15,
        ),
        paper_monte_carlo_adverse_selection_edge_loss=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_ADVERSE_SELECTION_EDGE_LOSS"),
            0.5,
        ),
        paper_monte_carlo_slippage_std_cents=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_SLIPPAGE_STD_CENTS"),
            0.5,
        ),
        paper_monte_carlo_slippage_max_cents=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_SLIPPAGE_MAX_CENTS"),
            2.0,
        ),
        paper_monte_carlo_expected_latency_seconds=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_EXPECTED_LATENCY_SECONDS"),
            1.5,
        ),
        paper_monte_carlo_edge_decay_half_life_seconds=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_EDGE_DECAY_HALF_LIFE_SECONDS"),
            30.0,
        ),
        paper_monte_carlo_resolution_success_rate=_as_float(
            os.getenv("ARB_PAPER_MONTE_CARLO_RESOLUTION_SUCCESS_RATE"),
            0.95,
        ),
        paper_monte_carlo_seed=_as_int(
            os.getenv("ARB_PAPER_MONTE_CARLO_SEED"),
            0,
        ) or None,
        control_socket_port=_as_int(os.getenv("ARB_CONTROL_SOCKET_PORT"), 9120),
        stream_mode=_as_bool(os.getenv("ARB_STREAM_MODE"), False),
        stream_recompute_cooldown_ms=_as_int(os.getenv("ARB_STREAM_RECOMPUTE_COOLDOWN_MS"), 400),
        stream_poll_decision_clock=_as_bool(os.getenv("ARB_STREAM_POLL_DECISION_CLOCK"), True),
        stream_full_discovery_interval_seconds=_as_int(
            os.getenv("ARB_STREAM_FULL_DISCOVERY_INTERVAL_SECONDS"),
            600,
        ),
        stream_low_coverage_trigger_cycles=_as_int(
            os.getenv("ARB_STREAM_LOW_COVERAGE_TRIGGER_CYCLES"),
            3,
        ),
        stream_low_coverage_min_cross_pairs=_as_int(
            os.getenv("ARB_STREAM_LOW_COVERAGE_MIN_CROSS_PAIRS"),
            1,
        ),
        stream_low_coverage_min_parity_rules=_as_int(
            os.getenv("ARB_STREAM_LOW_COVERAGE_MIN_PARITY_RULES"),
            1,
        ),
        stream_low_coverage_recovery_cycles=_as_int(
            os.getenv("ARB_STREAM_LOW_COVERAGE_RECOVERY_CYCLES"),
            3,
        ),
        stream_low_coverage_enable_full_scan_fallback=_as_bool(
            os.getenv("ARB_STREAM_LOW_COVERAGE_ENABLE_FULL_SCAN_FALLBACK"),
            True,
        ),
        stream_poll_only_on_stale=_as_bool(
            os.getenv("ARB_STREAM_POLL_ONLY_ON_STALE"),
            True,
        ),
        stream_stale_degrade_seconds=_as_float(
            os.getenv("ARB_STREAM_STALE_DEGRADE_SECONDS"),
            0.0,
        ),
        stream_recovery_attempt_seconds=_as_float(
            os.getenv("ARB_STREAM_RECOVERY_ATTEMPT_SECONDS"),
            30.0,
        ),
        default_bankroll_usd=_as_float(os.getenv("ARB_DEFAULT_BANKROLL_USD"), 250.0),
        bankroll_by_venue=bankroll_by_venue,
        log_level=os.getenv("ARB_LOG_LEVEL", "INFO").upper(),
        strategy=StrategySettings(
            min_net_edge_per_contract=_as_float(os.getenv("ARB_MIN_NET_EDGE_PER_CONTRACT"), 0.01),
            min_expected_profit_usd=_as_float(os.getenv("ARB_MIN_EXPECTED_PROFIT_USD"), 1.0),
            enable_cross_venue=_as_bool(os.getenv("ARB_ENABLE_CROSS_VENUE"), True),
            cross_venue_min_match_score=_as_float(os.getenv("ARB_CROSS_VENUE_MIN_MATCH_SCORE"), 0.62),
            cross_venue_mapping_path=cross_map_path,
            cross_venue_mapping_required=_as_bool(os.getenv("ARB_CROSS_VENUE_MAPPING_REQUIRED"), False),
            cross_venue_max_edge_sanity=_as_float(os.getenv("ARB_CROSS_VENUE_MAX_EDGE_SANITY"), 0.30),
            enable_fuzzy_cross_venue_fallback=_as_bool(os.getenv("ARB_ENABLE_FUZZY_CROSS_VENUE_FALLBACK"), True),
            enable_maker_estimates=_as_bool(os.getenv("ARB_ENABLE_MAKER_ESTIMATES"), True),
            discovery_mode=_as_bool(os.getenv("ARB_DISCOVERY_MODE"), False),
            discovery_min_net_edge_per_contract=_as_float(
                os.getenv("ARB_DISCOVERY_MIN_NET_EDGE_PER_CONTRACT"),
                -0.01,
            ),
            discovery_min_expected_profit_usd=_as_float(
                os.getenv("ARB_DISCOVERY_MIN_EXPECTED_PROFIT_USD"),
                0.0,
            ),
            near_arb_total_cost_threshold=_as_float(os.getenv("ARB_NEAR_ARB_TOTAL_COST_THRESHOLD"), 1.01),
            enable_structural_arb=_as_bool(os.getenv("ARB_ENABLE_STRUCTURAL_ARB"), True),
            structural_rules_path=structural_rules_path,
            enable_constraint_pricing=_as_bool(os.getenv("ARB_ENABLE_CONSTRAINT_PRICING"), True),
            constraint_max_bruteforce_markets=_as_int(
                os.getenv("ARB_CONSTRAINT_MAX_BRUTEFORCE_MARKETS"),
                14,
            ),
            cross_venue_equivalence_min_match_score=_as_float(
                os.getenv("ARB_CROSS_VENUE_EQUIVALENCE_MIN_MATCH_SCORE"),
                0.9,
            ),
            assume_structural_buckets_exhaustive=_as_bool(
                os.getenv("ARB_ASSUME_STRUCTURAL_BUCKETS_EXHAUSTIVE"),
                True,
            ),
            enable_time_regime_switching=_as_bool(os.getenv("ARB_TIME_REGIME_ENABLED"), True),
            far_regime_days_threshold=_as_float(os.getenv("ARB_TIME_REGIME_FAR_DAYS"), 30.0),
            near_regime_days_threshold=_as_float(os.getenv("ARB_TIME_REGIME_NEAR_DAYS"), 7.0),
            far_regime_edge_multiplier=_as_float(os.getenv("ARB_TIME_REGIME_FAR_EDGE_MULTIPLIER"), 0.9),
            far_regime_expected_profit_multiplier=_as_float(
                os.getenv("ARB_TIME_REGIME_FAR_EXPECTED_PROFIT_MULTIPLIER"),
                0.9,
            ),
            far_regime_fill_probability_delta=_as_float(
                os.getenv("ARB_TIME_REGIME_FAR_FILL_PROBABILITY_DELTA"),
                -0.03,
            ),
            far_regime_realized_profit_multiplier=_as_float(
                os.getenv("ARB_TIME_REGIME_FAR_REALIZED_PROFIT_MULTIPLIER"),
                0.9,
            ),
            near_regime_edge_multiplier=_as_float(os.getenv("ARB_TIME_REGIME_NEAR_EDGE_MULTIPLIER"), 1.15),
            near_regime_expected_profit_multiplier=_as_float(
                os.getenv("ARB_TIME_REGIME_NEAR_EXPECTED_PROFIT_MULTIPLIER"),
                1.1,
            ),
            near_regime_fill_probability_delta=_as_float(
                os.getenv("ARB_TIME_REGIME_NEAR_FILL_PROBABILITY_DELTA"),
                0.03,
            ),
            near_regime_realized_profit_multiplier=_as_float(
                os.getenv("ARB_TIME_REGIME_NEAR_REALIZED_PROFIT_MULTIPLIER"),
                1.1,
            ),
            enable_bucket_quality_model=_as_bool(os.getenv("ARB_BUCKET_QUALITY_ENABLED"), True),
            bucket_quality_history_glob=os.getenv("ARB_BUCKET_QUALITY_HISTORY_GLOB", "arb_bot/output/paper_*.csv"),
            bucket_quality_history_max_files=_as_int(os.getenv("ARB_BUCKET_QUALITY_HISTORY_MAX_FILES"), 50),
            bucket_quality_min_observations=_as_int(os.getenv("ARB_BUCKET_QUALITY_MIN_OBSERVATIONS"), 8),
            bucket_quality_max_active_buckets=_as_int(os.getenv("ARB_BUCKET_QUALITY_MAX_ACTIVE_BUCKETS"), 0),
            bucket_quality_explore_fraction=_as_float(os.getenv("ARB_BUCKET_QUALITY_EXPLORE_FRACTION"), 0.15),
            bucket_quality_prior_mean_realized_profit=_as_float(
                os.getenv("ARB_BUCKET_QUALITY_PRIOR_MEAN_REALIZED_PROFIT"),
                0.002,
            ),
            bucket_quality_prior_strength=_as_float(
                os.getenv("ARB_BUCKET_QUALITY_PRIOR_STRENGTH"),
                12.0,
            ),
            bucket_quality_min_score=_as_float(os.getenv("ARB_BUCKET_QUALITY_MIN_SCORE"), -0.02),
            bucket_quality_leg_count_penalty=_as_float(
                os.getenv("ARB_BUCKET_QUALITY_LEG_COUNT_PENALTY"),
                0.00025,
            ),
            bucket_quality_live_update_interval=_as_int(
                os.getenv("ARB_BUCKET_QUALITY_LIVE_UPDATE_INTERVAL"),
                25,
            ),
            strict_mapping_temporal_join_seconds=_as_float(
                os.getenv("ARB_STRICT_MAPPING_TEMPORAL_JOIN_SECONDS"),
                600.0,
            ),
            cross_lane_min_covered_pairs=_as_int(
                os.getenv("ARB_CROSS_LANE_MIN_COVERED_PAIRS"),
                0,
            ),
            parity_lane_min_covered_rules=_as_int(
                os.getenv("ARB_PARITY_LANE_MIN_COVERED_RULES"),
                0,
            ),
            max_bucket_legs=_as_int(
                os.getenv("ARB_MAX_BUCKET_LEGS"),
                0,
            ),
            max_consecutive_bucket_failures=_as_int(
                os.getenv("ARB_MAX_BUCKET_CONSECUTIVE_FAILURES"),
                3,
            ),
        ),
        lanes=OpportunityLaneSettings(
            intra_venue=_lane("INTRA"),
            cross_venue=_lane("CROSS"),
            structural_bucket=_lane("STRUCTURAL_BUCKET"),
            structural_event_tree=_lane("STRUCTURAL_EVENT_TREE"),
            structural_parity=_lane("STRUCTURAL_PARITY"),
        ),
        sizing=SizingSettings(
            max_dollars_per_trade=_as_float(os.getenv("ARB_MAX_DOLLARS_PER_TRADE"), 40.0),
            max_contracts_per_trade=_as_int(os.getenv("ARB_MAX_CONTRACTS_PER_TRADE"), 100),
            max_bankroll_fraction_per_trade=_as_float(
                os.getenv("ARB_MAX_BANKROLL_FRACTION_PER_TRADE"),
                0.08,
            ),
            max_liquidity_fraction_per_trade=_as_float(
                os.getenv("ARB_MAX_LIQUIDITY_FRACTION_PER_TRADE"),
                0.5,
            ),
            enable_execution_aware_kelly=_as_bool(os.getenv("ARB_ENABLE_EXECUTION_AWARE_KELLY"), True),
            kelly_fraction_cap=_as_float(os.getenv("ARB_KELLY_FRACTION_CAP"), 1.0),
            kelly_failure_loss_floor_fraction=_as_float(
                os.getenv("ARB_KELLY_FAILURE_LOSS_FLOOR_FRACTION"),
                0.0005,
            ),
            enable_kelly_confidence_shrinkage=_as_bool(
                os.getenv("ARB_ENABLE_KELLY_CONFIDENCE_SHRINKAGE"),
                True,
            ),
            kelly_confidence_window=_as_int(os.getenv("ARB_KELLY_CONFIDENCE_WINDOW"), 200),
            kelly_confidence_sensitivity=_as_float(
                os.getenv("ARB_KELLY_CONFIDENCE_SENSITIVITY"),
                1.0,
            ),
            kelly_confidence_floor=_as_float(os.getenv("ARB_KELLY_CONFIDENCE_FLOOR"), 0.25),
            enable_lane_kelly_autotune=_as_bool(
                os.getenv("ARB_ENABLE_LANE_KELLY_AUTOTUNE"),
                True,
            ),
            lane_kelly_autotune_step=_as_float(
                os.getenv("ARB_LANE_KELLY_AUTOTUNE_STEP"),
                0.02,
            ),
            lane_kelly_autotune_decay_step=_as_float(
                os.getenv("ARB_LANE_KELLY_AUTOTUNE_DECAY_STEP"),
                0.01,
            ),
            lane_kelly_autotune_max_floor=_as_float(
                os.getenv("ARB_LANE_KELLY_AUTOTUNE_MAX_FLOOR"),
                0.2,
            ),
            lane_kelly_autotune_min_detected=_as_int(
                os.getenv("ARB_LANE_KELLY_AUTOTUNE_MIN_DETECTED"),
                12,
            ),
            lane_kelly_autotune_kelly_zero_ratio=_as_float(
                os.getenv("ARB_LANE_KELLY_AUTOTUNE_KELLY_ZERO_RATIO"),
                0.5,
            ),
            cluster_budget_fraction=_as_float(os.getenv("ARB_CLUSTER_BUDGET_FRACTION"), 0.35),
            max_open_positions_per_cluster=_as_int(os.getenv("ARB_MAX_OPEN_POSITIONS_PER_CLUSTER"), 3),
            cluster_exposure_penalty_lambda=_as_float(
                os.getenv("ARB_CLUSTER_EXPOSURE_PENALTY_LAMBDA"),
                0.75,
            ),
        ),
        risk=RiskSettings(
            max_exposure_per_venue_usd=_as_float(os.getenv("ARB_MAX_EXPOSURE_PER_VENUE_USD"), 200.0),
            max_open_markets_per_venue=_as_int(os.getenv("ARB_MAX_OPEN_MARKETS_PER_VENUE"), 10),
            non_intra_open_market_reserve_per_venue=_as_int(
                os.getenv("ARB_NON_INTRA_OPEN_MARKET_RESERVE_PER_VENUE"),
                2,
            ),
            market_cooldown_seconds=_as_int(os.getenv("ARB_MARKET_COOLDOWN_SECONDS"), 900),
            market_cooldown_scope=(os.getenv("ARB_MARKET_COOLDOWN_SCOPE") or "market").strip().lower(),
            opportunity_cooldown_seconds=_as_int(os.getenv("ARB_OPPORTUNITY_COOLDOWN_SECONDS"), 0),
            kill_switch_file=os.getenv("ARB_KILL_SWITCH_FILE", ".kill_switch"),
            kill_switch_env_var=os.getenv("ARB_KILL_SWITCH_ENV_VAR", "ARB_KILL_SWITCH"),
            venue_kill_switch_env_prefix=os.getenv("ARB_VENUE_KILL_SWITCH_ENV_PREFIX", "ARB_KILL_"),
            daily_loss_cap_usd=_as_float(os.getenv("ARB_DAILY_LOSS_CAP_USD"), 0.0),
            max_consecutive_failures=_as_int(os.getenv("ARB_MAX_CONSECUTIVE_FAILURES"), 0),
            canary_mode=_as_bool(os.getenv("ARB_CANARY_MODE"), False),
            canary_max_dollars_per_trade=_as_float(os.getenv("ARB_CANARY_MAX_DOLLARS_PER_TRADE"), 10.0),
            canary_max_contracts_per_trade=_as_int(os.getenv("ARB_CANARY_MAX_CONTRACTS_PER_TRADE"), 25),
        ),
        universe=UniverseRankingSettings(
            enabled=_as_bool(os.getenv("ARB_UNIVERSE_RANKING_ENABLED"), True),
            hotset_size=_as_int(os.getenv("ARB_UNIVERSE_HOTSET_SIZE"), 0),
            enable_cold_scan_fallback=_as_bool(os.getenv("ARB_UNIVERSE_ENABLE_COLD_SCAN_FALLBACK"), True),
            volume_weight=_as_float(os.getenv("ARB_UNIVERSE_SCORE_VOLUME_WEIGHT"), 1.0),
            liquidity_weight=_as_float(os.getenv("ARB_UNIVERSE_SCORE_LIQUIDITY_WEIGHT"), 1.0),
            spread_weight=_as_float(os.getenv("ARB_UNIVERSE_SCORE_SPREAD_WEIGHT"), 1.0),
            staleness_weight=_as_float(os.getenv("ARB_UNIVERSE_SCORE_STALENESS_WEIGHT"), 0.05),
        ),
        fill_model=FillModelSettings(
            enabled=_as_bool(os.getenv("ARB_FILL_MODEL_ENABLED"), True),
            min_fill_probability=_as_float(os.getenv("ARB_FILL_MIN_PROBABILITY"), 0.30),
            queue_depth_factor=_as_float(os.getenv("ARB_FILL_QUEUE_DEPTH_FACTOR"), 1.0),
            stale_quote_half_life_seconds=_as_float(os.getenv("ARB_FILL_STALE_HALF_LIFE_SECONDS"), 120.0),
            spread_penalty_weight=_as_float(os.getenv("ARB_FILL_SPREAD_PENALTY_WEIGHT"), 1.0),
            transform_source_penalty=_as_float(os.getenv("ARB_FILL_TRANSFORM_SOURCE_PENALTY"), 0.05),
            partial_fill_penalty_per_contract=_as_float(os.getenv("ARB_FILL_PARTIAL_PENALTY_PER_CONTRACT"), 0.0003),
            min_fill_quality_score=_as_float(os.getenv("ARB_FILL_MIN_QUALITY_SCORE"), -0.5),
            min_expected_realized_profit_usd=_as_float(os.getenv("ARB_FILL_MIN_REALIZED_PROFIT_USD"), 0.0),
            same_venue_fill_correlation=_as_float(
                os.getenv("ARB_FILL_SAME_VENUE_CORRELATION"),
                0.7,
            ),
            enable_lane_fill_autotune=_as_bool(
                os.getenv("ARB_ENABLE_LANE_FILL_AUTOTUNE"),
                True,
            ),
            lane_fill_autotune_step=_as_float(
                os.getenv("ARB_LANE_FILL_AUTOTUNE_STEP"),
                0.02,
            ),
            lane_fill_autotune_decay_step=_as_float(
                os.getenv("ARB_LANE_FILL_AUTOTUNE_DECAY_STEP"),
                0.01,
            ),
            lane_fill_autotune_min_fill_probability=_as_float(
                os.getenv("ARB_LANE_FILL_AUTOTUNE_MIN_FILL_PROBABILITY"),
                0.15,
            ),
            lane_fill_autotune_min_detected=_as_int(
                os.getenv("ARB_LANE_FILL_AUTOTUNE_MIN_DETECTED"),
                12,
            ),
            lane_fill_autotune_fill_skip_ratio=_as_float(
                os.getenv("ARB_LANE_FILL_AUTOTUNE_FILL_SKIP_RATIO"),
                0.5,
            ),
            lane_fill_autotune_max_relaxation=_as_float(
                os.getenv("ARB_LANE_FILL_AUTOTUNE_MAX_RELAXATION"),
                0.2,
            ),
        ),
        kalshi=KalshiSettings(
            enabled=_as_bool(os.getenv("KALSHI_ENABLED"), True),
            api_base_url=kalshi_api_base_url,
            ws_url=kalshi_ws_url,
            enable_stream=_as_bool(os.getenv("KALSHI_ENABLE_STREAM"), True),
            stream_reconnect_delay_seconds=_as_float(os.getenv("KALSHI_STREAM_RECONNECT_DELAY_SECONDS"), 2.0),
            stream_ping_interval_seconds=_as_float(os.getenv("KALSHI_STREAM_PING_INTERVAL_SECONDS"), 20.0),
            stream_subscription_batch_size=_as_int(os.getenv("KALSHI_STREAM_SUBSCRIPTION_BATCH_SIZE"), 200),
            stream_subscription_retry_attempts=_as_int(
                os.getenv("KALSHI_STREAM_SUBSCRIPTION_RETRY_ATTEMPTS"),
                3,
            ),
            stream_subscription_ack_timeout_seconds=_as_float(
                os.getenv("KALSHI_STREAM_SUBSCRIPTION_ACK_TIMEOUT_SECONDS"),
                2.5,
            ),
            stream_max_tickers_per_socket=_as_int(os.getenv("KALSHI_STREAM_MAX_TICKERS_PER_SOCKET"), 250),
            stream_priority_tickers=kalshi_stream_priority_tickers,
            stream_pinned_tickers=kalshi_pinned_tickers,
            stream_priority_refresh_limit=_as_int(os.getenv("KALSHI_STREAM_PRIORITY_REFRESH_LIMIT"), 300),
            stream_allow_rest_topup=_as_bool(os.getenv("KALSHI_STREAM_ALLOW_REST_TOPUP"), False),
            stream_bootstrap_scan_pages=_as_int(os.getenv("KALSHI_STREAM_BOOTSTRAP_SCAN_PAGES"), 3),
            stream_bootstrap_enrich_limit=_as_int(os.getenv("KALSHI_STREAM_BOOTSTRAP_ENRICH_LIMIT"), 0),
            stream_subscribe_all=_as_bool(os.getenv("KALSHI_STREAM_SUBSCRIBE_ALL"), False),
            market_tickers=_as_csv(os.getenv("KALSHI_MARKET_TICKERS")),
            market_limit=_as_int(os.getenv("KALSHI_MARKET_LIMIT"), 40),
            use_orderbook_quotes=_as_bool(os.getenv("KALSHI_USE_ORDERBOOK_QUOTES"), True),
            max_orderbook_concurrency=_as_int(os.getenv("KALSHI_MAX_ORDERBOOK_CONCURRENCY"), 10),
            market_scan_pages=_as_int(os.getenv("KALSHI_MARKET_SCAN_PAGES"), 1),
            market_page_size=_as_int(os.getenv("KALSHI_MARKET_PAGE_SIZE"), 100),
            request_pause_seconds=_as_float(os.getenv("KALSHI_REQUEST_PAUSE_SECONDS"), 0.1),
            min_liquidity=_as_float(os.getenv("KALSHI_MIN_LIQUIDITY"), 1.0),
            exclude_ticker_prefixes=_as_csv(os.getenv("KALSHI_EXCLUDE_TICKER_PREFIXES")),
            include_event_tickers=_as_csv(os.getenv("KALSHI_INCLUDE_EVENT_TICKERS")),
            require_nondegenerate_quotes=_as_bool(os.getenv("KALSHI_REQUIRE_NONDEGENERATE_QUOTES"), True),
            maker_tick_size=_as_float(os.getenv("KALSHI_MAKER_TICK_SIZE"), 0.01),
            maker_aggressiveness_ticks=_as_int(os.getenv("KALSHI_MAKER_AGGRESSIVENESS_TICKS"), 2),
            taker_fee_per_contract=_as_float(os.getenv("KALSHI_TAKER_FEE_PER_CONTRACT"), 0.0),
            key_id=os.getenv("KALSHI_KEY_ID"),
            private_key_path=kalshi_key_path,
            private_key_pem=os.getenv("KALSHI_PRIVATE_KEY_PEM"),
            events_429_circuit_threshold=_as_int(os.getenv("KALSHI_EVENTS_429_CIRCUIT_THRESHOLD"), 6),
            events_429_circuit_cooldown_seconds=_as_float(
                os.getenv("KALSHI_EVENTS_429_CIRCUIT_COOLDOWN_SECONDS"),
                180.0,
            ),
        ),
        polymarket=PolymarketSettings(
            enabled=_as_bool(os.getenv("POLYMARKET_ENABLED"), False),
            gamma_base_url=os.getenv("POLYMARKET_GAMMA_BASE_URL", "https://gamma-api.polymarket.com"),
            clob_base_url=os.getenv("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com"),
            ws_base_url=os.getenv("POLYMARKET_WS_BASE_URL", "wss://ws-subscriptions-clob.polymarket.com/ws"),
            market_limit=_as_int(os.getenv("POLYMARKET_MARKET_LIMIT"), 25),
            market_scan_pages=_as_int(os.getenv("POLYMARKET_MARKET_SCAN_PAGES"), 1),
            market_page_size=_as_int(os.getenv("POLYMARKET_MARKET_PAGE_SIZE"), 100),
            priority_backfill_scan_pages=_as_int(
                os.getenv("POLYMARKET_PRIORITY_BACKFILL_SCAN_PAGES"),
                120,
            ),
            min_liquidity=_as_float(os.getenv("POLYMARKET_MIN_LIQUIDITY"), 1.0),
            min_pairs_before_event_fallback=_as_int(os.getenv("POLYMARKET_MIN_PAIRS_BEFORE_EVENT_FALLBACK"), 10),
            quote_depth_contracts=_as_float(os.getenv("POLYMARKET_QUOTE_DEPTH_CONTRACTS"), 20.0),
            max_pair_staleness_seconds=_as_float(os.getenv("POLYMARKET_MAX_PAIR_STALENESS_SECONDS"), 2.0),
            max_orderbook_concurrency=_as_int(os.getenv("POLYMARKET_MAX_ORDERBOOK_CONCURRENCY"), 8),
            book_retry_attempts=_as_int(os.getenv("POLYMARKET_BOOK_RETRY_ATTEMPTS"), 3),
            book_retry_base_delay_seconds=_as_float(os.getenv("POLYMARKET_BOOK_RETRY_BASE_DELAY_SECONDS"), 0.2),
            book_retry_max_delay_seconds=_as_float(os.getenv("POLYMARKET_BOOK_RETRY_MAX_DELAY_SECONDS"), 2.0),
            require_strict_yes_no_labels=_as_bool(os.getenv("POLYMARKET_REQUIRE_STRICT_YES_NO_LABELS"), True),
            maker_tick_size=_as_float(os.getenv("POLYMARKET_MAKER_TICK_SIZE"), 0.001),
            maker_aggressiveness_ticks=_as_int(os.getenv("POLYMARKET_MAKER_AGGRESSIVENESS_TICKS"), 2),
            enable_stream=_as_bool(os.getenv("POLYMARKET_ENABLE_STREAM"), True),
            ws_custom_feature_enabled=_as_bool(os.getenv("POLYMARKET_WS_CUSTOM_FEATURE_ENABLED"), False),
            taker_fee_per_contract=_as_float(os.getenv("POLYMARKET_TAKER_FEE_PER_CONTRACT"), 0.0),
            market_ids=_as_csv(os.getenv("POLYMARKET_MARKET_IDS")),
            priority_market_ids=polymarket_priority_market_ids,
            pinned_market_ids=polymarket_pinned_market_ids,
            chain_id=_as_int(os.getenv("POLYMARKET_CHAIN_ID"), 137),
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
            funder=os.getenv("POLYMARKET_FUNDER"),
            api_key=os.getenv("POLYMARKET_API_KEY"),
            api_secret=os.getenv("POLYMARKET_API_SECRET"),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
        ),
        forecastex=ForecastExSettings(
            enabled=_as_bool(os.getenv("FORECASTEX_ENABLED"), False),
            host=os.getenv("FORECASTEX_HOST", "127.0.0.1"),
            port=_as_int(os.getenv("FORECASTEX_PORT"), 7496),
            client_id=_as_int(os.getenv("FORECASTEX_CLIENT_ID"), 1),
            account_id=os.getenv("FORECASTEX_ACCOUNT_ID", ""),
            market_limit=_as_int(os.getenv("FORECASTEX_MARKET_LIMIT"), 50),
            enable_stream=_as_bool(os.getenv("FORECASTEX_ENABLE_STREAM"), True),
            contract_type=os.getenv("FORECASTEX_CONTRACT_TYPE", "forecast").strip().lower(),
            default_tif=os.getenv("FORECASTEX_DEFAULT_TIF", "GTC").strip().upper(),
            fee_per_contract=_as_float(os.getenv("FORECASTEX_FEE_PER_CONTRACT"), 0.005),
            payout_per_contract=_as_float(os.getenv("FORECASTEX_PAYOUT_PER_CONTRACT"), 1.0),
            priority_symbols=_as_csv(os.getenv("FORECASTEX_PRIORITY_SYMBOLS")),
            exclude_symbols=_as_csv(os.getenv("FORECASTEX_EXCLUDE_SYMBOLS")),
        ),
    )
