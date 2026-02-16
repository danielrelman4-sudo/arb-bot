"""Offline backtester that replays recorded cycle data from SQLite.

Loads edges, filter decisions, and trade outcomes from a cycle-recorder
database, then re-applies filters with modified CryptoSettings parameters
to evaluate strategy changes in seconds (no live data or MC simulation
required).

Python 3.9 compatible.  No external dependencies beyond stdlib.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestTrade:
    """A single trade produced by the backtest replay."""

    cycle_id: int
    cycle_number: int
    timestamp: float
    ticker: str
    side: str
    edge_cents: float
    model_prob: float
    market_prob: float
    contracts: int
    entry_price: float
    model_uncertainty: float
    time_to_expiry_minutes: float
    regime: Optional[str]
    actual_outcome: Optional[str] = None
    pnl: Optional[float] = None


@dataclass
class BacktestResult:
    """Aggregate results from a single backtest run."""

    config_label: str
    settings_overrides: Dict[str, Any]
    trades: List[BacktestTrade]
    total_pnl: float
    win_rate: float
    num_trades: int
    num_wins: int
    num_losses: int
    avg_edge: float
    sharpe_approx: float
    trades_by_regime: Dict[str, int]
    pnl_by_regime: Dict[str, float]
    edges_total: int
    edges_killed_by: Dict[str, int]


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    db_path: str
    settings_overrides: Dict[str, Any] = field(default_factory=dict)
    apply_regime_min_edge: bool = True
    apply_zscore: bool = True
    apply_counter_trend: bool = True
    apply_confidence: bool = True
    require_settlement: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REGIME_KELLY_KEYS = {
    "mean_reverting": "regime_kelly_mean_reverting",
    "trending_up": "regime_kelly_trending_up",
    "trending_down": "regime_kelly_trending_down",
    "high_vol": "regime_kelly_high_vol",
}

_REGIME_MIN_EDGE_KEYS = {
    "mean_reverting": "regime_min_edge_mean_reverting",
    "trending_up": "regime_min_edge_trending",
    "trending_down": "regime_min_edge_trending",
    "high_vol": "regime_min_edge_high_vol",
}

_TRENDING_REGIMES = {"trending_up", "trending_down"}

# Default sample count for Baker-McHale uncertainty haircut
_BAKER_MCHALE_N_DEFAULT = 20


def _get(settings: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from the merged settings dict with a default."""
    return settings.get(key, default)


def _is_counter_trend(
    side: str,
    direction: Optional[str],
    regime: Optional[str],
) -> bool:
    """Return True if the edge opposes the prevailing trend.

    ``direction`` comes from the market snapshot (above/below/up/down).
    ``regime`` is trending_up or trending_down.
    """
    if regime not in _TRENDING_REGIMES:
        return False
    if direction is None:
        return False

    trend_dir = 1 if regime == "trending_up" else -1

    if direction in ("above", "up"):
        trade_dir = 1 if side == "yes" else -1
    else:  # below, down
        trade_dir = -1 if side == "yes" else 1

    return trade_dir * trend_dir < 0


def _fmt_pnl(value: float) -> str:
    """Format PnL with sign and dollar symbol."""
    if value >= 0:
        return "+${:.2f}".format(value)
    return "-${:.2f}".format(abs(value))


def _fmt_pct(value: float) -> str:
    return "{:.1f}%".format(value * 100)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Replay recorded cycle data with modified parameters.

    Usage::

        bt = Backtester("/path/to/recording.db")
        result = bt.run(BacktestConfig(
            db_path="/path/to/recording.db",
            settings_overrides={"min_edge_pct": 0.10, "kelly_fraction_cap": 0.08},
        ))
        bt.print_report(result)
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Execute a single backtest run with the given configuration."""

        conn = sqlite3.connect(config.db_path or self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            return self._run_impl(conn, config)
        finally:
            conn.close()

    def sweep(
        self,
        param_name: str,
        values: List[Any],
        base_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[BacktestResult]:
        """Sweep a single parameter across a list of values.

        Returns one ``BacktestResult`` per value.
        """
        results: List[BacktestResult] = []
        for val in values:
            overrides = dict(base_overrides or {})
            overrides[param_name] = val
            cfg = BacktestConfig(
                db_path=self._db_path,
                settings_overrides=overrides,
            )
            results.append(self.run(cfg))
        return results

    def multi_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        base_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[BacktestResult]:
        """Cartesian product sweep across multiple parameters."""
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        results: List[BacktestResult] = []
        for combo in itertools.product(*value_lists):
            overrides = dict(base_overrides or {})
            for k, v in zip(keys, combo):
                overrides[k] = v
            cfg = BacktestConfig(
                db_path=self._db_path,
                settings_overrides=overrides,
            )
            results.append(self.run(cfg))
        return results

    def print_report(self, result: BacktestResult) -> None:
        """Print a human-readable backtest report to stdout."""
        # Header
        label = result.config_label
        print("=== Backtest Report: {} ===".format(label))

        # Period
        if result.trades:
            t_min = min(t.timestamp for t in result.trades)
            t_max = max(t.timestamp for t in result.trades)
            start_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(t_min))
            end_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(t_max))
        else:
            start_str = "N/A"
            end_str = "N/A"
        print("Period: {} - {}".format(start_str, end_str))

        # Trades summary
        print(
            "Trades: {} ({}W / {}L) | Win rate: {}".format(
                result.num_trades,
                result.num_wins,
                result.num_losses,
                _fmt_pct(result.win_rate),
            )
        )
        print(
            "Total PnL: {} | Avg edge: {:.1f}c | Sharpe: {:.2f}".format(
                _fmt_pnl(result.total_pnl),
                result.avg_edge,
                result.sharpe_approx,
            )
        )

        # Regime breakdown
        print("")
        print("Regime breakdown:")
        for regime in ["mean_reverting", "trending_up", "trending_down", "high_vol"]:
            count = result.trades_by_regime.get(regime, 0)
            pnl = result.pnl_by_regime.get(regime, 0.0)
            print(
                "  {:<17s} {:>3d} trades, {}".format(
                    regime + ":", count, _fmt_pnl(pnl)
                )
            )

        # Filter stats
        print("")
        total = result.edges_total
        if total > 0:
            print("Filter stats ({} raw edges):".format(total))
            for filt in ["regime_min_edge", "zscore", "counter_trend", "confidence"]:
                killed = result.edges_killed_by.get(filt, 0)
                print(
                    "  {:<17s} killed {:>4d} ({})".format(
                        filt + ":", killed, _fmt_pct(killed / total)
                    )
                )
            survived = result.edges_killed_by.get("survived", 0)
            traded = result.num_trades
            print(
                "  {:<17s}       {:>4d} ({})".format(
                    "survived:", survived, _fmt_pct(survived / total)
                )
            )
            print(
                "  {:<17s}       {:>4d} ({})".format(
                    "traded:", traded, _fmt_pct(traded / total)
                )
            )
        print("")

    def print_sweep_report(self, results: List[BacktestResult]) -> None:
        """Print a comparative sweep table to stdout."""
        if not results:
            print("No results to display.")
            return

        # Determine the sweep parameter from config labels
        print("=== Parameter Sweep ===")
        header = "  {:<12s} {:>6s}  {:<9s} {:>7s}  {:>10s}  {:>6s}".format(
            "Value", "Trades", "W/L", "WinRate", "PnL", "Sharpe"
        )
        print(header)

        for r in results:
            wl = "{}/{}".format(r.num_wins, r.num_losses)
            print(
                "  {:<12s} {:>6d}  {:<9s} {:>7s}  {:>10s}  {:>6.2f}".format(
                    r.config_label,
                    r.num_trades,
                    wl,
                    _fmt_pct(r.win_rate),
                    _fmt_pnl(r.total_pnl),
                    r.sharpe_approx,
                )
            )
        print("")

    def export_trades_csv(self, result: BacktestResult, path: str) -> None:
        """Export all BacktestTrade records to a CSV file."""
        fieldnames = [
            "cycle_id",
            "cycle_number",
            "timestamp",
            "ticker",
            "side",
            "edge_cents",
            "model_prob",
            "market_prob",
            "contracts",
            "entry_price",
            "model_uncertainty",
            "time_to_expiry_minutes",
            "regime",
            "actual_outcome",
            "pnl",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in result.trades:
                writer.writerow({
                    "cycle_id": t.cycle_id,
                    "cycle_number": t.cycle_number,
                    "timestamp": t.timestamp,
                    "ticker": t.ticker,
                    "side": t.side,
                    "edge_cents": t.edge_cents,
                    "model_prob": t.model_prob,
                    "market_prob": t.market_prob,
                    "contracts": t.contracts,
                    "entry_price": t.entry_price,
                    "model_uncertainty": t.model_uncertainty,
                    "time_to_expiry_minutes": t.time_to_expiry_minutes,
                    "regime": t.regime or "",
                    "actual_outcome": t.actual_outcome or "",
                    "pnl": t.pnl if t.pnl is not None else "",
                })

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _run_impl(
        self,
        conn: sqlite3.Connection,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Core replay logic."""

        # 1. Load original settings from the most recent session
        settings = self._load_settings(conn, config.settings_overrides)

        # 2. Load all cycles ordered by cycle_number
        cycles = conn.execute(
            """SELECT id, cycle_number, timestamp, regime, regime_confidence,
                      regime_is_transitioning, bankroll
               FROM cycles ORDER BY cycle_number"""
        ).fetchall()

        # 3. Preload direction map: (cycle_id, ticker) -> direction
        direction_map = self._load_direction_map(conn)

        # 4. Preload settlement data: edge_id -> (actual_outcome, pnl)
        settlement_map = self._load_settlement_map(conn)

        # Tracking state
        bankroll = _get(settings, "bankroll", 500.0)
        initial_bankroll = bankroll
        trades: List[BacktestTrade] = []
        edges_total = 0
        killed_by: Dict[str, int] = {
            "regime_min_edge": 0,
            "zscore": 0,
            "counter_trend": 0,
            "confidence": 0,
            "survived": 0,
        }

        # Position tracking for concurrency limits
        open_positions: Dict[str, int] = {}  # ticker -> contracts
        positions_per_underlying: Dict[str, int] = {}  # underlying -> count

        max_concurrent = int(_get(settings, "max_concurrent_positions", 10))
        max_per_underlying = int(_get(settings, "max_positions_per_underlying", 3))
        max_per_market = float(_get(settings, "max_position_per_market", 50.0))

        # 5. Replay each cycle
        for cycle in cycles:
            cycle_id = cycle["id"]
            cycle_number = cycle["cycle_number"]
            cycle_ts = cycle["timestamp"]
            regime = cycle["regime"]
            regime_confidence = cycle["regime_confidence"]
            regime_is_transitioning = bool(cycle["regime_is_transitioning"])
            cycle_bankroll = cycle["bankroll"]

            # Load edges for this cycle
            edges = conn.execute(
                """SELECT id, ticker, side, model_prob, market_implied_prob,
                          blended_prob, edge_cents, model_uncertainty,
                          time_to_expiry_minutes, yes_buy_price, no_buy_price,
                          spread_cost, staleness_score,
                          passed_regime_min_edge, passed_zscore,
                          passed_counter_trend, passed_confidence,
                          zscore_value, confidence_score, confidence_agreement,
                          filter_reject_reason, survived_all_filters,
                          contracts_sized, was_traded
                   FROM edges WHERE cycle_id = ?""",
                (cycle_id,),
            ).fetchall()

            if not edges:
                continue

            edges_total += len(edges)

            # Re-apply filters with modified settings
            surviving = []
            for edge in edges:
                ticker = edge["ticker"]
                side = edge["side"]
                edge_cents = edge["edge_cents"] or 0.0
                direction = direction_map.get((cycle_id, ticker))

                # --- Filter 1: Regime min edge ---
                if config.apply_regime_min_edge and _get(settings, "regime_min_edge_enabled", False):
                    if regime and regime in _REGIME_MIN_EDGE_KEYS:
                        setting_key = _REGIME_MIN_EDGE_KEYS[regime]
                        min_edge = float(_get(settings, setting_key, 0.0))
                        if min_edge > 0 and edge_cents < min_edge:
                            killed_by["regime_min_edge"] += 1
                            continue

                # --- Filter 2: Z-score ---
                if config.apply_zscore and _get(settings, "zscore_filter_enabled", True):
                    zscore_val = edge["zscore_value"]
                    zscore_max = float(_get(settings, "zscore_max", 2.0))
                    if zscore_val is not None and zscore_val > zscore_max:
                        killed_by["zscore"] += 1
                        continue

                # --- Filter 3: Counter-trend ---
                if config.apply_counter_trend and _get(settings, "regime_skip_counter_trend", True):
                    ct_min_conf = float(
                        _get(settings, "regime_skip_counter_trend_min_conf", 0.6)
                    )
                    if (
                        regime in _TRENDING_REGIMES
                        and regime_confidence is not None
                        and regime_confidence >= ct_min_conf
                        and _is_counter_trend(side, direction, regime)
                    ):
                        killed_by["counter_trend"] += 1
                        continue

                # --- Filter 4: Confidence ---
                if config.apply_confidence and _get(settings, "confidence_scoring_enabled", False):
                    conf_score = edge["confidence_score"]
                    conf_min = float(_get(settings, "confidence_min_score", 0.65))
                    if conf_score is not None and conf_score < conf_min:
                        killed_by["confidence"] += 1
                        continue

                surviving.append(edge)

            killed_by["survived"] += len(surviving)

            # Size and filter surviving edges
            for edge in surviving:
                ticker = edge["ticker"]
                side = edge["side"]
                edge_cents = edge["edge_cents"] or 0.0
                model_prob = edge["model_prob"] or 0.5
                market_prob = edge["market_implied_prob"] or 0.5
                model_uncertainty = edge["model_uncertainty"] or 0.0
                tte = edge["time_to_expiry_minutes"] or 0.0

                # Determine entry price
                if side == "yes":
                    entry_price = edge["yes_buy_price"] or 0.0
                else:
                    entry_price = edge["no_buy_price"] or 0.0

                if entry_price <= 0 or entry_price >= 1.0:
                    continue

                # Check position limits
                if len(open_positions) >= max_concurrent:
                    continue
                # Extract underlying from ticker (e.g. KXBTC15M-... -> KXBTC)
                underlying = _extract_underlying(ticker)
                if positions_per_underlying.get(underlying, 0) >= max_per_underlying:
                    continue

                # Compute position size using Kelly + regime adjustments
                cost = entry_price  # price per contract in dollars (0-1 range)
                if cost <= 0:
                    continue

                # 1. Raw Kelly fraction
                kelly_f = edge_cents / (cost * 100.0)

                # 2. Effective Kelly cap (boosted for mean_reverting if conf > 0.7)
                effective_kelly_cap = float(_get(settings, "kelly_fraction_cap", 0.10))
                if (
                    regime == "mean_reverting"
                    and regime_confidence is not None
                    and regime_confidence > 0.7
                ):
                    boost = float(
                        _get(settings, "regime_kelly_cap_boost_mean_reverting", 1.25)
                    )
                    if boost != 1.0:
                        effective_kelly_cap *= boost

                # 3. Cap Kelly fraction
                kelly_f = min(kelly_f, effective_kelly_cap)
                kelly_f = max(0.0, kelly_f)

                # 4. Regime Kelly multiplier
                if _get(settings, "regime_sizing_enabled", False) and regime:
                    regime_key = _REGIME_KELLY_KEYS.get(regime)
                    if regime_key:
                        mult = float(_get(settings, regime_key, 1.0))
                        kelly_f *= mult

                # 5. Uncertainty haircut (Baker-McHale)
                if model_uncertainty > 0 and 0 < cost < 1.0:
                    n = 1.0 / (cost * (1.0 - cost))
                    shrinkage = 1.0 / (1.0 + n * model_uncertainty * model_uncertainty)
                    kelly_f *= shrinkage

                # 6. Transition caution
                if (
                    regime_is_transitioning
                    and float(
                        _get(settings, "regime_transition_sizing_multiplier", 0.3)
                    )
                    < 1.0
                ):
                    kelly_f *= float(
                        _get(settings, "regime_transition_sizing_multiplier", 0.3)
                    )

                # 7. Dollar amount and contracts
                dollar_amount = kelly_f * bankroll
                dollar_amount = min(dollar_amount, max_per_market)
                dollar_amount = min(dollar_amount, bankroll)

                contracts = int(dollar_amount / cost) if cost > 0 else 0
                contracts = max(0, contracts)
                if contracts == 0:
                    continue

                # Check settlement availability
                edge_id = edge["id"]
                settlement = settlement_map.get(edge_id)
                if config.require_settlement and settlement is None:
                    continue

                actual_outcome: Optional[str] = None
                pnl: Optional[float] = None
                if settlement is not None:
                    actual_outcome = settlement[0]
                    # Recompute PnL based on our contract count
                    if actual_outcome == "yes":
                        pnl = (1.0 - entry_price) * contracts if side == "yes" else -entry_price * contracts
                    elif actual_outcome == "no":
                        pnl = -entry_price * contracts if side == "yes" else (1.0 - entry_price) * contracts
                    else:
                        # Unknown outcome, use stored PnL scaled to our contracts
                        stored_pnl = settlement[1]
                        stored_contracts = settlement[2]
                        if stored_contracts and stored_contracts > 0:
                            pnl = stored_pnl * (contracts / stored_contracts)

                # Deduct cost from bankroll
                trade_cost = entry_price * contracts
                if trade_cost > bankroll:
                    contracts = int(bankroll / entry_price)
                    if contracts <= 0:
                        continue
                    trade_cost = entry_price * contracts

                bankroll -= trade_cost

                # Track position
                open_positions[ticker] = contracts
                positions_per_underlying[underlying] = (
                    positions_per_underlying.get(underlying, 0) + 1
                )

                bt_trade = BacktestTrade(
                    cycle_id=cycle_id,
                    cycle_number=cycle_number,
                    timestamp=cycle_ts,
                    ticker=ticker,
                    side=side,
                    edge_cents=edge_cents,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    contracts=contracts,
                    entry_price=entry_price,
                    model_uncertainty=model_uncertainty,
                    time_to_expiry_minutes=tte,
                    regime=regime,
                    actual_outcome=actual_outcome,
                    pnl=pnl,
                )
                trades.append(bt_trade)

                # Settle immediately (these are short-expiry crypto markets)
                if pnl is not None:
                    bankroll += trade_cost + pnl

                # Release position (settled)
                if ticker in open_positions:
                    del open_positions[ticker]
                    positions_per_underlying[underlying] = max(
                        0, positions_per_underlying.get(underlying, 1) - 1
                    )

        # 6. Aggregate results
        return self._aggregate(trades, config, edges_total, killed_by)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_settings(
        self,
        conn: sqlite3.Connection,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load original CryptoSettings from the DB and apply overrides."""
        row = conn.execute(
            "SELECT settings_json FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            settings: Dict[str, Any] = {}
        else:
            try:
                settings = json.loads(row["settings_json"])
            except (json.JSONDecodeError, TypeError):
                settings = {}
        # Apply overrides
        settings.update(overrides)
        return settings

    def _load_direction_map(
        self,
        conn: sqlite3.Connection,
    ) -> Dict[Tuple[int, str], Optional[str]]:
        """Build a map of (cycle_id, ticker) -> direction from market_snapshots."""
        rows = conn.execute(
            "SELECT cycle_id, ticker, direction FROM market_snapshots"
        ).fetchall()
        return {(r["cycle_id"], r["ticker"]): r["direction"] for r in rows}

    def _load_settlement_map(
        self,
        conn: sqlite3.Connection,
    ) -> Dict[int, Tuple[Optional[str], Optional[float], Optional[int]]]:
        """Build a map of edge_id -> (actual_outcome, pnl, contracts) from trades."""
        rows = conn.execute(
            """SELECT edge_id, actual_outcome, pnl, contracts
               FROM trades WHERE settled = 1 AND edge_id IS NOT NULL"""
        ).fetchall()
        return {
            r["edge_id"]: (r["actual_outcome"], r["pnl"], r["contracts"])
            for r in rows
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        trades: List[BacktestTrade],
        config: BacktestConfig,
        edges_total: int,
        killed_by: Dict[str, int],
    ) -> BacktestResult:
        """Compute aggregate stats from trade list."""

        settled_trades = [t for t in trades if t.pnl is not None]
        total_pnl = sum(t.pnl for t in settled_trades)  # type: ignore[arg-type]
        num_trades = len(settled_trades)
        num_wins = sum(1 for t in settled_trades if t.pnl is not None and t.pnl > 0)
        num_losses = num_trades - num_wins
        win_rate = num_wins / num_trades if num_trades > 0 else 0.0
        avg_edge = (
            sum(t.edge_cents for t in settled_trades) / num_trades
            if num_trades > 0
            else 0.0
        )

        # Sharpe approximation: mean / std of per-trade PnLs
        sharpe = 0.0
        if num_trades > 1:
            pnls = [t.pnl for t in settled_trades if t.pnl is not None]
            mean_pnl = sum(pnls) / len(pnls)
            var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 0.0
            if std_pnl > 0:
                sharpe = mean_pnl / std_pnl

        # Per-regime breakdown
        trades_by_regime: Dict[str, int] = {}
        pnl_by_regime: Dict[str, float] = {}
        for t in settled_trades:
            r = t.regime or "unknown"
            trades_by_regime[r] = trades_by_regime.get(r, 0) + 1
            pnl_by_regime[r] = pnl_by_regime.get(r, 0.0) + (t.pnl or 0.0)

        # Config label
        label = self._make_label(config.settings_overrides)

        return BacktestResult(
            config_label=label,
            settings_overrides=config.settings_overrides,
            trades=trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            num_trades=num_trades,
            num_wins=num_wins,
            num_losses=num_losses,
            avg_edge=avg_edge,
            sharpe_approx=sharpe,
            trades_by_regime=trades_by_regime,
            pnl_by_regime=pnl_by_regime,
            edges_total=edges_total,
            edges_killed_by=killed_by,
        )

    @staticmethod
    def _make_label(overrides: Dict[str, Any]) -> str:
        """Build a concise label from override keys/values."""
        if not overrides:
            return "baseline"
        parts = []
        for k, v in overrides.items():
            short_key = k.replace("regime_", "r_").replace("kelly_", "k_")
            if isinstance(v, float):
                parts.append("{}={:.3g}".format(short_key, v))
            else:
                parts.append("{}={}".format(short_key, v))
        return " ".join(parts)


def _extract_underlying(ticker: str) -> str:
    """Extract the underlying series from a ticker string.

    Examples:
        KXBTC15M-26FEB15-... -> KXBTC
        KXETH15M-...         -> KXETH
    """
    # Find the first digit or dash that marks the end of the series prefix
    for i, ch in enumerate(ticker):
        if ch.isdigit() or ch == "-":
            # Walk back to find letters
            prefix = ticker[:i]
            # Remove trailing interval suffix (e.g. "15M" portion is digits+letter)
            # The underlying is the alpha-only prefix like "KXBTC"
            alpha = ""
            for c in prefix:
                if c.isalpha():
                    alpha += c
                else:
                    break
            return alpha if alpha else prefix
    return ticker
