"""Backtest: GARCH vol forecast accuracy vs EWMA/realized vol.

Loads 30 days of 1-minute BTC klines, rolls forward through time, and
at each evaluation point compares:
  1. GARCH multi-step forecast of integrated vol over next H minutes
  2. EWMA (exponentially weighted) vol forecast
  3. Simple realized vol (backward-looking, used as naive baseline)

Against the actual realized vol measured over the next H minutes.

Outputs MAE, RMSE, and directional accuracy for each method.

Usage:
    python3 -m arb_bot.crypto.backtest_garch
    python3 -m arb_bot.crypto.backtest_garch --horizon 30 --symbol SOL
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from arb_bot.crypto.garch_vol import GarchForecaster


_MINUTES_PER_YEAR = 365.25 * 24.0 * 60.0


def _load_klines(path: str) -> np.ndarray:
    """Load 1-minute close prices from a klines CSV."""
    prices = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row["close"]))
    return np.array(prices, dtype=np.float64)


def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series."""
    return np.diff(np.log(prices))


def _realized_vol_forward(
    returns: np.ndarray, start: int, horizon: int,
) -> float | None:
    """Compute realized vol over [start, start+horizon) returns.

    Returns annualized vol, or None if not enough data.
    """
    end = start + horizon
    if end > len(returns):
        return None
    chunk = returns[start:end]
    var = float(np.var(chunk, ddof=1))
    # Annualize: var is per-minute, multiply by minutes_per_year
    return math.sqrt(var * _MINUTES_PER_YEAR)


def _ewma_vol(
    returns: np.ndarray, end_idx: int, span: int = 60,
) -> float:
    """EWMA volatility estimate (annualized) using returns[:end_idx]."""
    if end_idx < 2:
        return 0.5  # fallback
    r = returns[:end_idx]
    # Use last `span` returns for EWMA
    r = r[-span:]
    alpha = 2.0 / (len(r) + 1)
    ewma_var = 0.0
    for ret in r:
        ewma_var = alpha * ret**2 + (1 - alpha) * ewma_var
    return math.sqrt(max(ewma_var, 1e-20) * _MINUTES_PER_YEAR)


def run_backtest(
    klines_path: str,
    lookback: int = 1440,
    horizon: int = 15,
    step: int = 60,
    min_obs: int = 120,
) -> dict:
    """Run GARCH vs EWMA vs RV backtest.

    Parameters
    ----------
    klines_path : path to klines CSV
    lookback : GARCH fitting window (minutes)
    horizon : forecast horizon (minutes)
    step : evaluation frequency (minutes)
    min_obs : minimum observations for GARCH fitting

    Returns
    -------
    dict with MAE, RMSE, directional accuracy for each method.
    """
    prices = _load_klines(klines_path)
    returns = _log_returns(prices)
    n = len(returns)

    print(f"  Loaded {len(prices)} prices → {n} returns")
    print(f"  Lookback: {lookback}m, Horizon: {horizon}m, Step: {step}m")
    print(f"  Evaluation points: ~{(n - lookback - horizon) // step}")

    garch = GarchForecaster(
        min_obs=min_obs,
        lookback_obs=lookback,
        interval_seconds=60,
    )

    # Tracking arrays
    garch_errors = []
    ewma_errors = []
    rv_errors = []
    garch_forecasts = []
    ewma_forecasts = []
    rv_forecasts = []
    realized_vols = []
    garch_fit_failures = 0

    eval_start = lookback
    eval_end = n - horizon

    for t in range(eval_start, eval_end, step):
        # Realized vol over next H minutes (ground truth)
        rv_forward = _realized_vol_forward(returns, t, horizon)
        if rv_forward is None:
            continue

        # Backward-looking returns for fitting
        r_back = returns[t - lookback:t]

        # ── GARCH forecast ──────────────────────────────
        params = garch.fit(r_back)
        if params is None:
            garch_fit_failures += 1
            continue

        forecast = garch.forecast(params, r_back, horizon_steps=horizon)
        garch_vol = forecast.sigma_annualized

        # ── EWMA forecast ───────────────────────────────
        ewma_vol = _ewma_vol(returns, t, span=lookback)

        # ── Simple realized vol (backward, as baseline) ─
        rv_back = math.sqrt(float(np.var(r_back, ddof=1)) * _MINUTES_PER_YEAR)

        # Record
        realized_vols.append(rv_forward)
        garch_forecasts.append(garch_vol)
        ewma_forecasts.append(ewma_vol)
        rv_forecasts.append(rv_back)

        garch_errors.append(garch_vol - rv_forward)
        ewma_errors.append(ewma_vol - rv_forward)
        rv_errors.append(rv_back - rv_forward)

    # ── Compute metrics ─────────────────────────────────
    n_eval = len(realized_vols)
    if n_eval == 0:
        print("  No evaluation points!")
        return {}

    realized_arr = np.array(realized_vols)
    garch_arr = np.array(garch_forecasts)
    ewma_arr = np.array(ewma_forecasts)
    rv_arr = np.array(rv_forecasts)

    def _metrics(forecast_arr, errors, label):
        err = np.array(errors)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))

        # Directional accuracy: did the forecast correctly predict
        # whether vol was above or below the median?
        median_rv = float(np.median(realized_arr))
        dir_correct = np.sum(
            (forecast_arr > median_rv) == (realized_arr > median_rv)
        )
        dir_acc = dir_correct / n_eval

        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        rho, pval = spearmanr(forecast_arr, realized_arr)

        return {
            "label": label,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "dir_accuracy": dir_acc,
            "spearman_rho": rho,
            "spearman_pval": pval,
        }

    garch_m = _metrics(garch_arr, garch_errors, "GARCH(1,1)")
    ewma_m = _metrics(ewma_arr, ewma_errors, "EWMA")
    rv_m = _metrics(rv_arr, rv_errors, "Realized (backward)")

    # ── Print results ───────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  GARCH Vol Forecast Backtest — {n_eval} evaluation points")
    print(f"  Horizon: {horizon}m | Lookback: {lookback}m | Step: {step}m")
    print(f"  GARCH fit failures: {garch_fit_failures}")
    print(f"  {'='*65}")
    print(f"  {'Method':<22} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Dir%':>8} {'ρ':>8} {'p-val':>10}")
    print(f"  {'-'*65}")
    for m in [garch_m, ewma_m, rv_m]:
        print(
            f"  {m['label']:<22} "
            f"{m['mae']:.4f}  "
            f"{m['rmse']:.4f}  "
            f"{m['bias']:+.4f}  "
            f"{m['dir_accuracy']:.1%}  "
            f"{m['spearman_rho']:.4f}  "
            f"{m['spearman_pval']:.2e}"
        )
    print(f"  {'-'*65}")

    # Improvement over EWMA
    garch_improve_mae = (ewma_m["mae"] - garch_m["mae"]) / ewma_m["mae"] * 100
    garch_improve_rmse = (ewma_m["rmse"] - garch_m["rmse"]) / ewma_m["rmse"] * 100
    print(f"\n  GARCH vs EWMA improvement:")
    print(f"    MAE:  {garch_improve_mae:+.1f}%")
    print(f"    RMSE: {garch_improve_rmse:+.1f}%")
    print(f"    Dir accuracy: {garch_m['dir_accuracy']:.1%} vs {ewma_m['dir_accuracy']:.1%}")
    print(f"    Rank corr:    {garch_m['spearman_rho']:.4f} vs {ewma_m['spearman_rho']:.4f}")

    # Vol regime analysis
    print(f"\n  Vol regime breakdown:")
    low_vol_mask = realized_arr < np.percentile(realized_arr, 33)
    mid_vol_mask = (realized_arr >= np.percentile(realized_arr, 33)) & (realized_arr < np.percentile(realized_arr, 67))
    high_vol_mask = realized_arr >= np.percentile(realized_arr, 67)

    for label, mask in [("Low vol (bottom 33%)", low_vol_mask),
                        ("Mid vol (33-67%)", mid_vol_mask),
                        ("High vol (top 33%)", high_vol_mask)]:
        if mask.sum() == 0:
            continue
        g_mae = float(np.mean(np.abs(garch_arr[mask] - realized_arr[mask])))
        e_mae = float(np.mean(np.abs(ewma_arr[mask] - realized_arr[mask])))
        improve = (e_mae - g_mae) / e_mae * 100 if e_mae > 0 else 0
        print(f"    {label}: GARCH MAE={g_mae:.4f} EWMA MAE={e_mae:.4f} ({improve:+.1f}%)")

    return {
        "n_eval": n_eval,
        "garch": garch_m,
        "ewma": ewma_m,
        "rv": rv_m,
        "garch_fit_failures": garch_fit_failures,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backtest GARCH vol forecasts vs EWMA/realized vol",
    )
    parser.add_argument(
        "--symbol", choices=["BTC", "SOL"], default="BTC",
        help="Which symbol to backtest (default: BTC)",
    )
    parser.add_argument(
        "--horizon", type=int, default=15,
        help="Forecast horizon in minutes (default: 15)",
    )
    parser.add_argument(
        "--lookback", type=int, default=1440,
        help="GARCH fitting lookback in minutes (default: 1440 = 24h)",
    )
    parser.add_argument(
        "--step", type=int, default=60,
        help="Evaluation step in minutes (default: 60)",
    )
    args = parser.parse_args()

    symbol_map = {
        "BTC": "historical_klines_btcusdt_30d.csv",
        "SOL": "historical_klines_solusdt_30d.csv",
    }

    base = Path(__file__).resolve().parent.parent / "output"
    klines_path = base / symbol_map[args.symbol]

    if not klines_path.exists():
        print(f"  ERROR: {klines_path} not found")
        print(f"  Run: python3 -m arb_bot.crypto.fetch_historical")
        sys.exit(1)

    print(f"\n  GARCH Vol Forecast Backtest — {args.symbol}")
    print(f"  Data: {klines_path.name}")

    results = run_backtest(
        str(klines_path),
        lookback=args.lookback,
        horizon=args.horizon,
        step=args.step,
    )

    # Run at multiple horizons for comparison
    if args.horizon == 15:
        print(f"\n\n  {'='*65}")
        print(f"  Multi-horizon comparison (5m, 15m, 30m, 60m)")
        print(f"  {'='*65}")
        for h in [5, 15, 30, 60]:
            r = run_backtest(
                str(klines_path),
                lookback=args.lookback,
                horizon=h,
                step=max(60, h * 2),
            )
            if r:
                print(f"\n  H={h}m: GARCH MAE={r['garch']['mae']:.4f} "
                      f"EWMA MAE={r['ewma']['mae']:.4f} "
                      f"GARCH ρ={r['garch']['spearman_rho']:.4f} "
                      f"EWMA ρ={r['ewma']['spearman_rho']:.4f}")


if __name__ == "__main__":
    main()
