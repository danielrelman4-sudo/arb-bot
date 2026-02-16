#!/usr/bin/env python3
"""
Fetch 30 days of 1-minute klines from Binance US REST API for BTCUSDT and SOLUSDT.
Saves raw kline CSVs and a summary profile with hourly volume averages and volatility.

Usage:
    python3 arb_bot/crypto/fetch_historical.py
"""

import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api.binance.us/api/v3/klines"
SYMBOLS = ["BTCUSDT", "SOLUSDT"]
INTERVAL = "1m"
LIMIT = 1000  # max per request
DAYS = 30
ONE_MINUTE_MS = 60_000
CANDLES_NEEDED = DAYS * 24 * 60  # 43200

# Rate-limit: Binance US allows 10 req/s on weight; we stay conservative
REQUEST_DELAY = 0.15  # seconds between requests

# Paths (relative to repo root)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = REPO_ROOT / "arb_bot" / "output"

CSV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ms_to_iso(ms: int) -> str:
    """Convert millisecond timestamp to human-readable ISO string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_klines(
    client: httpx.Client,
    symbol: str,
    start_time: int,
    end_time: int,
) -> list[list]:
    """Fetch up to 1000 klines from Binance US."""
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT,
        "startTime": start_time,
        "endTime": end_time,
    }
    resp = client.get(BASE_URL, params=params)
    resp.raise_for_status()
    return resp.json()


def fetch_all_klines(symbol: str) -> list[dict]:
    """
    Paginate through 30 days of 1-min klines for a symbol.
    Returns list of dicts with CSV_COLUMNS keys.
    """
    now_ms = int(time.time() * 1000)
    end_ms = now_ms
    start_ms = now_ms - (DAYS * 24 * 60 * 60 * 1000)

    all_rows: list[dict] = []
    cursor = start_ms
    request_num = 0
    total_requests_est = math.ceil(CANDLES_NEEDED / LIMIT)

    print(f"\n{'='*60}")
    print(f"Fetching {symbol} | {DAYS}d of 1m klines ({CANDLES_NEEDED:,} candles)")
    print(f"Range: {ms_to_iso(start_ms)} -> {ms_to_iso(end_ms)}")
    print(f"Estimated requests: ~{total_requests_est}")
    print(f"{'='*60}")

    with httpx.Client(timeout=30.0) as client:
        while cursor < end_ms:
            request_num += 1
            batch_end = min(cursor + LIMIT * ONE_MINUTE_MS, end_ms)

            try:
                raw = fetch_klines(client, symbol, cursor, batch_end)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 30
                    print(f"  [!] Rate limited. Sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                raise

            if not raw:
                print(f"  [{request_num}/{total_requests_est}] No data returned, stopping.")
                break

            for candle in raw:
                # Binance kline format: [open_time, open, high, low, close, volume,
                #   close_time, quote_asset_vol, num_trades, taker_buy_base, taker_buy_quote, ignore]
                row = {
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "close_time": int(candle[6]),
                    "quote_volume": float(candle[7]),
                    "num_trades": int(candle[8]),
                    "taker_buy_base_vol": float(candle[9]),
                    "taker_buy_quote_vol": float(candle[10]),
                }
                all_rows.append(row)

            last_ts = int(raw[-1][0])
            # Move cursor past the last candle to avoid duplicates
            cursor = last_ts + ONE_MINUTE_MS

            if request_num % 5 == 0 or request_num == 1:
                pct = min(100, len(all_rows) / CANDLES_NEEDED * 100)
                print(
                    f"  [{request_num:>3}/{total_requests_est}]  "
                    f"candles={len(all_rows):>6,}  "
                    f"last={ms_to_iso(last_ts)}  "
                    f"progress={pct:5.1f}%"
                )

            time.sleep(REQUEST_DELAY)

    # Deduplicate by timestamp (in case of overlap at boundaries)
    seen = set()
    deduped = []
    for row in all_rows:
        ts = row["timestamp"]
        if ts not in seen:
            seen.add(ts)
            deduped.append(row)
    deduped.sort(key=lambda r: r["timestamp"])

    print(f"  Done: {len(deduped):,} unique candles fetched for {symbol}")
    return deduped


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def save_klines_csv(rows: list[dict], path: Path) -> None:
    """Save kline rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows -> {path}")


def compute_profile_summary(data_by_symbol: dict[str, list[dict]]) -> list[dict]:
    """
    Compute hourly profile: average volume, average num_trades,
    1-min log-return volatility (annualized), and average true range.

    Returns list of dicts for CSV output.
    """
    summary_rows = []

    for symbol, rows in data_by_symbol.items():
        # Group by hour-of-day (UTC)
        hourly_buckets: dict[int, list[dict]] = {h: [] for h in range(24)}
        for row in rows:
            hour = datetime.fromtimestamp(
                row["timestamp"] / 1000, tz=timezone.utc
            ).hour
            hourly_buckets[hour].append(row)

        for hour in range(24):
            bucket = hourly_buckets[hour]
            if not bucket:
                continue

            n = len(bucket)
            avg_volume = sum(r["volume"] for r in bucket) / n
            avg_quote_volume = sum(r["quote_volume"] for r in bucket) / n
            avg_num_trades = sum(r["num_trades"] for r in bucket) / n
            avg_taker_ratio = (
                sum(r["taker_buy_base_vol"] for r in bucket)
                / max(sum(r["volume"] for r in bucket), 1e-12)
            )

            # 1-min log returns for volatility
            log_returns = []
            for i in range(1, len(bucket)):
                prev_close = bucket[i - 1]["close"]
                curr_close = bucket[i]["close"]
                if prev_close > 0 and curr_close > 0:
                    log_returns.append(math.log(curr_close / prev_close))

            if log_returns:
                mean_ret = sum(log_returns) / len(log_returns)
                var = sum((r - mean_ret) ** 2 for r in log_returns) / len(log_returns)
                vol_1m = math.sqrt(var)
                # Annualize: sqrt(minutes_per_year) where 1 year ~ 525600 minutes
                vol_annualized = vol_1m * math.sqrt(525_600)
            else:
                vol_1m = 0.0
                vol_annualized = 0.0

            # Average True Range (1-min)
            atr_vals = []
            for r in bucket:
                tr = max(
                    r["high"] - r["low"],
                    abs(r["high"] - r["close"]),
                    abs(r["low"] - r["close"]),
                )
                atr_vals.append(tr)
            avg_atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0.0

            summary_rows.append({
                "symbol": symbol,
                "hour_utc": hour,
                "candle_count": n,
                "avg_volume": round(avg_volume, 6),
                "avg_quote_volume": round(avg_quote_volume, 2),
                "avg_num_trades": round(avg_num_trades, 1),
                "avg_taker_buy_ratio": round(avg_taker_ratio, 4),
                "vol_1m_stdev": round(vol_1m, 8),
                "vol_annualized": round(vol_annualized, 6),
                "avg_atr": round(avg_atr, 6),
            })

    return summary_rows


def save_profile_summary(rows: list[dict], path: Path) -> None:
    """Save profile summary to CSV."""
    if not rows:
        print("  No summary rows to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} summary rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Historical kline fetcher — Binance US")
    print(f"Symbols: {SYMBOLS}")
    print(f"Period: {DAYS} days of 1-minute candles")
    print(f"Output dir: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_by_symbol: dict[str, list[dict]] = {}

    for symbol in SYMBOLS:
        rows = fetch_all_klines(symbol)
        data_by_symbol[symbol] = rows

        filename = f"historical_klines_{symbol.lower()}_{DAYS}d.csv"
        save_klines_csv(rows, OUTPUT_DIR / filename)

    # Compute and save profile summary
    print(f"\n{'='*60}")
    print("Computing hourly profile summary...")
    summary = compute_profile_summary(data_by_symbol)
    save_profile_summary(summary, OUTPUT_DIR / "historical_profile_summary.csv")

    # Print a quick overview
    print(f"\n{'='*60}")
    print("Summary overview:")
    print(f"{'='*60}")
    for symbol in SYMBOLS:
        rows = data_by_symbol[symbol]
        if not rows:
            print(f"  {symbol}: no data")
            continue
        first_ts = ms_to_iso(rows[0]["timestamp"])
        last_ts = ms_to_iso(rows[-1]["timestamp"])
        total_vol = sum(r["volume"] for r in rows)
        total_trades = sum(r["num_trades"] for r in rows)
        avg_close = sum(r["close"] for r in rows) / len(rows)
        print(f"  {symbol}:")
        print(f"    Candles : {len(rows):>10,}")
        print(f"    Range   : {first_ts} -> {last_ts}")
        print(f"    Avg close: {avg_close:>14,.2f}")
        print(f"    Total vol: {total_vol:>14,.4f}")
        print(f"    Total trades: {total_trades:>10,}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
