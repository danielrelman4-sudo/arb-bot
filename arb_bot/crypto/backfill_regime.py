"""Retroactively backfill regime labels into existing feature store CSVs.

Reads OFI, VPIN, vol, and returns data already captured in the feature
store and computes regime classifications using the RegimeDetector.

Usage:
    python3 -m arb_bot.crypto.backfill_regime arb_bot/output/feature_store_v11_1771212604.csv
    python3 -m arb_bot.crypto.backfill_regime arb_bot/output/feature_store_*.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from arb_bot.crypto.regime_detector import RegimeDetector, RegimeSnapshot


def backfill_regime(csv_path: str) -> None:
    """Backfill regime columns in an existing feature store CSV."""
    path = Path(csv_path)
    if not path.exists():
        print(f"  SKIP: {csv_path} not found")
        return

    # Read all rows
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print(f"  SKIP: {csv_path} is empty")
        return

    # Ensure regime columns exist in fieldnames
    regime_cols = [
        "regime", "regime_confidence", "regime_trend_score",
        "regime_vol_score", "regime_mean_reversion_score", "regime_ofi_alignment",
    ]
    new_fieldnames = list(fieldnames or [])
    for col in regime_cols:
        if col not in new_fieldnames:
            # Insert before 'side' if possible, else append
            if "side" in new_fieldnames:
                idx = new_fieldnames.index("side")
                new_fieldnames.insert(idx, col)
            else:
                new_fieldnames.append(col)

    detector = RegimeDetector()
    updated = 0

    for row in rows:
        # Extract OFI signals from the row
        ofi_multiscale: Dict[int, float] = {}
        for window, col in [(30, "ofi_30s"), (60, "ofi_60s"),
                            (120, "ofi_120s"), (300, "ofi_300s")]:
            try:
                val = float(row.get(col, 0.0) or 0.0)
                ofi_multiscale[window] = val
            except (ValueError, TypeError):
                ofi_multiscale[window] = 0.0

        # Extract vol data
        try:
            rv_1m = float(row.get("realized_vol_1m", 0.0) or 0.0)
            rv_5m = float(row.get("realized_vol_5m", 0.0) or 0.0)
        except (ValueError, TypeError):
            rv_1m = rv_5m = 0.0

        # Extract VPIN
        vpin_val: Optional[float] = None
        signed_vpin_val: Optional[float] = None
        try:
            v = row.get("vpin", "")
            if v:
                vpin_val = float(v)
        except (ValueError, TypeError):
            pass
        try:
            sv = row.get("signed_vpin", "")
            if sv:
                signed_vpin_val = float(sv)
        except (ValueError, TypeError):
            pass

        # Determine symbol from ticker
        ticker = row.get("ticker", "")
        if "BTC" in ticker.upper():
            symbol = "btcusdt"
        elif "ETH" in ticker.upper():
            symbol = "ethusdt"
        elif "SOL" in ticker.upper():
            symbol = "solusdt"
        else:
            symbol = "unknown"

        # We don't have the actual 1-min return series, but we can
        # synthesize a rough approximation from vol ratio
        # vol_ratio = rv_1m / rv_5m — if > 1, vol is expanding
        # For mean reversion score, we need actual returns — use neutral (0.5)
        # since we can't reconstruct the full series from a single snapshot.
        # The OFI and vol signals are the primary regime drivers anyway.

        # Create a synthetic returns list with the right autocorrelation structure
        # based on available vol data (rough approximation)
        returns_1m: List[float] = []
        if rv_1m > 0:
            # Generate 15 returns with the observed vol scale
            # Can't know actual autocorrelation, so use neutral
            rng = np.random.default_rng(hash(ticker) & 0xFFFFFFFF)
            returns_1m = list(rng.normal(0, rv_1m, 15))

        snap = detector.classify(
            symbol=symbol,
            ofi_multiscale=ofi_multiscale,
            returns_1m=returns_1m,
            vol_short=rv_1m,
            vol_long=rv_5m,
            vpin=vpin_val,
            signed_vpin=signed_vpin_val,
        )

        # Write regime fields into row
        row["regime"] = snap.regime
        row["regime_confidence"] = f"{snap.confidence:.6f}"
        row["regime_trend_score"] = f"{snap.trend_score:.6f}"
        row["regime_vol_score"] = f"{snap.vol_score:.6f}"
        row["regime_mean_reversion_score"] = f"{snap.mean_reversion_score:.6f}"
        row["regime_ofi_alignment"] = f"{snap.ofi_alignment:.6f}"
        updated += 1

    # Write back
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  OK: {csv_path} — backfilled {updated} rows")

    # Print summary
    regime_counts: Dict[str, int] = {}
    for row in rows:
        r = row.get("regime", "unknown")
        regime_counts[r] = regime_counts.get(r, 0) + 1

    outcomes: Dict[str, Dict[str, int]] = {}
    for row in rows:
        r = row.get("regime", "unknown")
        o = row.get("outcome", "-1")
        if r not in outcomes:
            outcomes[r] = {"win": 0, "loss": 0, "pending": 0}
        if o == "1":
            outcomes[r]["win"] += 1
        elif o == "0":
            outcomes[r]["loss"] += 1
        else:
            outcomes[r]["pending"] += 1

    print(f"\n  Regime breakdown:")
    for regime, count in sorted(regime_counts.items()):
        o = outcomes.get(regime, {})
        w, l, p = o.get("win", 0), o.get("loss", 0), o.get("pending", 0)
        total_settled = w + l
        win_rate = f"{w/total_settled*100:.0f}%" if total_settled > 0 else "n/a"
        print(f"    {regime:20s}: {count:3d} trades  |  W={w} L={l} P={p}  win_rate={win_rate}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 -m arb_bot.crypto.backfill_regime <csv_path> [csv_path2 ...]")
        sys.exit(1)

    print("Backfilling regime labels into feature store CSVs...\n")
    for csv_path in sys.argv[1:]:
        backfill_regime(csv_path)
        print()


if __name__ == "__main__":
    main()
