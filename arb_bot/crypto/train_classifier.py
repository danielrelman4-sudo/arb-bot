"""Offline classifier training script.

Merges feature store CSVs, backfills outcomes from paper run logs,
and trains an XGBoost classifier for model-path or momentum trades.

Supports per-cell training via --cell flag:
    python3 -m arb_bot.crypto.train_classifier --cell yes_15min
    python3 -m arb_bot.crypto.train_classifier --cell no_daily

Usage:
    python3 -m arb_bot.crypto.train_classifier [--strategy model|momentum] [--cell CELL] [--verbose]
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from arb_bot.crypto.feature_store import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    _REGIME_ENCODE,
    get_extended_feature_names,
    load_training_data_from_files,
)


# ── Backfill logic ──────────────────────────────────────────────────────

# Pattern: SETTLED (real) YES KXBTCD-26FEB1618-T68999.99 24@18.50¢ PnL=$-4.4400 outcome=NO
_SETTLED_RE = re.compile(
    r"SETTLED \(real\) (YES|NO) (\S+) \d+@[\d.]+¢ .+ outcome=(YES|NO)"
)


def parse_settlements_from_logs(log_paths: List[str]) -> Dict[str, int]:
    """Parse paper run logs for settlement outcomes.

    Returns
    -------
    Dict mapping ticker → outcome (1=win, 0=loss).
    A trade is a WIN when our_side == actual_outcome.
    """
    settlements: Dict[str, int] = {}

    for log_path in log_paths:
        try:
            with open(log_path, "r") as f:
                for line in f:
                    m = _SETTLED_RE.search(line)
                    if not m:
                        continue
                    our_side, ticker, actual_outcome = m.groups()
                    won = 1 if our_side == actual_outcome else 0
                    settlements[ticker] = won
        except (FileNotFoundError, OSError):
            continue

    return settlements


def backfill_csv_outcomes(
    csv_paths: List[str],
    settlements: Dict[str, int],
) -> int:
    """Update outcome=-1 rows in feature store CSVs with known settlements.

    Returns number of rows updated.
    """
    total_updated = 0

    for csv_path in csv_paths:
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except (FileNotFoundError, OSError):
            continue

        updated = 0
        for row in rows:
            if str(row.get("outcome", "-1")) != "-1":
                continue
            ticker = row.get("ticker", "")
            if ticker in settlements:
                row["outcome"] = str(settlements[ticker])
                updated += 1

        if updated > 0:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            total_updated += updated

    return total_updated


# ── Training ────────────────────────────────────────────────────────────


def train_and_report(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[dict],
    feature_names: List[str],
    output_path: str,
    verbose: bool = False,
) -> None:
    """Train classifier and print detailed report."""
    from arb_bot.crypto.classifier import BinaryClassifier, HAS_XGBOOST

    if not HAS_XGBOOST:
        print("ERROR: XGBoost not installed. Run: pip3 install xgboost")
        sys.exit(1)

    n_samples = len(y)
    n_wins = int(np.sum(y))
    n_losses = n_samples - n_wins

    print(f"\n{'='*60}")
    print(f"  Classifier Training")
    print(f"{'='*60}")
    print(f"  Samples: {n_samples} ({n_wins} wins, {n_losses} losses, {n_wins/n_samples*100:.0f}% win rate)")

    # Per-cell breakdown
    cell_counts: Dict[str, Dict[str, int]] = {}
    for i, meta in enumerate(metadata):
        cell = meta.get("strategy_cell", "unknown") or "unknown"
        if cell not in cell_counts:
            cell_counts[cell] = {"total": 0, "wins": 0}
        cell_counts[cell]["total"] += 1
        if y[i] == 1:
            cell_counts[cell]["wins"] += 1

    print(f"\n  Per-cell breakdown:")
    for cell in sorted(cell_counts):
        s = cell_counts[cell]
        wr = s["wins"] / s["total"] * 100 if s["total"] else 0
        print(f"    {cell:15s}  {s['total']:3d} trades  {s['wins']:3d} wins ({wr:4.0f}%)")

    # Train
    print(f"\n  Training XGBoost (80/20 split, isotonic calibration)...")

    # Use conservative hyperparams for small dataset
    clf = BinaryClassifier(
        max_depth=3,             # Shallow trees — small dataset
        n_estimators=50,         # Fewer trees — prevent overfit
        learning_rate=0.1,
        min_child_weight=3,      # Regularization
        subsample=0.8,
        use_isotonic_calibration=True,
        model_path=output_path,
    )

    report = clf.train(X, y, feature_names=feature_names)

    print(f"  Accuracy:   {report.accuracy*100:.1f}%")
    print(f"  Brier:      {report.brier_score:.4f}")
    print(f"  Log loss:   {report.log_loss:.4f}")

    # AUC-ROC if sklearn available
    try:
        from sklearn.metrics import roc_auc_score
        # Re-predict on full dataset for AUC (informational only — includes train data)
        all_probs = np.array([clf.predict(X[i]).probability for i in range(len(X))])
        auc = roc_auc_score(y, all_probs)
        print(f"  AUC-ROC:    {auc:.3f} (full dataset, informational)")
    except Exception:
        pass

    # Feature importances
    if report.feature_importances:
        sorted_imp = sorted(report.feature_importances.items(), key=lambda x: -x[1])
        print(f"\n  Top 15 Feature Importances:")
        for rank, (name, imp) in enumerate(sorted_imp[:15], 1):
            bar = "█" * int(imp * 100)
            print(f"    {rank:2d}. {name:30s} {imp:.3f} {bar}")

    # Per-cell validation accuracy (approximate — based on full predictions)
    print(f"\n  Per-cell predictions (full dataset):")
    for cell in sorted(cell_counts):
        cell_idx = [i for i, m in enumerate(metadata) if (m.get("strategy_cell", "") or "unknown") == cell]
        if not cell_idx:
            continue
        cell_y = y[cell_idx]
        cell_preds = np.array([clf.predict(X[i]).probability for i in cell_idx])
        cell_pred_binary = (cell_preds >= 0.5).astype(int)
        cell_acc = np.mean(cell_pred_binary == cell_y)
        cell_avg_prob = np.mean(cell_preds)
        print(f"    {cell:15s}  acc={cell_acc*100:.0f}%  avg_P(win)={cell_avg_prob:.2f}  n={len(cell_idx)}")

    # Save model
    print(f"\n  Model saved: {output_path}")
    if Path(output_path).with_suffix(".iso.npz").exists():
        print(f"  Isotonic calibration saved: {Path(output_path).with_suffix('.iso.npz')}")

    print(f"{'='*60}\n")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crypto classifier offline")
    parser.add_argument(
        "--strategy", choices=["model", "momentum"], default="model",
        help="Strategy to filter training data (default: model)"
    )
    parser.add_argument(
        "--data-dir", default="arb_bot/output",
        help="Directory containing feature_store_v11_*.csv files"
    )
    parser.add_argument(
        "--cell",
        choices=["yes_15min", "yes_daily", "no_15min", "no_daily"],
        default=None,
        help="Train per-cell classifier (filters data to single strategy cell)"
    )
    parser.add_argument(
        "--output", default="",
        help="Output model path (default: arb_bot/output/classifier_model_<strategy>[_<cell>].json)"
    )
    parser.add_argument(
        "--backfill", action="store_true", default=True,
        help="Backfill unsettled outcomes from paper run logs (default: True)"
    )
    parser.add_argument(
        "--no-backfill", action="store_false", dest="backfill",
        help="Skip backfill step"
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output:
        output_path = args.output
    elif args.cell:
        output_path = str(data_dir / f"classifier_model_{args.strategy}_{args.cell}.json")
    else:
        output_path = str(data_dir / f"classifier_model_{args.strategy}.json")

    # Find feature store CSVs
    csv_pattern = str(data_dir / "feature_store_v11_*.csv")
    csv_paths = sorted(glob.glob(csv_pattern))
    if not csv_paths:
        print(f"No feature store CSVs found matching: {csv_pattern}")
        sys.exit(1)

    print(f"Found {len(csv_paths)} feature store CSV files in {data_dir}")

    # Step 1: Backfill outcomes from paper run logs
    if args.backfill:
        log_pattern = str(data_dir / "paper_run_*.log")
        log_paths = sorted(glob.glob(log_pattern))
        if log_paths:
            print(f"Parsing {len(log_paths)} paper run logs for settlements...")
            settlements = parse_settlements_from_logs(log_paths)
            print(f"  Found {len(settlements)} settlements in logs")

            updated = backfill_csv_outcomes(csv_paths, settlements)
            print(f"  Backfilled {updated} unsettled rows in feature store CSVs")
        else:
            print("No paper run logs found for backfill")

    # Step 2: Load merged training data
    feature_names = get_extended_feature_names()
    X, y, metadata = load_training_data_from_files(csv_paths, strategy_filter=args.strategy)

    if len(y) == 0:
        print(f"No settled {args.strategy}-path trades found. Need more data.")
        sys.exit(1)

    print(f"Loaded {len(y)} settled {args.strategy}-path trades")

    # Step 2b: Filter to single strategy cell if --cell is provided
    if args.cell:
        cell_mask = [
            i for i, m in enumerate(metadata)
            if (m.get("strategy_cell", "") or "") == args.cell
        ]
        if len(cell_mask) < 20:
            print(f"\nWARNING: Cell '{args.cell}' has only {len(cell_mask)} samples "
                  f"(need >= 20). Skipping training.")
            print(f"Available cells:")
            cell_counts: Dict[str, int] = {}
            for m in metadata:
                c = m.get("strategy_cell", "unknown") or "unknown"
                cell_counts[c] = cell_counts.get(c, 0) + 1
            for c in sorted(cell_counts):
                print(f"  {c}: {cell_counts[c]} samples")
            sys.exit(0)

        X = X[cell_mask]
        y = y[cell_mask]
        metadata = [metadata[i] for i in cell_mask]
        print(f"Filtered to cell '{args.cell}': {len(y)} samples")

    # Step 3: Train and report
    train_and_report(X, y, metadata, feature_names, output_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
