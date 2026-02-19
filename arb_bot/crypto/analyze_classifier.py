"""Comprehensive classifier analysis script.

Runs stratified k-fold CV, calibration analysis, feature importance
comparison, correlation analysis, learning curves, and dead feature
detection across all strategy cells.

Usage:
    python3 -m arb_bot.crypto.analyze_classifier [--data-dir DIR] [--cell CELL] [--k-folds K] [--verbose]
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# sklearn imports
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss as sklearn_log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.inspection import permutation_importance

import xgboost as xgb

# Reuse existing infrastructure
from arb_bot.crypto.feature_store import (
    FEATURE_COLUMNS,
    get_extended_feature_names,
    load_training_data_from_files,
)
from arb_bot.crypto.train_classifier import (
    backfill_csv_outcomes,
    parse_settlements_from_logs,
)


# ── Data Types ─────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    """Metrics from a single CV fold."""

    accuracy: float = 0.0
    auc_roc: float = 0.0
    brier: float = 0.0
    log_loss_val: float = 0.0
    precision_0: float = 0.0  # Loss class
    recall_0: float = 0.0
    f1_0: float = 0.0
    precision_1: float = 0.0  # Win class
    recall_1: float = 0.0
    f1_1: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=int))
    y_true: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_prob: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class CVSummary:
    """Aggregated cross-validation results."""

    folds: List[FoldResult] = field(default_factory=list)
    label: str = "global"
    n_samples: int = 0
    n_wins: int = 0


# ── Model Factory ─────────────────────────────────────────────────────


def make_xgb_model() -> xgb.XGBClassifier:
    """Create XGBClassifier with production hyperparameters.

    These match exactly what train_classifier.py uses (lines 154-160).
    """
    return xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )


# ── Cross-Validation ──────────────────────────────────────────────────


def run_stratified_kfold(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    feature_names: List[str],
) -> CVSummary:
    """Run stratified k-fold CV and collect per-fold metrics.

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features)
    y : Binary labels
    k : Number of folds
    feature_names : Feature name list (for verbose logging)

    Returns CVSummary with all fold results.
    """
    n_samples = len(y)

    # Auto-reduce k for small datasets
    effective_k = k
    if n_samples < k * 5:
        effective_k = max(2, n_samples // 5)

    skf = StratifiedKFold(n_splits=effective_k, shuffle=True, random_state=42)
    folds: List[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_xgb_model()
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Core metrics
        acc = float(accuracy_score(y_test, y_pred))
        brier = float(brier_score_loss(y_test, y_prob))
        ll = float(sklearn_log_loss(y_test, y_prob))

        # AUC-ROC (handle single-class edge case)
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = float("nan")

        # Per-class precision/recall/F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0.0
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        folds.append(FoldResult(
            accuracy=acc,
            auc_roc=auc,
            brier=brier,
            log_loss_val=ll,
            precision_0=float(prec[0]),
            recall_0=float(rec[0]),
            f1_0=float(f1[0]),
            precision_1=float(prec[1]),
            recall_1=float(rec[1]),
            f1_1=float(f1[1]),
            confusion=cm,
            y_true=y_test,
            y_prob=y_prob,
        ))

    return CVSummary(
        folds=folds,
        label="global",
        n_samples=n_samples,
        n_wins=int(np.sum(y)),
    )


# ── Calibration Analysis ──────────────────────────────────────────────


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Tuple[float, float, int]]]:
    """Compute Expected and Maximum Calibration Error.

    Returns (ece, mce, bin_details) where bin_details is list of
    (mean_predicted, fraction_positive, count) tuples.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    mce = 0.0
    bin_details: List[Tuple[float, float, int]] = []
    total = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_pred = float(np.mean(y_prob[mask]))
        avg_true = float(np.mean(y_true[mask]))
        gap = abs(avg_pred - avg_true)
        ece += gap * count / total
        mce = max(mce, gap)
        bin_details.append((avg_pred, avg_true, count))

    return ece, mce, bin_details


# ── Feature Importance ─────────────────────────────────────────────────


def compute_permutation_importance_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Train model on 80/20 split, compute XGBoost gain + permutation importance.

    Returns (xgb_importances, perm_importances) where:
        xgb_importances: feature_name -> gain importance
        perm_importances: feature_name -> (mean_importance, std_importance)
    """
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = make_xgb_model()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # XGBoost native importances
    xgb_imp: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(feature_names, model.feature_importances_):
            xgb_imp[name] = float(imp)

    # Permutation importance on holdout
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring="accuracy",
        random_state=42,
    )
    perm_imp: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(feature_names):
        perm_imp[name] = (
            float(result.importances_mean[i]),
            float(result.importances_std[i]),
        )

    return xgb_imp, perm_imp


# ── Dead Features ──────────────────────────────────────────────────────


def detect_dead_features(
    X: np.ndarray,
    feature_names: List[str],
) -> List[Tuple[str, str]]:
    """Identify features with zero variance or always zero.

    Returns list of (feature_name, reason) tuples.
    """
    dead: List[Tuple[str, str]] = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.all(col == 0.0):
            dead.append((name, "always_zero"))
        elif np.var(col) == 0.0:
            dead.append((name, f"constant={col[0]:.4f}"))
        elif np.var(col) < 1e-10:
            dead.append((name, f"near_zero_var={np.var(col):.2e}"))
    return dead


# ── Feature Correlations ──────────────────────────────────────────────


def compute_feature_correlations(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """Find feature pairs with |Pearson r| > threshold.

    Returns list of (feat_a, feat_b, correlation) sorted by |r| descending.
    """
    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    valid_indices = np.where(valid_mask)[0]
    valid_names = [feature_names[i] for i in valid_indices]

    if len(valid_indices) < 2:
        return []

    corr_matrix = np.corrcoef(X[:, valid_mask].T)

    pairs: List[Tuple[str, str, float]] = []
    n = len(valid_names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if np.isnan(r) or np.isinf(r):
                continue
            if abs(r) > threshold:
                pairs.append((valid_names[i], valid_names[j], float(r)))

    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


# ── Learning Curve ─────────────────────────────────────────────────────


def run_learning_curve_analysis(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
) -> List[Tuple[int, float, float]]:
    """Compute learning curve at various data fractions.

    Returns list of (n_samples, mean_cv_accuracy, std_cv_accuracy).
    """
    fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Adjust k for small datasets
    effective_k = k
    min_samples_per_fold = int(len(y) * fractions[0] * (1 - 1 / k))
    if min_samples_per_fold < 5:
        effective_k = max(2, int(len(y) * fractions[0]) // 5)

    model = make_xgb_model()
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=fractions,
            cv=StratifiedKFold(n_splits=effective_k, shuffle=True, random_state=42),
            scoring="accuracy",
            random_state=42,
        )
    except ValueError:
        # May fail on very small datasets
        return []

    results: List[Tuple[int, float, float]] = []
    for i in range(len(train_sizes)):
        results.append((
            int(train_sizes[i]),
            float(np.mean(test_scores[i])),
            float(np.std(test_scores[i])),
        ))

    return results


# ── Output Formatting ──────────────────────────────────────────────────


def print_section_header(title: str) -> None:
    """Print a visually distinct section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[dict],
) -> None:
    """Print dataset summary with per-cell breakdown."""
    n_samples = len(y)
    n_wins = int(np.sum(y))
    n_losses = n_samples - n_wins
    wr = n_wins / n_samples * 100 if n_samples else 0

    print_section_header(f"Dataset Summary")
    print(f"  Total samples: {n_samples} ({n_wins} wins, {n_losses} losses, {wr:.1f}% win rate)")
    print(f"  Features: {X.shape[1]}")

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
    print(f"    {'Cell':15s}  {'Trades':>7s}  {'Wins':>5s}  {'Win Rate':>8s}")
    print(f"    {'─' * 15}  {'─' * 7}  {'─' * 5}  {'─' * 8}")
    for cell in sorted(cell_counts):
        s = cell_counts[cell]
        wr_cell = s["wins"] / s["total"] * 100 if s["total"] else 0
        print(f"    {cell:15s}  {s['total']:7d}  {s['wins']:5d}  {wr_cell:7.1f}%")


def print_cv_summary(summary: CVSummary) -> None:
    """Print cross-validation results as a formatted table."""
    folds = summary.folds
    if not folds:
        print("  No fold results available.")
        return

    k = len(folds)

    def _stats(vals: List[float]) -> Tuple[float, float, float, float]:
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            return (float("nan"),) * 4
        return (
            float(np.mean(clean)),
            float(np.std(clean)),
            float(np.min(clean)),
            float(np.max(clean)),
        )

    metrics = {
        "Accuracy": [f.accuracy for f in folds],
        "AUC-ROC": [f.auc_roc for f in folds],
        "Brier Score": [f.brier for f in folds],
        "Log Loss": [f.log_loss_val for f in folds],
    }

    print(f"\n  Stratified {k}-Fold CV: {summary.label}")
    print(f"  ({summary.n_samples} samples, {summary.n_wins} wins, "
          f"{summary.n_samples - summary.n_wins} losses)")
    print(f"\n  {'Metric':15s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}")
    print(f"  {'─' * 15}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}")

    for name, vals in metrics.items():
        mean, std, mn, mx = _stats(vals)
        print(f"  {name:15s}  {mean:7.3f}  {std:7.3f}  {mn:7.3f}  {mx:7.3f}")

    # Per-class metrics
    print(f"\n  Per-Class Metrics (averaged across folds):")
    print(f"  {'Class':8s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}")
    print(f"  {'─' * 8}  {'─' * 9}  {'─' * 7}  {'─' * 7}")

    avg_p0 = float(np.mean([f.precision_0 for f in folds]))
    avg_r0 = float(np.mean([f.recall_0 for f in folds]))
    avg_f0 = float(np.mean([f.f1_0 for f in folds]))
    avg_p1 = float(np.mean([f.precision_1 for f in folds]))
    avg_r1 = float(np.mean([f.recall_1 for f in folds]))
    avg_f1 = float(np.mean([f.f1_1 for f in folds]))

    print(f"  {'Loss':8s}  {avg_p0:9.3f}  {avg_r0:7.3f}  {avg_f0:7.3f}")
    print(f"  {'Win':8s}  {avg_p1:9.3f}  {avg_r1:7.3f}  {avg_f1:7.3f}")

    # Confusion matrix (summed across folds)
    cm_total = np.sum([f.confusion for f in folds], axis=0)
    print(f"\n  Confusion Matrix (summed across folds):")
    print(f"  {'':18s}  {'Pred Loss':>10s}  {'Pred Win':>10s}")
    print(f"  {'─' * 18}  {'─' * 10}  {'─' * 10}")
    print(f"  {'Actual Loss':18s}  {cm_total[0, 0]:10d}  {cm_total[0, 1]:10d}")
    print(f"  {'Actual Win':18s}  {cm_total[1, 0]:10d}  {cm_total[1, 1]:10d}")


def print_calibration_analysis(
    ece: float,
    mce: float,
    bin_details: List[Tuple[float, float, int]],
) -> None:
    """Print calibration metrics and text reliability diagram."""
    print(f"\n  Calibration Analysis")
    print(f"  ECE = {ece:.4f}  (Expected Calibration Error — lower is better)")
    print(f"  MCE = {mce:.4f}  (Maximum Calibration Error)")

    if bin_details:
        print(f"\n  {'Bin':>5s}  {'Pred':>6s}  {'Actual':>7s}  {'Count':>6s}  {'Gap':>6s}")
        print(f"  {'─' * 5}  {'─' * 6}  {'─' * 7}  {'─' * 6}  {'─' * 6}")
        for avg_pred, avg_true, count in bin_details:
            gap = abs(avg_pred - avg_true)
            flag = " ***" if gap > 0.1 else ""
            print(f"  {avg_pred:5.2f}  {avg_pred:6.3f}  {avg_true:7.3f}  {count:6d}  {gap:6.3f}{flag}")

    # Text reliability diagram (20x10 grid)
    if bin_details:
        print(f"\n  Reliability Diagram (. = actual, / = perfect calibration):")
        grid_h, grid_w = 10, 20
        grid = [[" "] * grid_w for _ in range(grid_h)]

        # Draw perfect calibration diagonal
        for i in range(grid_w):
            row = grid_h - 1 - int(i / grid_w * grid_h)
            row = max(0, min(grid_h - 1, row))
            grid[row][i] = "/"

        # Plot actual calibration points
        for avg_pred, avg_true, count in bin_details:
            col = int(avg_pred * (grid_w - 1))
            row = grid_h - 1 - int(avg_true * (grid_h - 1))
            col = max(0, min(grid_w - 1, col))
            row = max(0, min(grid_h - 1, row))
            grid[row][col] = "*"

        print(f"  1.0 |{''.join(grid[0])}|")
        for r in range(1, grid_h - 1):
            label = f"{1.0 - r / (grid_h - 1):.1f}"
            print(f"  {label} |{''.join(grid[r])}|")
        print(f"  0.0 |{''.join(grid[-1])}|")
        print(f"       {'─' * grid_w}")
        print(f"       0.0{'':>{grid_w - 6}}1.0")


def print_feature_importance_comparison(
    xgb_imp: Dict[str, float],
    perm_imp: Dict[str, Tuple[float, float]],
    feature_names: List[str],
) -> None:
    """Print side-by-side XGBoost gain vs permutation importance."""
    print_section_header("Feature Importance: XGBoost Gain vs Permutation")

    # Rank by XGBoost gain
    xgb_sorted = sorted(xgb_imp.items(), key=lambda x: -x[1])
    # Rank by permutation mean
    perm_sorted = sorted(perm_imp.items(), key=lambda x: -x[1][0])

    # Build rank maps
    xgb_rank = {name: rank for rank, (name, _) in enumerate(xgb_sorted, 1)}
    perm_rank = {name: rank for rank, (name, _) in enumerate(perm_sorted, 1)}

    # Top 20 by XGBoost gain
    print(f"\n  Top 20 by XGBoost Gain:")
    print(f"  {'Rank':>4s}  {'Feature':30s}  {'Gain':>7s}  {'Bar':20s}  {'Perm Rank':>9s}  {'Perm Mean':>9s}")
    print(f"  {'─' * 4}  {'─' * 30}  {'─' * 7}  {'─' * 20}  {'─' * 9}  {'─' * 9}")
    max_gain = xgb_sorted[0][1] if xgb_sorted else 1.0
    for rank, (name, gain) in enumerate(xgb_sorted[:20], 1):
        bar_len = int(gain / max_gain * 20) if max_gain > 0 else 0
        bar = "█" * bar_len
        p_rank = perm_rank.get(name, -1)
        p_mean = perm_imp.get(name, (0, 0))[0]
        rank_delta = abs(rank - p_rank)
        arrow = " ↕" if rank_delta > 5 else ""
        print(f"  {rank:4d}  {name:30s}  {gain:7.4f}  {bar:20s}  {p_rank:9d}  {p_mean:9.4f}{arrow}")

    # Top 20 by permutation importance
    print(f"\n  Top 20 by Permutation Importance:")
    print(f"  {'Rank':>4s}  {'Feature':30s}  {'Perm Mean':>9s}  {'Perm Std':>8s}  {'XGB Rank':>8s}")
    print(f"  {'─' * 4}  {'─' * 30}  {'─' * 9}  {'─' * 8}  {'─' * 8}")
    for rank, (name, (mean, std)) in enumerate(perm_sorted[:20], 1):
        x_rank = xgb_rank.get(name, -1)
        print(f"  {rank:4d}  {name:30s}  {mean:9.4f}  {std:8.4f}  {x_rank:8d}")

    # Rank correlation (Spearman)
    common = [n for n in feature_names if n in xgb_rank and n in perm_rank]
    if len(common) >= 5:
        from scipy.stats import spearmanr
        xgb_ranks = [xgb_rank[n] for n in common]
        perm_ranks = [perm_rank[n] for n in common]
        rho, pval = spearmanr(xgb_ranks, perm_ranks)
        print(f"\n  Spearman Rank Correlation: ρ = {rho:.3f} (p = {pval:.4f})")

    # Top-10 overlap
    xgb_top10 = set(n for n, _ in xgb_sorted[:10])
    perm_top10 = set(n for n, _ in perm_sorted[:10])
    overlap = xgb_top10 & perm_top10
    print(f"  Top-10 Overlap: {len(overlap)}/10 features shared")
    if overlap:
        print(f"    Shared: {', '.join(sorted(overlap))}")

    # Features with high gain but negative permutation importance (overfitting signal)
    overfit_suspects = [
        (name, gain, perm_imp[name][0])
        for name, gain in xgb_sorted[:20]
        if perm_imp.get(name, (0, 0))[0] <= 0
    ]
    if overfit_suspects:
        print(f"\n  ⚠ Potential Overfitting (high gain, ≤0 permutation importance):")
        for name, gain, perm in overfit_suspects:
            print(f"    {name:30s}  gain={gain:.4f}  perm={perm:.4f}")


def print_dead_features(dead: List[Tuple[str, str]]) -> None:
    """Print dead/low-variance features."""
    print_section_header(f"Dead/Low-Variance Features ({len(dead)} found)")
    if not dead:
        print("  None — all features have non-zero variance.")
        return
    for name, reason in dead:
        print(f"  {name:35s}  {reason}")


def print_correlation_pairs(pairs: List[Tuple[str, str, float]]) -> None:
    """Print highly correlated feature pairs."""
    print_section_header(f"Highly Correlated Feature Pairs (|r| > 0.80)")
    if not pairs:
        print("  None found.")
        return
    print(f"  {'Feature A':30s}  {'Feature B':30s}  {'r':>7s}")
    print(f"  {'─' * 30}  {'─' * 30}  {'─' * 7}")
    for feat_a, feat_b, r in pairs:
        print(f"  {feat_a:30s}  {feat_b:30s}  {r:7.3f}")

    # Suggest removals
    seen = set()
    candidates = set()
    for feat_a, feat_b, r in pairs:
        if feat_a not in seen and feat_b not in seen:
            candidates.add(feat_b)
        seen.add(feat_a)
        seen.add(feat_b)
    if candidates:
        print(f"\n  Removal candidates (keep first in pair, drop second):")
        for name in sorted(candidates):
            print(f"    - {name}")


def print_learning_curve(results: List[Tuple[int, float, float]]) -> None:
    """Print learning curve results."""
    print_section_header("Learning Curve")
    if not results:
        print("  Could not compute learning curve (too few samples).")
        return

    print(f"  {'Samples':>8s}  {'CV Accuracy':>11s}  {'Std':>7s}  {'Trend':20s}")
    print(f"  {'─' * 8}  {'─' * 11}  {'─' * 7}  {'─' * 20}")

    for i, (n_samples, mean_acc, std_acc) in enumerate(results):
        bar_len = int(mean_acc * 20)
        bar = "▓" * bar_len + "░" * (20 - bar_len)
        print(f"  {n_samples:8d}  {mean_acc:11.4f}  {std_acc:7.4f}  {bar}")

    # Trend analysis
    if len(results) >= 3:
        first_acc = results[0][1]
        last_acc = results[-1][1]
        delta = last_acc - first_acc
        pct_change = delta / first_acc * 100 if first_acc > 0 else 0

        mid = len(results) // 2
        first_half_delta = results[mid][1] - results[0][1]
        second_half_delta = results[-1][1] - results[mid][1]

        if delta > 0.01:
            if second_half_delta > first_half_delta * 0.5:
                assessment = "Still climbing — more data should help"
            else:
                assessment = "Growth decelerating — approaching plateau"
        elif delta > 0:
            assessment = "Near plateau — marginal gains from more data"
        else:
            assessment = "Flat or declining — more data unlikely to help"

        print(f"\n  Overall: {first_acc:.3f} → {last_acc:.3f} ({delta:+.3f}, {pct_change:+.1f}%)")
        print(f"  Assessment: {assessment}")


# ── Analysis Runner ────────────────────────────────────────────────────


class AnalysisRunner:
    """Orchestrates the full analysis pipeline."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: List[dict],
        feature_names: List[str],
        k_folds: int = 5,
        verbose: bool = False,
    ) -> None:
        self.X = X
        self.y = y
        self.metadata = metadata
        self.feature_names = feature_names
        self.k_folds = k_folds
        self.verbose = verbose

    def run_all(self, cell_filter: Optional[str] = None) -> None:
        """Run the complete analysis pipeline."""
        # 0. Dataset summary
        print_dataset_summary(self.X, self.y, self.metadata)

        # 1. Dead feature detection (fast)
        dead = detect_dead_features(self.X, self.feature_names)
        print_dead_features(dead)

        # 2. Feature correlations
        pairs = compute_feature_correlations(self.X, self.feature_names, threshold=0.8)
        print_correlation_pairs(pairs)

        # 3. Stratified K-Fold CV (global)
        print_section_header(
            f"Stratified {self.k_folds}-Fold Cross-Validation: Global"
        )
        global_summary = run_stratified_kfold(
            self.X, self.y, self.k_folds, self.feature_names,
        )
        print_cv_summary(global_summary)

        # 4. Calibration analysis (using all CV fold predictions)
        all_y_true = np.concatenate([f.y_true for f in global_summary.folds])
        all_y_prob = np.concatenate([f.y_prob for f in global_summary.folds])
        ece, mce, bin_details = compute_calibration_metrics(all_y_true, all_y_prob)
        print_calibration_analysis(ece, mce, bin_details)

        # 5. Feature importance comparison
        print("\n  Computing feature importance (XGBoost gain + permutation)...")
        xgb_imp, perm_imp = compute_permutation_importance_report(
            self.X, self.y, self.feature_names,
        )
        print_feature_importance_comparison(xgb_imp, perm_imp, self.feature_names)

        # 6. Learning curve
        print("\n  Computing learning curve...")
        lc_results = run_learning_curve_analysis(self.X, self.y, self.k_folds)
        print_learning_curve(lc_results)

        # 7. Per-cell deep dive (unless --cell already filtered)
        if cell_filter is None:
            self._run_per_cell_analysis()

    def _run_per_cell_analysis(self) -> None:
        """Run full analysis for each strategy cell separately."""
        cells: Dict[str, List[int]] = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            cell = meta.get("strategy_cell", "unknown") or "unknown"
            cells[cell].append(i)

        for cell_name in sorted(cells.keys()):
            indices = cells[cell_name]
            X_cell = self.X[indices]
            y_cell = self.y[np.array(indices)]

            n_samples = len(indices)
            n_wins = int(np.sum(y_cell))
            wr = n_wins / n_samples * 100 if n_samples else 0

            print_section_header(
                f"Per-Cell Deep Dive: {cell_name} "
                f"({n_samples} samples, {n_wins} wins, {wr:.0f}% WR)"
            )

            if n_samples < 20:
                print(f"  SKIPPED: Only {n_samples} samples (need >= 20)")
                continue

            # Adjust k for small cells
            k = min(self.k_folds, max(2, n_samples // 10))

            # CV
            summary = run_stratified_kfold(X_cell, y_cell, k, self.feature_names)
            summary.label = cell_name
            summary.n_samples = n_samples
            summary.n_wins = n_wins
            print_cv_summary(summary)

            # Calibration
            all_y_true = np.concatenate([f.y_true for f in summary.folds])
            all_y_prob = np.concatenate([f.y_prob for f in summary.folds])
            if len(all_y_true) >= 20:
                ece, mce, bins = compute_calibration_metrics(all_y_true, all_y_prob)
                print_calibration_analysis(ece, mce, bins)

            # Dead features specific to this cell
            cell_dead = detect_dead_features(X_cell, self.feature_names)
            extra_dead = [
                (n, r) for n, r in cell_dead
                if n not in {d[0] for d in detect_dead_features(self.X, self.feature_names)}
            ]
            if extra_dead:
                print(f"\n  Additional dead features in {cell_name}:")
                for name, reason in extra_dead:
                    print(f"    {name:35s}  {reason}")


# ── CLI Entry Point ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive classifier analysis for crypto strategy cells"
    )
    parser.add_argument(
        "--data-dir", default="arb_bot/output",
        help="Directory containing feature_store_v11_*.csv files",
    )
    parser.add_argument(
        "--cell",
        choices=["yes_15min", "yes_daily", "no_15min", "no_daily"],
        default=None,
        help="Analyze a single strategy cell only",
    )
    parser.add_argument(
        "--k-folds", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--strategy", choices=["model", "momentum"], default="model",
        help="Strategy to filter training data (default: model)",
    )
    parser.add_argument(
        "--backfill", action="store_true", default=True,
        help="Backfill outcomes from paper run logs first",
    )
    parser.add_argument(
        "--no-backfill", action="store_false", dest="backfill",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Find CSVs
    csv_pattern = str(data_dir / "feature_store_v11_*.csv")
    csv_paths = sorted(glob.glob(csv_pattern))
    if not csv_paths:
        print(f"No feature store CSVs found: {csv_pattern}")
        sys.exit(1)

    print(f"Found {len(csv_paths)} feature store CSV files in {data_dir}")

    # Backfill
    if args.backfill:
        log_pattern = str(data_dir / "paper_run_*.log")
        log_paths = sorted(glob.glob(log_pattern))
        if log_paths:
            print(f"Parsing {len(log_paths)} paper run logs for settlements...")
            settlements = parse_settlements_from_logs(log_paths)
            print(f"  Found {len(settlements)} settlements in logs")
            updated = backfill_csv_outcomes(csv_paths, settlements)
            if updated > 0:
                print(f"  Backfilled {updated} rows")
        else:
            print("No paper run logs found for backfill")

    # Load data
    feature_names = get_extended_feature_names()
    X, y, metadata = load_training_data_from_files(
        csv_paths, strategy_filter=args.strategy,
    )

    if len(y) == 0:
        print(f"No settled {args.strategy}-path trades found. Need more data.")
        sys.exit(1)

    print(f"Loaded {len(y)} settled {args.strategy}-path trades")

    # Filter to cell if requested
    if args.cell:
        cell_mask = [
            i for i, m in enumerate(metadata)
            if (m.get("strategy_cell", "") or "") == args.cell
        ]
        if len(cell_mask) < 20:
            print(f"Cell '{args.cell}' has only {len(cell_mask)} samples (need >= 20)")
            # Show available cells
            cell_counts: Dict[str, int] = {}
            for m in metadata:
                c = m.get("strategy_cell", "unknown") or "unknown"
                cell_counts[c] = cell_counts.get(c, 0) + 1
            print("Available cells:")
            for c in sorted(cell_counts):
                print(f"  {c}: {cell_counts[c]} samples")
            sys.exit(1)
        X = X[cell_mask]
        y = y[cell_mask]
        metadata = [metadata[i] for i in cell_mask]
        print(f"Filtered to cell '{args.cell}': {len(y)} samples")

    # Run analysis
    runner = AnalysisRunner(
        X, y, metadata, feature_names,
        k_folds=args.k_folds, verbose=args.verbose,
    )
    runner.run_all(cell_filter=args.cell)

    print(f"\n{'=' * 70}")
    print(f"  Analysis complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
