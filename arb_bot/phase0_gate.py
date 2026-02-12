from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUMMARY_RE = re.compile(
    r"paper summary cycles=(?P<cycles>\d+) quotes=(?P<quotes>\d+) opportunities=(?P<opportunities>\d+) "
    r"near=(?P<near>\d+) dry_trades=(?P<dry_trades>\d+) settled=(?P<settled>\d+) "
    r"skipped=(?P<skipped>\d+) simulated_pnl=(?P<sim_pnl>-?\d+(?:\.\d+)?)"
)
SESSION_RE = re.compile(
    r"paper session complete cycles=(?P<cycles>\d+) opportunities=(?P<opportunities>\d+) near=(?P<near>\d+) "
    r"dry_trades=(?P<dry_trades>\d+) settled=(?P<settled>\d+) simulated_pnl=(?P<sim_pnl>-?\d+(?:\.\d+)?)"
)
TIMEOUT_RE = re.compile(r"quote refresh timed out after (?P<seconds>\d+(?:\.\d+)?)s")
STALE_RE = re.compile(r"no stream quote updates for (?P<seconds>\d+(?:\.\d+)?)s")


@dataclass(frozen=True)
class GateThresholds:
    min_cycles: int = 10
    min_detected_rows: int = 200
    max_timeout_rate_per_cycle: float = 0.75
    min_dual_venue_rate: float = 0.5
    min_cross_pairs_covered: int = 20
    min_parity_rules_covered: int = 4
    min_opened_total: int = 3
    min_opened_lanes: int = 2
    lane_open_requirement_detected_min: int = 40


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _segment(text: str, start_key: str, end_keys: list[str]) -> str | None:
    start = text.find(start_key)
    if start < 0:
        return None
    start += len(start_key)
    end = len(text)
    for key in end_keys:
        idx = text.find(key, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def _parse_counts_map(fragment: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    if not fragment:
        return parsed
    for part in fragment.split(","):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = _safe_int(raw_value)
    return parsed


def _parse_coverage_map(fragment: str) -> dict[str, tuple[int, int] | int]:
    parsed: dict[str, tuple[int, int] | int] = {}
    if not fragment:
        return parsed
    for part in fragment.split(","):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        if "/" in raw_value:
            left, right = raw_value.split("/", 1)
            parsed[key] = (_safe_int(left), _safe_int(right))
        else:
            parsed[key] = _safe_int(raw_value)
    return parsed


def analyze_log(log_path: Path) -> dict[str, Any]:
    timeout_count = 0
    timeout_seconds: list[float] = []
    stale_seconds: list[float] = []
    cycle_snapshots = 0
    dual_venue_snapshots = 0
    max_quotes_by_venue: Counter[str] = Counter()
    max_cross_pairs_covered = 0
    max_parity_rules_covered = 0
    max_cross_k_covered = 0
    max_cross_p_covered = 0
    summary: dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            timeout_match = TIMEOUT_RE.search(line)
            if timeout_match:
                timeout_count += 1
                timeout_seconds.append(float(timeout_match.group("seconds")))

            stale_match = STALE_RE.search(line)
            if stale_match:
                stale_seconds.append(float(stale_match.group("seconds")))

            summary_match = SUMMARY_RE.search(line)
            if summary_match:
                summary = {k: _safe_int(v) for k, v in summary_match.groupdict().items() if k != "sim_pnl"}
                summary["sim_pnl"] = float(summary_match.group("sim_pnl"))

            session_match = SESSION_RE.search(line)
            if session_match and "cycles" not in summary:
                summary = {k: _safe_int(v) for k, v in session_match.groupdict().items() if k != "sim_pnl"}
                summary["sim_pnl"] = float(session_match.group("sim_pnl"))

            if "quotes_by_venue=" not in line or "coverage=" not in line:
                continue

            quotes_fragment = _segment(
                line,
                "quotes_by_venue=",
                [" coverage=", " by_kind=", " near_by_kind=", " by_family=", " near_by_family="],
            )
            coverage_fragment = _segment(
                line,
                "coverage=",
                [" by_kind=", " near_by_kind=", " by_family=", " near_by_family=", " cross_parity_split="],
            )
            if quotes_fragment is None or coverage_fragment is None:
                continue

            cycle_snapshots += 1
            quotes_by_venue = _parse_counts_map(quotes_fragment)
            coverage = _parse_coverage_map(coverage_fragment)

            kalshi_quotes = quotes_by_venue.get("kalshi", 0)
            polymarket_quotes = quotes_by_venue.get("polymarket", 0)
            if kalshi_quotes > 0 and polymarket_quotes > 0:
                dual_venue_snapshots += 1

            for venue, count in quotes_by_venue.items():
                if count > max_quotes_by_venue[venue]:
                    max_quotes_by_venue[venue] = count

            cross_pairs = coverage.get("cross_pairs")
            if isinstance(cross_pairs, tuple):
                max_cross_pairs_covered = max(max_cross_pairs_covered, cross_pairs[0])

            parity_rules = coverage.get("parity_rules")
            if isinstance(parity_rules, tuple):
                max_parity_rules_covered = max(max_parity_rules_covered, parity_rules[0])

            cross_k = coverage.get("cross_k")
            if isinstance(cross_k, tuple):
                max_cross_k_covered = max(max_cross_k_covered, cross_k[0])

            cross_p = coverage.get("cross_p")
            if isinstance(cross_p, tuple):
                max_cross_p_covered = max(max_cross_p_covered, cross_p[0])

    return {
        "timeout_count": timeout_count,
        "timeout_seconds_max": max(timeout_seconds) if timeout_seconds else 0.0,
        "stale_seconds_max": max(stale_seconds) if stale_seconds else 0.0,
        "cycle_snapshots": cycle_snapshots,
        "dual_venue_snapshots": dual_venue_snapshots,
        "max_quotes_by_venue": dict(max_quotes_by_venue),
        "max_cross_pairs_covered": max_cross_pairs_covered,
        "max_parity_rules_covered": max_parity_rules_covered,
        "max_cross_k_covered": max_cross_k_covered,
        "max_cross_p_covered": max_cross_p_covered,
        "summary": summary,
    }


def analyze_csv(csv_path: Path) -> dict[str, Any]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    action_counts = Counter(row.get("action", "") for row in rows)
    kind_counts = Counter(row.get("kind", "") for row in rows)
    opened_by_kind = Counter(row.get("kind", "") for row in rows if row.get("action") == "dry_run")
    opened_by_family = Counter(
        row.get("opportunity_family", "") for row in rows if row.get("action") == "dry_run"
    )
    skip_reason_counts = Counter(row.get("reason", "") for row in rows if row.get("action") == "skipped")

    return {
        "row_count": len(rows),
        "action_counts": dict(action_counts),
        "kind_counts": dict(kind_counts),
        "opened_by_kind": dict(opened_by_kind),
        "opened_by_family": dict(opened_by_family),
        "skip_reason_counts": dict(skip_reason_counts),
    }


def evaluate_phase0_gate(
    log_metrics: dict[str, Any],
    csv_metrics: dict[str, Any],
    thresholds: GateThresholds,
) -> list[GateCheck]:
    checks: list[GateCheck] = []

    summary = log_metrics.get("summary", {})
    cycles = _safe_int(summary.get("cycles", 0))
    if cycles <= 0:
        cycles = _safe_int(log_metrics.get("cycle_snapshots", 0))

    detected_rows = _safe_int(csv_metrics.get("row_count", 0))
    timeout_count = _safe_int(log_metrics.get("timeout_count", 0))
    timeout_rate = timeout_count / max(1, cycles)

    snapshots = _safe_int(log_metrics.get("cycle_snapshots", 0))
    dual_snapshots = _safe_int(log_metrics.get("dual_venue_snapshots", 0))
    dual_rate = dual_snapshots / max(1, snapshots)

    max_cross_pairs = _safe_int(log_metrics.get("max_cross_pairs_covered", 0))
    max_parity_rules = _safe_int(log_metrics.get("max_parity_rules_covered", 0))

    action_counts = csv_metrics.get("action_counts", {})
    opened_total = _safe_int(action_counts.get("dry_run", 0))
    opened_by_kind: dict[str, int] = {
        key: _safe_int(value) for key, value in (csv_metrics.get("opened_by_kind", {}) or {}).items()
    }
    opened_lanes = sum(1 for value in opened_by_kind.values() if value > 0)

    kind_counts: dict[str, int] = {
        key: _safe_int(value) for key, value in (csv_metrics.get("kind_counts", {}) or {}).items()
    }

    checks.append(
        GateCheck(
            name="min_cycles",
            passed=cycles >= thresholds.min_cycles,
            detail=f"cycles={cycles} required>={thresholds.min_cycles}",
        )
    )
    checks.append(
        GateCheck(
            name="min_detected_rows",
            passed=detected_rows >= thresholds.min_detected_rows,
            detail=f"detected_rows={detected_rows} required>={thresholds.min_detected_rows}",
        )
    )
    checks.append(
        GateCheck(
            name="max_timeout_rate_per_cycle",
            passed=timeout_rate <= thresholds.max_timeout_rate_per_cycle,
            detail=(
                f"timeout_rate={timeout_rate:.2f} ({timeout_count}/{max(1, cycles)}) "
                f"allowed<={thresholds.max_timeout_rate_per_cycle:.2f}"
            ),
        )
    )
    checks.append(
        GateCheck(
            name="min_dual_venue_rate",
            passed=dual_rate >= thresholds.min_dual_venue_rate,
            detail=(
                f"dual_venue_rate={dual_rate:.2f} ({dual_snapshots}/{max(1, snapshots)}) "
                f"required>={thresholds.min_dual_venue_rate:.2f}"
            ),
        )
    )
    checks.append(
        GateCheck(
            name="min_cross_pairs_covered",
            passed=max_cross_pairs >= thresholds.min_cross_pairs_covered,
            detail=f"cross_pairs_covered={max_cross_pairs} required>={thresholds.min_cross_pairs_covered}",
        )
    )
    checks.append(
        GateCheck(
            name="min_parity_rules_covered",
            passed=max_parity_rules >= thresholds.min_parity_rules_covered,
            detail=f"parity_rules_covered={max_parity_rules} required>={thresholds.min_parity_rules_covered}",
        )
    )
    checks.append(
        GateCheck(
            name="min_opened_total",
            passed=opened_total >= thresholds.min_opened_total,
            detail=f"opened_total={opened_total} required>={thresholds.min_opened_total}",
        )
    )
    checks.append(
        GateCheck(
            name="min_opened_lanes",
            passed=opened_lanes >= thresholds.min_opened_lanes,
            detail=f"opened_lanes={opened_lanes} required>={thresholds.min_opened_lanes}",
        )
    )

    for kind in sorted(kind_counts):
        detected = kind_counts[kind]
        opened = opened_by_kind.get(kind, 0)
        if detected < thresholds.lane_open_requirement_detected_min:
            continue
        checks.append(
            GateCheck(
                name=f"lane_open_requirement::{kind}",
                passed=opened > 0,
                detail=(
                    f"kind={kind} detected={detected} opened={opened} "
                    f"required_open>0 when detected>={thresholds.lane_open_requirement_detected_min}"
                ),
            )
        )

    return checks


def _default_log_for_csv(csv_path: Path) -> Path:
    return csv_path.with_suffix(".log")


def _print_report(
    csv_path: Path,
    log_path: Path,
    thresholds: GateThresholds,
    log_metrics: dict[str, Any],
    csv_metrics: dict[str, Any],
    checks: list[GateCheck],
) -> None:
    passed = all(check.passed for check in checks)
    print(f"Phase 0 gate: {'PASS' if passed else 'FAIL'}")
    print(f"csv={csv_path}")
    print(f"log={log_path}")
    print(
        "metrics "
        f"cycles={log_metrics.get('summary', {}).get('cycles', log_metrics.get('cycle_snapshots', 0))} "
        f"timeouts={log_metrics.get('timeout_count', 0)} "
        f"dual_venue_snapshots={log_metrics.get('dual_venue_snapshots', 0)}/{max(1, log_metrics.get('cycle_snapshots', 0))} "
        f"cross_pairs_covered={log_metrics.get('max_cross_pairs_covered', 0)} "
        f"parity_rules_covered={log_metrics.get('max_parity_rules_covered', 0)} "
        f"opened_total={csv_metrics.get('action_counts', {}).get('dry_run', 0)}"
    )
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase-0 run readiness gates from paper log/csv")
    parser.add_argument("--csv", required=True, help="Path to paper CSV output")
    parser.add_argument("--log", default=None, help="Path to log output (defaults to CSV basename with .log)")
    parser.add_argument("--json-output", default=None, help="Optional path to write JSON report")

    parser.add_argument("--min-cycles", type=int, default=GateThresholds.min_cycles)
    parser.add_argument("--min-detected-rows", type=int, default=GateThresholds.min_detected_rows)
    parser.add_argument("--max-timeout-rate-per-cycle", type=float, default=GateThresholds.max_timeout_rate_per_cycle)
    parser.add_argument("--min-dual-venue-rate", type=float, default=GateThresholds.min_dual_venue_rate)
    parser.add_argument("--min-cross-pairs-covered", type=int, default=GateThresholds.min_cross_pairs_covered)
    parser.add_argument("--min-parity-rules-covered", type=int, default=GateThresholds.min_parity_rules_covered)
    parser.add_argument("--min-opened-total", type=int, default=GateThresholds.min_opened_total)
    parser.add_argument("--min-opened-lanes", type=int, default=GateThresholds.min_opened_lanes)
    parser.add_argument(
        "--lane-open-requirement-detected-min",
        type=int,
        default=GateThresholds.lane_open_requirement_detected_min,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    log_path = Path(args.log) if args.log else _default_log_for_csv(csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        return 2

    thresholds = GateThresholds(
        min_cycles=args.min_cycles,
        min_detected_rows=args.min_detected_rows,
        max_timeout_rate_per_cycle=args.max_timeout_rate_per_cycle,
        min_dual_venue_rate=args.min_dual_venue_rate,
        min_cross_pairs_covered=args.min_cross_pairs_covered,
        min_parity_rules_covered=args.min_parity_rules_covered,
        min_opened_total=args.min_opened_total,
        min_opened_lanes=args.min_opened_lanes,
        lane_open_requirement_detected_min=args.lane_open_requirement_detected_min,
    )

    log_metrics = analyze_log(log_path)
    csv_metrics = analyze_csv(csv_path)
    checks = evaluate_phase0_gate(log_metrics=log_metrics, csv_metrics=csv_metrics, thresholds=thresholds)
    passed = all(check.passed for check in checks)

    _print_report(
        csv_path=csv_path,
        log_path=log_path,
        thresholds=thresholds,
        log_metrics=log_metrics,
        csv_metrics=csv_metrics,
        checks=checks,
    )

    if args.json_output:
        output = {
            "passed": passed,
            "thresholds": asdict(thresholds),
            "checks": [asdict(check) for check in checks],
            "log_metrics": log_metrics,
            "csv_metrics": csv_metrics,
            "csv_path": str(csv_path),
            "log_path": str(log_path),
        }
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
