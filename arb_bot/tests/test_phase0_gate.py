from __future__ import annotations

import csv
from pathlib import Path

from arb_bot.phase0_gate import GateThresholds, analyze_csv, analyze_log, evaluate_phase0_gate


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["action", "kind", "opportunity_family", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_analyze_log_extracts_coverage_and_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-02-11 10:00:00 | INFO | arb_bot.engine | Found 3 opportunities "
                "(source=paper-stream-poll-refresh scan=full quotes_by_venue=kalshi=120,polymarket=220 "
                "coverage=cross_pairs=21/1714,cross_k=190/1714,cross_p=178/1714,parity_rules=5/12,parity_mkts=16/24 by_kind=intra_venue=3)",
                "2026-02-11 10:01:00 | WARNING | arb_bot.paper | paper quote refresh timed out after 5.0s (source=paper-stream-poll-refresh)",
                "2026-02-11 10:02:00 | INFO | __main__ | paper summary cycles=12 quotes=400 opportunities=300 near=1200 dry_trades=6 settled=0 skipped=294 simulated_pnl=0.00 csv=foo.csv",
            ]
        ),
        encoding="utf-8",
    )

    metrics = analyze_log(log_path)
    assert metrics["timeout_count"] == 1
    assert metrics["max_cross_pairs_covered"] == 21
    assert metrics["max_parity_rules_covered"] == 5
    assert metrics["dual_venue_snapshots"] == 1
    assert metrics["summary"]["cycles"] == 12


def test_phase0_gate_fails_when_detected_lanes_have_no_opens(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-02-11 10:00:00 | INFO | arb_bot.engine | Found 10 opportunities "
                "(source=paper-stream-poll-refresh scan=full quotes_by_venue=kalshi=180,polymarket=250 "
                "coverage=cross_pairs=25/1714,cross_k=200/1714,cross_p=180/1714,parity_rules=6/12,parity_mkts=16/24 "
                "by_kind=intra_venue=5,cross_venue=3,structural_bucket=2)",
                "2026-02-11 10:01:00 | INFO | __main__ | paper summary cycles=12 quotes=400 opportunities=300 near=1200 dry_trades=1 settled=0 skipped=299 simulated_pnl=0.00 csv=foo.csv",
            ]
        ),
        encoding="utf-8",
    )

    csv_path = tmp_path / "run.csv"
    rows = []
    for _ in range(120):
        rows.append({"action": "skipped", "kind": "intra_venue", "opportunity_family": "intra_venue", "reason": "kelly_fraction_zero"})
    for _ in range(80):
        rows.append({"action": "skipped", "kind": "cross_venue", "opportunity_family": "cross_parity", "reason": "fill_probability_below_threshold"})
    rows.append({"action": "dry_run", "kind": "intra_venue", "opportunity_family": "intra_venue", "reason": ""})
    _write_csv(csv_path, rows)

    checks = evaluate_phase0_gate(
        log_metrics=analyze_log(log_path),
        csv_metrics=analyze_csv(csv_path),
        thresholds=GateThresholds(
            min_cycles=10,
            min_detected_rows=100,
            max_timeout_rate_per_cycle=1.0,
            min_dual_venue_rate=0.1,
            min_cross_pairs_covered=10,
            min_parity_rules_covered=4,
            min_opened_total=1,
            min_opened_lanes=1,
            lane_open_requirement_detected_min=50,
        ),
    )

    by_name = {check.name: check for check in checks}
    assert by_name["lane_open_requirement::cross_venue"].passed is False
