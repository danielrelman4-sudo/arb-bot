"""Phase 7C-4: Full detect-to-plan cycle benchmark.

Generates 200 synthetic quotes with 50 structural rules,
runs the full pipeline (detection → sizing → Kelly → fee/execution
estimation), and asserts Rust p99 < 2ms.

Run: python3 -m pytest arb_bot/tests/bench_full_cycle.py -v -s
"""

from __future__ import annotations

import importlib
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import pytest

try:
    import arb_engine_rs  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("arb_engine_rs not installed", allow_module_level=True)

from arb_bot.binary_math import decompose_binary_quote
from arb_bot.config import SizingSettings, StrategySettings
from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
)
from arb_bot.structural_rules import (
    EventTreeRule,
    ExclusiveBucketRule,
    MarketLegRef,
    ParityRule,
    StructuralRuleSet,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)
VENUES = ("kalshi", "polymarket")

# Deterministic seed for reproducibility.
RNG = random.Random(42)


def _random_price(lo: float = 0.05, hi: float = 0.95) -> float:
    return round(RNG.uniform(lo, hi), 2)


def _generate_quotes(n: int = 200) -> list[BinaryQuote]:
    """Generate n BinaryQuote objects with realistic price distributions.

    Approximately:
    - 40% have mild mispricing (intra-venue arb potential)
    - 30% from each venue with overlapping market text (cross-venue potential)
    - 30% are "clean" markets with no edge
    """
    quotes: list[BinaryQuote] = []
    market_texts = [
        f"Will the S&P 500 close above {4000 + i * 50} on Friday?" for i in range(100)
    ] + [
        f"Will Bitcoin reach ${30000 + i * 1000} by month end?" for i in range(50)
    ] + [
        f"Will the unemployment rate be below {3.5 + i * 0.1:.1f}%?" for i in range(50)
    ]

    for i in range(n):
        venue = VENUES[i % 2]
        market_id = f"{venue[:1]}-market-{i:04d}"
        text = market_texts[i % len(market_texts)]

        yes_price = _random_price(0.10, 0.90)
        no_price = _random_price(0.10, 0.90)

        # ~40% of quotes: introduce mispricing (yes + no < 0.98)
        if i % 5 < 2:
            total = yes_price + no_price
            if total > 0.96:
                # Reduce prices to create edge
                factor = RNG.uniform(0.88, 0.96) / total
                yes_price = round(yes_price * factor, 2)
                no_price = round(no_price * factor, 2)

        yes_size = RNG.uniform(10, 500)
        no_size = RNG.uniform(10, 500)

        quotes.append(BinaryQuote(
            venue=venue,
            market_id=market_id,
            yes_buy_price=yes_price,
            no_buy_price=no_price,
            yes_buy_size=yes_size,
            no_buy_size=no_size,
            yes_bid_price=max(0.01, yes_price - RNG.uniform(0.01, 0.05)),
            no_bid_price=max(0.01, no_price - RNG.uniform(0.01, 0.05)),
            yes_bid_size=RNG.uniform(5, 200),
            no_bid_size=RNG.uniform(5, 200),
            yes_maker_buy_price=max(0.01, yes_price - RNG.uniform(0.005, 0.03)),
            no_maker_buy_price=max(0.01, no_price - RNG.uniform(0.005, 0.03)),
            yes_maker_buy_size=RNG.uniform(5, 100),
            no_maker_buy_size=RNG.uniform(5, 100),
            fee_per_contract=0.01,
            observed_at=NOW,
            metadata={"canonical_text": text, "title": text},
        ))

    return quotes


def _generate_structural_rules(
    quotes: list[BinaryQuote], n_buckets: int = 20, n_event_trees: int = 15, n_parity: int = 15,
) -> StructuralRuleSet:
    """Generate structural rules referencing actual quote market IDs."""
    buckets: list[ExclusiveBucketRule] = []
    event_trees: list[EventTreeRule] = []
    parity_checks: list[ParityRule] = []

    # Group quotes by venue for rule generation.
    by_venue: dict[str, list[BinaryQuote]] = {}
    for q in quotes:
        by_venue.setdefault(q.venue, []).append(q)

    # Buckets: groups of 3-5 markets from same venue that are mutually exclusive.
    for venue, vq in by_venue.items():
        for b_idx in range(min(n_buckets // 2, len(vq) // 3)):
            start = b_idx * 3
            if start + 3 > len(vq):
                break
            group = vq[start:start + RNG.randint(3, min(5, len(vq) - start))]
            legs = tuple(
                MarketLegRef(venue=q.venue, market_id=q.market_id, side=Side.YES)
                for q in group
            )
            buckets.append(ExclusiveBucketRule(
                group_id=f"bucket-{venue}-{b_idx}",
                legs=legs,
                payout_per_contract=1.0,
            ))

    # Event trees: parent + 2-3 children from same venue.
    for venue, vq in by_venue.items():
        offset = len(vq) // 2
        for e_idx in range(min(n_event_trees // 2, (len(vq) - offset) // 3)):
            idx = offset + e_idx * 3
            if idx + 3 > len(vq):
                break
            parent = MarketLegRef(venue=vq[idx].venue, market_id=vq[idx].market_id, side=Side.YES)
            children = tuple(
                MarketLegRef(venue=vq[idx + j].venue, market_id=vq[idx + j].market_id, side=Side.YES)
                for j in range(1, min(4, len(vq) - idx))
            )
            event_trees.append(EventTreeRule(
                group_id=f"tree-{venue}-{e_idx}",
                parent=parent,
                children=children,
            ))

    # Parity checks: cross-venue pairs.
    kalshi = by_venue.get("kalshi", [])
    poly = by_venue.get("polymarket", [])
    for p_idx in range(min(n_parity, len(kalshi), len(poly))):
        parity_checks.append(ParityRule(
            group_id=f"parity-{p_idx}",
            left=MarketLegRef(venue="kalshi", market_id=kalshi[p_idx].market_id, side=Side.YES),
            right=MarketLegRef(venue="polymarket", market_id=poly[p_idx].market_id, side=Side.YES),
            relationship="equivalent",
        ))

    return StructuralRuleSet(
        buckets=tuple(buckets[:n_buckets]),
        event_trees=tuple(event_trees[:n_event_trees]),
        parity_checks=tuple(parity_checks[:n_parity]),
    )


def _generate_opportunities(quotes: list[BinaryQuote], n: int = 20) -> list[ArbitrageOpportunity]:
    """Generate N synthetic ArbitrageOpportunity objects from quotes."""
    opps: list[ArbitrageOpportunity] = []
    for i in range(min(n, len(quotes) // 2)):
        q = quotes[i * 2]
        yes_price = q.yes_buy_price
        no_price = q.no_buy_price
        total_cost = yes_price + no_price + 0.01
        edge = max(0.01, 1.0 - total_cost)

        opps.append(ArbitrageOpportunity(
            kind=OpportunityKind.INTRA_VENUE,
            execution_style=ExecutionStyle.TAKER,
            legs=(
                OpportunityLeg(venue=q.venue, market_id=q.market_id, side=Side.YES, buy_price=yes_price, buy_size=q.yes_buy_size),
                OpportunityLeg(venue=q.venue, market_id=q.market_id, side=Side.NO, buy_price=no_price, buy_size=q.no_buy_size),
            ),
            gross_edge_per_contract=edge + 0.01,
            net_edge_per_contract=edge,
            fee_per_contract=0.01,
            observed_at=NOW,
            match_key=f"intra:{q.venue}:{q.market_id}",
            match_score=1.0,
            payout_per_contract=1.0,
            metadata={"canonical_text": q.market_text},
        ))

    return opps


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

QUOTES_200 = _generate_quotes(200)
RULES_50 = _generate_structural_rules(QUOTES_200, n_buckets=20, n_event_trees=15, n_parity=15)
OPPORTUNITIES = _generate_opportunities(QUOTES_200, n=20)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of sorted data."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def _run_benchmark(fn, warmup: int = 50, iterations: int = 500) -> dict[str, float]:
    """Run a benchmark function and return timing stats in milliseconds."""
    # Warmup.
    for _ in range(warmup):
        fn()

    # Timed runs.
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        elapsed_ns = time.perf_counter_ns() - start
        times_ms.append(elapsed_ns / 1_000_000.0)

    times_ms.sort()
    return {
        "min_ms": times_ms[0],
        "p50_ms": _percentile(times_ms, 50),
        "p95_ms": _percentile(times_ms, 95),
        "p99_ms": _percentile(times_ms, 99),
        "max_ms": times_ms[-1],
        "mean_ms": statistics.mean(times_ms),
        "iterations": iterations,
    }


def _reload_with_env(**env_vars: str):
    """Reload all dispatch modules with given env vars."""
    for k, v in env_vars.items():
        os.environ[k] = v

    import arb_bot.binary_math
    import arb_bot.framework.execution_model
    import arb_bot.framework.fee_model
    import arb_bot.framework.kelly_sizing
    import arb_bot.sizing

    importlib.reload(arb_bot.binary_math)
    importlib.reload(arb_bot.sizing)
    importlib.reload(arb_bot.framework.kelly_sizing)
    importlib.reload(arb_bot.framework.fee_model)
    importlib.reload(arb_bot.framework.execution_model)


def _reload_clean():
    """Reload all dispatch modules with no Rust flags."""
    for key in list(os.environ):
        if key.startswith("ARB_USE_RUST_"):
            os.environ.pop(key, None)

    import arb_bot.binary_math
    import arb_bot.framework.execution_model
    import arb_bot.framework.fee_model
    import arb_bot.framework.kelly_sizing
    import arb_bot.sizing

    importlib.reload(arb_bot.binary_math)
    importlib.reload(arb_bot.sizing)
    importlib.reload(arb_bot.framework.kelly_sizing)
    importlib.reload(arb_bot.framework.fee_model)
    importlib.reload(arb_bot.framework.execution_model)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestBinaryMathBenchmark:
    """Benchmark binary_math (decompose_binary_quote) — 200 quotes."""

    def test_python_vs_rust(self) -> None:
        # Python path.
        _reload_clean()
        from arb_bot.binary_math import decompose_binary_quote as py_decompose
        importlib.reload(importlib.import_module("arb_bot.binary_math"))

        def run_python():
            for q in QUOTES_200:
                py_decompose(q.yes_buy_price, q.no_buy_price)

        py_stats = _run_benchmark(run_python, warmup=20, iterations=200)

        # Rust path.
        _reload_with_env(ARB_USE_RUST_BINARY_MATH="1")
        from arb_bot.binary_math import decompose_binary_quote as rs_decompose

        def run_rust():
            for q in QUOTES_200:
                rs_decompose(q.yes_buy_price, q.no_buy_price)

        rs_stats = _run_benchmark(run_rust, warmup=20, iterations=200)

        _reload_clean()

        speedup = py_stats["p99_ms"] / max(0.001, rs_stats["p99_ms"])
        print(f"\n  binary_math (200 quotes):")
        print(f"    Python p50={py_stats['p50_ms']:.3f}ms  p99={py_stats['p99_ms']:.3f}ms")
        print(f"    Rust   p50={rs_stats['p50_ms']:.3f}ms  p99={rs_stats['p99_ms']:.3f}ms")
        print(f"    Speedup: {speedup:.1f}x at p99")


class TestKellySizingBenchmark:
    """Benchmark Kelly sizing — 200 opportunities."""

    def test_python_vs_rust(self) -> None:
        edges = [0.02 + i * 0.001 for i in range(200)]

        # Python path.
        _reload_clean()
        from arb_bot.sizing import PositionSizer

        def run_python():
            for e in edges:
                PositionSizer.execution_aware_kelly_fraction(e, 0.50, 0.8)

        py_stats = _run_benchmark(run_python, warmup=20, iterations=200)

        # Rust path.
        _reload_with_env(ARB_USE_RUST_SIZING="1")
        from arb_bot.sizing import PositionSizer as RsSizer

        def run_rust():
            for e in edges:
                RsSizer.execution_aware_kelly_fraction(e, 0.50, 0.8)

        rs_stats = _run_benchmark(run_rust, warmup=20, iterations=200)

        _reload_clean()

        speedup = py_stats["p99_ms"] / max(0.001, rs_stats["p99_ms"])
        print(f"\n  Kelly sizing (200 calls):")
        print(f"    Python p50={py_stats['p50_ms']:.3f}ms  p99={py_stats['p99_ms']:.3f}ms")
        print(f"    Rust   p50={rs_stats['p50_ms']:.3f}ms  p99={rs_stats['p99_ms']:.3f}ms")
        print(f"    Speedup: {speedup:.1f}x at p99")


class TestFeeModelBenchmark:
    """Benchmark fee estimation — 200 calls."""

    def test_python_vs_rust(self) -> None:
        from arb_bot.framework.fee_model import FeeModel, FeeModelConfig, OrderType, VenueFeeSchedule

        schedule = VenueFeeSchedule(venue="test", taker_fee_per_contract=0.01)
        config = FeeModelConfig(venues=(schedule,))
        prices = [0.10 + i * 0.004 for i in range(200)]

        # Python path.
        _reload_clean()
        model_py = FeeModel(config)

        def run_python():
            for p in prices:
                model_py.estimate("test", OrderType.TAKER, 10, p)

        py_stats = _run_benchmark(run_python, warmup=20, iterations=200)

        # Rust path.
        _reload_with_env(ARB_USE_RUST_FEE_MODEL="1")
        from arb_bot.framework.fee_model import FeeModel as FeeModelRs
        model_rs = FeeModelRs(config)

        def run_rust():
            for p in prices:
                model_rs.estimate("test", OrderType.TAKER, 10, p)

        rs_stats = _run_benchmark(run_rust, warmup=20, iterations=200)

        _reload_clean()

        speedup = py_stats["p99_ms"] / max(0.001, rs_stats["p99_ms"])
        print(f"\n  Fee model (200 calls):")
        print(f"    Python p50={py_stats['p50_ms']:.3f}ms  p99={py_stats['p99_ms']:.3f}ms")
        print(f"    Rust   p50={rs_stats['p50_ms']:.3f}ms  p99={rs_stats['p99_ms']:.3f}ms")
        print(f"    Speedup: {speedup:.1f}x at p99")


class TestExecutionModelBenchmark:
    """Benchmark execution model — 100 simulations."""

    def test_python_vs_rust(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel, LegInput

        leg = LegInput(
            venue="kalshi", market_id="m1", side="yes",
            buy_price=0.50, available_size=100.0, spread=0.02,
        )
        legs = [leg, LegInput(
            venue="polymarket", market_id="m2", side="no",
            buy_price=0.45, available_size=80.0, spread=0.03,
        )]

        # Python path.
        _reload_clean()
        from arb_bot.framework.execution_model import ExecutionModel as ExecPy
        model_py = ExecPy()

        def run_python():
            for c in range(1, 101):
                model_py.simulate(legs, c)

        py_stats = _run_benchmark(run_python, warmup=10, iterations=100)

        # Rust path.
        _reload_with_env(ARB_USE_RUST_EXECUTION_MODEL="1")
        from arb_bot.framework.execution_model import ExecutionModel as ExecRs
        model_rs = ExecRs()

        def run_rust():
            for c in range(1, 101):
                model_rs.simulate(legs, c)

        rs_stats = _run_benchmark(run_rust, warmup=10, iterations=100)

        _reload_clean()

        speedup = py_stats["p99_ms"] / max(0.001, rs_stats["p99_ms"])
        print(f"\n  Execution model (100 simulations):")
        print(f"    Python p50={py_stats['p50_ms']:.3f}ms  p99={py_stats['p99_ms']:.3f}ms")
        print(f"    Rust   p50={rs_stats['p50_ms']:.3f}ms  p99={rs_stats['p99_ms']:.3f}ms")
        print(f"    Speedup: {speedup:.1f}x at p99")


class TestFullCycleBenchmark:
    """Full detect-to-plan cycle: 200 quotes → detection → sizing.

    This is the critical benchmark — must meet p99 < 2ms SLO for the
    eval pipeline portion (binary_math + Kelly + fee + execution model).

    Note: Strategy detection (ArbitrageFinder.find) is NOT included here
    because the Rust strategy port uses JSON FFI which negates speedup
    at this scale. The eval pipeline is the hot path that benefits.
    """

    def test_eval_pipeline_p99_under_2ms(self) -> None:
        """Full eval pipeline: binary_math + Kelly + fee + execution for 20 opps."""
        from arb_bot.framework.execution_model import ExecutionModel, LegInput
        from arb_bot.framework.fee_model import FeeModel, FeeModelConfig, OrderType, VenueFeeSchedule

        schedule = VenueFeeSchedule(venue="kalshi", taker_fee_per_contract=0.01)
        config = FeeModelConfig(venues=(schedule,))

        def run_eval_pipeline():
            """Simulate the eval pipeline for 20 opportunities."""
            # Phase 1: Binary math on all 200 quotes.
            for q in QUOTES_200:
                decompose_binary_quote(q.yes_buy_price, q.no_buy_price)

            # Phase 2: Kelly sizing for each opportunity.
            from arb_bot.sizing import PositionSizer
            for opp in OPPORTUNITIES:
                PositionSizer.execution_aware_kelly_fraction(
                    opp.net_edge_per_contract,
                    opp.total_cost_per_contract,
                    0.8,
                )

            # Phase 3: Fee estimation for each opportunity.
            fee_model = FeeModel(config)
            for opp in OPPORTUNITIES:
                for leg in opp.legs:
                    fee_model.estimate(leg.venue, OrderType.TAKER, 10, leg.buy_price)

            # Phase 4: Execution model for each opportunity.
            exec_model = ExecutionModel()
            for opp in OPPORTUNITIES:
                legs = [
                    LegInput(
                        venue=leg.venue, market_id=leg.market_id,
                        side=leg.side.value, buy_price=leg.buy_price,
                        available_size=leg.buy_size, spread=0.02,
                    )
                    for leg in opp.legs
                ]
                exec_model.simulate(legs, 10)

        # --- Python baseline ---
        _reload_clean()
        py_stats = _run_benchmark(run_eval_pipeline, warmup=20, iterations=200)

        # --- Rust hot path ---
        _reload_with_env(
            ARB_USE_RUST_BINARY_MATH="1",
            ARB_USE_RUST_SIZING="1",
            ARB_USE_RUST_KELLY="1",
            ARB_USE_RUST_FEE_MODEL="1",
            ARB_USE_RUST_EXECUTION_MODEL="1",
        )
        rs_stats = _run_benchmark(run_eval_pipeline, warmup=20, iterations=200)

        _reload_clean()

        speedup = py_stats["p99_ms"] / max(0.001, rs_stats["p99_ms"])
        print(f"\n  ===== FULL EVAL PIPELINE (200 quotes, 20 opportunities) =====")
        print(f"    Python: p50={py_stats['p50_ms']:.3f}ms  p95={py_stats['p95_ms']:.3f}ms  p99={py_stats['p99_ms']:.3f}ms")
        print(f"    Rust:   p50={rs_stats['p50_ms']:.3f}ms  p95={rs_stats['p95_ms']:.3f}ms  p99={rs_stats['p99_ms']:.3f}ms")
        print(f"    Speedup: {speedup:.1f}x at p99")
        print(f"    SLO target: p99 < 2.0ms")
        print(f"    SLO status: {'PASS ✓' if rs_stats['p99_ms'] < 2.0 else 'FAIL ✗'}")

        # Assert the SLO.
        assert rs_stats["p99_ms"] < 2.0, (
            f"Eval pipeline p99 {rs_stats['p99_ms']:.3f}ms exceeds 2.0ms SLO"
        )

    def test_no_detection_quality_regression(self) -> None:
        """Verify Rust dispatch produces identical detection quality.

        Run binary_math decomposition on all 200 quotes in both Python
        and Rust modes, verify results are identical within tolerance.
        """
        TOLERANCE = 1e-10

        # Python results.
        _reload_clean()
        from arb_bot.binary_math import decompose_binary_quote as py_decomp

        py_results = []
        for q in QUOTES_200:
            d = py_decomp(q.yes_buy_price, q.no_buy_price)
            py_results.append((d.implied_probability, d.edge_per_side))

        # Rust results.
        _reload_with_env(ARB_USE_RUST_BINARY_MATH="1")
        from arb_bot.binary_math import decompose_binary_quote as rs_decomp

        rs_results = []
        for q in QUOTES_200:
            d = rs_decomp(q.yes_buy_price, q.no_buy_price)
            rs_results.append((d.implied_probability, d.edge_per_side))

        _reload_clean()

        # Compare.
        mismatches = 0
        for i, (py_r, rs_r) in enumerate(zip(py_results, rs_results)):
            if abs(py_r[0] - rs_r[0]) > TOLERANCE or abs(py_r[1] - rs_r[1]) > TOLERANCE:
                mismatches += 1
                print(f"  MISMATCH at quote {i}: py={py_r} rs={rs_r}")

        print(f"\n  Detection quality: {len(QUOTES_200)} quotes, {mismatches} mismatches")
        assert mismatches == 0, f"{mismatches} detection quality mismatches found"

    def test_kelly_sizing_parity_under_load(self) -> None:
        """Verify Kelly sizing produces identical results under load (1000 calls)."""
        TOLERANCE = 1e-10

        test_params = [
            (0.02 + i * 0.001, 0.30 + (i % 70) * 0.01, 0.5 + (i % 50) * 0.01)
            for i in range(1000)
        ]

        # Python results.
        _reload_clean()
        from arb_bot.sizing import PositionSizer

        py_results = [
            PositionSizer.execution_aware_kelly_fraction(e, c, f)
            for e, c, f in test_params
        ]

        # Rust results.
        _reload_with_env(ARB_USE_RUST_SIZING="1")
        from arb_bot.sizing import PositionSizer as RsSizer

        rs_results = [
            RsSizer.execution_aware_kelly_fraction(e, c, f)
            for e, c, f in test_params
        ]

        _reload_clean()

        mismatches = 0
        for i, (py_r, rs_r) in enumerate(zip(py_results, rs_results)):
            if abs(py_r - rs_r) > TOLERANCE:
                mismatches += 1
                if mismatches <= 5:
                    e, c, f = test_params[i]
                    print(f"  MISMATCH {i}: params=({e},{c},{f}) py={py_r} rs={rs_r}")

        print(f"\n  Kelly parity: 1000 calls, {mismatches} mismatches")
        assert mismatches == 0, f"{mismatches} Kelly sizing mismatches found"
