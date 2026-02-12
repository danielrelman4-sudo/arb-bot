"""Parity tests: Rust arb_engine_rs vs Python implementations (Phase 7).

Every test calls the same function in both Python and Rust with identical
inputs and asserts outputs match within tolerance (1e-12 for floats).

Tests are skipped when the Rust module is not installed.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if Rust extension is not installed
# ---------------------------------------------------------------------------

try:
    import arb_engine_rs  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("arb_engine_rs not installed", allow_module_level=True)

from arb_bot.binary_math import (
    EffectiveBuy,
    QuoteDecomposition,
    QuoteDiagnostics,
    build_quote_diagnostics,
    choose_effective_buy_price,
    decompose_binary_quote,
    reconstruct_binary_quote,
)

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

TOLERANCE = 1e-12


def _close(a: float, b: float) -> bool:
    return abs(a - b) < TOLERANCE


def _close_optional(a: Optional[float], b: Optional[float]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return _close(a, b)


# ---------------------------------------------------------------------------
# Module-level smoke tests
# ---------------------------------------------------------------------------


class TestModuleAttributes:
    def test_version_is_string(self) -> None:
        v = arb_engine_rs.version()
        assert isinstance(v, str)
        assert v == "0.1.0"

    def test_available_functions(self) -> None:
        funcs = arb_engine_rs.available_functions()
        assert "decompose_binary_quote" in funcs
        assert "reconstruct_binary_quote" in funcs
        assert "choose_effective_buy_price" in funcs
        assert "build_quote_diagnostics" in funcs


# ---------------------------------------------------------------------------
# decompose_binary_quote parity
# ---------------------------------------------------------------------------

DECOMPOSE_CASES: List[Tuple[float, float]] = [
    (0.57, 0.48),  # Standard case.
    (0.50, 0.50),  # Fair market.
    (0.01, 0.99),  # Near boundaries.
    (0.99, 0.01),  # Near boundaries reversed.
    (1.0, 0.0),    # Extreme.
    (0.0, 1.0),    # Extreme reversed.
    (0.0, 0.0),    # Zero-zero.
    (1.0, 1.0),    # One-one (high vig).
    (0.55, 0.55),  # Symmetric spread.
    (0.33, 0.77),  # Arbitrary values.
    (0.123456789, 0.876543211),  # Many decimal places.
]


class TestDecomposeParity:
    @pytest.mark.parametrize("yes_price,no_price", DECOMPOSE_CASES)
    def test_decompose(self, yes_price: float, no_price: float) -> None:
        py = decompose_binary_quote(yes_price, no_price)
        rs = arb_engine_rs.decompose_binary_quote(yes_price, no_price)

        # Rust returns (implied_prob, edge_per_side) tuple.
        assert _close(py.implied_probability, rs[0]), (
            f"implied_prob mismatch: py={py.implied_probability} rs={rs[0]}"
        )
        assert _close(py.edge_per_side, rs[1]), (
            f"edge_per_side mismatch: py={py.edge_per_side} rs={rs[1]}"
        )

    def test_decompose_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            decompose_binary_quote(-0.1, 0.5)
        with pytest.raises(ValueError):
            arb_engine_rs.decompose_binary_quote(-0.1, 0.5)

        with pytest.raises(ValueError):
            decompose_binary_quote(0.5, 1.5)
        with pytest.raises(ValueError):
            arb_engine_rs.decompose_binary_quote(0.5, 1.5)


# ---------------------------------------------------------------------------
# reconstruct_binary_quote parity
# ---------------------------------------------------------------------------

RECONSTRUCT_CASES: List[Tuple[float, float]] = [
    (0.545, 0.025),
    (0.50, 0.0),
    (0.50, 0.10),
    (0.80, -0.05),
    (0.20, 0.02),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.5, 0.5),    # Edge pushes past 1.0 — tests clamping.
    (0.5, -0.5),   # Edge pushes below 0.0 — tests clamping.
]


class TestReconstructParity:
    @pytest.mark.parametrize("implied_prob,edge", RECONSTRUCT_CASES)
    def test_reconstruct(self, implied_prob: float, edge: float) -> None:
        py = reconstruct_binary_quote(implied_prob, edge)
        rs = arb_engine_rs.reconstruct_binary_quote(implied_prob, edge)

        assert _close(py[0], rs[0]), f"yes mismatch: py={py[0]} rs={rs[0]}"
        assert _close(py[1], rs[1]), f"no mismatch: py={py[1]} rs={rs[1]}"

    def test_roundtrip(self) -> None:
        """decompose then reconstruct should reproduce original."""
        for yes, no in DECOMPOSE_CASES:
            py_d = decompose_binary_quote(yes, no)
            py_r = reconstruct_binary_quote(py_d.implied_probability, py_d.edge_per_side)
            rs_d = arb_engine_rs.decompose_binary_quote(yes, no)
            rs_r = arb_engine_rs.reconstruct_binary_quote(rs_d[0], rs_d[1])
            assert _close(py_r[0], rs_r[0]), f"roundtrip yes: py={py_r[0]} rs={rs_r[0]}"
            assert _close(py_r[1], rs_r[1]), f"roundtrip no: py={py_r[1]} rs={rs_r[1]}"


# ---------------------------------------------------------------------------
# choose_effective_buy_price parity
# ---------------------------------------------------------------------------

EBP_CASES: List[Dict[str, Any]] = [
    # Direct ask only.
    dict(side="yes", ask=0.55, ask_size=100.0, opp=None, opp_size=0.0),
    # Opposite bid transform only.
    dict(side="yes", ask=None, ask_size=0.0, opp=0.45, opp_size=75.0),
    # Both — direct wins (lower price).
    dict(side="yes", ask=0.40, ask_size=50.0, opp=0.55, opp_size=75.0),
    # Both — transform wins (lower price).
    dict(side="yes", ask=0.60, ask_size=50.0, opp=0.55, opp_size=75.0),
    # No side.
    dict(side="no", ask=0.55, ask_size=100.0, opp=None, opp_size=0.0),
    dict(side="no", ask=None, ask_size=0.0, opp=0.45, opp_size=75.0),
    # Neither available.
    dict(side="yes", ask=None, ask_size=0.0, opp=None, opp_size=0.0),
    # Zero size — should be filtered out.
    dict(side="yes", ask=0.55, ask_size=0.0, opp=0.45, opp_size=0.0),
    # Tie on price — favor deeper size.
    dict(side="yes", ask=0.55, ask_size=50.0, opp=0.45, opp_size=100.0),
    # Edge: boundary prices.
    dict(side="yes", ask=0.0, ask_size=10.0, opp=1.0, opp_size=10.0),
    dict(side="yes", ask=1.0, ask_size=10.0, opp=0.0, opp_size=10.0),
]


class TestChooseEffectiveBuyParity:
    @pytest.mark.parametrize("case", EBP_CASES, ids=[str(i) for i in range(len(EBP_CASES))])
    def test_parity(self, case: Dict[str, Any]) -> None:
        py = choose_effective_buy_price(
            case["side"], case["ask"], case["ask_size"], case["opp"], case["opp_size"],
        )
        rs = arb_engine_rs.choose_effective_buy_price(
            case["side"], case["ask"], case["ask_size"], case["opp"], case["opp_size"],
        )

        if py is None:
            assert rs is None, f"Python returned None but Rust returned {rs}"
            return

        assert rs is not None, f"Rust returned None but Python returned {py}"
        # Rust returns (price, size, source) tuple.
        assert _close(py.price, rs[0]), f"price mismatch: py={py.price} rs={rs[0]}"
        assert _close(py.size, rs[1]), f"size mismatch: py={py.size} rs={rs[1]}"
        assert py.source == rs[2], f"source mismatch: py={py.source} rs={rs[2]}"


# ---------------------------------------------------------------------------
# build_quote_diagnostics parity
# ---------------------------------------------------------------------------

DIAG_CASES: List[Dict[str, Any]] = [
    # Full bids.
    dict(yb=0.57, nb=0.48, ybd=0.53, nbd=0.44),
    # No bids.
    dict(yb=0.57, nb=0.48, ybd=None, nbd=None),
    # Only yes bid (partial → treated as no bids).
    dict(yb=0.57, nb=0.48, ybd=0.53, nbd=None),
    # Fair market.
    dict(yb=0.50, nb=0.50, ybd=0.48, nbd=0.48),
    # Extreme spread.
    dict(yb=0.99, nb=0.99, ybd=0.01, nbd=0.01),
    # Zero spread.
    dict(yb=0.55, nb=0.45, ybd=0.55, nbd=0.45),
    # Boundary values.
    dict(yb=0.0, nb=1.0, ybd=0.0, nbd=1.0),
    dict(yb=1.0, nb=0.0, ybd=1.0, nbd=0.0),
]


class TestDiagnosticsParity:
    @pytest.mark.parametrize("case", DIAG_CASES, ids=[str(i) for i in range(len(DIAG_CASES))])
    def test_parity(self, case: Dict[str, Any]) -> None:
        py = build_quote_diagnostics(case["yb"], case["nb"], case["ybd"], case["nbd"])
        rs = arb_engine_rs.build_quote_diagnostics(
            case["yb"], case["nb"], case["ybd"], case["nbd"],
        )

        # Rust returns a dict.
        assert _close(py.ask_implied_probability, rs["ask_implied_probability"]), (
            f"ask_ip: py={py.ask_implied_probability} rs={rs['ask_implied_probability']}"
        )
        assert _close(py.ask_edge_per_side, rs["ask_edge_per_side"]), (
            f"ask_edge: py={py.ask_edge_per_side} rs={rs['ask_edge_per_side']}"
        )
        assert _close_optional(py.bid_implied_probability, rs["bid_implied_probability"]), (
            f"bid_ip: py={py.bid_implied_probability} rs={rs['bid_implied_probability']}"
        )
        assert _close_optional(py.bid_edge_per_side, rs["bid_edge_per_side"]), (
            f"bid_edge: py={py.bid_edge_per_side} rs={rs['bid_edge_per_side']}"
        )
        assert _close_optional(py.midpoint_consistency_gap, rs["midpoint_consistency_gap"]), (
            f"gap: py={py.midpoint_consistency_gap} rs={rs['midpoint_consistency_gap']}"
        )
        assert _close_optional(py.yes_spread, rs["yes_spread"]), (
            f"yes_spread: py={py.yes_spread} rs={rs['yes_spread']}"
        )
        assert _close_optional(py.no_spread, rs["no_spread"]), (
            f"no_spread: py={py.no_spread} rs={rs['no_spread']}"
        )
        assert _close_optional(py.spread_asymmetry, rs["spread_asymmetry"]), (
            f"spread_asymmetry: py={py.spread_asymmetry} rs={rs['spread_asymmetry']}"
        )


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------


class TestPerformance:
    """Measure Rust vs Python speed for binary_math functions.

    Not a strict assertion (host-dependent), but logs timings.
    """

    N = 10_000

    def test_decompose_perf(self) -> None:
        inputs = [(0.5 + i * 0.00001, 0.5 - i * 0.00001) for i in range(self.N)]

        start = time.perf_counter()
        for yes, no in inputs:
            decompose_binary_quote(yes, no)
        py_time = time.perf_counter() - start

        start = time.perf_counter()
        for yes, no in inputs:
            arb_engine_rs.decompose_binary_quote(yes, no)
        rs_time = time.perf_counter() - start

        print(f"\ndecompose: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")
        # Rust should be faster (but not enforcing — CI variance).

    def test_choose_effective_buy_perf(self) -> None:
        start = time.perf_counter()
        for i in range(self.N):
            p = 0.4 + i * 0.00001
            choose_effective_buy_price("yes", p, 100.0, 1.0 - p, 75.0)
        py_time = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(self.N):
            p = 0.4 + i * 0.00001
            arb_engine_rs.choose_effective_buy_price("yes", p, 100.0, 1.0 - p, 75.0)
        rs_time = time.perf_counter() - start

        print(f"\nchoose_effective_buy: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")

    def test_diagnostics_perf(self) -> None:
        start = time.perf_counter()
        for i in range(self.N):
            p = 0.5 + i * 0.00001
            build_quote_diagnostics(p, 1.0 - p, p - 0.02, 1.0 - p - 0.02)
        py_time = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(self.N):
            p = 0.5 + i * 0.00001
            arb_engine_rs.build_quote_diagnostics(p, 1.0 - p, p - 0.02, 1.0 - p - 0.02)
        rs_time = time.perf_counter() - start

        print(f"\ndiagnostics: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")


# ---------------------------------------------------------------------------
# Dispatch mechanism tests
# ---------------------------------------------------------------------------


class TestDispatch:
    """Test the env-var-controlled dispatch mechanism in binary_math.py."""

    def test_dispatch_flag_exists(self) -> None:
        from arb_bot import binary_math
        assert hasattr(binary_math, "_RUST_ACTIVE")

    def test_dispatch_returns_correct_types_when_active(self) -> None:
        """When dispatch is active, functions still return Python dataclasses."""
        import os
        import importlib

        os.environ["ARB_USE_RUST_BINARY_MATH"] = "1"
        try:
            from arb_bot import binary_math
            importlib.reload(binary_math)

            r = binary_math.decompose_binary_quote(0.57, 0.48)
            assert isinstance(r, binary_math.QuoteDecomposition)
            assert _close(r.implied_probability, 0.545)

            r2 = binary_math.reconstruct_binary_quote(0.545, 0.025)
            assert isinstance(r2, tuple)
            assert len(r2) == 2

            r3 = binary_math.choose_effective_buy_price("yes", 0.55, 100.0, None, 0.0)
            assert isinstance(r3, binary_math.EffectiveBuy)
            assert r3.source == "direct_ask"

            r4 = binary_math.build_quote_diagnostics(0.57, 0.48, 0.53, 0.44)
            assert isinstance(r4, binary_math.QuoteDiagnostics)
            assert r4.bid_implied_probability is not None

            r5 = binary_math.build_quote_diagnostics(0.57, 0.48)
            assert r5.bid_implied_probability is None
        finally:
            os.environ.pop("ARB_USE_RUST_BINARY_MATH", None)
            importlib.reload(binary_math)

    def test_dispatch_inactive_without_env_var(self) -> None:
        """Without env var, should use Python implementations."""
        import os
        import importlib

        os.environ.pop("ARB_USE_RUST_BINARY_MATH", None)
        from arb_bot import binary_math
        importlib.reload(binary_math)

        assert binary_math._RUST_ACTIVE is False
