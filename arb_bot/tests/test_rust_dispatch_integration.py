"""Integration tests for Rust dispatch wiring (Phase 7C).

Tests that each module's _RUST_ACTIVE flag and dispatch mechanism works
correctly with env var control. Verifies staged cutover and rollback.

Tests are skipped when the Rust module is not installed.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from typing import Any

import pytest

try:
    import arb_engine_rs  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("arb_engine_rs not installed", allow_module_level=True)


TOLERANCE = 1e-10


def _close(a: float, b: float) -> bool:
    return abs(a - b) < TOLERANCE


# ---------------------------------------------------------------------------
# Env var helpers
# ---------------------------------------------------------------------------


def _set_env(**env_vars: str):
    """Context-free env var setter. Returns list of keys to clean up."""
    keys = []
    for k, v in env_vars.items():
        os.environ[k] = v
        keys.append(k)
    return keys


def _clear_env(*keys: str):
    for k in keys:
        os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Binary math dispatch tests (existing, but verify in integration)
# ---------------------------------------------------------------------------


class TestBinaryMathDispatch:
    def test_inactive_by_default(self) -> None:
        _clear_env("ARB_USE_RUST_BINARY_MATH", "ARB_USE_RUST_ALL")
        from arb_bot import binary_math
        importlib.reload(binary_math)
        assert binary_math._RUST_ACTIVE is False

    def test_active_with_env_var(self) -> None:
        keys = _set_env(ARB_USE_RUST_BINARY_MATH="1")
        try:
            from arb_bot import binary_math
            importlib.reload(binary_math)
            assert binary_math._RUST_ACTIVE is True
            # Verify it still returns correct types.
            d = binary_math.decompose_binary_quote(0.57, 0.48)
            assert isinstance(d, binary_math.QuoteDecomposition)
            assert _close(d.implied_probability, 0.545)
        finally:
            _clear_env(*keys)
            importlib.reload(binary_math)

    def test_rollback(self) -> None:
        keys = _set_env(ARB_USE_RUST_BINARY_MATH="1")
        from arb_bot import binary_math
        importlib.reload(binary_math)
        assert binary_math._RUST_ACTIVE is True

        _clear_env(*keys)
        importlib.reload(binary_math)
        assert binary_math._RUST_ACTIVE is False


# ---------------------------------------------------------------------------
# Sizing dispatch tests
# ---------------------------------------------------------------------------


class TestSizingDispatch:
    def test_inactive_by_default(self) -> None:
        _clear_env("ARB_USE_RUST_SIZING", "ARB_USE_RUST_ALL")
        from arb_bot import sizing
        importlib.reload(sizing)
        assert sizing._RUST_ACTIVE is False

    def test_active_with_env_var(self) -> None:
        keys = _set_env(ARB_USE_RUST_SIZING="1")
        try:
            from arb_bot import sizing
            importlib.reload(sizing)
            assert sizing._RUST_ACTIVE is True
            # Verify kelly fraction still works.
            f = sizing.PositionSizer.execution_aware_kelly_fraction(
                0.50, 0.50, 0.8, None,
            )
            assert isinstance(f, float)
            assert f > 0.0
        finally:
            _clear_env(*keys)
            importlib.reload(sizing)

    def test_active_with_all_flag(self) -> None:
        keys = _set_env(ARB_USE_RUST_ALL="1")
        try:
            from arb_bot import sizing
            importlib.reload(sizing)
            assert sizing._RUST_ACTIVE is True
        finally:
            _clear_env(*keys)
            importlib.reload(sizing)


# ---------------------------------------------------------------------------
# Kelly sizing dispatch tests
# ---------------------------------------------------------------------------


class TestKellyDispatch:
    def test_inactive_by_default(self) -> None:
        _clear_env("ARB_USE_RUST_KELLY", "ARB_USE_RUST_ALL")
        from arb_bot import kelly_sizing
        importlib.reload(kelly_sizing)
        assert kelly_sizing._RUST_ACTIVE is False

    def test_active_with_env_var(self) -> None:
        keys = _set_env(ARB_USE_RUST_KELLY="1")
        try:
            from arb_bot import kelly_sizing
            importlib.reload(kelly_sizing)
            assert kelly_sizing._RUST_ACTIVE is True

            sizer = kelly_sizing.TailRiskKelly()
            result = sizer.compute(edge=0.50, cost=0.50, fill_prob=0.8)
            assert isinstance(result, kelly_sizing.KellySizingResult)
            assert not result.blocked
            assert result.adjusted_fraction > 0.0
        finally:
            _clear_env(*keys)
            importlib.reload(kelly_sizing)

    def test_parity_with_python(self) -> None:
        """Verify Rust and Python produce identical results."""
        # Run Python first.
        _clear_env("ARB_USE_RUST_KELLY", "ARB_USE_RUST_ALL")
        from arb_bot import kelly_sizing
        importlib.reload(kelly_sizing)
        sizer_py = kelly_sizing.TailRiskKelly()
        py = sizer_py.compute(edge=0.50, cost=0.50, fill_prob=0.8, model_uncertainty=0.1)

        # Run Rust.
        keys = _set_env(ARB_USE_RUST_KELLY="1")
        try:
            importlib.reload(kelly_sizing)
            sizer_rs = kelly_sizing.TailRiskKelly()
            rs = sizer_rs.compute(edge=0.50, cost=0.50, fill_prob=0.8, model_uncertainty=0.1)

            assert _close(py.raw_kelly, rs.raw_kelly), (
                f"raw_kelly: py={py.raw_kelly} rs={rs.raw_kelly}"
            )
            assert _close(py.adjusted_fraction, rs.adjusted_fraction), (
                f"adjusted_fraction: py={py.adjusted_fraction} rs={rs.adjusted_fraction}"
            )
        finally:
            _clear_env(*keys)
            importlib.reload(kelly_sizing)


# ---------------------------------------------------------------------------
# Fee model dispatch tests
# ---------------------------------------------------------------------------


class TestFeeModelDispatch:
    def test_inactive_by_default(self) -> None:
        _clear_env("ARB_USE_RUST_FEE_MODEL", "ARB_USE_RUST_ALL")
        from arb_bot import fee_model
        importlib.reload(fee_model)
        assert fee_model._RUST_ACTIVE is False

    def test_active_with_env_var(self) -> None:
        keys = _set_env(ARB_USE_RUST_FEE_MODEL="1")
        try:
            from arb_bot import fee_model
            importlib.reload(fee_model)
            assert fee_model._RUST_ACTIVE is True

            schedule = fee_model.VenueFeeSchedule(
                venue="test", taker_fee_per_contract=0.01,
            )
            model = fee_model.FeeModel(
                fee_model.FeeModelConfig(venues=(schedule,)),
            )
            est = model.estimate("test", fee_model.OrderType.TAKER, 10, 0.50)
            assert isinstance(est, fee_model.FeeEstimate)
            assert _close(est.total_fee, 0.10)
        finally:
            _clear_env(*keys)
            importlib.reload(fee_model)


# ---------------------------------------------------------------------------
# Execution model dispatch tests
# ---------------------------------------------------------------------------


class TestExecutionModelDispatch:
    def test_inactive_by_default(self) -> None:
        _clear_env("ARB_USE_RUST_EXECUTION_MODEL", "ARB_USE_RUST_ALL")
        from arb_bot import execution_model
        importlib.reload(execution_model)
        assert execution_model._RUST_ACTIVE is False

    def test_active_with_env_var(self) -> None:
        keys = _set_env(ARB_USE_RUST_EXECUTION_MODEL="1")
        try:
            from arb_bot import execution_model
            importlib.reload(execution_model)
            assert execution_model._RUST_ACTIVE is True

            model = execution_model.ExecutionModel()
            leg = execution_model.LegInput(
                venue="kalshi", market_id="m1", side="yes",
                buy_price=0.50, available_size=100.0, spread=0.02,
            )
            est = model.simulate([leg], 10)
            assert isinstance(est, execution_model.ExecutionEstimate)
            assert est.all_fill_probability > 0.0
        finally:
            _clear_env(*keys)
            importlib.reload(execution_model)


# ---------------------------------------------------------------------------
# Staged cutover tests
# ---------------------------------------------------------------------------


class TestStagedCutover:
    """Verify staged cutover via individual env vars."""

    def test_individual_modules_independent(self) -> None:
        """Each module can be enabled/disabled independently."""
        _clear_env(
            "ARB_USE_RUST_BINARY_MATH", "ARB_USE_RUST_SIZING",
            "ARB_USE_RUST_KELLY", "ARB_USE_RUST_FEE_MODEL",
            "ARB_USE_RUST_EXECUTION_MODEL", "ARB_USE_RUST_ALL",
        )

        # Enable only sizing.
        keys = _set_env(ARB_USE_RUST_SIZING="1")
        try:
            from arb_bot import sizing, kelly_sizing, fee_model, execution_model
            importlib.reload(sizing)
            importlib.reload(kelly_sizing)
            importlib.reload(fee_model)
            importlib.reload(execution_model)

            assert sizing._RUST_ACTIVE is True
            assert kelly_sizing._RUST_ACTIVE is False
            assert fee_model._RUST_ACTIVE is False
            assert execution_model._RUST_ACTIVE is False
        finally:
            _clear_env(*keys)
            importlib.reload(sizing)
            importlib.reload(kelly_sizing)
            importlib.reload(fee_model)
            importlib.reload(execution_model)

    def test_all_flag_enables_everything(self) -> None:
        """ARB_USE_RUST_ALL=1 enables all modules."""
        keys = _set_env(ARB_USE_RUST_ALL="1")
        try:
            from arb_bot import sizing, kelly_sizing, fee_model, execution_model
            importlib.reload(sizing)
            importlib.reload(kelly_sizing)
            importlib.reload(fee_model)
            importlib.reload(execution_model)

            assert sizing._RUST_ACTIVE is True
            assert kelly_sizing._RUST_ACTIVE is True
            assert fee_model._RUST_ACTIVE is True
            assert execution_model._RUST_ACTIVE is True
        finally:
            _clear_env(*keys)
            importlib.reload(sizing)
            importlib.reload(kelly_sizing)
            importlib.reload(fee_model)
            importlib.reload(execution_model)

    def test_rollback_all(self) -> None:
        """Clearing env vars rolls back to Python implementations."""
        keys = _set_env(ARB_USE_RUST_ALL="1")
        from arb_bot import sizing, kelly_sizing, fee_model, execution_model
        importlib.reload(sizing)
        importlib.reload(kelly_sizing)
        importlib.reload(fee_model)
        importlib.reload(execution_model)

        assert sizing._RUST_ACTIVE is True

        _clear_env(*keys)
        importlib.reload(sizing)
        importlib.reload(kelly_sizing)
        importlib.reload(fee_model)
        importlib.reload(execution_model)

        assert sizing._RUST_ACTIVE is False
        assert kelly_sizing._RUST_ACTIVE is False
        assert fee_model._RUST_ACTIVE is False
        assert execution_model._RUST_ACTIVE is False


# ---------------------------------------------------------------------------
# Performance comparison (Rust dispatch vs Python)
# ---------------------------------------------------------------------------


class TestDispatchPerformance:
    """Compare performance of Rust dispatch vs pure Python."""

    N = 5000

    def test_kelly_dispatch_perf(self) -> None:
        from arb_bot import sizing
        _clear_env("ARB_USE_RUST_SIZING", "ARB_USE_RUST_ALL")
        importlib.reload(sizing)

        # Python.
        start = time.perf_counter()
        for i in range(self.N):
            e = 0.30 + i * 0.00005
            sizing.PositionSizer.execution_aware_kelly_fraction(e, 0.50, 0.8)
        py_time = time.perf_counter() - start

        # Rust.
        keys = _set_env(ARB_USE_RUST_SIZING="1")
        try:
            importlib.reload(sizing)
            start = time.perf_counter()
            for i in range(self.N):
                e = 0.30 + i * 0.00005
                sizing.PositionSizer.execution_aware_kelly_fraction(e, 0.50, 0.8)
            rs_time = time.perf_counter() - start
        finally:
            _clear_env(*keys)
            importlib.reload(sizing)

        print(f"\nkelly dispatch: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")
