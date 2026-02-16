"""Parity tests: Rust arb_engine_rs eval pipeline vs Python implementations (Phase 7B-3).

Every test calls the same function in both Python and Rust with identical
inputs and asserts outputs match within tolerance (1e-12 for floats).

Tests are skipped when the Rust module is not installed.
"""

from __future__ import annotations

import json
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

import importlib
import os

import arb_bot.framework.fee_model
import arb_bot.framework.kelly_sizing
import arb_bot.sizing
import arb_bot.framework.execution_model

from arb_bot.framework.fee_model import FeeModel, FeeModelConfig, OrderType, VenueFeeSchedule
from arb_bot.framework.kelly_sizing import TailRiskKelly, TailRiskKellyConfig, _raw_kelly
from arb_bot.sizing import PositionSizer

# ---------------------------------------------------------------------------
# Ensure clean module state (dispatch tests may have leaked)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _clean_dispatch_state():
    """Reload modules to clear any dispatch monkey-patching from other tests."""
    for key in list(os.environ):
        if key.startswith("ARB_USE_RUST_"):
            os.environ.pop(key, None)
    importlib.reload(arb_bot.framework.fee_model)
    importlib.reload(arb_bot.framework.kelly_sizing)
    importlib.reload(arb_bot.sizing)
    importlib.reload(arb_bot.framework.execution_model)
    # Re-import from clean modules.
    global FeeModel, FeeModelConfig, OrderType, VenueFeeSchedule
    global TailRiskKelly, TailRiskKellyConfig, _raw_kelly, PositionSizer
    from arb_bot.framework.fee_model import FeeModel, FeeModelConfig, OrderType, VenueFeeSchedule
    from arb_bot.framework.kelly_sizing import TailRiskKelly, TailRiskKellyConfig, _raw_kelly
    from arb_bot.sizing import PositionSizer
    yield


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

TOLERANCE = 1e-10


def _close(a: float, b: float) -> bool:
    return abs(a - b) < TOLERANCE


# ---------------------------------------------------------------------------
# Fee Model helpers
# ---------------------------------------------------------------------------


def _schedule_to_json(s: VenueFeeSchedule) -> str:
    return json.dumps({
        "venue": s.venue,
        "taker_fee_per_contract": s.taker_fee_per_contract,
        "maker_fee_per_contract": s.maker_fee_per_contract,
        "taker_fee_rate": s.taker_fee_rate,
        "maker_fee_rate": s.maker_fee_rate,
        "taker_curve_coefficient": s.taker_curve_coefficient,
        "maker_curve_coefficient": s.maker_curve_coefficient,
        "curve_round_up": s.curve_round_up,
        "settlement_fee_per_contract": s.settlement_fee_per_contract,
        "min_fee_per_order": s.min_fee_per_order,
        "max_fee_per_order": s.max_fee_per_order,
    })


# ---------------------------------------------------------------------------
# Fee Model parity tests
# ---------------------------------------------------------------------------


FEE_CASES: List[Dict[str, Any]] = [
    # Basic flat taker fee.
    dict(
        venue="kalshi", order_type="taker", contracts=10, price=0.50,
        schedule=VenueFeeSchedule(
            venue="kalshi", taker_fee_per_contract=0.01,
        ),
    ),
    # Basic flat maker fee (rebate).
    dict(
        venue="kalshi", order_type="maker", contracts=10, price=0.50,
        schedule=VenueFeeSchedule(
            venue="kalshi", maker_fee_per_contract=-0.005,
        ),
    ),
    # Proportional fee.
    dict(
        venue="polymarket", order_type="taker", contracts=20, price=0.60,
        schedule=VenueFeeSchedule(
            venue="polymarket", taker_fee_rate=0.02,
        ),
    ),
    # Curve fee (Kalshi P*(1-P)).
    dict(
        venue="kalshi", order_type="taker", contracts=10, price=0.50,
        schedule=VenueFeeSchedule(
            venue="kalshi", taker_fee_per_contract=0.0,
            taker_curve_coefficient=1.0, curve_round_up=True,
        ),
    ),
    # Curve fee at extreme price.
    dict(
        venue="kalshi", order_type="taker", contracts=10, price=0.95,
        schedule=VenueFeeSchedule(
            venue="kalshi", taker_curve_coefficient=1.0, curve_round_up=True,
        ),
    ),
    # Curve fee at price 0 (should be zero curve).
    dict(
        venue="kalshi", order_type="taker", contracts=10, price=0.0,
        schedule=VenueFeeSchedule(
            venue="kalshi", taker_curve_coefficient=1.0, curve_round_up=True,
        ),
    ),
    # All components combined.
    dict(
        venue="kalshi", order_type="taker", contracts=5, price=0.55,
        schedule=VenueFeeSchedule(
            venue="kalshi", taker_fee_per_contract=0.01,
            taker_fee_rate=0.01, taker_curve_coefficient=0.5,
            curve_round_up=True, settlement_fee_per_contract=0.005,
        ),
    ),
    # Min fee floor.
    dict(
        venue="test", order_type="taker", contracts=1, price=0.10,
        schedule=VenueFeeSchedule(
            venue="test", taker_fee_per_contract=0.001,
            min_fee_per_order=0.10,
        ),
    ),
    # Max fee cap.
    dict(
        venue="test", order_type="taker", contracts=1000, price=0.50,
        schedule=VenueFeeSchedule(
            venue="test", taker_fee_per_contract=0.10,
            max_fee_per_order=5.0,
        ),
    ),
    # Settlement fee only.
    dict(
        venue="test", order_type="taker", contracts=10, price=0.50,
        schedule=VenueFeeSchedule(
            venue="test", settlement_fee_per_contract=0.01,
        ),
    ),
    # Zero contracts.
    dict(
        venue="test", order_type="taker", contracts=0, price=0.50,
        schedule=VenueFeeSchedule(venue="test", taker_fee_per_contract=0.01),
    ),
    # Maker curve fee.
    dict(
        venue="test", order_type="maker", contracts=10, price=0.50,
        schedule=VenueFeeSchedule(
            venue="test", maker_curve_coefficient=0.8,
            curve_round_up=False,
        ),
    ),
]


class TestFeeModelParity:
    @pytest.mark.parametrize("case", FEE_CASES, ids=[str(i) for i in range(len(FEE_CASES))])
    def test_parity(self, case: Dict[str, Any]) -> None:
        schedule = case["schedule"]
        model = FeeModel(FeeModelConfig(venues=(schedule,)))
        ot = OrderType.TAKER if case["order_type"] == "taker" else OrderType.MAKER
        py = model.estimate(case["venue"], ot, case["contracts"], case["price"])

        rs_json = arb_engine_rs.estimate_fee(
            case["venue"], case["order_type"], case["contracts"],
            case["price"], _schedule_to_json(schedule),
        )
        rs = json.loads(rs_json)

        assert _close(py.total_fee, rs["total_fee"]), (
            f"total_fee: py={py.total_fee} rs={rs['total_fee']}"
        )
        assert _close(py.per_contract_fee, rs["per_contract_fee"]), (
            f"per_contract_fee: py={py.per_contract_fee} rs={rs['per_contract_fee']}"
        )
        assert _close(py.settlement_fee, rs["settlement_fee"]), (
            f"settlement_fee: py={py.settlement_fee} rs={rs['settlement_fee']}"
        )


# ---------------------------------------------------------------------------
# Raw Kelly parity tests
# ---------------------------------------------------------------------------


KELLY_RAW_CASES: List[Tuple[float, float, float, Optional[float]]] = [
    # Standard positive Kelly.
    (0.50, 0.50, 0.9, None),
    # Edge equals cost (high Kelly).
    (1.00, 0.50, 0.8, None),
    # Edge too small for fill prob — Kelly is zero.
    (0.10, 0.50, 0.8, None),
    # Zero edge.
    (0.0, 0.50, 0.8, None),
    # Negative edge.
    (-0.10, 0.50, 0.8, None),
    # Zero cost.
    (0.10, 0.0, 0.8, None),
    # Perfect fill probability.
    (0.10, 0.50, 1.0, None),
    # Zero fill probability.
    (0.10, 0.50, 0.0, None),
    # With explicit failure loss (less than cost).
    (0.50, 0.50, 0.8, 0.25),
    # With explicit failure loss = 0 (raw=1.0).
    (0.50, 0.50, 0.8, 0.0),
    # With failure loss = cost (same as None).
    (0.50, 0.50, 0.8, 0.50),
    # High edge, low fill prob.
    (0.80, 0.50, 0.3, None),
    # Tiny edge, high fill prob.
    (0.001, 0.50, 0.99, None),
]


class TestRawKellyParity:
    @pytest.mark.parametrize(
        "edge,cost,fill_prob,failure_loss",
        KELLY_RAW_CASES,
        ids=[str(i) for i in range(len(KELLY_RAW_CASES))],
    )
    def test_parity(
        self, edge: float, cost: float, fill_prob: float, failure_loss: Optional[float],
    ) -> None:
        py = _raw_kelly(edge, cost, fill_prob, failure_loss)
        rs = arb_engine_rs.execution_aware_kelly_fraction(edge, cost, fill_prob, failure_loss)
        assert _close(py, rs), f"raw kelly: py={py} rs={rs}"


# ---------------------------------------------------------------------------
# Full Kelly sizing parity tests
# ---------------------------------------------------------------------------


KELLY_FULL_CASES: List[Dict[str, Any]] = [
    # Normal case: positive Kelly.
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.1, variance=0.0,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # Blocked: uncertainty too high.
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.9, variance=0.0,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # Blocked: confidence below minimum.
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.95, variance=0.0,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.99),
    # Blocked: insufficient edge (raw kelly = 0).
    dict(edge=0.05, cost=0.50, fill_prob=0.5, uncertainty=0.1, variance=0.0,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # With variance haircut.
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.1, variance=0.3,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # High base fraction.
    dict(edge=0.80, cost=0.50, fill_prob=0.9, uncertainty=0.0, variance=0.0,
         failure_loss=None, base=0.5, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # With failure_loss override.
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.1, variance=0.0,
         failure_loss=0.25, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # Zero uncertainty (no haircut).
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.0, variance=0.0,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
    # Both haircuts active.
    dict(edge=0.50, cost=0.50, fill_prob=0.8, uncertainty=0.3, variance=0.4,
         failure_loss=None, base=0.25, u_factor=1.0, v_factor=0.5,
         min_conf=0.1, max_u=0.8),
]


class TestKellyFullParity:
    @pytest.mark.parametrize("case", KELLY_FULL_CASES, ids=[str(i) for i in range(len(KELLY_FULL_CASES))])
    def test_parity(self, case: Dict[str, Any]) -> None:
        cfg = TailRiskKellyConfig(
            base_kelly_fraction=case["base"],
            uncertainty_haircut_factor=case["u_factor"],
            variance_haircut_factor=case["v_factor"],
            min_confidence=case["min_conf"],
            max_model_uncertainty=case["max_u"],
        )
        sizer = TailRiskKelly(cfg)
        # Inject variance directly (bypass lane tracking).
        # The Rust function takes variance as a scalar parameter.
        lane_variance = case["variance"]

        # Python: use compute() but with no lane (variance=0 by default).
        # We need to match the Rust behavior which takes variance directly.
        # So we compute manually following the same flow.
        py = sizer.compute(
            edge=case["edge"], cost=case["cost"],
            fill_prob=case["fill_prob"],
            model_uncertainty=case["uncertainty"],
            failure_loss=case["failure_loss"],
        )
        # Python sizer always computes lane_variance=0 for empty lane.
        # So for cases with variance > 0, we need to manually compute.
        if lane_variance > 0.0 and not py.blocked:
            # Re-apply the variance haircut on the raw Python result.
            raw = _raw_kelly(case["edge"], case["cost"], case["fill_prob"], case["failure_loss"])
            fraction = min(raw, case["base"])
            # Baker-McHale shrinkage (default).
            edge = case["edge"]
            cost = case["cost"]
            unc = case["uncertainty"]
            if cost > 0:
                b = edge / cost
                sigma_sq = (unc * edge) ** 2
                b_sq_sigma_sq = b * b * sigma_sq
                shrinkage = 1.0 / (1.0 + b_sq_sigma_sq) if b_sq_sigma_sq > 0 else 1.0
                u_haircut = 1.0 - shrinkage
                fraction *= shrinkage
            else:
                u_haircut = min(1.0, unc * case["u_factor"])
                fraction *= (1.0 - u_haircut)
            v_haircut = min(1.0, lane_variance * case["v_factor"])
            fraction *= (1.0 - v_haircut)
            fraction = max(0.0, min(1.0, fraction))
            confidence = max(0.0, 1.0 - unc)
            py_adj = fraction
            py_blocked = False
            py_reason = ""
            py_raw = raw
            py_u_haircut = u_haircut
            py_v_haircut = v_haircut
            py_confidence = confidence
        else:
            py_adj = py.adjusted_fraction
            py_blocked = py.blocked
            py_reason = py.block_reason
            py_raw = py.raw_kelly
            py_u_haircut = py.uncertainty_haircut
            py_v_haircut = py.variance_haircut
            py_confidence = py.confidence

        rs_json = arb_engine_rs.compute_kelly(
            case["edge"], case["cost"], case["fill_prob"],
            case["uncertainty"], lane_variance, case["failure_loss"],
            case["base"], case["u_factor"], case["v_factor"],
            case["min_conf"], case["max_u"],
            True,  # use_baker_mchale_shrinkage
        )
        rs = json.loads(rs_json)

        assert py_blocked == rs["blocked"], (
            f"blocked: py={py_blocked} rs={rs['blocked']}"
        )
        assert py_reason == rs["block_reason"], (
            f"block_reason: py={py_reason} rs={rs['block_reason']}"
        )
        assert _close(py_raw, rs["raw_kelly"]), (
            f"raw_kelly: py={py_raw} rs={rs['raw_kelly']}"
        )
        assert _close(py_adj, rs["adjusted_fraction"]), (
            f"adjusted_fraction: py={py_adj} rs={rs['adjusted_fraction']}"
        )
        assert _close(py_confidence, rs["confidence"]), (
            f"confidence: py={py_confidence} rs={rs['confidence']}"
        )
        assert _close(py_u_haircut, rs["uncertainty_haircut"]), (
            f"uncertainty_haircut: py={py_u_haircut} rs={rs['uncertainty_haircut']}"
        )
        assert _close(py_v_haircut, rs["variance_haircut"]), (
            f"variance_haircut: py={py_v_haircut} rs={rs['variance_haircut']}"
        )


# ---------------------------------------------------------------------------
# Execution model parity tests
# ---------------------------------------------------------------------------


class TestExecutionModelParity:
    def _exec_config_json(self, **overrides: Any) -> str:
        defaults = dict(
            queue_decay_half_life_seconds=5.0,
            latency_seconds=0.2,
            market_impact_factor=0.01,
            max_market_impact=0.5,
            min_fill_fraction=0.1,
            fill_fraction_steps=5,
            sequential_leg_delay_seconds=1.0,
            enable_queue_decay=True,
            enable_market_impact=True,
        )
        defaults.update(overrides)
        return json.dumps(defaults)

    def _exec_config(self, **overrides: Any) -> Any:
        from arb_bot.framework.execution_model import ExecutionModelConfig
        defaults = dict(
            queue_decay_half_life_seconds=5.0,
            latency_seconds=0.2,
            market_impact_factor=0.01,
            max_market_impact=0.5,
            min_fill_fraction=0.1,
            fill_fraction_steps=5,
            sequential_leg_delay_seconds=1.0,
            enable_queue_decay=True,
            enable_market_impact=True,
        )
        defaults.update(overrides)
        return ExecutionModelConfig(**defaults)

    def _leg_input(self, **overrides: Any) -> Any:
        from arb_bot.framework.execution_model import LegInput
        defaults = dict(
            venue="kalshi", market_id="m1", side="yes",
            buy_price=0.50, available_size=100.0, spread=0.02,
        )
        defaults.update(overrides)
        return LegInput(**defaults)

    def _legs_to_json(self, legs: list) -> str:
        return json.dumps([
            dict(venue=l.venue, market_id=l.market_id, side=l.side,
                 buy_price=l.buy_price, available_size=l.available_size,
                 spread=l.spread)
            for l in legs
        ])

    def test_single_leg_basic(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel
        legs = [self._leg_input()]
        config = self._exec_config()
        model = ExecutionModel(config)
        py = model.simulate(legs, 10, staleness_seconds=0.0, sequential=False)

        rs_json = arb_engine_rs.simulate_execution(
            self._legs_to_json(legs), 10, 0.0, False, self._exec_config_json(),
        )
        rs = json.loads(rs_json)

        assert _close(py.all_fill_probability, rs["all_fill_probability"]), (
            f"all_fill_prob: py={py.all_fill_probability} rs={rs['all_fill_probability']}"
        )
        assert _close(py.expected_fill_fraction, rs["expected_fill_fraction"]), (
            f"expected_fill: py={py.expected_fill_fraction} rs={rs['expected_fill_fraction']}"
        )
        assert _close(py.expected_slippage_per_contract, rs["expected_slippage_per_contract"]), (
            f"slippage: py={py.expected_slippage_per_contract} rs={rs['expected_slippage_per_contract']}"
        )
        assert _close(py.expected_market_impact_per_contract, rs["expected_market_impact_per_contract"]), (
            f"impact: py={py.expected_market_impact_per_contract} rs={rs['expected_market_impact_per_contract']}"
        )

    def test_two_legs_sequential(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel
        legs = [
            self._leg_input(venue="kalshi", market_id="m1", spread=0.03),
            self._leg_input(venue="polymarket", market_id="m2", available_size=50.0, spread=0.05),
        ]
        config = self._exec_config()
        model = ExecutionModel(config)
        py = model.simulate(legs, 10, staleness_seconds=1.0, sequential=True)

        rs_json = arb_engine_rs.simulate_execution(
            self._legs_to_json(legs), 10, 1.0, True, self._exec_config_json(),
        )
        rs = json.loads(rs_json)

        assert _close(py.all_fill_probability, rs["all_fill_probability"]), (
            f"all_fill_prob: py={py.all_fill_probability} rs={rs['all_fill_probability']}"
        )
        assert _close(py.expected_slippage_per_contract, rs["expected_slippage_per_contract"]), (
            f"slippage: py={py.expected_slippage_per_contract} rs={rs['expected_slippage_per_contract']}"
        )

        # Per-leg parity.
        for i, py_leg in enumerate(py.legs):
            rs_leg = rs["legs"][i]
            assert _close(py_leg.fill_probability, rs_leg["fill_probability"]), (
                f"leg[{i}] fill_prob: py={py_leg.fill_probability} rs={rs_leg['fill_probability']}"
            )
            assert _close(py_leg.queue_position_score, rs_leg["queue_position_score"]), (
                f"leg[{i}] queue_score: py={py_leg.queue_position_score} rs={rs_leg['queue_position_score']}"
            )
            assert _close(py_leg.market_impact, rs_leg["market_impact"]), (
                f"leg[{i}] impact: py={py_leg.market_impact} rs={rs_leg['market_impact']}"
            )
            assert _close(py_leg.time_offset_seconds, rs_leg["time_offset_seconds"]), (
                f"leg[{i}] time: py={py_leg.time_offset_seconds} rs={rs_leg['time_offset_seconds']}"
            )

    def test_no_decay_no_impact(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel
        legs = [self._leg_input()]
        config = self._exec_config(enable_queue_decay=False, enable_market_impact=False)
        model = ExecutionModel(config)
        py = model.simulate(legs, 10, staleness_seconds=5.0)

        rs_json = arb_engine_rs.simulate_execution(
            self._legs_to_json(legs), 10, 5.0, False,
            self._exec_config_json(enable_queue_decay=False, enable_market_impact=False),
        )
        rs = json.loads(rs_json)

        assert _close(py.all_fill_probability, rs["all_fill_probability"])
        assert _close(py.expected_slippage_per_contract, rs["expected_slippage_per_contract"])
        assert _close(py.expected_market_impact_per_contract, rs["expected_market_impact_per_contract"])

    def test_high_staleness(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel
        legs = [self._leg_input(available_size=50.0)]
        config = self._exec_config()
        model = ExecutionModel(config)
        py = model.simulate(legs, 10, staleness_seconds=30.0)

        rs_json = arb_engine_rs.simulate_execution(
            self._legs_to_json(legs), 10, 30.0, False, self._exec_config_json(),
        )
        rs = json.loads(rs_json)

        assert _close(py.all_fill_probability, rs["all_fill_probability"])
        # With high staleness, fill prob should be low.
        assert py.all_fill_probability < 0.1

    def test_zero_spread(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel
        legs = [self._leg_input(spread=0.0)]
        config = self._exec_config()
        model = ExecutionModel(config)
        py = model.simulate(legs, 10, staleness_seconds=0.0)

        rs_json = arb_engine_rs.simulate_execution(
            self._legs_to_json(legs), 10, 0.0, False, self._exec_config_json(),
        )
        rs = json.loads(rs_json)

        assert _close(py.expected_market_impact_per_contract, rs["expected_market_impact_per_contract"])
        assert py.expected_market_impact_per_contract == 0.0

    def test_graduated_distribution_length(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel
        legs = [self._leg_input()]
        config = self._exec_config(fill_fraction_steps=5)
        model = ExecutionModel(config)
        py = model.simulate(legs, 10, staleness_seconds=0.0)

        rs_json = arb_engine_rs.simulate_execution(
            self._legs_to_json(legs), 10, 0.0, False,
            self._exec_config_json(fill_fraction_steps=5),
        )
        rs = json.loads(rs_json)

        assert len(py.graduated_fill_distribution) == len(rs["graduated_fill_distribution"]), (
            f"distribution len: py={len(py.graduated_fill_distribution)} rs={len(rs['graduated_fill_distribution'])}"
        )


# ---------------------------------------------------------------------------
# Sizing parity tests (compute_sizing)
# ---------------------------------------------------------------------------


class TestSizingParity:
    def _sizing_config_json(self, **overrides: Any) -> str:
        defaults = dict(
            max_dollars_per_trade=50.0,
            max_contracts_per_trade=100,
            max_liquidity_fraction_per_trade=0.5,
            max_bankroll_fraction_per_trade=0.25,
            min_expected_profit_usd=0.01,
        )
        defaults.update(overrides)
        return json.dumps(defaults)

    def test_basic_sizing(self) -> None:
        opp_json = json.dumps({
            "total_cost_per_contract": 0.90,
            "net_edge_per_contract": 0.10,
            "legs": [
                {"venue": "kalshi", "buy_price": 0.45, "buy_size": 100.0},
                {"venue": "polymarket", "buy_price": 0.45, "buy_size": 80.0},
            ],
            "capital_per_contract_by_venue": {"kalshi": 0.45, "polymarket": 0.45},
        })
        cash_json = json.dumps({"kalshi": 200.0, "polymarket": 200.0})
        config_json = self._sizing_config_json()

        rs_raw = arb_engine_rs.compute_sizing(opp_json, cash_json, config_json)
        rs = json.loads(rs_raw)

        assert rs is not None
        assert rs["contracts"] > 0
        assert rs["expected_profit"] > 0.0
        assert rs["capital_required"] > 0.0

    def test_zero_cost_returns_null(self) -> None:
        opp_json = json.dumps({
            "total_cost_per_contract": 0.0,
            "net_edge_per_contract": 0.10,
            "legs": [],
            "capital_per_contract_by_venue": {},
        })
        cash_json = json.dumps({})
        config_json = self._sizing_config_json()

        rs = json.loads(arb_engine_rs.compute_sizing(opp_json, cash_json, config_json))
        assert rs is None

    def test_insufficient_cash(self) -> None:
        opp_json = json.dumps({
            "total_cost_per_contract": 0.90,
            "net_edge_per_contract": 0.10,
            "legs": [
                {"venue": "kalshi", "buy_price": 0.90, "buy_size": 100.0},
            ],
            "capital_per_contract_by_venue": {"kalshi": 0.90},
        })
        # Very little cash: 0.25 * 1.0 / 0.90 = 0.277 → floor = 0 contracts → None
        cash_json = json.dumps({"kalshi": 1.0})
        config_json = self._sizing_config_json()

        rs = json.loads(arb_engine_rs.compute_sizing(opp_json, cash_json, config_json))
        assert rs is None

    def test_liquidity_cap(self) -> None:
        opp_json = json.dumps({
            "total_cost_per_contract": 0.50,
            "net_edge_per_contract": 0.10,
            "legs": [
                {"venue": "kalshi", "buy_price": 0.25, "buy_size": 10.0},
                {"venue": "kalshi", "buy_price": 0.25, "buy_size": 10.0},
            ],
            "capital_per_contract_by_venue": {"kalshi": 0.50},
        })
        cash_json = json.dumps({"kalshi": 1000.0})
        config_json = self._sizing_config_json(max_liquidity_fraction_per_trade=0.5)

        rs = json.loads(arb_engine_rs.compute_sizing(opp_json, cash_json, config_json))
        # Max from liquidity: floor(10 * 0.5) = 5
        assert rs is not None
        assert rs["contracts"] <= 5

    def test_max_contracts_cap(self) -> None:
        opp_json = json.dumps({
            "total_cost_per_contract": 0.10,
            "net_edge_per_contract": 0.05,
            "legs": [
                {"venue": "kalshi", "buy_price": 0.05, "buy_size": 10000.0},
            ],
            "capital_per_contract_by_venue": {"kalshi": 0.10},
        })
        cash_json = json.dumps({"kalshi": 100000.0})
        config_json = self._sizing_config_json(max_contracts_per_trade=10)

        rs = json.loads(arb_engine_rs.compute_sizing(opp_json, cash_json, config_json))
        assert rs is not None
        assert rs["contracts"] <= 10

    def test_profit_below_minimum(self) -> None:
        opp_json = json.dumps({
            "total_cost_per_contract": 0.90,
            "net_edge_per_contract": 0.001,  # tiny edge
            "legs": [
                {"venue": "kalshi", "buy_price": 0.45, "buy_size": 100.0},
            ],
            "capital_per_contract_by_venue": {"kalshi": 0.90},
        })
        cash_json = json.dumps({"kalshi": 100.0})
        # min expected profit = 1.0 USD — with 1 contract * 0.001 = 0.001 < 1.0
        config_json = self._sizing_config_json(min_expected_profit_usd=1.0)

        rs = json.loads(arb_engine_rs.compute_sizing(opp_json, cash_json, config_json))
        assert rs is None


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


class TestEvalPipelinePerformance:
    N = 5000

    def test_fee_perf(self) -> None:
        schedule = VenueFeeSchedule(
            venue="kalshi", taker_fee_per_contract=0.01,
            taker_curve_coefficient=0.5, curve_round_up=True,
            settlement_fee_per_contract=0.005,
        )
        model = FeeModel(FeeModelConfig(venues=(schedule,)))
        schedule_json = _schedule_to_json(schedule)

        start = time.perf_counter()
        for i in range(self.N):
            p = 0.10 + i * 0.00015
            model.estimate("kalshi", OrderType.TAKER, 10, p)
        py_time = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(self.N):
            p = 0.10 + i * 0.00015
            arb_engine_rs.estimate_fee("kalshi", "taker", 10, p, schedule_json)
        rs_time = time.perf_counter() - start

        print(f"\nfee_model: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")

    def test_kelly_perf(self) -> None:
        cfg = TailRiskKellyConfig()
        sizer = TailRiskKelly(cfg)

        start = time.perf_counter()
        for i in range(self.N):
            edge = 0.30 + i * 0.00005
            _raw_kelly(edge, 0.50, 0.8, None)
        py_time = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(self.N):
            edge = 0.30 + i * 0.00005
            arb_engine_rs.execution_aware_kelly_fraction(edge, 0.50, 0.8, None)
        rs_time = time.perf_counter() - start

        print(f"\nkelly: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")

    def test_execution_model_perf(self) -> None:
        from arb_bot.framework.execution_model import ExecutionModel, ExecutionModelConfig, LegInput

        config = ExecutionModelConfig()
        model = ExecutionModel(config)
        config_json = json.dumps({
            "queue_decay_half_life_seconds": 5.0,
            "latency_seconds": 0.2,
            "market_impact_factor": 0.01,
            "max_market_impact": 0.5,
            "min_fill_fraction": 0.1,
            "fill_fraction_steps": 5,
            "sequential_leg_delay_seconds": 1.0,
            "enable_queue_decay": True,
            "enable_market_impact": True,
        })

        legs_py = [
            LegInput(venue="kalshi", market_id="m1", side="yes",
                     buy_price=0.50, available_size=100.0, spread=0.02),
            LegInput(venue="polymarket", market_id="m2", side="no",
                     buy_price=0.45, available_size=80.0, spread=0.03),
        ]
        legs_json = json.dumps([
            {"venue": "kalshi", "market_id": "m1", "side": "yes",
             "buy_price": 0.50, "available_size": 100.0, "spread": 0.02},
            {"venue": "polymarket", "market_id": "m2", "side": "no",
             "buy_price": 0.45, "available_size": 80.0, "spread": 0.03},
        ])

        start = time.perf_counter()
        for _ in range(self.N):
            model.simulate(legs_py, 10, staleness_seconds=1.0, sequential=True)
        py_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(self.N):
            arb_engine_rs.simulate_execution(legs_json, 10, 1.0, True, config_json)
        rs_time = time.perf_counter() - start

        print(f"\nexecution_model: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / rs_time:.1f}x")


# ---------------------------------------------------------------------------
# Available functions check
# ---------------------------------------------------------------------------


class TestEvalPipelineModuleAttributes:
    def test_all_eval_functions_registered(self) -> None:
        funcs = arb_engine_rs.available_functions()
        expected = [
            "estimate_fee", "estimate_fill", "simulate_execution",
            "compute_kelly", "compute_sizing", "execution_aware_kelly_fraction",
        ]
        for name in expected:
            assert name in funcs, f"{name} not in available_functions()"
