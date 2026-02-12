"""Tests for Phase 4C: Confidence-scaled max-loss cap."""

from __future__ import annotations

import pytest

from arb_bot.loss_cap import LossCapConfig, LossCapManager, LossCapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mgr(**kw) -> LossCapManager:
    return LossCapManager(LossCapConfig(**kw))


BANKROLL = 10_000.0
COST = 0.55


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LossCapConfig()
        assert cfg.base_loss_fraction == 0.02
        assert cfg.min_loss_fraction == 0.002
        assert cfg.confidence_exponent == 1.0
        assert cfg.absolute_max_loss == 500.0
        assert cfg.absolute_min_loss == 1.0

    def test_custom(self) -> None:
        cfg = LossCapConfig(base_loss_fraction=0.05, absolute_max_loss=1000.0)
        assert cfg.base_loss_fraction == 0.05
        assert cfg.absolute_max_loss == 1000.0

    def test_frozen(self) -> None:
        cfg = LossCapConfig()
        with pytest.raises(AttributeError):
            cfg.base_loss_fraction = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Compute — basic
# ---------------------------------------------------------------------------


class TestComputeBasic:
    def test_full_confidence(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=10)
        # loss_fraction = 0.002 + (0.02 - 0.002) * 1.0 = 0.02
        assert r.loss_fraction == pytest.approx(0.02)
        assert r.max_loss_dollars == pytest.approx(200.0)
        assert r.capped is False

    def test_zero_confidence(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=BANKROLL, confidence=0.0,
                      cost_per_contract=COST, contracts=10)
        # loss_fraction = 0.002 + (0.02 - 0.002) * 0.0 = 0.002
        assert r.loss_fraction == pytest.approx(0.002)
        assert r.max_loss_dollars == pytest.approx(20.0)

    def test_half_confidence(self) -> None:
        m = _mgr(confidence_exponent=1.0)
        r = m.compute(bankroll=BANKROLL, confidence=0.5,
                      cost_per_contract=COST, contracts=10)
        assert r.loss_fraction == pytest.approx(0.002 + 0.018 * 0.5)

    def test_higher_confidence_more_loss_budget(self) -> None:
        m = _mgr()
        low = m.compute(bankroll=BANKROLL, confidence=0.3,
                        cost_per_contract=COST, contracts=10)
        high = m.compute(bankroll=BANKROLL, confidence=0.9,
                         cost_per_contract=COST, contracts=10)
        assert high.max_loss_dollars > low.max_loss_dollars


# ---------------------------------------------------------------------------
# Compute — capping
# ---------------------------------------------------------------------------


class TestComputeCapping:
    def test_not_capped_when_within_budget(self) -> None:
        m = _mgr(base_loss_fraction=0.10)  # 10% of 10k = 1000
        r = m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=5)
        # 5 * 0.55 = 2.75 total risk, well within 500 (abs max)
        assert r.capped is False
        assert r.allowed_contracts == 5

    def test_capped_by_loss_budget(self) -> None:
        m = _mgr(base_loss_fraction=0.01, absolute_max_loss=10000.0)
        # Budget = 10000 * 0.01 = 100. Each contract risks 0.55.
        # max_allowed = int(100 / 0.55) = 181
        r = m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=200)
        assert r.capped is True
        assert r.allowed_contracts == 181
        assert r.cap_reason == "loss_budget_exceeded"

    def test_capped_by_absolute_max(self) -> None:
        m = _mgr(base_loss_fraction=0.50, absolute_max_loss=50.0)
        # Bankroll budget = 5000, but abs max = 50.
        r = m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=200)
        assert r.max_loss_dollars == pytest.approx(50.0)
        assert r.allowed_contracts == int(50.0 / 0.55)

    def test_below_absolute_min_loss(self) -> None:
        m = _mgr(absolute_min_loss=100.0, base_loss_fraction=0.001)
        # Budget = 10000 * 0.001 = 10 < 100 min.
        r = m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=5)
        assert r.capped is True
        assert r.allowed_contracts == 0
        assert r.cap_reason == "below_absolute_min_loss"


# ---------------------------------------------------------------------------
# Compute — edge cases
# ---------------------------------------------------------------------------


class TestComputeEdgeCases:
    def test_zero_cost(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=BANKROLL, confidence=0.8,
                      cost_per_contract=0.0, contracts=10)
        assert r.capped is True
        assert r.cap_reason == "zero_loss_per_contract"
        assert r.allowed_contracts == 0

    def test_zero_bankroll(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=0.0, confidence=0.8,
                      cost_per_contract=COST, contracts=10)
        assert r.capped is True
        assert r.allowed_contracts == 0

    def test_zero_contracts_requested(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=BANKROLL, confidence=0.8,
                      cost_per_contract=COST, contracts=0)
        assert r.capped is False
        assert r.allowed_contracts == 0

    def test_confidence_clamped_above_1(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=BANKROLL, confidence=1.5,
                      cost_per_contract=COST, contracts=10)
        assert r.confidence == 1.0

    def test_confidence_clamped_below_0(self) -> None:
        m = _mgr()
        r = m.compute(bankroll=BANKROLL, confidence=-0.5,
                      cost_per_contract=COST, contracts=10)
        assert r.confidence == 0.0

    def test_failure_loss_fraction(self) -> None:
        m = _mgr(base_loss_fraction=0.02, absolute_max_loss=10000.0)
        # Full failure: 0.55 per contract.
        full = m.compute(bankroll=BANKROLL, confidence=1.0,
                         cost_per_contract=COST, contracts=500,
                         failure_loss_fraction=1.0)
        # Half failure: 0.275 per contract → more contracts allowed.
        half = m.compute(bankroll=BANKROLL, confidence=1.0,
                         cost_per_contract=COST, contracts=500,
                         failure_loss_fraction=0.5)
        assert half.allowed_contracts > full.allowed_contracts


# ---------------------------------------------------------------------------
# Confidence exponent
# ---------------------------------------------------------------------------


class TestConfidenceExponent:
    def test_exponent_2_conservative(self) -> None:
        m = _mgr(confidence_exponent=2.0)
        # At conf=0.5: scale = 0.25 (sub-linear).
        r = m.compute(bankroll=BANKROLL, confidence=0.5,
                      cost_per_contract=COST, contracts=10)
        expected_frac = 0.002 + 0.018 * 0.25
        assert r.loss_fraction == pytest.approx(expected_frac)

    def test_exponent_05_aggressive(self) -> None:
        m = _mgr(confidence_exponent=0.5)
        # At conf=0.5: scale = sqrt(0.5) ≈ 0.707 (concave).
        r = m.compute(bankroll=BANKROLL, confidence=0.5,
                      cost_per_contract=COST, contracts=10)
        import math
        expected_frac = 0.002 + 0.018 * math.sqrt(0.5)
        assert r.loss_fraction == pytest.approx(expected_frac)

    def test_exponent_1_linear(self) -> None:
        m = _mgr(confidence_exponent=1.0)
        r = m.compute(bankroll=BANKROLL, confidence=0.5,
                      cost_per_contract=COST, contracts=10)
        expected_frac = 0.002 + 0.018 * 0.5
        assert r.loss_fraction == pytest.approx(expected_frac)


# ---------------------------------------------------------------------------
# Recent cap rate
# ---------------------------------------------------------------------------


class TestRecentCapRate:
    def test_no_history(self) -> None:
        m = _mgr()
        assert m.recent_cap_rate() == 0.0

    def test_all_capped(self) -> None:
        m = _mgr(base_loss_fraction=0.0001, absolute_max_loss=0.01,
                 absolute_min_loss=1.0)
        for _ in range(10):
            m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=100)
        assert m.recent_cap_rate() == 1.0

    def test_none_capped(self) -> None:
        m = _mgr(base_loss_fraction=0.10, absolute_max_loss=10000.0)
        for _ in range(10):
            m.compute(bankroll=BANKROLL, confidence=1.0,
                      cost_per_contract=COST, contracts=1)
        assert m.recent_cap_rate() == 0.0

    def test_mixed(self) -> None:
        m = _mgr(base_loss_fraction=0.01, absolute_max_loss=10000.0)
        # Budget ≈ 100. 1 contract = 0.55 risk → not capped.
        m.compute(bankroll=BANKROLL, confidence=1.0,
                  cost_per_contract=COST, contracts=1)
        # 200 contracts = 110 risk → capped.
        m.compute(bankroll=BANKROLL, confidence=1.0,
                  cost_per_contract=COST, contracts=200)
        assert m.recent_cap_rate() == 0.5


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_history(self) -> None:
        m = _mgr()
        m.compute(bankroll=BANKROLL, confidence=0.8,
                  cost_per_contract=COST, contracts=10)
        m.clear()
        assert m.recent_cap_rate() == 0.0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = LossCapConfig(base_loss_fraction=0.05)
        m = LossCapManager(cfg)
        assert m.config.base_loss_fraction == 0.05


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_scaling_across_confidence_levels(self) -> None:
        """Verify smooth scaling from low to high confidence."""
        m = _mgr()
        prev_allowed = 0
        for conf_pct in range(10, 101, 10):
            conf = conf_pct / 100.0
            r = m.compute(bankroll=BANKROLL, confidence=conf,
                          cost_per_contract=COST, contracts=1000)
            assert r.allowed_contracts >= prev_allowed
            prev_allowed = r.allowed_contracts

    def test_small_bankroll_tight_caps(self) -> None:
        """Small bankroll should tightly cap position sizes."""
        m = _mgr()
        r = m.compute(bankroll=100.0, confidence=1.0,
                      cost_per_contract=COST, contracts=100)
        # Budget = 100 * 0.02 = 2.0. At 0.55/contract → 3 contracts.
        assert r.allowed_contracts <= 4
        assert r.capped is True
