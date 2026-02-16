"""Tests for strategy cell classification, config, model adjustments, gates, and sizing."""

from __future__ import annotations

import dataclasses
import math

import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.edge_detector import blend_probabilities
from arb_bot.crypto.strategy_cell import (
    CellConfig,
    StrategyCell,
    classify_cell,
    get_cell_config,
)


# ── Helpers ────────────────────────────────────────────────────────

def _default_settings(**overrides) -> CryptoSettings:
    """Build a CryptoSettings with all defaults, overriding as needed."""
    return CryptoSettings(**overrides) if overrides else CryptoSettings()


# ═══════════════════════════════════════════════════════════════════
# 1. classify_cell() — 4 tests, one per cell
# ═══════════════════════════════════════════════════════════════════

class TestClassifyCell:
    def test_yes_15min(self) -> None:
        assert classify_cell("yes", is_daily=False) is StrategyCell.YES_15MIN

    def test_yes_daily(self) -> None:
        assert classify_cell("yes", is_daily=True) is StrategyCell.YES_DAILY

    def test_no_15min(self) -> None:
        assert classify_cell("no", is_daily=False) is StrategyCell.NO_15MIN

    def test_no_daily(self) -> None:
        assert classify_cell("no", is_daily=True) is StrategyCell.NO_DAILY


# ═══════════════════════════════════════════════════════════════════
# 2. get_cell_config() — 4 tests, verify field mapping for each cell
# ═══════════════════════════════════════════════════════════════════

class TestGetCellConfig:
    def test_yes_15min_fields(self) -> None:
        s = _default_settings()
        cfg = get_cell_config(StrategyCell.YES_15MIN, s)

        assert cfg.model_weight == s.cell_yes_15min_model_weight
        assert cfg.prob_haircut == s.cell_yes_15min_prob_haircut
        assert cfg.empirical_window_minutes == s.cell_yes_15min_empirical_window
        assert cfg.min_edge_pct == s.cell_yes_15min_min_edge_pct
        assert cfg.require_ofi_alignment is True
        assert cfg.ofi_alignment_min == s.cell_yes_15min_ofi_min
        assert cfg.require_trend_confirmation is False
        assert cfg.trend_window_minutes == 0.0
        assert cfg.require_price_past_strike is True
        assert cfg.kelly_multiplier == s.cell_yes_15min_kelly_multiplier
        assert cfg.max_position == s.cell_yes_15min_max_position

    def test_yes_daily_fields(self) -> None:
        s = _default_settings()
        cfg = get_cell_config(StrategyCell.YES_DAILY, s)

        assert cfg.model_weight == s.cell_yes_daily_model_weight
        assert cfg.prob_haircut == s.cell_yes_daily_prob_haircut
        assert cfg.empirical_window_minutes == s.cell_yes_daily_empirical_window
        assert cfg.min_edge_pct == s.cell_yes_daily_min_edge_pct
        assert cfg.require_ofi_alignment is False
        assert cfg.ofi_alignment_min == 0.0
        assert cfg.require_trend_confirmation is True
        assert cfg.trend_window_minutes == s.cell_yes_daily_trend_window_minutes
        assert cfg.require_price_past_strike is False
        assert cfg.kelly_multiplier == s.cell_yes_daily_kelly_multiplier
        assert cfg.max_position == s.cell_yes_daily_max_position

    def test_no_15min_fields(self) -> None:
        s = _default_settings()
        cfg = get_cell_config(StrategyCell.NO_15MIN, s)

        assert cfg.model_weight == s.cell_no_15min_model_weight
        assert cfg.prob_haircut == s.cell_no_15min_prob_haircut
        assert cfg.empirical_window_minutes == s.cell_no_15min_empirical_window
        assert cfg.min_edge_pct == s.cell_no_15min_min_edge_pct
        assert cfg.require_ofi_alignment is False
        assert cfg.require_trend_confirmation is False
        assert cfg.require_price_past_strike is False
        assert cfg.kelly_multiplier == s.cell_no_15min_kelly_multiplier
        assert cfg.max_position == s.cell_no_15min_max_position

    def test_no_daily_fields(self) -> None:
        s = _default_settings()
        cfg = get_cell_config(StrategyCell.NO_DAILY, s)

        assert cfg.model_weight == s.cell_no_daily_model_weight
        assert cfg.prob_haircut == s.cell_no_daily_prob_haircut
        assert cfg.empirical_window_minutes == s.cell_no_daily_empirical_window
        assert cfg.min_edge_pct == s.cell_no_daily_min_edge_pct
        assert cfg.require_ofi_alignment is False
        assert cfg.require_trend_confirmation is False
        assert cfg.require_price_past_strike is False
        assert cfg.kelly_multiplier == s.cell_no_daily_kelly_multiplier
        assert cfg.max_position == s.cell_no_daily_max_position


# ═══════════════════════════════════════════════════════════════════
# 3. CellConfig defaults — frozen dataclass, all fields set
# ═══════════════════════════════════════════════════════════════════

class TestCellConfigDefaults:
    def test_frozen_cannot_mutate(self) -> None:
        cfg = get_cell_config(StrategyCell.NO_DAILY, _default_settings())
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.model_weight = 0.99  # type: ignore[misc]

    def test_all_fields_present(self) -> None:
        cfg = get_cell_config(StrategyCell.YES_15MIN, _default_settings())
        field_names = {f.name for f in dataclasses.fields(cfg)}
        expected = {
            "model_weight",
            "prob_haircut",
            "empirical_window_minutes",
            "min_edge_pct",
            "require_ofi_alignment",
            "ofi_alignment_min",
            "require_trend_confirmation",
            "trend_window_minutes",
            "require_price_past_strike",
            "kelly_multiplier",
            "max_position",
        }
        assert field_names == expected


# ═══════════════════════════════════════════════════════════════════
# 4. Model adjustments — 8 tests
# ═══════════════════════════════════════════════════════════════════

class TestModelAdjustments:
    """Test that per-cell blending weight, haircut, and empirical window
    produce the expected behavioural differences."""

    def test_yes_daily_blending_weight_is_0_4(self) -> None:
        """YES/daily defaults to 0.4 (heavy market deference)."""
        cfg = get_cell_config(StrategyCell.YES_DAILY, _default_settings())
        assert cfg.model_weight == pytest.approx(0.4)

    def test_no_daily_blending_weight_is_0_75(self) -> None:
        """NO/daily defaults to 0.75 (model underconfident => boost)."""
        cfg = get_cell_config(StrategyCell.NO_DAILY, _default_settings())
        assert cfg.model_weight == pytest.approx(0.75)

    def test_yes_daily_prob_haircut_reduces_blended(self) -> None:
        """YES/daily 0.85 haircut reduces the blended probability."""
        cfg = get_cell_config(StrategyCell.YES_DAILY, _default_settings())
        model_prob, market_prob, uncertainty = 0.70, 0.50, 0.05
        blended = blend_probabilities(
            model_prob, market_prob, uncertainty,
            base_model_weight=cfg.model_weight,
        )
        haircut_blended = blended * cfg.prob_haircut
        assert haircut_blended < blended
        assert cfg.prob_haircut == pytest.approx(0.85)

    def test_no_daily_no_haircut(self) -> None:
        """NO/daily haircut is 1.0 — no adjustment."""
        cfg = get_cell_config(StrategyCell.NO_DAILY, _default_settings())
        model_prob, market_prob, uncertainty = 0.70, 0.50, 0.05
        blended = blend_probabilities(
            model_prob, market_prob, uncertainty,
            base_model_weight=cfg.model_weight,
        )
        haircut_blended = blended * cfg.prob_haircut
        assert haircut_blended == pytest.approx(blended)

    def test_haircut_kills_marginal_yes_edge(self) -> None:
        """A marginal YES/daily edge flips negative after haircut.

        If market_implied is 0.50 and blended prob is only slightly above
        the buy price, the 0.85 haircut should push it below, killing the edge.
        """
        cfg = get_cell_config(StrategyCell.YES_DAILY, _default_settings())
        # model barely above market => small edge
        model_prob, market_prob, uncertainty = 0.58, 0.50, 0.05
        blended = blend_probabilities(
            model_prob, market_prob, uncertainty,
            base_model_weight=cfg.model_weight,
        )
        buy_price = 0.50  # yes buy price
        # blended should be above buy_price before haircut
        assert blended > buy_price, f"Pre-haircut blended {blended} should exceed {buy_price}"
        # After haircut, edge should be killed
        haircut_prob = blended * cfg.prob_haircut
        edge_after = haircut_prob - buy_price
        assert edge_after < 0.0, f"Haircut edge {edge_after} should be negative"

    def test_15min_empirical_window_override(self) -> None:
        """15-min cells override empirical window to 30 minutes."""
        s = _default_settings()
        cfg_y15 = get_cell_config(StrategyCell.YES_15MIN, s)
        cfg_n15 = get_cell_config(StrategyCell.NO_15MIN, s)
        assert cfg_y15.empirical_window_minutes == 30
        assert cfg_n15.empirical_window_minutes == 30

    def test_daily_empirical_window_uses_global(self) -> None:
        """Daily cells leave empirical_window_minutes at 0 (use global)."""
        s = _default_settings()
        cfg_yd = get_cell_config(StrategyCell.YES_DAILY, s)
        cfg_nd = get_cell_config(StrategyCell.NO_DAILY, s)
        assert cfg_yd.empirical_window_minutes == 0
        assert cfg_nd.empirical_window_minutes == 0

    def test_blending_yes_daily_less_model_than_no_daily(self) -> None:
        """YES/daily (0.4) should produce a lower model weight than NO/daily (0.75)
        in the blend, meaning YES defers more to market."""
        cfg_yes = get_cell_config(StrategyCell.YES_DAILY, _default_settings())
        cfg_no = get_cell_config(StrategyCell.NO_DAILY, _default_settings())

        model_prob, market_prob, uncertainty = 0.70, 0.50, 0.05
        blended_yes = blend_probabilities(
            model_prob, market_prob, uncertainty,
            base_model_weight=cfg_yes.model_weight,
        )
        blended_no = blend_probabilities(
            model_prob, market_prob, uncertainty,
            base_model_weight=cfg_no.model_weight,
        )
        # YES defers more to market (0.50), so blended is lower
        assert blended_yes < blended_no


# ═══════════════════════════════════════════════════════════════════
# 5. Signal gates — 6 tests
# ═══════════════════════════════════════════════════════════════════

class TestSignalGates:
    """Test that per-cell gate configuration is correct."""

    # ── OFI alignment gate ─────────────────────────────────────────

    def test_ofi_gate_enabled_for_yes_15min(self) -> None:
        cfg = get_cell_config(StrategyCell.YES_15MIN, _default_settings())
        assert cfg.require_ofi_alignment is True
        assert cfg.ofi_alignment_min == pytest.approx(0.3)

    def test_ofi_gate_disabled_for_all_other_cells(self) -> None:
        s = _default_settings()
        for cell in [StrategyCell.YES_DAILY, StrategyCell.NO_15MIN, StrategyCell.NO_DAILY]:
            cfg = get_cell_config(cell, s)
            assert cfg.require_ofi_alignment is False, f"OFI gate should be off for {cell}"

    # ── Price-past-strike gate ─────────────────────────────────────

    def test_price_past_strike_enabled_for_yes_15min(self) -> None:
        cfg = get_cell_config(StrategyCell.YES_15MIN, _default_settings())
        assert cfg.require_price_past_strike is True

    def test_price_past_strike_disabled_for_others(self) -> None:
        s = _default_settings()
        for cell in [StrategyCell.YES_DAILY, StrategyCell.NO_15MIN, StrategyCell.NO_DAILY]:
            cfg = get_cell_config(cell, s)
            assert cfg.require_price_past_strike is False, (
                f"price-past-strike should be off for {cell}"
            )

    # ── Trend confirmation gate ────────────────────────────────────

    def test_trend_gate_enabled_for_yes_daily(self) -> None:
        cfg = get_cell_config(StrategyCell.YES_DAILY, _default_settings())
        assert cfg.require_trend_confirmation is True
        assert cfg.trend_window_minutes == pytest.approx(10.0)

    def test_trend_gate_disabled_for_others(self) -> None:
        s = _default_settings()
        for cell in [StrategyCell.YES_15MIN, StrategyCell.NO_15MIN, StrategyCell.NO_DAILY]:
            cfg = get_cell_config(cell, s)
            assert cfg.require_trend_confirmation is False, (
                f"trend gate should be off for {cell}"
            )


# ═══════════════════════════════════════════════════════════════════
# 6. Sizing — 4 tests
# ═══════════════════════════════════════════════════════════════════

class TestSizing:
    """Per-cell Kelly multiplier and max position."""

    def test_kelly_multiplier_yes_15min(self) -> None:
        cfg = get_cell_config(StrategyCell.YES_15MIN, _default_settings())
        assert cfg.kelly_multiplier == pytest.approx(0.5)

    def test_kelly_multiplier_no_daily_full_size(self) -> None:
        """NO/daily has proven alpha => full Kelly (1.0)."""
        cfg = get_cell_config(StrategyCell.NO_DAILY, _default_settings())
        assert cfg.kelly_multiplier == pytest.approx(1.0)

    def test_max_position_no_daily_largest(self) -> None:
        """NO/daily proven cell has the largest position cap."""
        s = _default_settings()
        caps = {cell: get_cell_config(cell, s).max_position for cell in StrategyCell}
        assert caps[StrategyCell.NO_DAILY] == max(caps.values())
        assert caps[StrategyCell.NO_DAILY] == pytest.approx(50.0)

    def test_max_position_yes_cells_smaller(self) -> None:
        """YES cells (unproven/poor track record) have smaller caps."""
        s = _default_settings()
        yes_cap = get_cell_config(StrategyCell.YES_15MIN, s).max_position
        no_daily_cap = get_cell_config(StrategyCell.NO_DAILY, s).max_position
        assert yes_cap < no_daily_cap


# ═══════════════════════════════════════════════════════════════════
# 7. Integration — 2 tests
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_classify_then_config_roundtrip(self) -> None:
        """classify_cell -> get_cell_config produces a valid CellConfig for every cell."""
        s = _default_settings()
        for side in ("yes", "no"):
            for is_daily in (True, False):
                cell = classify_cell(side, is_daily)
                cfg = get_cell_config(cell, s)
                assert isinstance(cfg, CellConfig)
                assert cfg.model_weight > 0.0
                assert cfg.kelly_multiplier > 0.0
                assert cfg.max_position > 0.0

    def test_custom_settings_override_propagates(self) -> None:
        """dataclasses.replace on CryptoSettings propagates to CellConfig."""
        s = _default_settings()
        s2 = dataclasses.replace(s, cell_no_daily_kelly_multiplier=0.42)
        cfg = get_cell_config(StrategyCell.NO_DAILY, s2)
        assert cfg.kelly_multiplier == pytest.approx(0.42)
