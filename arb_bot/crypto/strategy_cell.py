"""Strategy cell classification for model-path trades.

Classifies each model-path edge into one of 4 strategy cells based on
(side, duration):

    | | 15-min (up/down) | Daily (above/below, TTE>30m) |
    |---|---|---|
    | YES | Microstructure-confirmed | Momentum-confirmed directional |
    | NO | Short-term vol fade | Big-move fade (proven alpha) |

Each cell has its own model adjustments (blending weight, probability
haircut, empirical window), filtering gates (OFI, trend, price-past-strike),
and sizing parameters (Kelly multiplier, max position).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arb_bot.crypto.config import CryptoSettings


class StrategyCell(str, Enum):
    """One of 4 model-path strategy cells."""

    YES_15MIN = "yes_15min"    # Microstructure-confirmed
    YES_DAILY = "yes_daily"    # Momentum-confirmed directional
    NO_15MIN = "no_15min"      # Short-term vol fade
    NO_DAILY = "no_daily"      # Big-move fade (proven alpha)


@dataclass(frozen=True)
class CellConfig:
    """Per-cell trading parameters loaded from CryptoSettings.

    Three layers of adjustment:
    1. Model — vol dampening, blending weight, probability haircut,
       empirical window, uncertainty multiplier
    2. Filtering — signal gates (OFI, trend, price-past-strike)
    3. Sizing — Kelly multiplier, max position

    Vol dampening addresses the ROOT CAUSE of YES overconfidence: the IID
    bootstrap ignores mean-reversion in 1-minute returns, inflating tail
    probabilities.  Each resampled return is multiplied by ``vol_dampening``
    before accumulation.  Values < 1.0 shrink returns toward zero, narrowing
    the terminal distribution and reducing the probability of reaching
    far-OTM strikes.
    """

    # Model adjustments (addresses root cause: model overconfidence)
    model_weight: float             # Blending weight (0-1), replaces global 0.7
    prob_haircut: float             # Multiply blended prob by this (1.0=no haircut)
    vol_dampening: float            # Scale resampled returns (1.0=no change, <1.0=shrink)
    uncertainty_multiplier: float   # Per-cell uncertainty multiplier (replaces global)
    empirical_window_minutes: int   # Empirical CDF lookback (0=use global)

    # Filtering gates
    min_edge_pct: float             # Minimum edge as fraction
    require_ofi_alignment: bool     # Require OFI in trade direction
    ofi_alignment_min: float        # Min |weighted OFI| (0-1)
    require_trend_confirmation: bool  # Require price moving toward strike
    trend_window_minutes: float     # Window for trend check
    require_price_past_strike: bool  # Price must already be past strike

    # Sizing
    kelly_multiplier: float         # Multiplied into Kelly fraction
    max_position: float             # Dollar cap per trade


def classify_cell(side: str, is_daily: bool) -> StrategyCell:
    """Classify a model-path edge into one of 4 strategy cells.

    Parameters
    ----------
    side: "yes" or "no"
    is_daily: True for daily (above/below, TTE>30m), False for 15-min (up/down)
    """
    if side == "yes":
        return StrategyCell.YES_DAILY if is_daily else StrategyCell.YES_15MIN
    else:
        return StrategyCell.NO_DAILY if is_daily else StrategyCell.NO_15MIN


def get_cell_config(cell: StrategyCell, settings: "CryptoSettings") -> CellConfig:
    """Build CellConfig from CryptoSettings for the given cell.

    Each cell maps to a set of ``cell_<cell>_*`` fields on CryptoSettings.
    """
    if cell is StrategyCell.YES_15MIN:
        return CellConfig(
            model_weight=settings.cell_yes_15min_model_weight,
            prob_haircut=settings.cell_yes_15min_prob_haircut,
            vol_dampening=settings.cell_yes_15min_vol_dampening,
            uncertainty_multiplier=settings.cell_yes_15min_uncertainty_mult,
            empirical_window_minutes=settings.cell_yes_15min_empirical_window,
            min_edge_pct=settings.cell_yes_15min_min_edge_pct,
            require_ofi_alignment=settings.cell_yes_15min_require_ofi,
            ofi_alignment_min=settings.cell_yes_15min_ofi_min,
            require_trend_confirmation=False,
            trend_window_minutes=0.0,
            require_price_past_strike=settings.cell_yes_15min_require_price_past_strike,
            kelly_multiplier=settings.cell_yes_15min_kelly_multiplier,
            max_position=settings.cell_yes_15min_max_position,
        )
    elif cell is StrategyCell.YES_DAILY:
        return CellConfig(
            model_weight=settings.cell_yes_daily_model_weight,
            prob_haircut=settings.cell_yes_daily_prob_haircut,
            vol_dampening=settings.cell_yes_daily_vol_dampening,
            uncertainty_multiplier=settings.cell_yes_daily_uncertainty_mult,
            empirical_window_minutes=settings.cell_yes_daily_empirical_window,
            min_edge_pct=settings.cell_yes_daily_min_edge_pct,
            require_ofi_alignment=False,
            ofi_alignment_min=0.0,
            require_trend_confirmation=settings.cell_yes_daily_require_trend,
            trend_window_minutes=settings.cell_yes_daily_trend_window_minutes,
            require_price_past_strike=False,
            kelly_multiplier=settings.cell_yes_daily_kelly_multiplier,
            max_position=settings.cell_yes_daily_max_position,
        )
    elif cell is StrategyCell.NO_15MIN:
        return CellConfig(
            model_weight=settings.cell_no_15min_model_weight,
            prob_haircut=settings.cell_no_15min_prob_haircut,
            vol_dampening=settings.cell_no_15min_vol_dampening,
            uncertainty_multiplier=settings.cell_no_15min_uncertainty_mult,
            empirical_window_minutes=settings.cell_no_15min_empirical_window,
            min_edge_pct=settings.cell_no_15min_min_edge_pct,
            require_ofi_alignment=False,
            ofi_alignment_min=0.0,
            require_trend_confirmation=False,
            trend_window_minutes=0.0,
            require_price_past_strike=False,
            kelly_multiplier=settings.cell_no_15min_kelly_multiplier,
            max_position=settings.cell_no_15min_max_position,
        )
    else:  # NO_DAILY
        return CellConfig(
            model_weight=settings.cell_no_daily_model_weight,
            prob_haircut=settings.cell_no_daily_prob_haircut,
            vol_dampening=settings.cell_no_daily_vol_dampening,
            uncertainty_multiplier=settings.cell_no_daily_uncertainty_mult,
            empirical_window_minutes=settings.cell_no_daily_empirical_window,
            min_edge_pct=settings.cell_no_daily_min_edge_pct,
            require_ofi_alignment=False,
            ofi_alignment_min=0.0,
            require_trend_confirmation=False,
            trend_window_minutes=0.0,
            require_price_past_strike=False,
            kelly_multiplier=settings.cell_no_daily_kelly_multiplier,
            max_position=settings.cell_no_daily_max_position,
        )
