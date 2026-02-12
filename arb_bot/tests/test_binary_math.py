import pytest

from arb_bot.binary_math import (
    build_quote_diagnostics,
    choose_effective_buy_price,
    decompose_binary_quote,
    reconstruct_binary_quote,
)


def test_quote_decomposition_roundtrip() -> None:
    decomposition = decompose_binary_quote(yes_price=0.57, no_price=0.48)
    assert decomposition.implied_probability == pytest.approx(0.545)
    assert decomposition.edge_per_side == pytest.approx(0.025)

    yes, no = reconstruct_binary_quote(
        implied_probability=decomposition.implied_probability,
        edge_per_side=decomposition.edge_per_side,
    )
    assert yes == pytest.approx(0.57)
    assert no == pytest.approx(0.48)


def test_effective_buy_prefers_opposite_bid_transform_when_cheaper() -> None:
    effective = choose_effective_buy_price(
        side="yes",
        direct_ask_price=0.62,
        direct_ask_size=100,
        opposite_bid_price=0.45,  # implies 1 - 0.45 = 0.55
        opposite_bid_size=80,
    )

    assert effective is not None
    assert effective.price == pytest.approx(0.55)
    assert effective.size == pytest.approx(80)
    assert effective.source == "opposite_bid_transform"


def test_effective_buy_falls_back_to_direct_ask() -> None:
    effective = choose_effective_buy_price(
        side="no",
        direct_ask_price=0.49,
        direct_ask_size=25,
        opposite_bid_price=None,
        opposite_bid_size=0,
    )

    assert effective is not None
    assert effective.price == pytest.approx(0.49)
    assert effective.size == pytest.approx(25)
    assert effective.source == "direct_ask"


def test_quote_diagnostics_calculations() -> None:
    diagnostics = build_quote_diagnostics(
        yes_buy_price=0.55,
        no_buy_price=0.50,
        yes_bid_price=0.53,
        no_bid_price=0.47,
    )

    assert diagnostics.ask_implied_probability == pytest.approx(0.525)
    assert diagnostics.ask_edge_per_side == pytest.approx(0.025)
    assert diagnostics.bid_implied_probability == pytest.approx(0.53)
    assert diagnostics.bid_edge_per_side == pytest.approx(0.0)
    assert diagnostics.midpoint_consistency_gap == pytest.approx(0.005)
    assert diagnostics.yes_spread == pytest.approx(0.02)
    assert diagnostics.no_spread == pytest.approx(0.03)
    assert diagnostics.spread_asymmetry == pytest.approx(-0.01)
