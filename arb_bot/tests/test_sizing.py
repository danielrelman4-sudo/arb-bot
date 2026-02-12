from datetime import datetime, timezone

from arb_bot.config import SizingSettings, StrategySettings
from arb_bot.models import ArbitrageOpportunity, ExecutionStyle, OpportunityKind, OpportunityLeg, Side
from arb_bot.sizing import PositionSizer


def test_position_sizer_respects_caps() -> None:
    sizer = PositionSizer(
        sizing=SizingSettings(
            max_dollars_per_trade=20,
            max_contracts_per_trade=50,
            max_bankroll_fraction_per_trade=0.2,
        ),
        strategy=StrategySettings(min_net_edge_per_contract=0.01, min_expected_profit_usd=0.5),
    )

    opp = ArbitrageOpportunity(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            OpportunityLeg(
                venue="kalshi",
                market_id="X",
                side=Side.YES,
                buy_price=0.45,
                buy_size=100,
            ),
            OpportunityLeg(
                venue="kalshi",
                market_id="X",
                side=Side.NO,
                buy_price=0.50,
                buy_size=100,
            ),
        ),
        gross_edge_per_contract=0.05,
        net_edge_per_contract=0.05,
        fee_per_contract=0.0,
        observed_at=datetime.now(timezone.utc),
        match_key="kalshi:X",
    )

    plan = sizer.build_trade_plan(opp, available_cash_by_venue={"kalshi": 100})
    assert plan is not None
    # Cost per contract is 0.95, so dollar cap (20) limits to 21 contracts.
    assert plan.contracts == 21


def test_cross_venue_sizing_respects_tightest_venue_cash() -> None:
    sizer = PositionSizer(
        sizing=SizingSettings(
            max_dollars_per_trade=100,
            max_contracts_per_trade=200,
            max_bankroll_fraction_per_trade=0.5,
        ),
        strategy=StrategySettings(min_net_edge_per_contract=0.01, min_expected_profit_usd=0.01),
    )

    opp = ArbitrageOpportunity(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            OpportunityLeg(
                venue="kalshi",
                market_id="K",
                side=Side.YES,
                buy_price=0.45,
                buy_size=100,
            ),
            OpportunityLeg(
                venue="polymarket",
                market_id="P",
                side=Side.NO,
                buy_price=0.40,
                buy_size=100,
            ),
        ),
        gross_edge_per_contract=0.15,
        net_edge_per_contract=0.15,
        fee_per_contract=0.0,
        observed_at=datetime.now(timezone.utc),
        match_key="btc-100k",
    )

    plan = sizer.build_trade_plan(
        opp,
        available_cash_by_venue={"kalshi": 40, "polymarket": 1000},
    )

    assert plan is not None
    # Kalshi side costs 0.45/contract. With 40 cash and 50% fraction, cap = floor(20 / 0.45) = 44.
    assert plan.contracts == 44


def test_position_sizer_applies_liquidity_fraction_cap() -> None:
    sizer = PositionSizer(
        sizing=SizingSettings(
            max_dollars_per_trade=1000,
            max_contracts_per_trade=500,
            max_bankroll_fraction_per_trade=1.0,
            max_liquidity_fraction_per_trade=0.5,
        ),
        strategy=StrategySettings(min_net_edge_per_contract=0.01, min_expected_profit_usd=0.01),
    )

    opp = ArbitrageOpportunity(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            OpportunityLeg(
                venue="kalshi",
                market_id="L1",
                side=Side.YES,
                buy_price=0.45,
                buy_size=9,
            ),
            OpportunityLeg(
                venue="kalshi",
                market_id="L1",
                side=Side.NO,
                buy_price=0.50,
                buy_size=9,
            ),
        ),
        gross_edge_per_contract=0.05,
        net_edge_per_contract=0.05,
        fee_per_contract=0.0,
        observed_at=datetime.now(timezone.utc),
        match_key="kalshi:L1",
    )

    plan = sizer.build_trade_plan(opp, available_cash_by_venue={"kalshi": 1000})
    assert plan is not None
    # Visible depth floor is 9 and we cap to 50%, so max contracts is floor(4.5) = 4.
    assert plan.contracts == 4


def test_execution_aware_kelly_fraction_downweights_low_fill_probability() -> None:
    high_fill = PositionSizer.execution_aware_kelly_fraction(
        edge_per_contract=0.20,
        cost_per_contract=0.80,
        fill_probability=0.95,
    )
    low_fill = PositionSizer.execution_aware_kelly_fraction(
        edge_per_contract=0.20,
        cost_per_contract=0.80,
        fill_probability=0.40,
    )

    assert 0.0 <= low_fill <= 1.0
    assert 0.0 <= high_fill <= 1.0
    assert high_fill > low_fill

    low_failure_loss = PositionSizer.execution_aware_kelly_fraction(
        edge_per_contract=0.02,
        cost_per_contract=0.80,
        fill_probability=0.90,
        failure_loss_per_contract=0.01,
    )
    full_failure_loss = PositionSizer.execution_aware_kelly_fraction(
        edge_per_contract=0.02,
        cost_per_contract=0.80,
        fill_probability=0.90,
        failure_loss_per_contract=0.80,
    )
    assert low_failure_loss >= full_failure_loss
