from datetime import datetime, timedelta, timezone

from arb_bot.config import FillModelSettings, UniverseRankingSettings
from arb_bot.fill_model import FillModel
from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
    TradeLegPlan,
    TradePlan,
)
from arb_bot.universe_ranking import rank_quotes


def test_fill_model_returns_fill_adjusted_ev() -> None:
    fill_model = FillModel(
        FillModelSettings(
            enabled=True,
            min_fill_probability=0.2,
            queue_depth_factor=2.0,
            stale_quote_half_life_seconds=30.0,
            spread_penalty_weight=1.0,
            transform_source_penalty=0.1,
            partial_fill_penalty_per_contract=0.02,
            min_expected_realized_profit_usd=0.0,
        )
    )

    opportunity = ArbitrageOpportunity(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            OpportunityLeg(
                venue="kalshi",
                market_id="A",
                side=Side.YES,
                buy_price=0.45,
                buy_size=40,
                metadata={"yes_spread": 0.02, "yes_buy_source": "direct_ask"},
            ),
            OpportunityLeg(
                venue="polymarket",
                market_id="B",
                side=Side.NO,
                buy_price=0.44,
                buy_size=30,
                metadata={"no_spread": 0.03, "no_buy_source": "opposite_bid_transform"},
            ),
        ),
        gross_edge_per_contract=0.11,
        net_edge_per_contract=0.10,
        fee_per_contract=0.01,
        observed_at=datetime.now(timezone.utc),
        match_key="A-B",
    )

    plan = TradePlan(
        kind=opportunity.kind,
        execution_style=opportunity.execution_style,
        legs=(
            TradeLegPlan("kalshi", "A", Side.YES, 20, 0.45, metadata=opportunity.legs[0].metadata),
            TradeLegPlan("polymarket", "B", Side.NO, 20, 0.44, metadata=opportunity.legs[1].metadata),
        ),
        contracts=20,
        capital_required=17.8,
        capital_required_by_venue={"kalshi": 9.0, "polymarket": 8.8},
        expected_profit=2.0,
        edge_per_contract=0.10,
    )

    estimate = fill_model.estimate(opportunity, plan)
    assert 0.0 <= estimate.all_fill_probability <= 1.0
    assert estimate.expected_realized_profit <= plan.expected_profit
    assert estimate.expected_realized_edge_per_contract <= plan.edge_per_contract


def test_universe_ranking_prioritizes_liquid_tight_fresh_quotes() -> None:
    now = datetime.now(timezone.utc)
    settings = UniverseRankingSettings(
        enabled=True,
        hotset_size=0,
        enable_cold_scan_fallback=True,
        volume_weight=1.0,
        liquidity_weight=1.0,
        spread_weight=2.0,
        staleness_weight=0.2,
    )

    hot = BinaryQuote(
        venue="kalshi",
        market_id="HOT",
        yes_buy_price=0.51,
        no_buy_price=0.49,
        yes_buy_size=1000,
        no_buy_size=1000,
        observed_at=now,
        metadata={"volume_24h": 100000, "liquidity": 50000, "yes_spread": 0.01, "no_spread": 0.01},
    )
    cold = BinaryQuote(
        venue="kalshi",
        market_id="COLD",
        yes_buy_price=0.58,
        no_buy_price=0.46,
        yes_buy_size=50,
        no_buy_size=50,
        observed_at=now - timedelta(minutes=10),
        metadata={"volume_24h": 500, "liquidity": 200, "yes_spread": 0.08, "no_spread": 0.07},
    )

    ranked = rank_quotes([cold, hot], settings, now=now)
    assert [quote.market_id for quote in ranked] == ["HOT", "COLD"]
