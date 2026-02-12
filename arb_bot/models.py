from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OpportunityKind(str, Enum):
    INTRA_VENUE = "intra_venue"
    CROSS_VENUE = "cross_venue"
    STRUCTURAL_BUCKET = "structural_bucket"
    STRUCTURAL_EVENT_TREE = "structural_event_tree"
    STRUCTURAL_PARITY = "structural_parity"


class ExecutionStyle(str, Enum):
    TAKER = "taker"
    MAKER_ESTIMATE = "maker_estimate"


@dataclass(frozen=True)
class BinaryQuote:
    venue: str
    market_id: str
    yes_buy_price: float
    no_buy_price: float
    yes_buy_size: float
    no_buy_size: float
    yes_bid_price: float | None = None
    no_bid_price: float | None = None
    yes_bid_size: float = 0.0
    no_bid_size: float = 0.0
    yes_maker_buy_price: float | None = None
    no_maker_buy_price: float | None = None
    yes_maker_buy_size: float = 0.0
    no_maker_buy_size: float = 0.0
    fee_per_contract: float = 0.0
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def market_text(self) -> str:
        for key in ("canonical_text", "title", "question", "name"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


@dataclass(frozen=True)
class OpportunityLeg:
    venue: str
    market_id: str
    side: Side
    buy_price: float
    buy_size: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArbitrageOpportunity:
    kind: OpportunityKind
    execution_style: ExecutionStyle
    legs: tuple[OpportunityLeg, ...]
    gross_edge_per_contract: float
    net_edge_per_contract: float
    fee_per_contract: float
    observed_at: datetime
    match_key: str
    match_score: float = 1.0
    payout_per_contract: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost_per_contract(self) -> float:
        return sum(leg.buy_price for leg in self.legs) + self.fee_per_contract

    @property
    def venues(self) -> set[str]:
        return {leg.venue for leg in self.legs}

    @property
    def market_ids(self) -> tuple[str, ...]:
        return tuple(leg.market_id for leg in self.legs)

    @property
    def capital_per_contract_by_venue(self) -> Dict[str, float]:
        by_venue: Dict[str, float] = {}
        for leg in self.legs:
            by_venue[leg.venue] = by_venue.get(leg.venue, 0.0) + leg.buy_price

        venues = list(by_venue.keys())
        if venues and self.fee_per_contract:
            fee_split = self.fee_per_contract / len(venues)
            for venue in venues:
                by_venue[venue] += fee_split

        return by_venue


@dataclass(frozen=True)
class TradeLegPlan:
    venue: str
    market_id: str
    side: Side
    contracts: int
    limit_price: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradePlan:
    kind: OpportunityKind
    execution_style: ExecutionStyle
    legs: tuple[TradeLegPlan, ...]
    contracts: int
    capital_required: float
    capital_required_by_venue: Dict[str, float]
    expected_profit: float
    edge_per_contract: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def venues(self) -> set[str]:
        return {leg.venue for leg in self.legs}

    @property
    def market_key(self) -> str:
        return ":".join(sorted(f"{leg.venue}/{leg.market_id}" for leg in self.legs))

    @property
    def venue(self) -> str | None:
        if len(self.venues) == 1:
            return next(iter(self.venues))
        return None

    @property
    def market_id(self) -> str:
        ids = {leg.market_id for leg in self.legs}
        if len(ids) == 1:
            return next(iter(ids))
        return self.market_key


@dataclass(frozen=True)
class LegExecutionResult:
    success: bool
    order_id: Optional[str]
    requested_contracts: int
    filled_contracts: int
    average_price: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairExecutionResult:
    venue: str
    market_id: str
    yes_leg: LegExecutionResult
    no_leg: LegExecutionResult
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.yes_leg.success and self.no_leg.success

    @property
    def filled_contracts(self) -> int:
        return min(self.yes_leg.filled_contracts, self.no_leg.filled_contracts)


@dataclass(frozen=True)
class PlannedLegExecutionResult:
    leg: TradeLegPlan
    result: LegExecutionResult


@dataclass(frozen=True)
class MultiLegExecutionResult:
    legs: tuple[PlannedLegExecutionResult, ...]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and all(item.result.success for item in self.legs)

    @property
    def filled_contracts(self) -> int:
        if not self.legs:
            return 0
        return min(item.result.filled_contracts for item in self.legs)


@dataclass
class EngineState:
    cash_by_venue: Dict[str, float]
    locked_capital_by_venue: Dict[str, float] = field(default_factory=dict)
    open_markets_by_venue: Dict[str, set[str]] = field(default_factory=dict)
    last_trade_ts_by_market: Dict[tuple[str, str], float] = field(default_factory=dict)
    last_trade_ts_by_opportunity: Dict[tuple[str, str], float] = field(default_factory=dict)

    def cash_for(self, venue: str) -> float:
        return self.cash_by_venue.get(venue, 0.0)

    def locked_for(self, venue: str) -> float:
        return self.locked_capital_by_venue.get(venue, 0.0)

    def mark_open_market(self, venue: str, market_id: str) -> None:
        self.open_markets_by_venue.setdefault(venue, set()).add(market_id)

    def unmark_open_market(self, venue: str, market_id: str) -> None:
        markets = self.open_markets_by_venue.get(venue)
        if not markets:
            return
        markets.discard(market_id)
        if not markets:
            self.open_markets_by_venue.pop(venue, None)

    def open_market_count(self, venue: str) -> int:
        return len(self.open_markets_by_venue.get(venue, set()))
