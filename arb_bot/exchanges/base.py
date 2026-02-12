from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from arb_bot.models import BinaryQuote, LegExecutionResult, PairExecutionResult, TradeLegPlan, TradePlan


class ExchangeAdapter(ABC):
    venue: str

    @abstractmethod
    async def fetch_quotes(self) -> list[BinaryQuote]:
        raise NotImplementedError

    @abstractmethod
    async def place_pair_order(self, plan: TradePlan) -> PairExecutionResult:
        raise NotImplementedError

    @abstractmethod
    async def place_single_order(self, leg: TradeLegPlan) -> LegExecutionResult:
        raise NotImplementedError

    async def fetch_quote(self, market_id: str) -> BinaryQuote | None:
        """Fetch a fresh quote for a single market. Returns None if unavailable."""
        quotes = await self.fetch_quotes()
        for q in quotes:
            if q.market_id == market_id:
                return q
        return None

    async def get_available_cash(self) -> float | None:
        return None

    def supports_streaming(self) -> bool:
        return False

    async def stream_quotes(self) -> AsyncIterator[BinaryQuote]:
        if False:
            yield
        return

    async def aclose(self) -> None:
        return None
