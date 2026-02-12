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
