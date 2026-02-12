"""Auto dependency discovery (Phase 6C).

Detect likely market-link constraints from text similarity and
co-movement signals, propose rules for human review.

Usage::

    disco = DependencyDiscovery(config)
    disco.add_market("m1", text="Will BTC exceed 50K by March?", venue="kalshi")
    disco.add_market("m2", text="Bitcoin above 50000 end of March", venue="poly")
    disco.record_price("m1", 0.55, ts=100.0)
    disco.record_price("m2", 0.54, ts=100.0)
    candidates = disco.discover()
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DependencyDiscoveryConfig:
    """Configuration for auto dependency discovery.

    Parameters
    ----------
    text_similarity_threshold:
        Minimum text similarity (0-1) to propose a link. Default 0.40.
    comovement_threshold:
        Minimum absolute price correlation to propose a link.
        Default 0.70.
    min_price_observations:
        Minimum price snapshots before computing co-movement.
        Default 5.
    max_candidates:
        Maximum candidates to return per discovery run. Default 50.
    max_markets:
        Maximum tracked markets. Default 10000.
    price_history_limit:
        Maximum price snapshots per market. Default 100.
    combined_score_threshold:
        Minimum combined score to include a candidate. Default 0.30.
    text_weight:
        Weight for text similarity in combined score. Default 0.5.
    comovement_weight:
        Weight for co-movement in combined score. Default 0.5.
    """

    text_similarity_threshold: float = 0.40
    comovement_threshold: float = 0.70
    min_price_observations: int = 5
    max_candidates: int = 50
    max_markets: int = 10000
    price_history_limit: int = 100
    combined_score_threshold: float = 0.30
    text_weight: float = 0.5
    comovement_weight: float = 0.5


# ---------------------------------------------------------------------------
# Market info
# ---------------------------------------------------------------------------


@dataclass
class MarketInfo:
    """Tracked market information."""

    market_id: str
    text: str
    venue: str
    tokens: Set[str] = field(default_factory=set)
    prices: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Candidate link
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateLink:
    """A proposed dependency between two markets."""

    market_a: str
    market_b: str
    venue_a: str
    venue_b: str
    text_similarity: float
    comovement: float
    combined_score: float
    cross_venue: bool


# ---------------------------------------------------------------------------
# Discovery report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryReport:
    """Report from a discovery run."""

    total_markets: int
    pairs_evaluated: int
    candidates_found: int
    candidates: Tuple[CandidateLink, ...]


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> Set[str]:
    """Tokenize and normalize text for similarity."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = set(text.split())
    # Remove very short tokens.
    return {t for t in tokens if len(t) >= 2}


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation between two price series."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xs = xs[-n:]
    ys = ys[-n:]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return num / (denom_x * denom_y)


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------


class DependencyDiscovery:
    """Discovers likely market-link dependencies.

    Uses text similarity (Jaccard on tokenized market descriptions)
    and price co-movement (Pearson correlation on price histories)
    to propose candidate links for human review.
    """

    def __init__(self, config: DependencyDiscoveryConfig | None = None) -> None:
        self._config = config or DependencyDiscoveryConfig()
        self._markets: Dict[str, MarketInfo] = {}

    @property
    def config(self) -> DependencyDiscoveryConfig:
        return self._config

    def add_market(
        self,
        market_id: str,
        text: str,
        venue: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Add a market for dependency discovery."""
        tokens = _tokenize(text)
        self._markets[market_id] = MarketInfo(
            market_id=market_id,
            text=text,
            venue=venue,
            tokens=tokens,
            metadata=dict(metadata or {}),
        )

    def record_price(
        self,
        market_id: str,
        price: float,
        ts: float | None = None,
    ) -> bool:
        """Record a price snapshot for a market.

        Returns False if market not found.
        """
        if ts is None:
            ts = time.time()
        info = self._markets.get(market_id)
        if info is None:
            return False

        info.prices.append(price)
        info.timestamps.append(ts)

        # Enforce history limit.
        limit = self._config.price_history_limit
        if len(info.prices) > limit:
            info.prices = info.prices[-limit:]
            info.timestamps = info.timestamps[-limit:]

        return True

    def text_similarity(self, market_a: str, market_b: str) -> float:
        """Compute text similarity between two markets."""
        a = self._markets.get(market_a)
        b = self._markets.get(market_b)
        if a is None or b is None:
            return 0.0
        return _jaccard_similarity(a.tokens, b.tokens)

    def comovement(self, market_a: str, market_b: str) -> float:
        """Compute price co-movement between two markets."""
        a = self._markets.get(market_a)
        b = self._markets.get(market_b)
        if a is None or b is None:
            return 0.0
        if len(a.prices) < self._config.min_price_observations:
            return 0.0
        if len(b.prices) < self._config.min_price_observations:
            return 0.0
        return _pearson_correlation(a.prices, b.prices)

    def discover(self) -> DiscoveryReport:
        """Run discovery to find candidate links.

        Evaluates all market pairs for text similarity and co-movement,
        returning ranked candidates above the combined score threshold.
        """
        cfg = self._config
        market_ids = list(self._markets.keys())
        n = len(market_ids)
        candidates: List[CandidateLink] = []
        pairs_evaluated = 0

        for i in range(n):
            for j in range(i + 1, n):
                mid_a = market_ids[i]
                mid_b = market_ids[j]
                info_a = self._markets[mid_a]
                info_b = self._markets[mid_b]
                pairs_evaluated += 1

                # Text similarity.
                text_sim = _jaccard_similarity(info_a.tokens, info_b.tokens)

                # Co-movement.
                comove = 0.0
                if (
                    len(info_a.prices) >= cfg.min_price_observations
                    and len(info_b.prices) >= cfg.min_price_observations
                ):
                    comove = abs(_pearson_correlation(info_a.prices, info_b.prices))

                # Combined score.
                combined = (
                    cfg.text_weight * text_sim
                    + cfg.comovement_weight * comove
                )

                if combined >= cfg.combined_score_threshold:
                    cross = info_a.venue != info_b.venue and info_a.venue != "" and info_b.venue != ""
                    candidates.append(CandidateLink(
                        market_a=mid_a,
                        market_b=mid_b,
                        venue_a=info_a.venue,
                        venue_b=info_b.venue,
                        text_similarity=text_sim,
                        comovement=comove,
                        combined_score=combined,
                        cross_venue=cross,
                    ))

        # Sort by combined score descending.
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        candidates = candidates[: cfg.max_candidates]

        return DiscoveryReport(
            total_markets=n,
            pairs_evaluated=pairs_evaluated,
            candidates_found=len(candidates),
            candidates=tuple(candidates),
        )

    def get_market(self, market_id: str) -> MarketInfo | None:
        """Get market info."""
        return self._markets.get(market_id)

    def market_count(self) -> int:
        """Total tracked markets."""
        return len(self._markets)

    def clear(self) -> None:
        """Clear all markets."""
        self._markets.clear()
