from __future__ import annotations

from arb_bot.cross_mapping_generator import generate_cross_venue_mapping_rows


def _k_market(market_id: str, title: str, subtitle: str) -> dict:
    return {
        "venue": "kalshi",
        "ticker": market_id,
        "title": title,
        "subtitle": subtitle,
        "status": "open",
        "liquidity": 100.0,
    }


def _p_market(market_id: str, question: str, event_title: str) -> dict:
    return {
        "venue": "polymarket",
        "conditionId": market_id,
        "question": question,
        "event_title": event_title,
        "active": True,
        "closed": False,
        "liquidity": 100.0,
    }


def test_generate_cross_mapping_rows_matches_simple_pairs() -> None:
    markets = [
        _k_market("KX-A", "Will Alice win the election?", "Alice"),
        _k_market("KX-B", "Will Bob win the election?", "Bob"),
        _p_market("0x1", "Will Alice win the election?", "Election"),
        _p_market("0x2", "Will Bob win the election?", "Election"),
    ]

    rows, diagnostics = generate_cross_venue_mapping_rows(
        markets,
        min_match_score=0.45,
        min_shared_tokens=2,
        min_score_gap=0.0,
    )

    assert len(rows) == 2
    mapped = {(row["kalshi_market_id"], row["polymarket_market_id"]) for row in rows}
    assert ("KX-A", "0x1") in mapped
    assert ("KX-B", "0x2") in mapped
    assert diagnostics.mappings_emitted == 2


def test_generate_cross_mapping_rows_skips_ambiguous_top_match() -> None:
    markets = [
        _k_market("KX-ALICE", "Will Alice win the election?", "Alice"),
        _p_market("0x1", "Will Alice win the election?", "Election"),
        _p_market("0x2", "Will Alice win election?", "Election"),
    ]

    rows, diagnostics = generate_cross_venue_mapping_rows(
        markets,
        min_match_score=0.45,
        min_shared_tokens=2,
        min_score_gap=0.25,
    )

    assert rows == []
    assert diagnostics.skip_reasons.get("ambiguous_top_match", 0) == 1
