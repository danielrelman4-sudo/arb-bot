from arb_bot.exchanges.polymarket import PolymarketAdapter


def test_extract_best_level_uses_min_ask_and_max_bid() -> None:
    asks = [
        {"price": "0.99", "size": "10"},
        {"price": "0.45", "size": "7"},
        {"price": "0.63", "size": "8"},
    ]
    bids = [
        {"price": "0.30", "size": "11"},
        {"price": "0.52", "size": "3"},
        {"price": "0.47", "size": "5"},
    ]

    ask_price, ask_size = PolymarketAdapter._extract_best_level(asks, side="ask")
    bid_price, bid_size = PolymarketAdapter._extract_best_level(bids, side="bid")

    assert ask_price == 0.45
    assert ask_size == 7.0
    assert bid_price == 0.52
    assert bid_size == 3.0


def test_extract_depth_level_returns_weighted_average_price() -> None:
    asks = [
        {"price": "0.40", "size": "5"},
        {"price": "0.42", "size": "5"},
        {"price": "0.45", "size": "5"},
    ]
    bids = [
        {"price": "0.60", "size": "3"},
        {"price": "0.58", "size": "4"},
        {"price": "0.55", "size": "10"},
    ]

    ask_price, ask_size = PolymarketAdapter._extract_depth_level(asks, side="ask", target_size=8)
    bid_price, bid_size = PolymarketAdapter._extract_depth_level(bids, side="bid", target_size=6)

    assert ask_size == 8.0
    assert round(ask_price or 0.0, 6) == 0.4075
    assert bid_size == 6.0
    assert round(bid_price or 0.0, 6) == 0.59


def test_resolve_yes_no_indices_strict_rejects_non_binary_labels() -> None:
    assert PolymarketAdapter._resolve_yes_no_indices(["yes", "no"], strict=True) == (0, 1)
    assert PolymarketAdapter._resolve_yes_no_indices(["up", "down"], strict=True) is None
    assert PolymarketAdapter._resolve_yes_no_indices(["yes", "yes"], strict=True) is None


def test_pair_is_fresh_checks_timestamp_skew() -> None:
    assert PolymarketAdapter._pair_is_fresh(100.0, 101.5, max_skew_seconds=2.0) is True
    assert PolymarketAdapter._pair_is_fresh(100.0, 103.5, max_skew_seconds=2.0) is False
