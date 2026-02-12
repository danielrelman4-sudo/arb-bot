from arb_bot.config import KalshiSettings
from arb_bot.exchanges.kalshi import KalshiAdapter


def test_normalize_price_treats_integer_levels_as_cents() -> None:
    assert KalshiAdapter._normalize_price(1) == 0.01
    assert KalshiAdapter._normalize_price(6) == 0.06
    assert KalshiAdapter._normalize_price(100) == 1.0


def test_normalize_price_keeps_decimal_fraction_inputs() -> None:
    assert KalshiAdapter._normalize_price(0.07) == 0.07
    assert KalshiAdapter._normalize_price(1.0) == 1.0


def test_coerce_timestamp_handles_iso_and_milliseconds() -> None:
    iso_value = "2026-02-10T12:34:56Z"
    ms_value = 1_706_000_000_000

    iso_ts = KalshiAdapter._coerce_timestamp(iso_value)
    ms_ts = KalshiAdapter._coerce_timestamp(ms_value)

    assert iso_ts is not None
    assert ms_ts is not None
    assert ms_ts < ms_value


def test_extract_resolution_ts_reads_known_market_fields() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    market = {
        "ticker": "TEST-1",
        "close_time": "2026-02-10T18:00:00Z",
    }

    resolution_ts = adapter._extract_resolution_ts(market)
    assert resolution_ts is not None


def test_quote_from_ticker_message_uses_cached_market_metadata() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    adapter._settings = KalshiSettings(
        min_liquidity=0.0,
        taker_fee_per_contract=0.0,
    )

    cache = {
        "TEST-STREAM-1": {
            "ticker": "TEST-STREAM-1",
            "title": "Streamed test market",
            "liquidity": 100.0,
        }
    }
    message = {
        "market_ticker": "TEST-STREAM-1",
        "yes_bid": 49,
        "yes_ask": 51,
    }

    quote = adapter._quote_from_ticker_message(message, cache)
    assert quote is not None
    assert quote.market_id == "TEST-STREAM-1"
    assert quote.yes_buy_price == 0.51
    assert quote.no_buy_price == 0.51
    assert cache["TEST-STREAM-1"]["yes_bid"] == 49
    assert cache["TEST-STREAM-1"]["yes_ask"] == 51


def test_quote_from_ticker_message_resolves_market_id_via_lookup() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    adapter._settings = KalshiSettings(
        min_liquidity=0.0,
        taker_fee_per_contract=0.0,
    )

    cache = {
        "TEST-STREAM-2": {
            "ticker": "TEST-STREAM-2",
            "market_id": "0xabc",
            "title": "Lookup test market",
            "liquidity": 100.0,
        }
    }
    lookup = {"0xabc": "TEST-STREAM-2"}
    message = {
        "market_id": "0xabc",
        "yes_bid": 49,
        "no_bid": 49,
    }

    quote = adapter._quote_from_ticker_message(message, cache, lookup)
    assert quote is not None
    assert quote.market_id == "TEST-STREAM-2"
    assert lookup["0xabc"] == "TEST-STREAM-2"
    assert cache["TEST-STREAM-2"]["yes_bid"] == 49
    assert cache["TEST-STREAM-2"]["no_bid"] == 49


def test_extract_ticker_messages_handles_wrapped_ticker_payload() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    raw = '{"type":"ticker","sid":1,"seq":2,"msg":{"market_ticker":"ABC","yes_bid":49,"yes_ask":51}}'

    messages = adapter._extract_ticker_messages(raw)
    assert len(messages) == 1
    assert messages[0]["market_ticker"] == "ABC"


def test_extract_ticker_messages_handles_list_payload() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    raw = (
        '{"type":"ticker","sid":1,"seq":2,"msg":['
        '{"market_ticker":"ABC","yes_bid":49,"yes_ask":51},'
        '{"market_ticker":"DEF","yes_bid":48,"yes_ask":52}'
        "]}"
    )

    messages = adapter._extract_ticker_messages(raw)
    assert len(messages) == 2
    assert {item["market_ticker"] for item in messages} == {"ABC", "DEF"}


def test_quote_from_ticker_message_handles_fp_scaled_prices() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    adapter._settings = KalshiSettings(
        min_liquidity=0.0,
        taker_fee_per_contract=0.0,
    )
    cache = {
        "TEST-STREAM-3": {
            "ticker": "TEST-STREAM-3",
            "title": "FP test market",
            "liquidity": 100.0,
        }
    }
    message = {
        "market_ticker": "TEST-STREAM-3",
        "yes_bid_fp": 490000,
        "no_bid_fp": 490000,
    }

    quote = adapter._quote_from_ticker_message(message, cache)
    assert quote is not None
    assert quote.yes_buy_price == 0.51
    assert quote.no_buy_price == 0.51


def test_canonical_ws_signing_path_defaults_to_trade_api_path() -> None:
    assert KalshiAdapter._canonical_ws_signing_path("wss://api.elections.kalshi.com") == "/trade-api/ws/v2"
    assert KalshiAdapter._canonical_ws_signing_path("wss://api.elections.kalshi.com/") == "/trade-api/ws/v2"
    assert (
        KalshiAdapter._canonical_ws_signing_path("wss://api.elections.kalshi.com/trade-api/ws/v2")
        == "/trade-api/ws/v2"
    )


def test_canonical_ws_signing_path_normalizes_short_ws_path() -> None:
    assert KalshiAdapter._canonical_ws_signing_path("wss://api.elections.kalshi.com/ws/v2") == "/trade-api/ws/v2"


def test_priority_refresh_tickers_round_robin_windows() -> None:
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    adapter._priority_refresh_cursor = 0

    universe = ["A", "B", "C", "D", "E"]
    first = adapter._select_priority_refresh_tickers(universe, 2)
    second = adapter._select_priority_refresh_tickers(universe, 2)
    third = adapter._select_priority_refresh_tickers(universe, 2)

    assert first == ["A", "B"]
    assert second == ["C", "D"]
    assert third == ["E", "A"]


def test_kalshi_host_candidates_stick_to_elections_host() -> None:
    candidates = KalshiAdapter._build_kalshi_host_candidates("https://api.kalshi.com/trade-api/v2")
    assert candidates == ["https://api.elections.kalshi.com/trade-api/v2"]


def test_kalshi_ws_candidates_stick_to_elections_host() -> None:
    candidates = KalshiAdapter._build_kalshi_ws_candidates("wss://api.kalshi.com/trade-api/ws/v2")
    assert "wss://api.elections.kalshi.com/trade-api/ws/v2" in candidates
    assert "wss://api.kalshi.com/trade-api/ws/v2" not in candidates


def test_kalshi_host_candidates_include_failover_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("KALSHI_ALLOW_HOST_FAILOVER", "true")
    monkeypatch.setattr(KalshiAdapter, "_hostname_resolves", staticmethod(lambda _host: True))

    candidates = KalshiAdapter._build_kalshi_host_candidates("https://api.kalshi.com/trade-api/v2")
    assert candidates == [
        "https://api.elections.kalshi.com/trade-api/v2",
        "https://api.kalshi.com/trade-api/v2",
    ]


def test_kalshi_ws_candidates_include_failover_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("KALSHI_ALLOW_HOST_FAILOVER", "true")
    monkeypatch.setattr(KalshiAdapter, "_hostname_resolves", staticmethod(lambda _host: True))

    candidates = KalshiAdapter._build_kalshi_ws_candidates("wss://api.kalshi.com/trade-api/ws/v2")
    assert "wss://api.elections.kalshi.com/trade-api/ws/v2" in candidates
    assert "wss://api.kalshi.com/trade-api/ws/v2" in candidates
