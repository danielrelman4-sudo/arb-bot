import json

from arb_bot.config import load_settings


def test_load_settings_normalizes_legacy_kalshi_hosts(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.kalshi.com/trade-api/v2")
    monkeypatch.setenv("KALSHI_WS_URL", "wss://api.kalshi.com/trade-api/ws/v2")

    settings = load_settings()

    assert settings.kalshi.api_base_url == "https://api.elections.kalshi.com/trade-api/v2"
    assert settings.kalshi.ws_url == "wss://api.elections.kalshi.com/trade-api/ws/v2"


def test_load_settings_stream_rest_topup_defaults_off(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.kalshi.stream_allow_rest_topup is False
    assert settings.kalshi.stream_bootstrap_scan_pages == 3
    assert settings.kalshi.stream_bootstrap_enrich_limit == 0


def test_load_settings_stream_rest_topup_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KALSHI_STREAM_ALLOW_REST_TOPUP", "true")
    monkeypatch.setenv("KALSHI_STREAM_BOOTSTRAP_SCAN_PAGES", "5")
    monkeypatch.setenv("KALSHI_STREAM_BOOTSTRAP_ENRICH_LIMIT", "25")

    settings = load_settings()

    assert settings.kalshi.stream_allow_rest_topup is True
    assert settings.kalshi.stream_bootstrap_scan_pages == 5
    assert settings.kalshi.stream_bootstrap_enrich_limit == 25


def test_load_settings_derives_polymarket_priority_ids_from_map_and_rules(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    cross_map = tmp_path / "cross.csv"
    cross_map.write_text(
        "group_id,kalshi_market_id,polymarket_market_id\n"
        "g1,K1,0xaaa\n"
        "g2,K2,0xbbb\n",
        encoding="utf-8",
    )
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "b1",
                        "legs": [
                            {"venue": "polymarket", "market_id": "0xccc", "side": "yes"},
                            {"venue": "kalshi", "market_id": "K3", "side": "yes"},
                        ],
                    }
                ],
                "event_trees": [],
                "cross_market_parity_checks": [
                    {
                        "group_id": "p1",
                        "relationship": "equivalent",
                        "left": {"venue": "kalshi", "market_id": "K4", "side": "yes"},
                        "right": {"venue": "polymarket", "market_id": "0xddd", "side": "yes"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARB_CROSS_VENUE_MAPPING_PATH", str(cross_map))
    monkeypatch.setenv("ARB_STRUCTURAL_RULES_PATH", str(rules))

    settings = load_settings()
    assert settings.polymarket.priority_market_ids[:4] == ["0xaaa", "0xbbb", "0xccc", "0xddd"]


def test_load_settings_polymarket_priority_ids_hard_cap(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    cross_map = tmp_path / "cross.csv"
    cross_map.write_text(
        "group_id,kalshi_market_id,polymarket_market_id\n"
        "g1,K1,0xaaa\n"
        "g2,K2,0xbbb\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARB_CROSS_VENUE_MAPPING_PATH", str(cross_map))
    monkeypatch.setenv("POLYMARKET_PRIORITY_MARKET_IDS_HARD_CAP", "1")

    settings = load_settings()
    assert settings.polymarket.priority_market_ids == ["0xaaa"]


def test_load_settings_parses_lane_specific_kelly_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARB_LANE_CROSS_USE_EXECUTION_AWARE_KELLY", "false")
    monkeypatch.setenv("ARB_LANE_CROSS_KELLY_FRACTION_MULTIPLIER", "1.7")
    monkeypatch.setenv("ARB_LANE_CROSS_KELLY_FRACTION_FLOOR", "0.08")
    monkeypatch.setenv("ARB_LANE_CROSS_KELLY_FRACTION_CAP", "0.55")

    settings = load_settings()
    lane = settings.lanes.cross_venue

    assert lane.use_execution_aware_kelly is False
    assert lane.kelly_fraction_multiplier == 1.7
    assert lane.kelly_fraction_floor == 0.08
    assert lane.kelly_fraction_cap == 0.55


def test_load_settings_parses_lane_fill_autotune_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARB_ENABLE_LANE_FILL_AUTOTUNE", "true")
    monkeypatch.setenv("ARB_LANE_FILL_AUTOTUNE_STEP", "0.03")
    monkeypatch.setenv("ARB_LANE_FILL_AUTOTUNE_DECAY_STEP", "0.02")
    monkeypatch.setenv("ARB_LANE_FILL_AUTOTUNE_MIN_FILL_PROBABILITY", "0.18")
    monkeypatch.setenv("ARB_LANE_FILL_AUTOTUNE_MIN_DETECTED", "25")
    monkeypatch.setenv("ARB_LANE_FILL_AUTOTUNE_FILL_SKIP_RATIO", "0.6")
    monkeypatch.setenv("ARB_LANE_FILL_AUTOTUNE_MAX_RELAXATION", "0.12")

    settings = load_settings()
    fill = settings.fill_model

    assert fill.enable_lane_fill_autotune is True
    assert fill.lane_fill_autotune_step == 0.03
    assert fill.lane_fill_autotune_decay_step == 0.02
    assert fill.lane_fill_autotune_min_fill_probability == 0.18
    assert fill.lane_fill_autotune_min_detected == 25
    assert fill.lane_fill_autotune_fill_skip_ratio == 0.6
    assert fill.lane_fill_autotune_max_relaxation == 0.12


def test_load_settings_parses_stream_hardening_and_checkpoint_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARB_STREAM_POLL_ONLY_ON_STALE", "false")
    monkeypatch.setenv("ARB_STREAM_STALE_DEGRADE_SECONDS", "90")
    monkeypatch.setenv("ARB_STREAM_RECOVERY_ATTEMPT_SECONDS", "45")
    monkeypatch.setenv("ARB_PAPER_CHECKPOINT_FLUSH_ROWS", "5")
    monkeypatch.setenv("ARB_PAPER_CHECKPOINT_FSYNC", "true")

    settings = load_settings()

    assert settings.stream_poll_only_on_stale is False
    assert settings.stream_stale_degrade_seconds == 90.0
    assert settings.stream_recovery_attempt_seconds == 45.0
    assert settings.paper_checkpoint_flush_rows == 5
    assert settings.paper_checkpoint_fsync is True


def test_load_settings_parses_non_intra_open_market_reserve(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARB_NON_INTRA_OPEN_MARKET_RESERVE_PER_VENUE", "4")

    settings = load_settings()

    assert settings.risk.non_intra_open_market_reserve_per_venue == 4
