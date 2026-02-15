"""Tests for crypto module configuration."""

from __future__ import annotations

import os
from unittest import mock

from arb_bot.crypto.config import CryptoSettings, load_crypto_settings


class TestCryptoSettingsDefaults:
    def test_defaults(self) -> None:
        s = CryptoSettings()
        assert s.enabled is False
        assert s.symbols == ["KXBTC", "KXETH"]
        assert s.mc_num_paths == 1000
        assert s.min_edge_pct == 0.06
        assert s.paper_mode is True
        assert s.bankroll == 500.0

    def test_frozen(self) -> None:
        s = CryptoSettings()
        try:
            s.enabled = True  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


class TestLoadCryptoSettings:
    def test_all_defaults(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            s = load_crypto_settings()
        assert s.enabled is False
        assert s.mc_num_paths == 1000

    def test_enabled_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_ENABLED": "true"}, clear=True):
            s = load_crypto_settings()
        assert s.enabled is True

    def test_symbols_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_SYMBOLS": "KXBTC,KXSOL"}, clear=True):
            s = load_crypto_settings()
        assert s.symbols == ["KXBTC", "KXSOL"]

    def test_numeric_overrides(self) -> None:
        env = {
            "ARB_CRYPTO_MC_NUM_PATHS": "5000",
            "ARB_CRYPTO_MIN_EDGE_PCT": "0.10",
            "ARB_CRYPTO_BANKROLL": "1000.0",
            "ARB_CRYPTO_SCAN_INTERVAL_SECONDS": "2.5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.mc_num_paths == 5000
        assert s.min_edge_pct == 0.10
        assert s.bankroll == 1000.0
        assert s.scan_interval_seconds == 2.5

    def test_paper_mode_off(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_PAPER_MODE": "false"}, clear=True):
            s = load_crypto_settings()
        assert s.paper_mode is False

    def test_price_feed_override(self) -> None:
        env = {
            "ARB_CRYPTO_PRICE_FEED_SYMBOLS": "btcusdt,solusdt",
            "ARB_CRYPTO_PRICE_HISTORY_MINUTES": "120",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.price_feed_symbols == ["btcusdt", "solusdt"]
        assert s.price_history_minutes == 120

    def test_edge_detection_params(self) -> None:
        env = {
            "ARB_CRYPTO_MAX_MODEL_UNCERTAINTY": "0.20",
            "ARB_CRYPTO_CONFIDENCE_LEVEL": "0.99",
            "ARB_CRYPTO_MIN_EDGE_CENTS": "0.05",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.max_model_uncertainty == 0.20
        assert s.confidence_level == 0.99
        assert s.min_edge_cents == 0.05
