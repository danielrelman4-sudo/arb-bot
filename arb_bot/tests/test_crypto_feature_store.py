"""Tests for the crypto feature store and training data pipeline (C1)."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from arb_bot.crypto.feature_store import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    FeatureStore,
    FeatureVector,
)
from arb_bot.crypto.config import CryptoSettings, load_crypto_settings


# ── Helpers ────────────────────────────────────────────────────────


def _make_fv(
    ticker: str = "KXBTCD-26FEB14-T97500",
    side: str = "yes",
    entry_price: float = 0.50,
    model_prob: float = 0.65,
    edge_cents: float = 0.10,
    **overrides,
) -> FeatureVector:
    """Create a FeatureVector with reasonable defaults."""
    defaults = dict(
        ticker=ticker,
        timestamp=time.time(),
        strike_distance_pct=-0.025,
        time_to_expiry_minutes=10.0,
        implied_probability=0.50,
        spread_cents=0.04,
        book_depth_yes=50,
        book_depth_no=40,
        model_probability=model_prob,
        model_uncertainty=0.05,
        edge_cents=edge_cents,
        blended_probability=0.60,
        staleness_score=0.3,
        vpin=0.45,
        signed_vpin=0.15,
        vpin_trend=0.02,
        ofi_30s=0.3,
        ofi_60s=0.25,
        ofi_120s=0.20,
        ofi_300s=0.15,
        aggressor_ratio=0.55,
        volume_acceleration=1.2,
        funding_rate=0.0001,
        funding_rate_8h_avg=0.0002,
        funding_rate_change=-0.0001,
        realized_vol_1m=0.004,
        realized_vol_5m=0.003,
        vol_ratio=1.33,
        leader_ofi=0.2,
        leader_return_5m=0.001,
        leader_vol_ratio=1.1,
        side=side,
        entry_price=entry_price,
        outcome=-1,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


# ── FeatureStore CSV creation ────────────────────────────────────


class TestFeatureStoreCSVCreation:
    def test_creates_csv_on_init(self, tmp_path: Path) -> None:
        """CSV file is created with headers on initialization."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)
        assert csv_path.exists()

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ALL_COLUMNS

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        csv_path = tmp_path / "sub" / "dir" / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)
        assert csv_path.exists()


# ── Recording entries ─────────────────────────────────────────────


class TestRecordEntry:
    def test_record_entry_writes_row(self, tmp_path: Path) -> None:
        """A recorded entry appears as a row in the CSV."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="KXBTCD-26FEB14-T97500")
        store.record_entry(fv)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "KXBTCD-26FEB14-T97500"
        assert rows[0]["outcome"] == "-1"

    def test_record_entry_stores_in_memory(self, tmp_path: Path) -> None:
        """The ticker is stored in _entries for later outcome labeling."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="KXBTCD-26FEB14-T97500")
        store.record_entry(fv)

        assert "KXBTCD-26FEB14-T97500" in store._entries

    def test_record_entry_preserves_feature_values(self, tmp_path: Path) -> None:
        """Feature values survive the CSV round-trip."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(
            model_probability=0.72,
            edge_cents=0.15,
            vpin=0.88,
            ofi_60s=-0.42,
        )
        store.record_entry(fv)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert abs(float(rows[0]["model_probability"]) - 0.72) < 1e-6
        assert abs(float(rows[0]["edge_cents"]) - 0.15) < 1e-6
        assert abs(float(rows[0]["vpin"]) - 0.88) < 1e-6
        assert abs(float(rows[0]["ofi_60s"]) - (-0.42)) < 1e-6

    def test_record_entry_ignores_empty_ticker(self, tmp_path: Path) -> None:
        """Empty ticker is ignored; no row written."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="")
        store.record_entry(fv)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0


# ── Recording outcomes ────────────────────────────────────────────


class TestRecordOutcome:
    def test_record_outcome_updates_csv(self, tmp_path: Path) -> None:
        """Outcome changes from -1 to the actual value."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="TICKER-A")
        store.record_entry(fv)
        store.record_outcome("TICKER-A", won=True)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["outcome"] == "1"

    def test_record_outcome_win(self, tmp_path: Path) -> None:
        """Won trade gets outcome=1."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="WIN-TRADE")
        store.record_entry(fv)
        store.record_outcome("WIN-TRADE", won=True)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["outcome"] == "1"

    def test_record_outcome_loss(self, tmp_path: Path) -> None:
        """Lost trade gets outcome=0."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="LOSS-TRADE")
        store.record_entry(fv)
        store.record_outcome("LOSS-TRADE", won=False)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["outcome"] == "0"

    def test_record_outcome_not_found(self, tmp_path: Path) -> None:
        """No error if ticker not present in CSV."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        # Should not raise
        store.record_outcome("NONEXISTENT", won=True)

    def test_multiple_entries_same_ticker(self, tmp_path: Path) -> None:
        """Most recent unsettled row for a ticker gets updated."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv1 = _make_fv(ticker="MULTI", edge_cents=0.05)
        fv2 = _make_fv(ticker="MULTI", edge_cents=0.15)

        store.record_entry(fv1)
        store.record_entry(fv2)

        store.record_outcome("MULTI", won=True)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        # First entry should still be unsettled
        assert rows[0]["outcome"] == "-1"
        # Second (most recent) entry should be settled
        assert rows[1]["outcome"] == "1"

    def test_record_outcome_removes_from_entries(self, tmp_path: Path) -> None:
        """After outcome, ticker is removed from in-memory entries."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="REMOVE-ME")
        store.record_entry(fv)
        assert "REMOVE-ME" in store._entries

        store.record_outcome("REMOVE-ME", won=False)
        assert "REMOVE-ME" not in store._entries


# ── Loading training data ─────────────────────────────────────────


class TestLoadTrainingData:
    def test_load_training_data_empty(self, tmp_path: Path) -> None:
        """Empty store returns empty arrays."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        X, y = store.load_training_data()
        assert X.shape == (0, len(FEATURE_COLUMNS))
        assert y.shape == (0,)

    def test_load_training_data_settled(self, tmp_path: Path) -> None:
        """Returns X, y for settled rows."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        for i in range(5):
            fv = _make_fv(ticker=f"SETTLED-{i}")
            store.record_entry(fv)
            store.record_outcome(f"SETTLED-{i}", won=(i % 2 == 0))

        X, y = store.load_training_data()
        assert X.shape[0] == 5
        assert y.shape[0] == 5
        # Verify labels: i=0 win, i=1 loss, i=2 win, ...
        assert y[0] == 1.0
        assert y[1] == 0.0
        assert y[2] == 1.0

    def test_load_training_data_filters_unsettled(self, tmp_path: Path) -> None:
        """Only settled rows appear in the output."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv_settled = _make_fv(ticker="SETTLED")
        fv_pending = _make_fv(ticker="PENDING")

        store.record_entry(fv_settled)
        store.record_entry(fv_pending)
        store.record_outcome("SETTLED", won=True)

        X, y = store.load_training_data()
        assert X.shape[0] == 1
        assert y[0] == 1.0

    def test_load_training_data_feature_columns(self, tmp_path: Path) -> None:
        """X has correct number of feature columns."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        fv = _make_fv(ticker="COL-CHECK")
        store.record_entry(fv)
        store.record_outcome("COL-CHECK", won=True)

        X, y = store.load_training_data()
        assert X.shape[1] == len(FEATURE_COLUMNS)

    def test_load_training_data_no_file(self, tmp_path: Path) -> None:
        """If CSV file is deleted, returns empty arrays."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)
        os.remove(csv_path)

        X, y = store.load_training_data()
        assert X.shape == (0, len(FEATURE_COLUMNS))
        assert y.shape == (0,)


# ── Counting and threshold ────────────────────────────────────────


class TestCounting:
    def test_count_settled(self, tmp_path: Path) -> None:
        """Correct count of settled rows."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        for i in range(3):
            fv = _make_fv(ticker=f"S-{i}")
            store.record_entry(fv)
            store.record_outcome(f"S-{i}", won=True)
        # Add one pending
        store.record_entry(_make_fv(ticker="PENDING"))

        assert store.count_settled() == 3

    def test_count_pending(self, tmp_path: Path) -> None:
        """Correct count of pending (unsettled) rows."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), flush_interval=1)

        store.record_entry(_make_fv(ticker="P1"))
        store.record_entry(_make_fv(ticker="P2"))
        fv = _make_fv(ticker="DONE")
        store.record_entry(fv)
        store.record_outcome("DONE", won=False)

        assert store.count_pending() == 2

    def test_has_enough_samples(self, tmp_path: Path) -> None:
        """True when >= min_samples settled."""
        csv_path = tmp_path / "features.csv"
        store = FeatureStore(path=str(csv_path), min_samples_for_classifier=3, flush_interval=1)

        for i in range(2):
            fv = _make_fv(ticker=f"TICK-{i}")
            store.record_entry(fv)
            store.record_outcome(f"TICK-{i}", won=True)

        assert not store.has_enough_samples()

        fv = _make_fv(ticker="TICK-2")
        store.record_entry(fv)
        store.record_outcome("TICK-2", won=False)

        assert store.has_enough_samples()


# ── FeatureVector completeness ────────────────────────────────────


class TestFeatureVectorCompleteness:
    def test_feature_vector_all_fields(self) -> None:
        """All FEATURE_COLUMNS are attributes of FeatureVector."""
        fv = FeatureVector()
        for col in FEATURE_COLUMNS:
            assert hasattr(fv, col), f"FeatureVector missing field: {col}"

    def test_all_columns_are_in_feature_vector(self) -> None:
        """All ALL_COLUMNS are attributes of FeatureVector."""
        fv = FeatureVector()
        for col in ALL_COLUMNS:
            assert hasattr(fv, col), f"FeatureVector missing field: {col}"


# ── Config defaults ───────────────────────────────────────────────


class TestFeatureStoreConfig:
    def test_config_defaults(self) -> None:
        """CryptoSettings has correct feature store defaults."""
        s = CryptoSettings()
        assert s.feature_store_enabled is False
        assert s.feature_store_path == "feature_store.csv"
        assert s.feature_store_min_samples == 200

    def test_config_from_env(self) -> None:
        """Feature store settings are loaded from env vars."""
        env = {
            "ARB_CRYPTO_FEATURE_STORE_ENABLED": "true",
            "ARB_CRYPTO_FEATURE_STORE_PATH": "/tmp/custom_features.csv",
            "ARB_CRYPTO_FEATURE_STORE_MIN_SAMPLES": "500",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.feature_store_enabled is True
        assert s.feature_store_path == "/tmp/custom_features.csv"
        assert s.feature_store_min_samples == 500


# ── Engine integration ────────────────────────────────────────────


class TestEngineFeatureStoreIntegration:
    def test_engine_creates_feature_store_when_enabled(self, tmp_path: Path) -> None:
        """Engine creates a FeatureStore when feature_store_enabled=True."""
        from arb_bot.crypto.engine import CryptoEngine

        csv_path = tmp_path / "features.csv"
        settings = CryptoSettings(
            enabled=True,
            paper_mode=True,
            bankroll=1000.0,
            feature_store_enabled=True,
            feature_store_path=str(csv_path),
            feature_store_min_samples=100,
        )
        engine = CryptoEngine(settings)
        assert engine._feature_store is not None
        assert engine._feature_store.path == csv_path

    def test_engine_no_feature_store_when_disabled(self) -> None:
        """Engine does not create FeatureStore when disabled."""
        from arb_bot.crypto.engine import CryptoEngine

        settings = CryptoSettings(
            enabled=True,
            paper_mode=True,
            bankroll=1000.0,
            feature_store_enabled=False,
        )
        engine = CryptoEngine(settings)
        assert engine._feature_store is None

    def test_engine_records_entry_on_paper_trade(self, tmp_path: Path) -> None:
        """Paper trade records a feature vector in the feature store."""
        from arb_bot.crypto.engine import CryptoEngine
        from arb_bot.crypto.edge_detector import CryptoEdge
        from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketMeta, parse_ticker
        from arb_bot.crypto.price_model import ProbabilityEstimate
        from arb_bot.crypto.price_feed import PriceTick

        csv_path = tmp_path / "features.csv"
        settings = CryptoSettings(
            enabled=True,
            paper_mode=True,
            bankroll=1000.0,
            paper_slippage_cents=0.0,
            feature_store_enabled=True,
            feature_store_path=str(csv_path),
            feature_store_min_samples=10,
            price_feed_symbols=["btcusdt"],
            symbols=["KXBTC"],
        )
        engine = CryptoEngine(settings)

        # Inject price data so _build_feature_vector can fetch current price
        ts = time.time()
        for i in range(10):
            engine._price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0, ts - 10 + i, 1.0)
            )

        meta = parse_ticker("KXBTCD-26FEB14-T97500")
        assert meta is not None
        now = datetime.now(timezone.utc)
        adjusted_meta = replace(meta, expiry=now + timedelta(minutes=10))
        market = CryptoMarket(ticker="KXBTCD-26FEB14-T97500", meta=adjusted_meta)

        edge = CryptoEdge(
            market=market,
            model_prob=ProbabilityEstimate(0.65, 0.60, 0.70, 0.05, 1000),
            market_implied_prob=0.50,
            edge=0.10,
            edge_cents=0.10,
            side="yes",
            recommended_price=0.50,
            model_uncertainty=0.05,
            time_to_expiry_minutes=10.0,
            yes_buy_price=0.50,
            no_buy_price=0.50,
        )

        engine._execute_paper_trade(edge, 5)
        engine._feature_store.flush()

        # Verify entry was recorded
        assert engine._feature_store.count_pending() == 1

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "KXBTCD-26FEB14-T97500"
        assert rows[0]["side"] == "yes"
        assert rows[0]["outcome"] == "-1"

    def test_engine_records_outcome_on_settle(self, tmp_path: Path) -> None:
        """Settlement records outcome in the feature store."""
        from arb_bot.crypto.engine import CryptoEngine
        from arb_bot.crypto.edge_detector import CryptoEdge
        from arb_bot.crypto.market_scanner import CryptoMarket, parse_ticker
        from arb_bot.crypto.price_model import ProbabilityEstimate
        from arb_bot.crypto.price_feed import PriceTick

        csv_path = tmp_path / "features.csv"
        settings = CryptoSettings(
            enabled=True,
            paper_mode=True,
            bankroll=1000.0,
            paper_slippage_cents=0.0,
            feature_store_enabled=True,
            feature_store_path=str(csv_path),
            feature_store_min_samples=10,
            price_feed_symbols=["btcusdt"],
            symbols=["KXBTC"],
        )
        engine = CryptoEngine(settings)

        ts = time.time()
        for i in range(10):
            engine._price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0, ts - 10 + i, 1.0)
            )

        meta = parse_ticker("KXBTCD-26FEB14-T97500")
        assert meta is not None
        now = datetime.now(timezone.utc)
        adjusted_meta = replace(meta, expiry=now + timedelta(minutes=10))
        market = CryptoMarket(ticker="KXBTCD-26FEB14-T97500", meta=adjusted_meta)

        edge = CryptoEdge(
            market=market,
            model_prob=ProbabilityEstimate(0.65, 0.60, 0.70, 0.05, 1000),
            market_implied_prob=0.50,
            edge=0.10,
            edge_cents=0.10,
            side="yes",
            recommended_price=0.50,
            model_uncertainty=0.05,
            time_to_expiry_minutes=10.0,
            yes_buy_price=0.50,
            no_buy_price=0.50,
        )

        engine._execute_paper_trade(edge, 5)
        engine._feature_store.flush()
        engine.settle_position_with_outcome("KXBTCD-26FEB14-T97500", settled_yes=True)

        # Verify outcome was recorded
        assert engine._feature_store.count_settled() == 1
        assert engine._feature_store.count_pending() == 0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["outcome"] == "1"
