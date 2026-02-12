"""Tests for Phase 5L: Secret and credential hardening."""

from __future__ import annotations

import pytest

from arb_bot.credential_manager import (
    CredentialManager,
    CredentialManagerConfig,
    CredentialStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mgr(**kw) -> CredentialManager:
    return CredentialManager(CredentialManagerConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = CredentialManagerConfig()
        assert cfg.expiry_warning_seconds == 86400.0
        assert cfg.check_interval == 300.0
        assert cfg.max_age_seconds == 2592000.0

    def test_frozen(self) -> None:
        cfg = CredentialManagerConfig()
        with pytest.raises(AttributeError):
            cfg.expiry_warning_seconds = 100.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="trading", created_at=100.0)
        assert mgr.credential_count() == 1

    def test_metadata(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="trading", created_at=100.0, expires_at=200.0)
        meta = mgr.get_credential("key_1")
        assert meta is not None
        assert meta.name == "key_1"
        assert meta.scope == "trading"
        assert meta.expires_at == 200.0

    def test_nonexistent(self) -> None:
        mgr = _mgr()
        assert mgr.get_credential("nope") is None


# ---------------------------------------------------------------------------
# Status evaluation
# ---------------------------------------------------------------------------


class TestStatus:
    def test_valid(self) -> None:
        mgr = _mgr(expiry_warning_seconds=10.0)
        mgr.register_credential("key_1", scope="read", created_at=100.0, expires_at=200.0)
        report = mgr.health_check(now=110.0)  # 90s until expiry > 10s warning.
        assert report.valid_count == 1
        assert report.any_expired is False

    def test_expired(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0, expires_at=200.0)
        report = mgr.health_check(now=250.0)
        assert report.expired_count == 1
        assert report.any_expired is True
        assert "key_1" in report.expired_credentials

    def test_expiring_soon(self) -> None:
        mgr = _mgr(expiry_warning_seconds=3600.0)
        mgr.register_credential("key_1", scope="read", created_at=100.0, expires_at=200.0)
        report = mgr.health_check(now=190.0)  # 10s until expiry < 3600s warning.
        assert report.expiring_soon_count == 1
        assert report.any_expiring_soon is True
        assert "key_1" in report.expiring_credentials

    def test_no_expiry(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0, expires_at=0.0)
        report = mgr.health_check(now=999999.0)
        assert report.valid_count == 1

    def test_revoked(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        mgr.revoke("key_1")
        report = mgr.health_check(now=110.0)
        assert report.revoked_count == 1


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


class TestUsage:
    def test_record_use(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        mgr.record_use("key_1", now=150.0)
        meta = mgr.get_credential("key_1")
        assert meta is not None
        assert meta.use_count == 1
        assert meta.last_used_at == 150.0

    def test_cumulative_use(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        mgr.record_use("key_1", now=150.0)
        mgr.record_use("key_1", now=160.0)
        meta = mgr.get_credential("key_1")
        assert meta is not None
        assert meta.use_count == 2


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestRotation:
    def test_record_rotation(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        mgr.record_rotation("key_1", now=200.0)
        meta = mgr.get_credential("key_1")
        assert meta is not None
        assert meta.last_rotated_at == 200.0
        assert meta.status == CredentialStatus.VALID

    def test_rotation_recommended(self) -> None:
        mgr = _mgr(max_age_seconds=86400.0)  # 1 day.
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        report = mgr.health_check(now=100.0 + 100000.0)  # > 1 day old.
        assert "key_1" in report.rotation_recommended

    def test_not_recommended_if_fresh(self) -> None:
        mgr = _mgr(max_age_seconds=86400.0)
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        report = mgr.health_check(now=200.0)  # 100s < 1 day.
        assert "key_1" not in report.rotation_recommended


# ---------------------------------------------------------------------------
# Scope coverage
# ---------------------------------------------------------------------------


class TestScopeCoverage:
    def test_all_scopes_covered(self) -> None:
        mgr = _mgr(required_scopes=("trading", "read"))
        mgr.register_credential("key_1", scope="trading", created_at=100.0)
        mgr.register_credential("key_2", scope="read", created_at=100.0)
        report = mgr.health_check(now=110.0)
        assert report.missing_scopes == ()

    def test_missing_scope(self) -> None:
        mgr = _mgr(required_scopes=("trading", "read", "admin"))
        mgr.register_credential("key_1", scope="trading", created_at=100.0)
        report = mgr.health_check(now=110.0)
        assert "read" in report.missing_scopes
        assert "admin" in report.missing_scopes

    def test_expired_scope_not_counted(self) -> None:
        mgr = _mgr(required_scopes=("trading",))
        mgr.register_credential("key_1", scope="trading", created_at=100.0, expires_at=150.0)
        report = mgr.health_check(now=200.0)  # Expired.
        assert "trading" in report.missing_scopes


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        mgr = _mgr()
        mgr.register_credential("key_1", scope="read", created_at=100.0)
        mgr.clear()
        assert mgr.credential_count() == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = CredentialManagerConfig(check_interval=60.0)
        mgr = CredentialManager(cfg)
        assert mgr.config.check_interval == 60.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_lifecycle(self) -> None:
        """Credential: valid → expiring → rotate → valid."""
        mgr = _mgr(
            expiry_warning_seconds=100.0,
            required_scopes=("trading",),
        )
        mgr.register_credential(
            "api_key", scope="trading",
            created_at=1000.0, expires_at=2000.0,
        )

        # Valid.
        report = mgr.health_check(now=1500.0)
        assert report.valid_count == 1
        assert report.missing_scopes == ()

        # Expiring soon.
        report = mgr.health_check(now=1950.0)
        assert report.expiring_soon_count == 1

        # Rotate — re-register with new expiry.
        mgr.record_rotation("api_key", now=1960.0)
        mgr.register_credential(
            "api_key", scope="trading",
            created_at=1960.0, expires_at=3000.0,
        )
        report = mgr.health_check(now=1970.0)
        assert report.valid_count == 1
        assert report.any_expiring_soon is False
