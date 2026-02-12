"""Secret and credential hardening (Phase 5L).

Key scope/expiry validation, credential health checks, and
rotation tracking. Does not store actual secrets — validates
metadata and tracks rotation status.

Usage::

    mgr = CredentialManager(config)
    mgr.register_credential("kalshi_api_key", scope="trading", expires_at=time.time() + 86400)
    report = mgr.health_check(now)
    if report.any_expired:
        # alert for rotation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Credential status
# ---------------------------------------------------------------------------


class CredentialStatus(str, Enum):
    """Status of a credential."""

    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    REVOKED = "revoked"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredentialManagerConfig:
    """Configuration for credential manager.

    Parameters
    ----------
    expiry_warning_seconds:
        Warn this many seconds before expiry. Default 86400 (24h).
    check_interval:
        Minimum seconds between health checks. Default 300.
    required_scopes:
        Scopes that must have valid credentials. Default empty.
    max_age_seconds:
        Maximum credential age before recommending rotation.
        Default 2592000 (30 days).
    """

    expiry_warning_seconds: float = 86400.0
    check_interval: float = 300.0
    required_scopes: Tuple[str, ...] = ()
    max_age_seconds: float = 2592000.0


# ---------------------------------------------------------------------------
# Credential metadata
# ---------------------------------------------------------------------------


@dataclass
class CredentialMeta:
    """Metadata for a credential (not the secret itself)."""

    name: str
    scope: str
    created_at: float
    expires_at: float  # 0 = no expiry.
    last_used_at: float = 0.0
    last_rotated_at: float = 0.0
    use_count: int = 0
    status: CredentialStatus = CredentialStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Health report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredentialHealthReport:
    """Health report for all credentials."""

    total_credentials: int
    valid_count: int
    expiring_soon_count: int
    expired_count: int
    revoked_count: int
    any_expired: bool
    any_expiring_soon: bool
    missing_scopes: Tuple[str, ...]
    rotation_recommended: Tuple[str, ...]
    expired_credentials: Tuple[str, ...]
    expiring_credentials: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class CredentialManager:
    """Manages credential metadata and health.

    Tracks expiry, scope coverage, rotation age, and usage.
    Does NOT store actual secrets.
    """

    def __init__(self, config: CredentialManagerConfig | None = None) -> None:
        self._config = config or CredentialManagerConfig()
        self._credentials: Dict[str, CredentialMeta] = {}

    @property
    def config(self) -> CredentialManagerConfig:
        return self._config

    def register_credential(
        self,
        name: str,
        scope: str,
        created_at: float | None = None,
        expires_at: float = 0.0,
    ) -> None:
        """Register a credential's metadata."""
        if created_at is None:
            created_at = time.time()
        self._credentials[name] = CredentialMeta(
            name=name,
            scope=scope,
            created_at=created_at,
            expires_at=expires_at,
            last_rotated_at=created_at,
        )

    def record_use(self, name: str, now: float | None = None) -> None:
        """Record credential usage."""
        if now is None:
            now = time.time()
        meta = self._credentials.get(name)
        if meta is not None:
            meta.last_used_at = now
            meta.use_count += 1

    def record_rotation(self, name: str, now: float | None = None) -> None:
        """Record that a credential was rotated."""
        if now is None:
            now = time.time()
        meta = self._credentials.get(name)
        if meta is not None:
            meta.last_rotated_at = now
            meta.status = CredentialStatus.VALID

    def revoke(self, name: str) -> None:
        """Mark a credential as revoked."""
        meta = self._credentials.get(name)
        if meta is not None:
            meta.status = CredentialStatus.REVOKED

    def _evaluate_status(
        self, meta: CredentialMeta, now: float
    ) -> CredentialStatus:
        """Evaluate credential status."""
        if meta.status == CredentialStatus.REVOKED:
            return CredentialStatus.REVOKED

        if meta.expires_at > 0:
            if now >= meta.expires_at:
                return CredentialStatus.EXPIRED
            if now >= meta.expires_at - self._config.expiry_warning_seconds:
                return CredentialStatus.EXPIRING_SOON

        return CredentialStatus.VALID

    def health_check(self, now: float | None = None) -> CredentialHealthReport:
        """Run health check on all credentials."""
        if now is None:
            now = time.time()

        cfg = self._config
        valid = 0
        expiring_soon = 0
        expired = 0
        revoked = 0
        expired_names: List[str] = []
        expiring_names: List[str] = []
        rotation_recommended: List[str] = []

        for meta in self._credentials.values():
            status = self._evaluate_status(meta, now)
            meta.status = status

            if status == CredentialStatus.VALID:
                valid += 1
            elif status == CredentialStatus.EXPIRING_SOON:
                expiring_soon += 1
                expiring_names.append(meta.name)
            elif status == CredentialStatus.EXPIRED:
                expired += 1
                expired_names.append(meta.name)
            elif status == CredentialStatus.REVOKED:
                revoked += 1

            # Check rotation age.
            age = now - meta.last_rotated_at
            if age > cfg.max_age_seconds and status != CredentialStatus.REVOKED:
                rotation_recommended.append(meta.name)

        # Check scope coverage.
        active_scopes = {
            meta.scope
            for meta in self._credentials.values()
            if meta.status in (CredentialStatus.VALID, CredentialStatus.EXPIRING_SOON)
        }
        missing = [s for s in cfg.required_scopes if s not in active_scopes]

        return CredentialHealthReport(
            total_credentials=len(self._credentials),
            valid_count=valid,
            expiring_soon_count=expiring_soon,
            expired_count=expired,
            revoked_count=revoked,
            any_expired=expired > 0,
            any_expiring_soon=expiring_soon > 0,
            missing_scopes=tuple(missing),
            rotation_recommended=tuple(rotation_recommended),
            expired_credentials=tuple(expired_names),
            expiring_credentials=tuple(expiring_names),
        )

    def get_credential(self, name: str) -> CredentialMeta | None:
        """Get credential metadata."""
        return self._credentials.get(name)

    def credential_count(self) -> int:
        """Total registered credentials."""
        return len(self._credentials)

    def clear(self) -> None:
        """Clear all credentials."""
        self._credentials.clear()
