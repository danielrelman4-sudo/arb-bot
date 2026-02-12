from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_TOKENS = ("SECRET", "PRIVATE", "PASSPHRASE", "TOKEN", "KEY", "FUNDER")
TRACKED_PREFIXES = ("ARB_", "KALSHI_", "POLYMARKET_")


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 64)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in {'"', "'"}):
            value = value[1:-1]
        values[key] = value
    return values


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in SECRET_TOKENS)


def _tracked_non_secret_env(values: dict[str, str]) -> dict[str, str]:
    tracked: dict[str, str] = {}
    for key, value in values.items():
        if not key.startswith(TRACKED_PREFIXES):
            continue
        if _is_secret_key(key):
            continue
        tracked[key] = value
    return dict(sorted(tracked.items()))


def _resolve_path(value: str | None, default: str | None = None) -> Path | None:
    candidate = value or default
    if not candidate:
        return None
    return Path(candidate).expanduser().resolve()


def _file_entry(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    exists = path.exists()
    sha = _file_sha256(path) if exists and path.is_file() else None
    return {"path": str(path), "exists": exists, "sha256": sha}


@dataclass(frozen=True)
class BaselineLock:
    version: int
    created_at_utc: str
    env_file: dict[str, Any]
    cross_map_file: dict[str, Any]
    structural_rules_file: dict[str, Any]
    profile_file: dict[str, Any]
    tracked_env_key_count: int
    tracked_env_fingerprint_sha256: str

    def to_json(self) -> str:
        payload = {
            "version": self.version,
            "created_at_utc": self.created_at_utc,
            "env_file": self.env_file,
            "cross_map_file": self.cross_map_file,
            "structural_rules_file": self.structural_rules_file,
            "profile_file": self.profile_file,
            "tracked_env_key_count": self.tracked_env_key_count,
            "tracked_env_fingerprint_sha256": self.tracked_env_fingerprint_sha256,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def create_lock(env_file: Path, output_path: Path, profile_path: Path | None = None) -> BaselineLock:
    env_values = _parse_env_file(env_file)
    tracked_env = _tracked_non_secret_env(env_values)
    tracked_blob = json.dumps(tracked_env, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tracked_sha = _sha256_bytes(tracked_blob)

    cross_map_path = _resolve_path(
        env_values.get("ARB_CROSS_VENUE_MAPPING_PATH"),
        default="arb_bot/config/cross_venue_map.generated.csv",
    )
    structural_rules_path = _resolve_path(
        env_values.get("ARB_STRUCTURAL_RULES_PATH"),
        default="arb_bot/config/structural_rules.generated.json",
    )

    lock = BaselineLock(
        version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        env_file=_file_entry(env_file.resolve()),
        cross_map_file=_file_entry(cross_map_path),
        structural_rules_file=_file_entry(structural_rules_path),
        profile_file=_file_entry(profile_path.resolve() if profile_path else None),
        tracked_env_key_count=len(tracked_env),
        tracked_env_fingerprint_sha256=tracked_sha,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(lock.to_json(), encoding="utf-8")
    return lock


def verify_lock(env_file: Path, lock_path: Path, profile_path: Path | None = None) -> tuple[bool, list[str]]:
    if not lock_path.exists():
        return False, [f"lock file missing: {lock_path}"]

    payload = json.loads(lock_path.read_text(encoding="utf-8"))

    env_values = _parse_env_file(env_file)
    tracked_env = _tracked_non_secret_env(env_values)
    tracked_blob = json.dumps(tracked_env, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tracked_sha = _sha256_bytes(tracked_blob)

    expected_tracked_sha = str(payload.get("tracked_env_fingerprint_sha256") or "")
    errors: list[str] = []
    if tracked_sha != expected_tracked_sha:
        errors.append("non-secret env fingerprint mismatch")

    def _verify_file(name: str, file_payload: dict[str, Any] | None, explicit_path: Path | None = None) -> None:
        if not isinstance(file_payload, dict):
            errors.append(f"{name}: missing payload in lock")
            return
        expected_path = file_payload.get("path")
        expected_exists = bool(file_payload.get("exists"))
        expected_sha = file_payload.get("sha256")

        if explicit_path is not None:
            candidate = explicit_path.resolve()
        elif expected_path:
            candidate = Path(expected_path).expanduser().resolve()
        else:
            candidate = None

        if candidate is None:
            if expected_path is not None:
                errors.append(f"{name}: lock has path but candidate is missing")
            return

        actual_exists = candidate.exists()
        if actual_exists != expected_exists:
            errors.append(f"{name}: exists mismatch expected={expected_exists} actual={actual_exists} path={candidate}")
            return

        if expected_exists:
            actual_sha = _file_sha256(candidate)
            if actual_sha != expected_sha:
                errors.append(f"{name}: sha256 mismatch path={candidate}")

    _verify_file("env_file", payload.get("env_file"), explicit_path=env_file.resolve())
    _verify_file("cross_map_file", payload.get("cross_map_file"))
    _verify_file("structural_rules_file", payload.get("structural_rules_file"))
    _verify_file("profile_file", payload.get("profile_file"), explicit_path=profile_path.resolve() if profile_path else None)

    return len(errors) == 0, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/verify frozen baseline lock for Phase-0 runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create", help="Create baseline lock")
    create.add_argument("--env-file", default=".env")
    create.add_argument("--output", default="arb_bot/config/phase0_baseline.lock.json")
    create.add_argument("--profile", default=None, help="Optional profile file used for this baseline")

    verify = sub.add_parser("verify", help="Verify baseline lock")
    verify.add_argument("--env-file", default=".env")
    verify.add_argument("--lock", default="arb_bot/config/phase0_baseline.lock.json")
    verify.add_argument("--profile", default=None, help="Optional profile file expected for this baseline")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cmd == "create":
        lock = create_lock(
            env_file=Path(args.env_file).expanduser(),
            output_path=Path(args.output).expanduser(),
            profile_path=Path(args.profile).expanduser() if args.profile else None,
        )
        print("baseline lock created")
        print(f"output={Path(args.output).expanduser()}")
        print(f"tracked_env_key_count={lock.tracked_env_key_count}")
        print(f"tracked_env_fingerprint_sha256={lock.tracked_env_fingerprint_sha256}")
        return 0

    ok, errors = verify_lock(
        env_file=Path(args.env_file).expanduser(),
        lock_path=Path(args.lock).expanduser(),
        profile_path=Path(args.profile).expanduser() if args.profile else None,
    )
    if ok:
        print("baseline lock verify: PASS")
        return 0

    print("baseline lock verify: FAIL")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
