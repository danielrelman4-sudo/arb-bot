from __future__ import annotations

from pathlib import Path

from arb_bot.baseline_lock import create_lock, verify_lock


def test_baseline_lock_create_and_verify(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    cross_map = tmp_path / "cross.csv"
    rules = tmp_path / "rules.json"
    profile = tmp_path / "profile.env"
    lock_path = tmp_path / "baseline.lock.json"

    cross_map.write_text("a,b\n1,2\n", encoding="utf-8")
    rules.write_text('{"ok":true}\n', encoding="utf-8")
    profile.write_text("ARB_STREAM_MODE=true\n", encoding="utf-8")
    env_path.write_text(
        "\n".join(
            [
                f"ARB_CROSS_VENUE_MAPPING_PATH={cross_map}",
                f"ARB_STRUCTURAL_RULES_PATH={rules}",
                "ARB_STREAM_MODE=true",
                "ARB_MIN_NET_EDGE_PER_CONTRACT=0.01",
                "KALSHI_KEY_ID=abc123",
                "POLYMARKET_API_KEY=secret-value",
            ]
        ),
        encoding="utf-8",
    )

    create_lock(env_file=env_path, output_path=lock_path, profile_path=profile)

    ok, errors = verify_lock(env_file=env_path, lock_path=lock_path, profile_path=profile)
    assert ok
    assert errors == []


def test_baseline_lock_detects_env_change(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    lock_path = tmp_path / "baseline.lock.json"

    env_path.write_text("ARB_STREAM_MODE=true\n", encoding="utf-8")
    create_lock(env_file=env_path, output_path=lock_path, profile_path=None)

    env_path.write_text("ARB_STREAM_MODE=false\n", encoding="utf-8")
    ok, errors = verify_lock(env_file=env_path, lock_path=lock_path, profile_path=None)

    assert not ok
    assert any("fingerprint mismatch" in error for error in errors)
