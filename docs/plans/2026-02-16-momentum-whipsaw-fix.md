# Momentum Whipsaw Fix + Sizing Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix OFI direction reversal whipsaw in v18 momentum strategy and fix model-path sizing to convert 98% win rate into positive P&L.

**Architecture:** Add OFI confirmation streak (3-cycle) and acceleration filter (|OFI_30s| > |OFI_120s|) as guards before momentum entry. Reduce Kelly cap, enable regime-conditional sizing, and buffer feature store writes.

**Tech Stack:** Python 3.9+, numpy, pytest, dataclasses

**Target branch:** `feature/strategy-cell-matrix` (the v18 momentum work lives here, NOT main)

**Design doc:** `docs/plans/2026-02-16-momentum-whipsaw-fix-design.md`

---

### Task 1: Add Config Fields for Streak + Acceleration + Max Contracts

**Files:**
- Modify: `arb_bot/crypto/config.py:305` (after momentum_cooldown_seconds)
- Modify: `arb_bot/crypto/config.py:558` (after momentum_cooldown_seconds env var)

**Step 1: Add 3 new fields to the CryptoSettings dataclass**

After line 305 (`momentum_cooldown_seconds: float = 120.0`), add:

```python
    momentum_min_ofi_streak: int = 3           # Consecutive aligned OFI cycles before entry
    momentum_require_ofi_acceleration: bool = True  # Skip if |OFI_30s| <= |OFI_120s|
    momentum_max_contracts: int = 100          # Hard cap on contracts per momentum trade
```

**Step 2: Add matching env var wiring in from_env()**

After line 558 (`momentum_cooldown_seconds=...`), add:

```python
        momentum_min_ofi_streak=_as_int(os.getenv("ARB_CRYPTO_MOMENTUM_MIN_OFI_STREAK"), 3),
        momentum_require_ofi_acceleration=_as_bool(os.getenv("ARB_CRYPTO_MOMENTUM_REQUIRE_OFI_ACCELERATION"), True),
        momentum_max_contracts=_as_int(os.getenv("ARB_CRYPTO_MOMENTUM_MAX_CONTRACTS"), 100),
```

**Step 3: Verify config loads**

Run: `python3 -c "from arb_bot.crypto.config import CryptoSettings; s = CryptoSettings(); print(s.momentum_min_ofi_streak, s.momentum_require_ofi_acceleration, s.momentum_max_contracts)"`

Expected: `3 True 100`

**Step 4: Commit**

```bash
git add arb_bot/crypto/config.py
git commit -m "feat(crypto): add momentum streak, acceleration, max_contracts config"
```

---

### Task 2: Add OFI Streak Tracker Tests

**Files:**
- Modify: `arb_bot/tests/test_momentum.py`

**Step 1: Write failing tests for OFI streak behavior**

Add these tests to the existing `test_momentum.py`. They test the streak logic we'll add to `_try_momentum_trades()`. Since the actual method is async and needs a full engine, we test the streak tracking as a standalone helper first.

```python
# ── OFI Streak Tests ──────────────────────────────────────────────


def test_ofi_streak_increments_on_consistent_direction():
    """Streak increments when OFI direction is consistent and aligned."""
    from arb_bot.crypto.engine import _update_ofi_streak

    streaks = {}
    last_dirs = {}

    # Cycle 1: direction=+1, alignment=0.8 -> streak=1
    s = _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    assert s == 1

    # Cycle 2: same direction -> streak=2
    s = _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.7, min_alignment=0.6)
    assert s == 2

    # Cycle 3: same direction -> streak=3
    s = _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.9, min_alignment=0.6)
    assert s == 3


def test_ofi_streak_resets_on_direction_flip():
    """Streak resets to 0 when OFI direction flips."""
    from arb_bot.crypto.engine import _update_ofi_streak

    streaks = {}
    last_dirs = {}

    _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    assert streaks["btcusdt"] == 2

    # Direction flips -> reset
    s = _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=-1, ofi_alignment=0.8, min_alignment=0.6)
    assert s == 0
    assert streaks["btcusdt"] == 0


def test_ofi_streak_resets_on_low_alignment():
    """Streak resets when alignment drops below threshold."""
    from arb_bot.crypto.engine import _update_ofi_streak

    streaks = {}
    last_dirs = {}

    _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    assert streaks["btcusdt"] == 2

    # Same direction but alignment too low -> reset
    s = _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.4, min_alignment=0.6)
    assert s == 0


def test_ofi_streak_per_symbol_independence():
    """Streaks are tracked independently per symbol."""
    from arb_bot.crypto.engine import _update_ofi_streak

    streaks = {}
    last_dirs = {}

    _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    _update_ofi_streak(streaks, last_dirs, "btcusdt", ofi_direction=1, ofi_alignment=0.8, min_alignment=0.6)
    _update_ofi_streak(streaks, last_dirs, "ethusdt", ofi_direction=-1, ofi_alignment=0.9, min_alignment=0.6)

    assert streaks["btcusdt"] == 2
    assert streaks["ethusdt"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -k "ofi_streak" -v`

Expected: FAIL with `ImportError: cannot import name '_update_ofi_streak'`

**Step 3: Commit test stubs**

```bash
git add arb_bot/tests/test_momentum.py
git commit -m "test(crypto): add OFI streak tracker tests (red)"
```

---

### Task 3: Implement OFI Streak Tracker

**Files:**
- Modify: `arb_bot/crypto/engine.py:61` (near top, after `_select_momentum_contract`)
- Modify: `arb_bot/crypto/engine.py:258` (CryptoEngine.__init__, after `_momentum_cooldowns`)
- Modify: `arb_bot/crypto/engine.py:2936-3010` (`_try_momentum_trades`)

**Step 1: Add `_update_ofi_streak` helper function**

After the `_compute_momentum_size` function (around line 130), add:

```python
def _update_ofi_streak(
    streaks: dict,
    last_dirs: dict,
    symbol: str,
    ofi_direction: int,
    ofi_alignment: float,
    min_alignment: float,
) -> int:
    """Update and return the OFI direction streak for a symbol.

    Streak increments when direction is consistent and alignment is above
    threshold. Resets to 0 on direction flip or alignment drop.
    """
    prev_dir = last_dirs.get(symbol)
    last_dirs[symbol] = ofi_direction

    if prev_dir is not None and ofi_direction == prev_dir and ofi_alignment >= min_alignment:
        streaks[symbol] = streaks.get(symbol, 0) + 1
    else:
        streaks[symbol] = 0 if prev_dir is not None else 1

    return streaks[symbol]
```

Wait — let me reconsider. On the very first observation there's no previous direction. We should return 1 on first aligned observation (that's the start of a streak), and 0 on first misaligned observation. Actually, looking at the tests: first call returns 1. That means the first aligned observation starts a streak at 1.

Correction to the logic:

```python
def _update_ofi_streak(
    streaks: dict,
    last_dirs: dict,
    symbol: str,
    ofi_direction: int,
    ofi_alignment: float,
    min_alignment: float,
) -> int:
    """Update and return the OFI direction streak for a symbol.

    Streak increments when direction is consistent and alignment is above
    threshold. Resets to 0 on direction flip or alignment drop.
    """
    prev_dir = last_dirs.get(symbol)
    last_dirs[symbol] = ofi_direction

    if ofi_alignment < min_alignment:
        streaks[symbol] = 0
        return 0

    if prev_dir is None or ofi_direction != prev_dir:
        # First observation or direction changed: start fresh
        # First observation with good alignment = 1, direction flip = 0
        streaks[symbol] = 1 if prev_dir is None else 0
        return streaks[symbol]

    # Consistent direction + good alignment: increment
    streaks[symbol] = streaks.get(symbol, 0) + 1
    return streaks[symbol]
```

**Step 2: Add streak state to CryptoEngine.__init__**

After line 258 (`self._momentum_cooldowns`), add:

```python
        self._ofi_direction_streak: Dict[str, int] = {}
        self._ofi_last_direction: Dict[str, int] = {}
```

**Step 3: Wire streak check into `_try_momentum_trades`**

In `_try_momentum_trades` (line ~2936), after computing `ofi_direction` and `ofi_magnitude` (around line 3007), BEFORE the existing trigger checks, add the streak update and gate:

```python
            # OFI confirmation streak: require N consecutive aligned cycles
            streak = _update_ofi_streak(
                self._ofi_direction_streak,
                self._ofi_last_direction,
                binance_sym,
                ofi_direction=ofi_direction,
                ofi_alignment=ofi_alignment,
                min_alignment=self._settings.momentum_ofi_alignment_min,
            )
            if streak < self._settings.momentum_min_ofi_streak:
                LOGGER.debug(
                    "CryptoEngine: momentum skip %s — OFI streak %d < %d",
                    binance_sym, streak, self._settings.momentum_min_ofi_streak,
                )
                continue
```

This goes BEFORE the existing alignment/magnitude trigger checks (which still serve as a secondary filter on the current cycle's values).

**Step 4: Run tests to verify they pass**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -k "ofi_streak" -v`

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add arb_bot/crypto/engine.py
git commit -m "feat(crypto): OFI confirmation streak for momentum entry"
```

---

### Task 4: Add OFI Acceleration Filter Tests

**Files:**
- Modify: `arb_bot/tests/test_momentum.py`

**Step 1: Write failing tests for acceleration filter**

```python
# ── OFI Acceleration Tests ────────────────────────────────────────


def test_ofi_acceleration_positive_when_short_stronger():
    """Acceleration is positive when 30s OFI magnitude exceeds 120s."""
    from arb_bot.crypto.engine import _check_ofi_acceleration

    ofi_multi = {30: 0.8, 60: 0.6, 120: 0.3, 300: 0.1}
    assert _check_ofi_acceleration(ofi_multi) is True  # |0.8| > |0.3|


def test_ofi_acceleration_negative_when_short_weaker():
    """Acceleration is negative (exhaustion) when 30s OFI < 120s."""
    from arb_bot.crypto.engine import _check_ofi_acceleration

    ofi_multi = {30: 0.2, 60: 0.5, 120: 0.6, 300: 0.4}
    assert _check_ofi_acceleration(ofi_multi) is False  # |0.2| < |0.6|


def test_ofi_acceleration_works_with_negative_ofi():
    """Acceleration uses absolute values, works for bearish flow."""
    from arb_bot.crypto.engine import _check_ofi_acceleration

    ofi_multi = {30: -0.7, 60: -0.5, 120: -0.3, 300: -0.1}
    assert _check_ofi_acceleration(ofi_multi) is True  # |-0.7| > |-0.3|


def test_ofi_acceleration_false_when_equal():
    """Acceleration is False when magnitudes are equal (not strengthening)."""
    from arb_bot.crypto.engine import _check_ofi_acceleration

    ofi_multi = {30: 0.5, 60: 0.5, 120: 0.5, 300: 0.5}
    assert _check_ofi_acceleration(ofi_multi) is False  # |0.5| == |0.5|


def test_ofi_acceleration_handles_missing_windows():
    """Returns True (permissive) when 30s or 120s window is missing."""
    from arb_bot.crypto.engine import _check_ofi_acceleration

    assert _check_ofi_acceleration({60: 0.5, 300: 0.1}) is True
    assert _check_ofi_acceleration({}) is True
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -k "ofi_acceleration" -v`

Expected: FAIL with `ImportError: cannot import name '_check_ofi_acceleration'`

**Step 3: Commit test stubs**

```bash
git add arb_bot/tests/test_momentum.py
git commit -m "test(crypto): add OFI acceleration filter tests (red)"
```

---

### Task 5: Implement OFI Acceleration Filter

**Files:**
- Modify: `arb_bot/crypto/engine.py` (after `_update_ofi_streak`)
- Modify: `arb_bot/crypto/engine.py:_try_momentum_trades` (after streak check)

**Step 1: Add `_check_ofi_acceleration` helper**

After `_update_ofi_streak`, add:

```python
def _check_ofi_acceleration(ofi_multiscale: dict) -> bool:
    """Check if OFI flow is accelerating (short-term stronger than medium-term).

    Compares |OFI at 30s| vs |OFI at 120s|. Returns True if short-term
    flow is stronger (momentum building) or if data is missing (permissive).
    """
    ofi_30 = ofi_multiscale.get(30)
    ofi_120 = ofi_multiscale.get(120)
    if ofi_30 is None or ofi_120 is None:
        return True  # Permissive when data missing
    return abs(ofi_30) > abs(ofi_120)
```

**Step 2: Wire into `_try_momentum_trades`**

After the streak check (added in Task 3), before the existing alignment/magnitude checks, add:

```python
            # OFI acceleration filter: skip if flow is fading (exhaustion)
            if self._settings.momentum_require_ofi_acceleration:
                if not _check_ofi_acceleration(ofi_multi):
                    LOGGER.debug(
                        "CryptoEngine: momentum skip %s — OFI decelerating",
                        binance_sym,
                    )
                    continue
```

**Step 3: Run tests**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -k "ofi_acceleration" -v`

Expected: 5 PASSED

**Step 4: Run full momentum test suite**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -v`

Expected: ALL PASSED (existing tests + new tests)

**Step 5: Commit**

```bash
git add arb_bot/crypto/engine.py
git commit -m "feat(crypto): OFI acceleration filter for momentum entry"
```

---

### Task 6: Add Momentum Max Contracts Cap

**Files:**
- Modify: `arb_bot/crypto/engine.py` — `_compute_momentum_size` (line ~116)
- Modify: `arb_bot/tests/test_momentum.py`

**Step 1: Write failing test**

```python
def test_momentum_size_capped_by_max_contracts():
    """Momentum size is capped by momentum_max_contracts."""
    settings = CryptoSettings(
        momentum_kelly_fraction=0.10,  # 10% to get large size
        momentum_max_position=500.0,   # High to not interfere
        momentum_max_contracts=50,     # Cap at 50
    )
    # bankroll=5000, price=0.10 -> uncapped = 5000*0.10*1.0/0.10 = 5000 contracts
    contracts = _compute_momentum_size(5000.0, 0.10, 1.0, settings)
    assert contracts == 50
```

**Step 2: Run to verify failure**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -k "max_contracts" -v`

Expected: FAIL — currently no cap, returns 5000

**Step 3: Add max_contracts clamp to `_compute_momentum_size`**

Modify `_compute_momentum_size` to add the final clamp:

```python
def _compute_momentum_size(
    bankroll: float,
    buy_price: float,
    ofi_alignment: float,
    settings,
) -> int:
    """Compute momentum position size: fixed fraction scaled by OFI alignment."""
    dollar_amount = bankroll * settings.momentum_kelly_fraction * ofi_alignment
    dollar_amount = min(dollar_amount, settings.momentum_max_position)
    dollar_amount = min(dollar_amount, bankroll)
    if buy_price <= 0:
        return 0
    contracts = int(dollar_amount / buy_price)
    # Hard contract cap
    max_contracts = getattr(settings, 'momentum_max_contracts', 100)
    return min(contracts, max_contracts)
```

**Step 4: Run test**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -k "max_contracts" -v`

Expected: PASS

**Step 5: Commit**

```bash
git add arb_bot/crypto/engine.py arb_bot/tests/test_momentum.py
git commit -m "feat(crypto): hard max_contracts cap for momentum trades"
```

---

### Task 7: Buffer Feature Store Writes

**Files:**
- Modify: `arb_bot/crypto/feature_store.py:138-178`
- Create: `arb_bot/tests/test_feature_store_buffer.py`

**Step 1: Write failing test**

Create `arb_bot/tests/test_feature_store_buffer.py`:

```python
"""Tests for buffered feature store writes."""
import tempfile
import os
import csv
from pathlib import Path

import pytest

from arb_bot.crypto.feature_store import FeatureStore, FeatureVector


def _make_fv(ticker: str) -> FeatureVector:
    """Create a minimal FeatureVector for testing."""
    return FeatureVector(
        ticker=ticker,
        timestamp=1000.0,
        side="yes",
        contracts=10,
        entry_price=0.25,
        model_probability=0.60,
        market_probability=0.50,
        edge_cents=0.10,
        tte_minutes=10.0,
        strategy="momentum",
    )


def test_buffered_writes_flush_at_interval(tmp_path):
    """Records buffer in memory and flush to CSV at interval."""
    csv_path = tmp_path / "test_fs.csv"
    fs = FeatureStore(path=str(csv_path), flush_interval=3)

    # Write 2 entries — should NOT flush yet
    fs.record_entry(_make_fv("TICK-1"))
    fs.record_entry(_make_fv("TICK-2"))

    # Only header row in CSV (entries still in buffer)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0

    # Write 3rd entry — triggers flush
    fs.record_entry(_make_fv("TICK-3"))

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_flush_on_explicit_call(tmp_path):
    """Explicit flush() writes all buffered entries."""
    csv_path = tmp_path / "test_fs.csv"
    fs = FeatureStore(path=str(csv_path), flush_interval=100)

    fs.record_entry(_make_fv("TICK-1"))
    fs.flush()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
```

**Step 2: Run to verify failure**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_feature_store_buffer.py -v`

Expected: FAIL — `FeatureStore.__init__() got an unexpected keyword argument 'flush_interval'`

**Step 3: Implement buffered writes**

Modify `FeatureStore.__init__` to accept `flush_interval` and buffer entries:

```python
    def __init__(
        self,
        path: str = "feature_store.csv",
        min_samples_for_classifier: int = 200,
        flush_interval: int = 50,
    ) -> None:
        self._path = Path(path)
        self._min_samples = min_samples_for_classifier
        self._flush_interval = flush_interval
        self._entries: dict[str, FeatureVector] = {}  # ticker -> pending FV
        self._write_buffer: list[dict] = []
        self._ensure_file()
```

Modify `record_entry` to buffer instead of immediately writing:

```python
    def record_entry(self, fv: FeatureVector) -> None:
        if not fv.ticker:
            LOGGER.warning("FeatureStore: ignoring entry with empty ticker")
            return

        self._entries[fv.ticker] = fv
        self._write_buffer.append(self._fv_to_row(fv))

        if len(self._write_buffer) >= self._flush_interval:
            self.flush()

        LOGGER.debug("FeatureStore: recorded entry for %s", fv.ticker)
```

Add `flush()` method:

```python
    def flush(self) -> None:
        """Write buffered entries to CSV."""
        if not self._write_buffer:
            return
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            for row in self._write_buffer:
                writer.writerow(row)
        self._write_buffer.clear()
```

**Step 4: Run tests**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_feature_store_buffer.py -v`

Expected: 2 PASSED

**Step 5: Commit**

```bash
git add arb_bot/crypto/feature_store.py arb_bot/tests/test_feature_store_buffer.py
git commit -m "feat(crypto): buffered feature store writes with flush interval"
```

---

### Task 8: Update Paper Test Config

**Files:**
- Modify: `arb_bot/crypto/paper_test.py:425` (kelly_fraction_cap)
- Modify: `arb_bot/crypto/paper_test.py:534-543` (regime settings)
- Modify: `arb_bot/crypto/paper_test.py:563-575` (momentum settings)

**Step 1: Update kelly_fraction_cap**

Line 425: change `kelly_fraction_cap=0.10` to `kelly_fraction_cap=0.06`

**Step 2: Update regime min edge thresholds**

Line 541: change `regime_min_edge_mean_reverting=0.10` to `regime_min_edge_mean_reverting=0.12`
Line 542: change `regime_min_edge_trending=0.15` to `regime_min_edge_trending=0.12`
Line 543: change `regime_min_edge_high_vol=0.15` to `regime_min_edge_high_vol=0.12`

**Step 3: Add new momentum config fields**

After line 575 (`momentum_cooldown_seconds=120.0`), add:

```python
        momentum_min_ofi_streak=3,
        momentum_require_ofi_acceleration=True,
        momentum_max_contracts=100,
```

**Step 4: Add flush call in paper_test shutdown**

Find the shutdown/cleanup section of `run_paper_test()` and add `feature_store.flush()` before exit. Search for where the function handles graceful shutdown (try/finally or signal handler).

**Step 5: Verify config prints correctly**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -c "from arb_bot.crypto.paper_test import *; print('config loads ok')"`

Expected: `config loads ok` (no import errors)

**Step 6: Commit**

```bash
git add arb_bot/crypto/paper_test.py
git commit -m "feat(crypto): paper test config for v19 whipsaw fix + sizing"
```

---

### Task 9: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all momentum tests**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_momentum.py -v`

Expected: ALL PASSED

**Step 2: Run feature store tests**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_feature_store_buffer.py -v`

Expected: ALL PASSED

**Step 3: Run crypto config tests**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/test_crypto_config.py -v`

Expected: ALL PASSED

**Step 4: Run broader test suite to check for regressions**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m pytest arb_bot/tests/ -v --timeout=30 2>&1 | tail -30`

Expected: No new failures. Note any pre-existing failures.

---

### Task 10: Relaunch Paper Run

**Files:** None (execution only)

**Step 1: Verify we're on the correct branch**

Run: `git branch --show-current`

Expected: `feature/strategy-cell-matrix` (or the worktree branch that tracks it)

**Step 2: Launch paper run**

Run: `cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/heuristic-antonelli && python3 -m arb_bot.crypto.paper_test --duration-minutes 480 --symbols BTC ETH SOL --mc-paths 1000`

Monitor first 5 minutes of output for:
- Config printout showing new values (kelly_cap=0.06, regime_sizing=ON, momentum streak/accel)
- Momentum trades being skipped for streak < 3 (look for "OFI streak" in logs)
- Momentum trades being skipped for deceleration (look for "OFI decelerating" in logs)
- No file permission errors on feature store
