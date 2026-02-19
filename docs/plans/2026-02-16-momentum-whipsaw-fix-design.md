# Momentum Whipsaw Fix + Sizing Improvements

**Date**: 2026-02-16
**Branch**: feature/strategy-cell-matrix (extends backlog/b2-monte-carlo-sim)
**Status**: Design approved, ready for implementation

## Problem

The v18 momentum strategy suffers from OFI direction reversals. The bot enters
a directional trade following OFI, but OFI flips within minutes. Since momentum
contracts are OTM ($0.15-$0.40) binary options with max 15-min TTE, a reversal
means the contract settles at $0.00.

Root cause: OFI weighting favors the 30s window (weight 4x) over 300s (weight
1x). Short bursts of directional volume trigger entries that don't persist.

Secondary issue: model-path trades lose money despite 97.9% win rate because
Kelly sizing allocates 2x more contracts to cheap losing trades vs expensive
winners. Regime-conditional sizing flags were implemented (commit 07808df) but
never enabled.

## Changes

### 1. OFI Confirmation Streak

**File**: `arb_bot/crypto/engine.py` — `_try_momentum_trades()`

Add per-symbol streak tracker requiring N consecutive cycles of consistent OFI
direction before triggering momentum entry.

New state:
- `self._ofi_direction_streak: Dict[str, int]` — streak count per symbol
- `self._ofi_last_direction: Dict[str, int]` — last direction per symbol

Logic per cycle:
1. Compute `ofi_direction` (weighted mean of multi-timescale OFI)
2. If direction matches previous AND alignment >= threshold: streak += 1
3. Else: streak = 0
4. Only enter if streak >= `momentum_min_ofi_streak`

New config:
- `momentum_min_ofi_streak: int = 3`

### 2. OFI Acceleration Filter

**File**: `arb_bot/crypto/engine.py` — `_try_momentum_trades()`

After streak check passes, verify flow is strengthening (not exhausting).

```
ofi_short = |OFI at 30s window|
ofi_long  = |OFI at 120s window|
acceleration = ofi_short - ofi_long
```

- `acceleration > 0`: flow strengthening — proceed
- `acceleration <= 0`: flow fading (exhaustion) — skip

New config:
- `momentum_require_ofi_acceleration: bool = True`

### 3. Sizing Fixes

**3a. Hard contract cap for momentum trades**

New config:
- `momentum_max_contracts: int = 100`

Applied as final clamp in `_compute_momentum_size()`.

**3b. Global Kelly cap reduction**

- `kelly_fraction_cap`: 0.10 -> 0.06

**3c. Enable regime-conditional sizing**

Enable flags that were already implemented but defaulted to False:
- `regime_sizing_enabled`: False -> True
- `regime_min_edge_enabled`: True (already on)

Uniform min edge at 12% across all regimes:
- `regime_min_edge_mean_reverting`: 0.10 -> 0.12
- `regime_min_edge_trending`: 0.20 -> 0.12
- `regime_min_edge_high_vol`: 0.30 -> 0.12

Kelly multipliers (unchanged, already correct):
- mean_reverting: 1.0x
- trending_up: 0.4x
- trending_down: 0.5x
- high_vol: 0.0x

### 4. Feature Store Buffered Writes

**File**: `arb_bot/crypto/feature_store.py`

Buffer writes instead of appending to CSV on every trade:
- `_feature_buffer: List[dict]` — in-memory buffer
- `_flush_interval: int = 50` — flush every 50 records
- `flush()` method for bulk CSV write
- Auto-flush on graceful shutdown

Fixes the file permission error that killed the paper run at cycle 2030.

## Paper Test Config Summary

| Setting | Old | New |
|---------|-----|-----|
| `momentum_min_ofi_streak` | (new) | 3 |
| `momentum_require_ofi_acceleration` | (new) | True |
| `momentum_max_contracts` | (new) | 100 |
| `kelly_fraction_cap` | 0.10 | 0.06 |
| `regime_sizing_enabled` | False | True |
| `regime_min_edge_mean_reverting` | 0.10 | 0.12 |
| `regime_min_edge_trending` | 0.20 | 0.12 |
| `regime_min_edge_high_vol` | 0.30 | 0.12 |

## Files Changed

- `arb_bot/crypto/config.py` — 3 new config fields + defaults
- `arb_bot/crypto/engine.py` — streak tracker, acceleration filter in `_try_momentum_trades()`
- `arb_bot/crypto/feature_store.py` — buffered writes
- `arb_bot/crypto/paper_test.py` — updated config for relaunch
- `arb_bot/tests/test_momentum.py` — tests for streak + acceleration

## Success Criteria

- Momentum win rate improves from ~20% toward 55%+ target
- Fewer false entries (streak filter should eliminate impulse-driven trades)
- Model-path P&L turns positive (regime sizing prevents oversized losing trades)
- Paper run completes full duration without file permission crash
