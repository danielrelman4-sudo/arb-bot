# v18 Micro-Momentum Design

## Problem

v17 defensive filters block all trading when VPIN > 0.85 (high toxicity). In practice, BTC VPIN stays above 0.85 for extended periods (hours), resulting in zero trades. High VPIN indicates strong directional flow — exactly the condition where momentum strategies profit.

## Solution

Add a parallel "momentum path" in the engine cycle that activates in a VPIN "momentum zone" (0.85–0.95). Instead of sitting out, follow OFI direction by buying the nearest OTM contract in the $0.15–$0.40 price sweet spot.

## VPIN Tiered Gate

| VPIN Range | Behavior |
|---|---|
| < 0.85 | Normal model-based trading (existing v17 logic) |
| 0.85 – 0.95 | Momentum zone: model-based halted, momentum trades allowed |
| > 0.95 | Full halt: nothing trades |

## Momentum Trigger (all must be true)

1. Regime = `high_vol` (from regime detector)
2. OFI alignment > 0.6 (3+ of 4 timescale windows agree)
3. |weighted OFI magnitude| > threshold (default 200)
4. VPIN in momentum zone (0.85–0.95)
5. Not in regime transition (`is_transitioning = False`)

## Contract Selection

1. Filter for "above"/"below" daily contracts with TTE <= 15 min
2. Direction: bullish OFI → YES on "above" strikes above spot; bearish → NO on "above" strikes below spot
3. Price sweet spot: buy price must be $0.15–$0.40
   - Below $0.15: dead money, low delta, won't move enough
   - Above $0.40: momentum already priced in, poor risk/reward
4. Sort by strike proximity to spot (nearest first = highest gamma)
5. Take first candidate

## Sizing

Fixed-fraction (not Kelly — no model probability available):
- Base: `momentum_kelly_fraction` (default 0.03 = 3% of bankroll)
- Scaled by OFI alignment: `size *= alignment` (0.6–1.0)
- Hard cap: `momentum_max_position` (default $25)
- One momentum trade per symbol per cycle

## Safety Rails

- Max concurrent momentum positions: 2 (across all symbols)
- Cooldown: 120s per symbol after last momentum trade settles
- No momentum during regime transition
- Feature store tags momentum trades with `strategy="momentum"`

## Config Fields

```
momentum_enabled: bool = False
momentum_vpin_floor: float = 0.85
momentum_vpin_ceiling: float = 0.95
momentum_ofi_alignment_min: float = 0.6
momentum_ofi_magnitude_min: float = 200.0
momentum_max_tte_minutes: float = 15.0
momentum_price_floor: float = 0.15
momentum_price_ceiling: float = 0.40
momentum_kelly_fraction: float = 0.03
momentum_max_position: float = 25.0
momentum_max_concurrent: int = 2
momentum_cooldown_seconds: float = 120.0
```

## Engine Integration

Momentum path in `_run_cycle()` after VPIN check:

```
1. VPIN check
   - VPIN > ceiling (0.95) → full halt, return
   - VPIN > floor (0.85) → momentum_zone = True, skip model path
   - VPIN <= floor → normal model path (v17)
2. If momentum_zone AND momentum_enabled:
   a. Check regime == high_vol
   b. Compute OFI alignment + magnitude
   c. Select best contract per symbol (nearest OTM, $0.15–$0.40)
   d. Size and execute momentum trade
   e. Record to feature store + cycle recorder
3. End cycle (settle, record, etc.)
```

## Files Changed

- `arb_bot/crypto/config.py` — 12 momentum config fields + env vars
- `arb_bot/crypto/engine.py` — Tiered VPIN gate, `_try_momentum_trades()`, contract selection
- `arb_bot/crypto/paper_test.py` — Enable momentum, set config
- `arb_bot/tests/test_momentum.py` — ~40 tests

## Key Metrics to Watch

- Momentum trade win rate (target > 55%)
- Average payout ratio (entry $0.20–$0.35 → settlement $1.00 or $0.00)
- VPIN zone utilization (% of momentum-zone cycles that trigger trades)
- OFI direction accuracy (did price move in OFI direction?)
