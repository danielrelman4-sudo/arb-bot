#!/usr/bin/env bash
# Launch the arb bot with the v11 live starter profile.
set -euo pipefail

cd "$(dirname "$0")"

# Load the profile into this shell's env
set -a
source arb_bot/config/profiles/v11_live_starter.env
set +a

exec python3 -m arb_bot.main --live --stream
