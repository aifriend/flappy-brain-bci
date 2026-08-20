#!/usr/bin/env bash
# Run the full Flappy Brain demo stack with one command:
#   simulated pilot (brain) -> play.py (decoder) -> mock game (visual)
#
# Usage:
#   scripts/run_demo.sh                  # live demo: game window + monitor window
#   scripts/run_demo.sh --headless       # no windows (CI / SSH)
#   scripts/run_demo.sh --duration 60    # auto-stop after N seconds (default: run until Ctrl-C)
#
# Prerequisites: .venv set up and a trained model.pkl (see README Quickstart).
# Tunables: edit the FLAGS blocks below.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

FLAP_PORT=5005     # play.py -> game
STATE_PORT=5006    # game -> pilot
HEADLESS=0
DURATION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --headless) HEADLESS=1; shift ;;
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- preflight -------------------------------------------------------------
[ -x "$PY" ] || { echo "error: $PY not found — create the venv first (see README)"; exit 1; }
[ -f model.pkl ] || { echo "error: model.pkl not found — run scripts/calibrate.py then scripts/train.py first"; exit 1; }
for port in "$FLAP_PORT" "$STATE_PORT"; do
  if command -v lsof >/dev/null && lsof -nP -iUDP:"$port" >/dev/null 2>&1; then
    echo "error: UDP port $port is already in use (stale demo process?)."
    echo "       check with: lsof -nP -iUDP:$port"
    exit 1
  fi
done

PIDS=()
cleanup() {
  echo
  echo "[demo] shutting down..."
  kill "${PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "$HEADLESS" = 1 ]; then
  export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
fi

DUR_PILOT=""; DUR_PLAY=""; DUR_GAME=""
if [ -n "$DURATION" ]; then
  DUR_PILOT="--duration $((DURATION + 15))"
  DUR_PLAY="--duration $((DURATION + 10))"
  DUR_GAME="--duration $DURATION"
fi

# --- tuned flags (from scripts/sweep.py results) ---------------------------
PLAY_FLAGS="--window 0.75 --threshold 0.55 --smoothing 0.5 --refractory 1.0"
GAME_FLAGS="--gravity 100 --flap-force 150 --terrain-scale 0.5 --seed 7"

echo "[demo] 1/3 starting pilot (simulated brain) on state port $STATE_PORT..."
$PY scripts/pilot.py --state-port "$STATE_PORT" $DUR_PILOT &
PILOT_PID=$!; PIDS+=($PILOT_PID)
sleep 4
kill -0 "$PILOT_PID" 2>/dev/null || { echo "error: pilot died at startup"; exit 1; }

echo "[demo] 2/3 starting decoder (play.py) -> flaps to 127.0.0.1:$FLAP_PORT..."
MONITOR_FLAG="--monitor"; [ "$HEADLESS" = 1 ] && MONITOR_FLAG=""
# shellcheck disable=SC2086
$PY scripts/play.py $MONITOR_FLAG $PLAY_FLAGS $DUR_PLAY &
PLAY_PID=$!; PIDS+=($PLAY_PID)
sleep 5
kill -0 "$PLAY_PID" 2>/dev/null || { echo "error: play.py died at startup"; exit 1; }

echo "[demo] 3/3 starting mock game (listening on :$FLAP_PORT, broadcasting state to :$STATE_PORT)..."
# shellcheck disable=SC2086
$PY scripts/mock_game.py --state-broadcast "$STATE_PORT" $GAME_FLAGS $DUR_GAME &
GAME_PID=$!; PIDS+=($GAME_PID)

echo "[demo] all three running — Ctrl-C to stop everything"
wait
