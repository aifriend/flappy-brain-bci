#!/usr/bin/env python3
"""Headless decoder-hyperparameter sweep against mock-game score.

For each (window, threshold, smoothing, refractory) combo, runs the full
closed loop — pilot.py (brain) + play.py (decoder) + mock_game.py (headless,
fixed seed) — and records score, game-overs, best life, flaps from the game
log. Sequential, distinct UDP port pair per run to avoid cross-talk.

Usage: .venv/bin/python scripts/sweep.py [--game-seconds 35] [--seed 7]
Total runtime ≈ (game_seconds + ~9 s startup) x number of combos.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python"

# trimmed grid around the known-good point; ~13 runs
GRID = [
    # (window, threshold, smoothing, refractory)
    (0.35, 0.55, 0.5, 1.0),
    (0.35, 0.60, 0.5, 1.0),
    (0.35, 0.70, 0.5, 1.0),
    (0.50, 0.55, 0.5, 1.0),
    (0.50, 0.60, 0.5, 1.0),   # current recommended
    (0.50, 0.70, 0.5, 1.0),
    (0.75, 0.55, 0.5, 1.0),
    (0.75, 0.60, 0.5, 1.0),
    (0.75, 0.70, 0.5, 1.0),
    (0.50, 0.60, 0.4, 1.0),
    (0.50, 0.60, 0.65, 1.0),
    (0.50, 0.60, 0.5, 0.6),
    (0.50, 0.60, 0.5, 1.5),
]

GAME_ARGS = ["--gravity", "100", "--flap-force", "150", "--terrain-scale", "0.5"]
PILOT_ARGS = ["--mode", "pulse", "--deadband", "25", "--lead", "0.7",
              "--pulse", "0.9", "--cooldown", "0.4"]
DONE_RE = re.compile(r"score (-?\d+) \(best life ([\d.]+)s\), (\d+) game-over\(s\), (\d+) UDP flap")


def run_combo(idx, combo, game_s, seed, logdir):
    window, threshold, smoothing, refractory = combo
    flap_port = 6100 + 2 * idx
    state_port = flap_port + 1
    tag = f"w{window}_t{threshold}_s{smoothing}_r{refractory}"
    procs = []
    try:
        procs.append(subprocess.Popen(
            [str(PY), "scripts/pilot.py", "--state-port", str(state_port),
             "--duration", str(game_s + 12)] + PILOT_ARGS,
            cwd=ROOT, stdout=open(logdir / f"{tag}_pilot.log", "w"),
            stderr=subprocess.STDOUT))
        time.sleep(1.0)
        procs.append(subprocess.Popen(
            [str(PY), "scripts/play.py", "--stream-name", "SimulatedEEG", "--no-log",
             "--duration", str(game_s + 7), "--port", str(flap_port),
             "--window", str(window), "--threshold", str(threshold),
             "--smoothing", str(smoothing), "--refractory", str(refractory)],
            cwd=ROOT, stdout=open(logdir / f"{tag}_play.log", "w"),
            stderr=subprocess.STDOUT, env={**__import__("os").environ, "MPLBACKEND": "Agg"}))
        time.sleep(4.0)
        game_log = logdir / f"{tag}_game.log"
        game = subprocess.Popen(
            [str(PY), "scripts/mock_game.py", "--duration", str(game_s),
             "--seed", str(seed), "--port", str(flap_port),
             "--state-broadcast", str(state_port)] + GAME_ARGS,
            cwd=ROOT, stdout=open(game_log, "w"), stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "SDL_VIDEODRIVER": "dummy"})
        game.wait()
        for p in procs:
            p.wait()
        m = DONE_RE.search(game_log.read_text())
        if not m:
            return (tag, None)
        score, best, deaths, flaps = m.groups()
        return (tag, dict(score=int(score), best_life=float(best),
                          deaths=int(deaths), flaps=int(flaps)))
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-seconds", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--logdir", default="/tmp/flappy_sweep")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, combo in enumerate(GRID):
        t0 = time.time()
        tag, res = run_combo(i, combo, args.game_seconds, args.seed, logdir)
        res = res or {}
        print(f"[{i+1}/{len(GRID)}] {tag}: {res} ({time.time()-t0:.0f}s)", flush=True)
        results.append((tag, combo, res))

    ok = [(t, c, r) for t, c, r in results if r]
    ok.sort(key=lambda x: (-x[2]["score"], x[2]["deaths"], -x[2]["best_life"]))
    print("\n=== sweep results (sorted by score, deaths) ===")
    print(f"{'config':<28} {'score':>5} {'deaths':>6} {'best_life':>9} {'flaps':>5}")
    for tag, _, r in ok:
        print(f"{tag:<28} {r['score']:>5} {r['deaths']:>6} {r['best_life']:>9.1f} {r['flaps']:>5}")


if __name__ == "__main__":
    main()
