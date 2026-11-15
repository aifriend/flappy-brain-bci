# flappy-brain-bci

Real-time motor-imagery EEG controller for the ISRC CN3 Hackathon 2026
"Flappy Brain" game: EEG streams in over LabStreamingLayer (LSL), a
CSP + LDA classifier detects the imagery, and flap commands go out to the
game over UDP. Includes a simulated EEG source and a pygame mock game, so
the whole loop runs on macOS/Linux with zero hardware.

[![test](https://github.com/aifriend/flappy-brain-bci/actions/workflows/test.yml/badge.svg)](https://github.com/aifriend/flappy-brain-bci/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **LSL input** — resolves any stream of type `EEG` (name/`--any-host`
  filters), background reader thread into a timestamped ring buffer;
  channel count and sampling rate discovered at runtime.
- **Classifier** — CSP (m=2) + LDA, numpy/scipy-only CSP implementation;
  optional **filter-bank CSP** (4–8 / 8–13 / 13–30 Hz, 12 features).
- **Two MI schemes** — `leftright` (LEFT vs RIGHT hand imagery, RIGHT = flap)
  and `feetrest` (REST vs both-feet imagery, FEET = flap); pick per
  participant (see *Calibration schemes*).
- **Calibration tool** — terminal-cued recording with configurable
  trials/cue/gap/epoch window, saves epochs + scheme metadata to `.npz`.
- **Live monitor dashboard** — scrolling band-passed EEG, smoothed flap-class
  probability with threshold line, flap markers, running flap count.
- **Mock game** — pygame Flappy-Bird clone speaking the same UDP protocol as
  the Unity game (mirrors its settings: flap force, gravity, time limit,
  spawn interval/jitter, seed).
- **Simulated EEG source** — fake LSL outlet with lateralized ERD, cue-driven
  or scripted, for headset-free development.
- **sklearn-free deployment** — models export to `.npz` arrays; inference
  needs only numpy + scipy (no pickle, no scikit-learn).
- **Session logging** — per-window CSV logs (probability, smoothed
  probability, flap flag) for post-hoc tuning and retraining.
- **Tested** — a unit-test suite including a full simulated LSL→classifier→UDP
  end-to-end run.

## Repository layout

```
flappy_brain/           the package
  lsl_input.py          resolve EEG stream, reader thread -> 4 s ring buffer (LSL timestamps)
  preprocessing.py      butterworth band-pass (SOS), optional notch
  csp.py                CSP + BandCSP (filter-bank), numpy/scipy only (importable for unpickling)
  model.py              Pipeline(BandCSP, LDA), CV + out-of-fold report, pickle, arrays export
  inference.py          numpy/scipy-only inference from exported .npz (no sklearn)
  calibration.py        cued LEFT/RIGHT or REST/FEET recording -> data/calib_*.npz
  online.py             sliding-window inference, smoothing, threshold + refractory, SessionLogger
  monitor.py            live dashboard (MonitorState data logic + LiveMonitor figure)
  output_udp.py         UDP flap sender ("1" to host:port)
  pilot.py              closed-loop demo controller (state parse, bang-bang/pulse modes)
  simulate.py           fake LSL EEG outlet; ch2/3 left motor, ch4 Cz, ch5/6 right motor
scripts/
  calibrate.py          record a cued calibration session (--scheme, --simulate)
  train.py              train + evaluate + export model
  play.py               live loop: LSL -> classifier -> UDP flaps (--monitor, --arrays, --log)
  pilot.py              simulated pilot: game state -> brain state (closed-loop demo)
  mock_game.py          pygame demo game speaking the game's UDP protocol
  sweep.py              headless decoder-hyperparameter sweep against game score
tests/                  unit tests + simulated end-to-end test
docs/                   screenshots (dashboard, mock game)
data/                   calibration .npz + session logs (gitignored)
```

## Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies: numpy, scipy, scikit-learn, pylsl, matplotlib, pygame-ce,
pytest. Notes:

- `pylsl` ships prebuilt wheels (bundled liblsl) for macOS/Linux/Windows.
- We depend on **`pygame-ce`** (drop-in `pygame` replacement, prebuilt
  manylinux/macOS wheels with bundled SDL) because stock pygame 2.6.1's font
  module is broken on Python 3.14.

## Quickstart (zero hardware)

Train a model on the built-in simulator first (`model.pkl` is gitignored —
everyone trains their own):

```bash
.venv/bin/python scripts/calibrate.py --simulate --trials 10
.venv/bin/python scripts/train.py --data data/calib_<timestamp>.npz
```

**One-command demo** (recommended): launches the closed-loop stack —
simulated pilot (the "brain"), the classifier, and the mock game — with the
tuned flags, correct startup order, port checks, and Ctrl-C cleanup:

```bash
scripts/run_demo.sh                  # live windows (game + monitor)
scripts/run_demo.sh --headless       # no windows (CI / SSH)
scripts/run_demo.sh --duration 60    # auto-stop after N s
```

Or run the stages by hand — two terminals. For a **playable** demo (bird
stays airborne), use the tuned combo below — gentler game physics, calmer
terrain, and a simulator cycle with a long RIGHT (flap) phase and short
dive phases. Start the pipeline first so the game doesn't idle unflapped
during startup:

```bash
# terminal 1 — the BCI pipeline with live dashboard (starts first)
.venv/bin/python scripts/play.py --simulate --monitor \
    --sim-rest 1.0 --sim-left 1.2 --sim-right 8 \
    --refractory 2.0 --window 0.75 --threshold 0.75

# terminal 2 — the mock game (start ~3 s later)
.venv/bin/python scripts/mock_game.py --gravity 10 --flap-force 18 --terrain-scale 0.5
```

Verified headless on three seeds: bird survives full 38 s runs with zero
game-overs, scoring 1–6 points. The simulator cycle flags (`--sim-rest`,
`--sim-left`, `--sim-right`, `--sim-feet`) tune the scripted phases;
`--terrain-scale` flattens the hills as a demo assist (default 1.0 keeps the
hilly terrain). Without these, the default physics (gravity 1800) need ~2
flaps/s while the default 16 s cycle only flaps during 4 s of RIGHT — the
bird dies almost instantly, which is expected, not a classifier bug.

Use `--scheme feetrest` on all three commands for the REST/FEET variant
(cycle rest → FEET). For the more impressive **closed-loop** demo (a
simulated pilot that actually watches the bird), see the next section.

## Closed-loop simulated pilot (recommended demo)

The scripted demo above is **open-loop**: the brain cycles on a timer, whether
the bird needs a flap or not. `scripts/pilot.py` closes the loop — a
simulated pilot watches the game's state and produces RIGHT imagery only when
the bird should climb, so the *real* pipeline (simulated EEG → LSL →
`play.py` classifier → UDP flaps → game) plays with realistic classifier
latency and noise. Three processes:

```bash
# terminal 1 — the brain (simulated EEG driven by game state)
.venv/bin/python scripts/pilot.py --state-port 5006

# terminal 2 — the decoder (NO --simulate: it resolves the pilot's stream)
.venv/bin/python scripts/play.py --monitor \
    --window 0.75 --threshold 0.55 --smoothing 0.5 --refractory 1.0

# terminal 3 — the game (start last, ~5 s after the decoder, with state broadcast)
.venv/bin/python scripts/mock_game.py --state-broadcast 5006 \
    --gravity 100 --flap-force 150 --terrain-scale 0.5
```

The game broadcasts a v2 CSV packet per frame to the pilot:

```
t,bird_y,bird_vy,corridor_center_y,score,alive,green_dx,green_dy,red_dx,red_dy
```

(`*_dx/dy` = nearest on-screen green/red neuron ahead of the bird, relative
to the bird; `9999` when none. Old 6-field v1 packets are still accepted.)

The pilot's controller (in `flappy_brain/pilot.py`) tracks a desired
altitude: the nearest green neuron's height when one is ahead within a ±90 px
chase band around the corridor center (bigger excursions end in terrain at
this latency), dodged around any red neuron sitting near the path, else the
corridor center. It emits a short climb-intent pulse when the predicted bird
position (velocity lead) dips below the target (deadband; screen y grows
downward), and — importantly for the leftright scheme — holds LEFT imagery
as the "sink" state, which keeps P(RIGHT) pinned near 0 far more reliably
than unmodeled rest (measured: near-zero false flaps, vs several per minute
with rest).

Measured headless (classifier latency ~1 s included, 60 s games):

- default physics (gravity 1800) + full hills: 41 game-overs in 60 s — a ~1 s
  classifier lag means the bird falls 900+ px before a flap lands; closed
  loop cannot fix that, as with a human reacting too slowly;
- moderate physics (gravity 100) + `--terrain-scale 0.5`, tuned decoder
  (window 0.75 / threshold 0.55 / smoothing 0.5 / refractory 1.0), three
  seeds: scores 2 / 5 / 7, 3 game-overs each, best lives 23.8–26.9 s.
  Compare the open-loop scripted demo, which needed gravity 10 + calm
  terrain to survive — the closed loop tolerates ~10× the gravity.

### Tuning against game score

`scripts/sweep.py` runs the whole loop headless over a trimmed grid of
decoder settings (window × threshold × smoothing × refractory) with a fixed
game seed and prints a score/game-over table — use it to re-tune after
changing the pilot, physics, or model:

```bash
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy .venv/bin/python scripts/sweep.py \
    --game-seconds 35 --seed 7
```

The recommended combo in the commands above came from such a sweep.

## Game-day runbook (Windows PC + NeuroCONCISE)

1. Windows PC: .NET 5.0 runtime installed; start the game (`ISRC CN3 2026.exe`)
   and the NeuroCONCISE software; confirm the EEG stream is on LSL (type `EEG`).
2. This machine (same LAN), in order:
   `calibrate.py` → `train.py` (check flap-class recall ≥ 0.8; if lower, redo
   with `--trials 30` or try `--filterbank`) → `play.py --host <game-pc-ip>
   --monitor` (optionally `--arrays model_arrays.npz` if sklearn is a problem
   on that machine).
3. Tune live with flags on `play.py`:
   - `--threshold` (default 0.7): lower if flaps are missed (recall!), raise on false flaps.
   - `--refractory` (default 1.0): min seconds between flaps.
   - `--smoothing` (default 0.3): lower = more responsive, higher = steadier.
   - `--window` / `--step` (default 1.0 / 0.25 s).
4. After each run, check the session CSV in `data/` to see where the
   flap-class probability sat relative to the threshold.
5. Ctrl-C stops `play.py` cleanly and prints windows classified / flaps sent.

### Fallback: LSL output instead of UDP

If the organizers prefer LSL input to the game, swap `output_udp.py` for a
small LSL outlet: stream name `MotorImagery`, type `Markers`, 1 float32
channel, irregular rate; push `1.0` = flap, `0.0` = idle. That is the only
module that changes.

## Calibration schemes

Two class schemes, label 1 = flap class in both:

- `leftright` (default): LEFT (0) vs RIGHT (1) hand imagery; RIGHT = flap.
- `feetrest`: REST (0) vs both-FEET (1) imagery; FEET = flap.

`calibrate.py --scheme` selects the cues and stores the scheme in the `.npz`
metadata; `train.py` picks it up (confusion-matrix labels, model metadata);
`play.py --scheme` selects the scripted simulator cycle and warns if the
model was trained for another scheme.

### Game-day scheme screening (pick what works for the pilot)

Record a quick session per scheme, train both, compare:

```bash
.venv/bin/python scripts/calibrate.py --scheme leftright --trials 10
.venv/bin/python scripts/train.py --data data/calib_<t1>.npz --out model_lr.pkl
.venv/bin/python scripts/calibrate.py --scheme feetrest --trials 10
.venv/bin/python scripts/train.py --data data/calib_<t2>.npz --out model_fr.pkl
```

Compare the out-of-fold confusion matrices and flap-class (RIGHT/FEET)
recall; play with the winner. ~10 trials/class per scheme takes about two
minutes per scheme and is usually enough to see which one separates.

### train.py output (simulated calibration data)

```
[train] 5-fold CV accuracy: 1.00 1.00 1.00 1.00 1.00  (mean 1.000 +/- 0.000)
[train] out-of-fold confusion matrix (rows=true, cols=pred):
[train]           pred  LEFT RIGHT
[train]   true  LEFT       8       0
[train]   true RIGHT       0       8
[train] per-class metrics (out-of-fold): precision/recall/f1 per class ...
[train] hint: a missed RIGHT = a fallen bird — if RIGHT recall is low, lower
       --threshold in play.py toward recall ...
```

The recall numbers are out-of-fold (`cross_val_predict`, same folds as the
accuracy) — tune `--threshold` toward flap-class recall, not raw accuracy.

## Live monitor

`play.py --monitor` opens a live dashboard that updates with every window step
(~4 Hz): the last 4 s of band-passed EEG for the first channels
(`--monitor-channels N`, default 4), the smoothed flap-class probability over
the last 15 s with the flap threshold as a dashed line, flap events as red
markers, and a running flap count in the title.

![live monitor dashboard](docs/dashboard.png)

Notes:

- The matplotlib window must run on the main thread — it does, since the
  online loop drives redraws itself; don't run `play.py` backgrounded.
- Closing the dashboard window **stops the loop** (same as Ctrl-C) — the
  classifier is not meant to run unwatched on game day.
- Headless runs (e.g. `MPLBACKEND=Agg`) work fine: no window is shown and the
  loop just runs without rendering.

## Mock game

`scripts/mock_game.py` is a pygame Flappy-Bird clone that listens for the same
UDP flap packets (`"1"` on `127.0.0.1:5005`) as the real Unity game — the
classifier can fly a bird without Windows. Green neurons = +1, red
neurons = −1, hilly terrain kills on contact (auto-restart). Settings mirror
the Unity game (`--flap-force`, `--gravity`, `--time-limit`,
`--spawn-interval`, `--spawn-jitter`, `--seed`); `--terrain-scale` (default
1.0) flattens the hills as a demo assist. Game-overs are logged with cause
(top/bottom terrain, altitude, corridor) and the exit line reports cumulative
score, best single-life time, game-over count, and UDP flaps received.

![mock game](docs/mock_game.png)

Useful flags: `--auto` (built-in autopilot, no EEG needed), `--seed N`
(reproducible neuron layouts), `--duration S`, and `--frames-dir DIR` for
headless rendering checks (`SDL_VIDEODRIVER=dummy`). Manual controls:
SPACE = flap, R = restart, ESC = quit.

## Deployment without sklearn

`train.py` always exports `model_arrays.npz` alongside the pickle (`--export
PATH` to override): CSP filters per band, LDA coef/intercept, fs, band edges,
scheme, layout version. `play.py --arrays model_arrays.npz` runs the
identical classifier with only numpy+scipy (flap probability = sigmoid of the
LDA score, exactly matching sklearn's binary `predict_proba`) — handy if the
machine next to the game can't install scikit-learn, and safer than
unpickling.

`train.py --filterbank` uses bands (4–8, 8–13, 13–30) Hz with a separate CSP
per band (12 features) instead of the default single 8–30 Hz band. Bands are
stored inside the pickle and in the exported npz, so `play.py` picks them up
automatically on both paths (`--filterbank` on play.py is only a consistency
check).

Session logs: every play session writes `data/session_<timestamp>.csv`
(`--log PATH` to override, `--no-log` to disable): a `#`-prefixed header with
model/threshold/refractory/smoothing/window/step, then one row per classified
window: `t_rel,p_right,p_smooth,flapped`. Use it for post-hoc threshold
tuning and as extra (labeled-by-behavior) data for retraining.

## Testing

```bash
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest tests/ -q
```

The suite covers CSP, preprocessing, filter-bank, arrays-export equivalence, session
logging, monitor data logic, mock-game mechanics, scheme handling, and a full
simulated end-to-end test (simulated LSL stream → calibration → training →
online loop → UDP flaps at a listener, flaps asserted inside the scripted
imagery phases). CI runs the same command on ubuntu-latest (see
`.github/workflows/test.yml`).

## License

MIT — see [LICENSE](LICENSE).
