# Flappy Brain — ISRC CN3 Hackathon 2026

Motor-imagery BCI controller for the hackathon's Unity Flappy-Bird game.
EEG comes in over LabStreamingLayer (LSL), a CSP + LDA classifier detects
motor imagery, and each detected imagery burst sends a flap command to the
game over UDP (ASCII `"1"` to `host:port`, default `127.0.0.1:5005`).

Two class schemes, selectable per participant (`--scheme` on calibrate/play;
label 1 = flap class in both):

- `leftright` (default): LEFT (0) vs RIGHT (1) hand imagery; RIGHT = flap.
- `feetrest`: REST (0) vs both-FEET (1) imagery; FEET = flap.

Pipeline: band-pass → CSP (m=2, 4 features/band) → LDA → sliding 1 s window /
250 ms step → exponential smoothing of the flap-class probability → flap when
smoothed probability > 0.7 with ≥ 1 s refractory. Optional filter-bank mode
splits 4–8 / 8–13 / 13–30 Hz (12 features).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest            # includes a ~40 s simulated end-to-end test
```

Channel count and sampling rate are discovered from the LSL stream info at
runtime — nothing is hard-coded to the NeuroCONCISE headset.

## Workflow

```bash
# 1. Calibrate (terminal cues, 20 trials/class by default; --scheme feetrest
#    for the REST/FEET variant)
.venv/bin/python scripts/calibrate.py --scheme leftright

# 2. Train (CV accuracy + confusion matrix + per-class recall; writes
#    model.pkl AND model_arrays.npz; scheme is stored in both)
.venv/bin/python scripts/train.py --data data/calib_<timestamp>.npz

# 3. Play (sends "1" to the game whenever the flap class is detected;
#    logs every classified window to data/session_<timestamp>.csv)
.venv/bin/python scripts/play.py --host <game-pc-ip> --port 5005
```

All scripts take `--stream-name` / `--any-host` to select the LSL stream.

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
accuracy) — tune `--threshold` toward RIGHT recall, not raw accuracy.

### numpy-only inference (`--arrays`) — no sklearn on the game machine

`train.py` always exports `model_arrays.npz` alongside the pickle (`--export
PATH` to override): CSP filters per band, LDA coef/intercept, fs, band edges,
layout version. `play.py --arrays model_arrays.npz` runs the identical
classifier with only numpy+scipy (P(RIGHT) = sigmoid of the LDA score, exactly
matching sklearn's binary `predict_proba`) — handy if the machine next to the
game can't install scikit-learn, and safer than unpickling.

### Filter-bank CSP (`--filterbank`)

`train.py --filterbank` uses bands (4–8, 8–13, 13–30) Hz with a separate CSP
per band (12 features) instead of the default single 8–30 Hz band. Bands are
stored inside the pickle and in the exported npz, so `play.py` picks them up
automatically on both paths (`--filterbank` on play.py is only a consistency
check). Default stays single-band 8–30 Hz.

### Session logs (`--log`)

Every play session writes `data/session_<timestamp>.csv` (`--log PATH` to
override, `--no-log` to disable): a `#`-prefixed header with model/threshold/
refractory/smoothing/window/step, then one row per classified window:
`t_rel,p_right,p_smooth,flapped`. Use it for post-hoc threshold tuning and as
extra (labeled-by-behavior) data for retraining.

### Testing without the headset

`--simulate` starts a fake LSL EEG source in-process (`SimulatedEEG`, 8 ch,
250 Hz, pink-ish noise + alpha). Channel layout: ch2/3 = left motor, ch4 =
vertex/Cz (feet area), ch5/6 = right motor. `leftright`: LEFT imagery
suppresses mu/beta on right-motor channels, RIGHT on left-motor
(contralateral ERD). `feetrest`: FEET imagery suppresses mu/beta on ch4 (Cz):

```bash
.venv/bin/python scripts/calibrate.py --simulate --scheme feetrest --trials 10
.venv/bin/python scripts/train.py --data data/calib_<timestamp>.npz --plot features.png
.venv/bin/python scripts/play.py --simulate --scheme feetrest
```

In calibration mode the simulator's ERD state follows the cues; in play mode
it runs a scripted cycle (`leftright`: rest → LEFT → rest → RIGHT; `feetrest`:
rest → FEET; 4 s each), so flaps should arrive only during flap-class phases.
`play.py --scheme` must match the model's scheme — a mismatch prints a
warning.

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

## Mock game (see it fly without Windows)

`scripts/mock_game.py` is a pygame Flappy-Bird clone that listens for the same
UDP flap packets (`"1"` on `127.0.0.1:5005`) as the real Unity game — the
classifier can fly a bird on macOS. Green neurons = +1, red neurons = −1,
hilly terrain kills on contact (auto-restart). Settings mirror the Unity game
(`--flap-force`, `--gravity`, `--time-limit`, `--spawn-interval`,
`--spawn-jitter`, `--seed`).

Two terminals (start the game first so no packets are missed):

```bash
# terminal 1 — the game
.venv/bin/python scripts/mock_game.py

# terminal 2 — the BCI pipeline (after calibrate + train)
.venv/bin/python scripts/play.py --simulate --monitor
```

![mock game](docs/mock_game.png)

Useful flags: `--auto` (built-in autopilot, no EEG needed), `--seed N`
(reproducible neuron layouts), `--duration S`, and `--frames-dir DIR` for
headless rendering checks (`SDL_VIDEODRIVER=dummy`). Manual controls:
SPACE = flap, R = restart, ESC = quit.

Note: we depend on `pygame-ce` (drop-in `pygame` replacement) because stock
pygame 2.6.1's font module is broken on Python 3.14.

## Game-day runbook (Windows PC + NeuroCONCISE)

1. Windows PC: .NET 5.0 runtime installed; start the game (`ISRC CN3 2026.exe`)
   and the NeuroCONCISE software; confirm the EEG stream is on LSL (type `EEG`).
2. This machine (same LAN), in order:
   `calibrate.py` → `train.py` (check RIGHT recall ≥ 0.8; if lower, redo with
   `--trials 30` or try `--filterbank`) → `play.py --host <game-pc-ip>
   --monitor` (optionally `--arrays model_arrays.npz` if sklearn is a problem
   on that machine).
3. Tune live with flags on `play.py`:
   - `--threshold` (default 0.7): lower if flaps are missed (recall!), raise on false flaps.
   - `--refractory` (default 1.0): min seconds between flaps.
   - `--smoothing` (default 0.3): lower = more responsive, higher = steadier.
   - `--window` / `--step` (default 1.0 / 0.25 s).
4. After each run, check the session CSV in `data/` to see where P(RIGHT) sat
   relative to the threshold.
5. Ctrl-C stops `play.py` cleanly and prints windows classified / flaps sent.

### Live monitor (`--monitor`)

`play.py --monitor` opens a live dashboard that updates with every window step
(~4 Hz): the last 4 s of band-passed EEG for the first channels
(`--monitor-channels N`, default 4), the smoothed P(RIGHT) over the last 15 s
with the flap threshold as a dashed line, flap events as red markers, and a
running flap count in the title.

![live monitor dashboard](docs/dashboard.png)

Notes:

- The matplotlib window must run on the main thread — it does, since the
  online loop drives redraws itself; don't run `play.py` backgrounded.
- Closing the dashboard window **stops the loop** (same as Ctrl-C) — the
  classifier is not meant to run unwatched on game day.
- Headless runs (e.g. `MPLBACKEND=Agg`) work fine: no window is shown and the
  loop just runs without rendering.

### Fallback: LSL output instead of UDP

If the organizers prefer LSL input to the game, swap `output_udp.py` for a
small LSL outlet: stream name `MotorImagery`, type `Markers`, 1 float32
channel, irregular rate; push `1.0` = flap, `0.0` = idle. That is the only
module that changes.

## Layout

```
flappy_brain/
  lsl_input.py      resolve EEG stream, reader thread -> 4 s ring buffer (with LSL timestamps)
  preprocessing.py  butterworth band-pass (SOS), optional notch
  csp.py            CSP + BandCSP (filter-bank), numpy/scipy only (importable for unpickling)
  model.py          Pipeline(BandCSP, LDA), CV + out-of-fold report, pickle, arrays export
  inference.py      numpy/scipy-only inference from exported .npz (no sklearn)
  calibration.py    cued LEFT/RIGHT recording -> data/calib_*.npz
  online.py         sliding-window inference, smoothing, threshold + refractory, SessionLogger
  monitor.py        live dashboard (MonitorState data logic + LiveMonitor figure)
  output_udp.py     UDP flap sender ("1" to host:port)
  simulate.py       fake LSL EEG outlet; ch2/3 left motor, ch4 Cz, ch5/6 right motor; L/R + feet ERD
scripts/            calibrate.py / train.py / play.py / mock_game.py (pygame demo game)
tests/              CSP + preprocessing + monitor + mock-game + inference + filterbank +
                    session-log unit tests, simulated end-to-end test
```
