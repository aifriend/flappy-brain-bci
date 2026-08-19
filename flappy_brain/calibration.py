"""Terminal-based cued calibration recording.

Two schemes (``--scheme``): ``leftright`` cues LEFT(0)/RIGHT(1) imagery,
``feetrest`` cues REST(0)/FEET(1) imagery; label 1 (RIGHT / FEET) is always
the flap class. Epochs ``tmin..tmax`` seconds post-cue are sliced out of the
live :class:`~flappy_brain.lsl_input.EEGBuffer` using LSL timestamps and saved
with scheme metadata to ``data/calib_<timestamp>.npz``.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pylsl

from .lsl_input import EEGBuffer
from .model import DEFAULT_SCHEME, scheme_label_names


def _sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - pylsl.local_clock()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def run_calibration(
    buffer: EEGBuffer,
    n_trials_per_class: int = 20,
    cue_duration: float = 4.0,
    gap_duration: float = 2.0,
    tmin: float = 0.5,
    tmax: float = 2.5,
    out_dir: str | Path = "data",
    out_path: str | Path | None = None,
    on_cue=None,
    scheme: str = DEFAULT_SCHEME,
    seed: int = 0,
) -> Path:
    """Record a calibration session from a running buffer.

    ``on_cue(label, active)`` is an optional hook called at cue start/end —
    used by the simulator to sync its imagery state with the cues.
    ``scheme`` selects the cue pair (``leftright``: LEFT/RIGHT, ``feetrest``:
    REST/FEET) and is stored in the saved metadata.
    Returns the path of the saved ``.npz``.
    """
    label_names = scheme_label_names(scheme)
    cue_text = {0: label_names[0], 1: label_names[1]}
    fs = buffer.fs
    expected = int(round((tmax - tmin) * fs))
    trials = [0] * n_trials_per_class + [1] * n_trials_per_class
    rng = np.random.default_rng(seed)
    rng.shuffle(trials)

    epochs, labels = [], []
    skipped = 0
    total = len(trials)
    print(
        f"[calib] scheme={scheme} ({label_names[0]} vs {label_names[1]}), "
        f"{n_trials_per_class} trials/class, cue {cue_duration}s + gap "
        f"{gap_duration}s, epoch {tmin}-{tmax}s post-cue. Get ready..."
    )
    time.sleep(1.0)

    for i, label in enumerate(trials, 1):
        cue_onset = pylsl.local_clock()
        print(f"[calib] trial {i:2d}/{total}: {cue_text[label]}", flush=True)
        if on_cue:
            on_cue(label, True)

        # Slice as soon as the epoch is fully buffered (cue may still be on screen).
        _sleep_until(cue_onset + tmax + 0.05)
        data, ts = buffer.get_time_range(cue_onset + tmin, cue_onset + tmax)

        _sleep_until(cue_onset + cue_duration)
        if on_cue:
            on_cue(label, False)

        if len(data) < expected * 0.8:
            print(f"[calib]   ! epoch too short ({len(data)}/{expected} samples) — skipped")
            skipped += 1
        else:
            if len(data) < expected:  # minor under-run: pad with last sample
                pad = np.repeat(data[-1:], expected - len(data), axis=0)
                data = np.vstack([data, pad])
            epochs.append(data[:expected].T)  # (n_channels, n_samples)
            labels.append(label)

        _sleep_until(cue_onset + cue_duration + gap_duration)

    if not epochs:
        raise RuntimeError("no usable epochs recorded — is the stream running?")
    if skipped:
        print(f"[calib] warning: {skipped}/{total} epochs skipped (short)")

    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(out_dir) / f"calib_{stamp}.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        epochs=np.stack(epochs),
        labels=np.asarray(labels, dtype=int),
        fs=np.float64(fs),
        label_names=np.array(label_names),
        scheme=np.str_(scheme),  # 1 = flap class in every scheme
    )
    print(f"[calib] saved {len(epochs)} epochs ({fs:g} Hz, scheme={scheme}) -> {out_path}")
    return out_path


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a saved calibration file -> (epochs, labels, fs)."""
    z = np.load(path)
    return z["epochs"], z["labels"], float(z["fs"])


def load_calibration_meta(path: str | Path) -> dict:
    """Metadata stored alongside a calibration file (scheme, label_names, ...).

    Missing keys (older files) fall back to the leftright defaults.
    """
    z = np.load(path)
    meta = {"scheme": str(z["scheme"]) if "scheme" in z.files else DEFAULT_SCHEME}
    if "label_names" in z.files:
        meta["label_names"] = [str(n) for n in z["label_names"]]
    else:
        meta["label_names"] = list(scheme_label_names(meta["scheme"]))
    # tolerate legacy scheme tag from the very first sessions
    if meta["scheme"] == "left_right_mi":
        meta["scheme"] = "leftright"
    return meta
