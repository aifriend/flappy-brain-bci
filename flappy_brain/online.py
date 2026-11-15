"""Online inference: sliding window -> CSP+LDA probability -> smoothing -> flap.

Decision rule: exponential smoothing of P(RIGHT motor imagery) — RIGHT is the
flap class; send one UDP flap when the smoothed probability crosses
``threshold``, then hold off for at least ``refractory_s`` seconds (one
sustained imagery burst = one flap).

The model receives the *raw* window — band-pass filtering lives inside the
pipeline (BandCSP) or the exported ArraysModel, so training and online
preprocessing can't diverge. The monitor observer still gets an 8–30 Hz
filtered window for display.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pylsl

from . import model as model_mod
from .lsl_input import EEGBuffer
from .output_udp import UDPSender
from .preprocessing import apply_sos, make_bandpass_sos


class SessionLogger:
    """Lightweight CSV session log: one row per classified window.

    Header comment lines record the run configuration, then columns:
    ``t_rel,p_right,p_smooth,flapped``. Buffered writes, flushed on close.
    """

    def __init__(self, path: str | Path, header: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", buffering=8192)
        self._f.write("# flappy-brain session log v1\n")
        for key, value in header.items():
            self._f.write(f"# {key}={value}\n")
        self._f.write("t_rel,p_right,p_smooth,flapped\n")
        self.rows = 0

    def log(self, t_rel: float, p_right: float, p_smooth: float, flapped: bool) -> None:
        self._f.write(f"{t_rel:.3f},{p_right:.4f},{p_smooth:.4f},{int(flapped)}\n")
        self.rows += 1

    def close(self) -> None:
        self._f.flush()
        self._f.close()

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_online(
    buffer: EEGBuffer,
    model,
    sender: UDPSender,
    window_s: float = 1.0,
    step_s: float = 0.25,
    threshold: float = 0.7,
    refractory_s: float = 1.0,
    smoothing: float = 0.3,
    duration: float | None = None,
    flap_callback=None,
    observer=None,
    logger: SessionLogger | None = None,
    verbose: bool = True,
    stale_s: float | None = 1.0,
    clock=pylsl.local_clock,
) -> dict:
    """Run the classify-and-flap loop until Ctrl-C or ``duration`` seconds.

    ``smoothing`` is the weight of the newest probability (0..1).
    ``flap_callback(wall_time, prob)`` is invoked on every flap (testing hook).
    ``observer(t_rel, filtered_window, p_smooth, flapped)`` is invoked on every
    classified window with an 8–30 Hz filtered ``(n_samples, n_channels)``
    window — e.g. ``LiveMonitor.update``. If it returns ``False`` the loop
    stops (used when the monitor window is closed).
    ``stale_s``: if the newest buffered sample is older than this many seconds
    (the stream stalled or dropped), the window is skipped instead of being
    re-classified, and smoothing restarts when data resumes. ``None`` disables
    the check; it is also skipped when the buffer supplies no timestamps.
    ``clock`` returns the current time in the buffer's timestamp domain.
    ``logger`` receives one row per classified window (see :class:`SessionLogger`).
    Returns ``{"windows": n_windows, "flaps": n_flaps}``.
    """
    fs = buffer.fs
    win = int(round(window_s * fs))
    right_col = model_mod.flap_class_index(model)
    display_sos = make_bandpass_sos(fs) if observer is not None else None

    n_windows = 0
    n_flaps = 0
    p_smooth: float | None = None
    last_flap = -np.inf
    start = time.monotonic()
    next_tick = start
    print(
        f"[online] window {window_s}s / step {step_s}s, threshold {threshold}, "
        f"refractory {refractory_s}s, smoothing {smoothing} — Ctrl-C to stop"
    )
    try:
        while True:
            next_tick += step_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            data, ts = buffer.get_last(win)
            if data.shape[0] < win:
                continue  # buffer still filling
            if (
                stale_s is not None
                and ts is not None
                and len(ts)
                and clock() - float(ts[-1]) > stale_s
            ):
                # Stream stalled: the buffer only holds old samples. Re-classifying
                # them could keep firing flaps, so skip and restart smoothing.
                p_smooth = None
                if duration is not None and time.monotonic() - start >= duration:
                    break
                continue
            X = data.T[None, :, :]  # raw (1, n_channels, n_samples); model filters
            p = float(model.predict_proba(X)[0, right_col])
            p_smooth = p if p_smooth is None else smoothing * p + (1 - smoothing) * p_smooth
            n_windows += 1

            now = time.monotonic()
            flapped = False
            if p_smooth > threshold and now - last_flap >= refractory_s:
                sender.send_flap()
                last_flap = now
                n_flaps += 1
                flapped = True
                wall = time.time()
                if verbose:
                    print(f"[online] {now - start:7.2f}s  FLAP  (P(MI)={p_smooth:.2f})", flush=True)
                if flap_callback:
                    flap_callback(wall, p_smooth)
            elif verbose and n_windows % 4 == 0:
                print(f"[online] {now - start:7.2f}s  P(MI)={p_smooth:.2f}", flush=True)

            if logger is not None:
                logger.log(now - start, p, p_smooth, flapped)

            if observer is not None:
                filtered = apply_sos(display_sos, data)
                if observer(now - start, filtered, p_smooth, flapped) is False:
                    print("[online] monitor window closed — stopping")
                    break

            if duration is not None and now - start >= duration:
                break
    except KeyboardInterrupt:
        print("\n[online] interrupted")
    print(f"[online] done: {n_windows} windows classified, {n_flaps} flaps sent")
    return {"windows": n_windows, "flaps": n_flaps}
