"""Simulated LSL EEG source for development without the headset.

Publishes a ``SimulatedEEG`` stream (type ``EEG``, 8 channels, 250 Hz float32):
pink-ish background noise + 10 Hz alpha on all channels, plus a strong
mu (10.5 Hz) / beta (22 Hz) rhythm on the motor channels.

Fixed channel layout (8 ch):
    ch 2, 3 = left-hemisphere motor cortex    (LEFT_MOTOR)
    ch 4    = vertex / Cz (feet area)         (CENTRAL)
    ch 5, 6 = right-hemisphere motor cortex   (RIGHT_MOTOR)

Two class schemes (``scheme`` constructor arg):

- ``leftright``: LEFT imagery suppresses mu/beta on RIGHT_MOTOR, RIGHT imagery
  on LEFT_MOTOR (contralateral ERD). Scripted cycle: rest -> LEFT -> rest ->
  RIGHT.
- ``feetrest``: FEET (both-feet) imagery suppresses mu/beta on CENTRAL (Cz)
  channels; rest = no suppression. Scripted cycle: rest -> FEET.

Label 1 (RIGHT / FEET) is the flap class in both schemes. Manual mode is
driven via :meth:`SimulatedEEGSource.set_state` (calibration cue hook, tests).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pylsl
from scipy import signal

STREAM_NAME = "SimulatedEEG"

LEFT_MOTOR = (2, 3)   # left-hemisphere motor channels
CENTRAL = (4,)        # vertex / Cz (feet area)
RIGHT_MOTOR = (5, 6)  # right-hemisphere motor channels

SCHEME_STATES = {
    "leftright": ("rest", "left", "right"),
    "feetrest": ("rest", "feet"),
}


class SimulatedEEGSource:
    """Generates fake EEG chunks, sample by sample, in stream time."""

    def __init__(
        self,
        n_channels: int = 8,
        fs: float = 250.0,
        scheme: str = "leftright",
        left_motor=LEFT_MOTOR,
        central=CENTRAL,
        right_motor=RIGHT_MOTOR,
        rest_duration: float = 4.0,
        imagery_duration: float = 4.0,
        left_duration: float | None = None,
        right_duration: float | None = None,
        scripted: bool = True,
        erd_depth: float = 0.15,
        seed: int = 0,
    ):
        if scheme not in SCHEME_STATES:
            raise ValueError(f"scheme must be one of {tuple(SCHEME_STATES)}, got {scheme!r}")
        self.n_channels = n_channels
        self.fs = float(fs)
        self.scheme = scheme
        self.left_motor = list(left_motor)
        self.central = list(central)
        self.right_motor = list(right_motor)
        self.rest_duration = rest_duration
        self.imagery_duration = imagery_duration
        # leftright scripted cycle may use asymmetric phase durations
        self.left_duration = left_duration if left_duration is not None else imagery_duration
        self.right_duration = right_duration if right_duration is not None else imagery_duration
        self.scripted = scripted
        self.erd_depth = erd_depth
        self._rng = np.random.default_rng(seed)
        self._sample = 0
        self._state = "rest"  # manual mode state
        # smoothed mu/beta amplitude per channel group (1.0 = full, erd_depth = suppressed)
        self._env = {"left": 1.0, "central": 1.0, "right": 1.0}
        # pink-ish noise state (one-pole low-pass of white noise)
        self._lp_b = [0.22]
        self._lp_a = [1.0, -0.95]
        self._lp_zi = np.zeros((1, n_channels))
        self._alpha_phase = self._rng.uniform(0, 2 * np.pi, n_channels)
        self._groups = (("left", self.left_motor), ("central", self.central), ("right", self.right_motor))
        self._mu_phase = {g: self._rng.uniform(0, 2 * np.pi, len(c)) for g, c in self._groups}
        self._beta_phase = {g: self._rng.uniform(0, 2 * np.pi, len(c)) for g, c in self._groups}

    # ------------------------------------------------------------- state
    @property
    def state(self) -> str:
        """Current brain state (manual mode) / base state."""
        return self._state

    def set_state(self, state: str) -> None:
        """Manual mode: force the imagery state (must belong to the scheme)."""
        if state not in SCHEME_STATES[self.scheme]:
            raise ValueError(
                f"state must be one of {SCHEME_STATES[self.scheme]} "
                f"for scheme {self.scheme!r}, got {state!r}"
            )
        self._state = state

    def state_at(self, t: float) -> str:
        """Scripted mode cycle (scheme-dependent)."""
        if not self.scripted:
            return self._state
        r, i = self.rest_duration, self.imagery_duration
        if self.scheme == "feetrest":
            return "rest" if (t % (r + i)) < r else "feet"
        l, rr = self.left_duration, self.right_duration
        phase = t % (2 * r + l + rr)
        if phase < r:
            return "rest"
        if phase < r + l:
            return "left"
        if phase < 2 * r + l:
            return "rest"
        return "right"

    def _is_suppressed(self, group: str, state: str) -> bool:
        """Contralateral / Cz ERD for the active scheme."""
        if self.scheme == "feetrest":
            return state == "feet" and group == "central"
        return (state == "right" and group == "left") or (state == "left" and group == "right")

    # ----------------------------------------------------------- synthesis
    def generate(self, n_samples: int) -> np.ndarray:
        """Next chunk, ``(n_samples, n_channels)``, advancing stream time."""
        t = (self._sample + np.arange(n_samples)) / self.fs
        self._sample += n_samples
        ch = self.n_channels

        # pink-ish background noise + a little white noise
        white = self._rng.standard_normal((n_samples, ch))
        pink, self._lp_zi = signal.lfilter(
            self._lp_b, self._lp_a, white, zi=self._lp_zi, axis=0
        )
        out = pink + 0.05 * white

        # 10 Hz alpha everywhere
        out += 0.5 * np.sin(2 * np.pi * 10.0 * t[:, None] + self._alpha_phase[None, :])

        # mu/beta on motor channels, suppressed per scheme during imagery (ERD)
        state = self.state_at(t[0])
        rate = min(1.0, (n_samples / self.fs) / 0.3)  # ~0.3 s transition
        for group, chans in self._groups:
            target = self.erd_depth if self._is_suppressed(group, state) else 1.0
            self._env[group] += (target - self._env[group]) * rate
            mu = np.sin(2 * np.pi * 10.5 * t[:, None] + self._mu_phase[group][None, :])
            beta = 0.6 * np.sin(2 * np.pi * 22.0 * t[:, None] + self._beta_phase[group][None, :])
            rhythm = 2.0 * self._env[group] * (mu + beta)
            for k, c in enumerate(chans):
                out[:, c] += rhythm[:, k]

        return out


def run_outlet(
    source: SimulatedEEGSource,
    name: str = STREAM_NAME,
    chunk_s: float = 0.1,
    stop_event: threading.Event | None = None,
) -> None:
    """Publish the source as an LSL outlet in real time (blocking)."""
    info = pylsl.StreamInfo(
        name=name,
        type="EEG",
        channel_count=source.n_channels,
        nominal_srate=source.fs,
        channel_format=pylsl.cf_float32,
        source_id="flappy-brain-sim",
    )
    outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=30)
    chunk = int(round(chunk_s * source.fs))
    if source.scripted:
        cycle = "rest/FEET" if source.scheme == "feetrest" else "rest/LEFT/rest/RIGHT"
        mode = f"[scripted {cycle} cycle]"
    else:
        mode = "[manual state control]"
    print(f"[sim] streaming '{name}' ({source.n_channels} ch @ {source.fs:g} Hz) {mode}")
    next_push = pylsl.local_clock()
    while not (stop_event and stop_event.is_set()):
        data = source.generate(chunk)
        outlet.push_chunk(data.astype(np.float32).tolist())
        next_push += chunk / source.fs
        delay = next_push - pylsl.local_clock()
        if delay > 0:
            time.sleep(delay)
        else:
            next_push = pylsl.local_clock()  # fell behind; resync
