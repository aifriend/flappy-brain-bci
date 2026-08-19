"""Resolve an EEG stream from LSL and continuously buffer it.

A background thread pulls chunks from the inlet into a numpy ring buffer
(~4 s by default). LSL timestamps are corrected into the local clock domain
and stored alongside the samples so calibration can slice epochs by cue time.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pylsl


def resolve_eeg_stream(
    name: str | None = None,
    any_host: bool = False,
    timeout: float = 30.0,
    resolve_wait: float = 1.0,
) -> pylsl.StreamInfo:
    """Find an LSL stream of type ``EEG`` (optionally by name).

    Unless ``any_host`` is set, streams on this machine are preferred when
    several match. Raises ``TimeoutError`` if nothing shows up in ``timeout``.
    """
    deadline = time.monotonic() + timeout
    local_host = socket.gethostname()
    while True:
        streams = pylsl.resolve_streams(resolve_wait)
        matches = [s for s in streams if s.type().lower() == "eeg"]
        if name is not None:
            matches = [s for s in matches if s.name() == name]
        if matches:
            if not any_host:
                local = [s for s in matches if s.hostname() in ("", local_host)]
                if local:
                    matches = local
            chosen = matches[0]
            print(
                f"[lsl] using stream '{chosen.name()}' (type={chosen.type()}, "
                f"{chosen.channel_count()} ch @ {chosen.nominal_srate()} Hz, "
                f"host={chosen.hostname() or '?'})"
            )
            return chosen
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no LSL stream of type 'EEG'"
                + (f" named '{name}'" if name else "")
                + f" found within {timeout:.0f} s"
            )


class EEGBuffer:
    """Ring buffer fed by a background LSL reader thread."""

    def __init__(self, stream: pylsl.StreamInfo, buffer_seconds: float = 4.0):
        self.inlet = pylsl.StreamInlet(stream, max_buflen=30, processing_flags=0)
        info = self.inlet.info()
        fs = float(info.nominal_srate())
        if fs <= 0:
            raise ValueError("stream reports irregular sampling rate; need a fixed-rate EEG stream")
        self.fs = fs
        self.n_channels = info.channel_count()
        self.capacity = int(round(buffer_seconds * self.fs)) + 1

        self._data = np.zeros((self.capacity, self.n_channels))
        self._ts = np.full(self.capacity, np.nan)
        self._write = 0
        self._count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tc_offset = 0.0
        self._tc_last = 0.0
        self.samples_received = 0

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        """Open the inlet, sync clocks, and start the reader thread."""
        self.inlet.open_stream(timeout=10.0)
        self._sync_clock()
        self._thread = threading.Thread(target=self._reader, name="lsl-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.inlet.close_stream()

    def __enter__(self) -> "EEGBuffer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def available(self) -> int:
        with self._lock:
            return self._count

    def get_last(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Most recent ``n_samples`` as ``(n, n_channels)`` + timestamps.

        Returns fewer rows if the buffer has not filled yet.
        """
        with self._lock:
            n = min(self._count, n_samples)
            idx = (self._write - n + np.arange(n)) % self.capacity
            return self._data[idx].copy(), self._ts[idx].copy()

    def get_time_range(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """All buffered samples with ``t0 <= timestamp <= t1``, time-ordered."""
        with self._lock:
            n = self._count
            idx = (self._write - n + np.arange(n)) % self.capacity
            ts = self._ts[idx]
            mask = (ts >= t0) & (ts <= t1)
            order = np.argsort(ts[mask])
            return self._data[idx][mask][order].copy(), ts[mask][order].copy()

    # ------------------------------------------------------------- internal
    def _sync_clock(self) -> None:
        try:
            self._tc_offset = self.inlet.time_correction(timeout=2.0)
            self._tc_last = time.monotonic()
        except Exception as exc:  # keep old offset; same-machine streams are ~0 anyway
            print(f"[lsl] time_correction failed: {exc}")

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.2)
            except Exception as exc:
                print(f"[lsl] pull_chunk error: {exc}")
                time.sleep(0.5)
                continue
            if time.monotonic() - self._tc_last > 5.0:
                self._sync_clock()
            if not samples:
                continue
            arr = np.asarray(samples, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.n_channels:
                continue  # malformed chunk; drop it
            ts = np.asarray(timestamps, dtype=float) + self._tc_offset
            self._append(arr, ts)

    def _append(self, data: np.ndarray, ts: np.ndarray) -> None:
        n = data.shape[0]
        if n > self.capacity:
            data, ts = data[-self.capacity :], ts[-self.capacity :]
            n = self.capacity
        with self._lock:
            idx = (self._write + np.arange(n)) % self.capacity
            self._data[idx] = data
            self._ts[idx] = ts
            self._write = (self._write + n) % self.capacity
            self._count = min(self._count + n, self.capacity)
            self.samples_received += n
