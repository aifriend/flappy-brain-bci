"""Band-pass (and optional notch) filtering for EEG windows.

Data convention everywhere in this package: ``(n_samples, n_channels)`` for
continuous buffers, ``(n_epochs, n_channels, n_samples)`` for epochs.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

DEFAULT_LOW_HZ = 8.0
DEFAULT_HIGH_HZ = 30.0
DEFAULT_ORDER = 4


def make_bandpass_sos(
    fs: float,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Butterworth band-pass as second-order sections (numerically stable)."""
    nyq = fs / 2.0
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(
            f"need 0 < low < high < Nyquist ({nyq} Hz), got {low_hz}-{high_hz} Hz"
        )
    return signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def make_notch_sos(fs: float, freq_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    """IIR notch (mains hum) as second-order sections."""
    b, a = signal.iirnotch(freq_hz, q, fs=fs)
    return signal.tf2sos(b, a)


def apply_sos(sos: np.ndarray, data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Zero-phase filter along the sample axis (default axis 0)."""
    data = np.asarray(data, dtype=float)
    n = data.shape[axis]
    if n < 4 * sos.shape[0] * 3:
        raise ValueError(f"window too short ({n} samples) to filter")
    return signal.sosfiltfilt(sos, data, axis=axis)


def bandpass(
    data: np.ndarray,
    fs: float,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
    axis: int = 0,
) -> np.ndarray:
    """Convenience one-shot: build SOS and filter along the sample axis."""
    return apply_sos(make_bandpass_sos(fs, low_hz, high_hz, order), data, axis=axis)
