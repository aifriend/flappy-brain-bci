"""Preprocessing unit tests: band-pass sanity with synthetic sines."""

import numpy as np

from flappy_brain.preprocessing import bandpass, make_bandpass_sos, make_notch_sos, apply_sos

FS = 250.0


def _rms_ratio(freq_in, fs=FS, dur_s=4.0):
    """RMS of filtered sine relative to original, measured away from edges."""
    t = np.arange(int(fs * dur_s)) / fs
    x = np.sin(2 * np.pi * freq_in * t)[:, None]  # (n_samples, 1 channel)
    y = bandpass(x, fs)
    sl = slice(int(fs), -int(fs))  # drop 1 s at each edge
    return np.sqrt(np.mean(y[sl] ** 2) / np.mean(x[sl] ** 2))


def test_in_band_sine_passes():
    assert _rms_ratio(12.0) > 0.8
    assert _rms_ratio(20.0) > 0.8


def test_out_of_band_attenuated():
    assert _rms_ratio(2.0) < 0.1
    assert _rms_ratio(50.0) < 0.1
    assert _rms_ratio(80.0) < 0.1


def test_multichannel_shape_preserved():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 8))
    y = bandpass(x, FS)
    assert y.shape == x.shape


def test_notch_removes_mains():
    t = np.arange(int(FS * 4)) / FS
    x = np.sin(2 * np.pi * 50.0 * t)[:, None]
    y = apply_sos(make_notch_sos(FS, 50.0), x)
    sl = slice(int(FS), -int(FS))
    ratio = np.sqrt(np.mean(y[sl] ** 2) / np.mean(x[sl] ** 2))
    assert ratio < 0.1


def test_bandpass_rejects_bad_band():
    import pytest

    with pytest.raises(ValueError):
        make_bandpass_sos(FS, 30.0, 8.0)  # low > high
