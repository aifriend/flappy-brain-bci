"""Numpy/scipy-only inference from an exported model .npz — no sklearn, no pickle.

Load the export written by ``flappy_brain.model.export_arrays`` (train.py
--export) and classify raw EEG windows. Drop-in for the pickled sklearn
Pipeline in the online loop: exposes ``classes_`` and ``predict_proba``.

Only numpy + scipy are needed, so the game-side machine doesn't need
scikit-learn installed. P(RIGHT) is computed as expit(LDA decision score),
which matches sklearn's binary ``LinearDiscriminantAnalysis.predict_proba``
exactly (verified against sklearn).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit

from .model import EXPORT_VERSION


def _logvar_features(filters: np.ndarray, X: np.ndarray) -> np.ndarray:
    """CSP log-variance features: X (n_epochs, n_channels, n_samples)."""
    Z = np.einsum("cf,ecs->efs", filters, X)
    var = Z.var(axis=2)
    var_sum = var.sum(axis=1, keepdims=True)
    var_sum[var_sum == 0] = 1.0
    return np.log(var / var_sum)


class ArraysModel:
    classes_ = np.array([0, 1])

    def __init__(self, fs: float, bands: np.ndarray, filters: list[np.ndarray],
                 lda_coef: np.ndarray, lda_intercept: np.ndarray, scheme: str = "leftright"):
        self.fs = float(fs)
        self.scheme = scheme
        self.bands = [tuple(b) for b in np.asarray(bands, dtype=float)]
        self.filters = filters
        self.lda_coef = np.asarray(lda_coef, dtype=float).ravel()
        self.lda_intercept = float(np.asarray(lda_intercept, dtype=float).ravel()[0])
        self._sos = [
            butter(4, [lo / (self.fs / 2), hi / (self.fs / 2)], btype="band", output="sos")
            for lo, hi in self.bands
        ]

    @classmethod
    def load(cls, path: str | Path) -> "ArraysModel":
        z = np.load(path)
        version = int(z["version"])
        if version != EXPORT_VERSION:
            raise ValueError(f"unsupported export version {version} (expected {EXPORT_VERSION})")
        bands = z["bands"]
        filters = [z[f"filters_{i}"] for i in range(len(bands))]
        scheme = str(z["scheme"]) if "scheme" in z.files else "leftright"
        return cls(z["fs"], bands, filters, z["lda_coef"], z["lda_intercept"], scheme=scheme)

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        feats = [
            _logvar_features(f, sosfiltfilt(sos, X, axis=-1))
            for f, sos in zip(self.filters, self._sos)
        ]
        return np.concatenate(feats, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, 2) probabilities [P(LEFT), P(RIGHT)]; expit of LDA scores."""
        d = self._features(X) @ self.lda_coef + self.lda_intercept
        p_right = expit(d)
        return np.column_stack([1.0 - p_right, p_right])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
