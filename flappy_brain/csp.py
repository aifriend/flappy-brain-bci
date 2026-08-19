"""Common Spatial Patterns (CSP) for 2-class motor imagery, numpy/scipy only.

Kept as module-level classes in ``flappy_brain.csp`` so pickled models can be
unpickled as long as this package is importable.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocessing import apply_sos, make_bandpass_sos


def _normalized_covariance(epoch: np.ndarray) -> np.ndarray:
    """Covariance of one ``(n_channels, n_samples)`` epoch, scaled by its trace."""
    cov = epoch @ epoch.T / (epoch.shape[1] - 1)
    trace = np.trace(cov)
    return cov / trace if trace > 0 else cov


class CSP(BaseEstimator, TransformerMixin):
    """CSP spatial filter + log-variance features.

    Parameters
    ----------
    n_components : int
        Total number of filters kept: the first/last ``n_components // 2``
        eigenvectors of the joint diagonalization (m = 2 -> 4 features).
    """

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(f"X must be (n_epochs, n_channels, n_samples), got {X.shape}")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"CSP here is 2-class only, got classes {classes}")
        n_channels = X.shape[1]
        if self.n_components > n_channels:
            raise ValueError("n_components exceeds number of channels")

        covs = []
        for cls in classes:
            epochs_c = X[y == cls]
            covs.append(
                np.mean([_normalized_covariance(ep) for ep in epochs_c], axis=0)
            )

        # Joint diagonalization: V^T (R0+R1) V = I, V^T R0 V diagonal.
        # Eigenvalues near 1 -> variance dominated by class 0, near 0 -> class 1.
        eigenvalues, V = eigh(covs[0], covs[0] + covs[1])
        order = np.argsort(eigenvalues)[::-1]
        V = V[:, order]
        self.eigenvalues_ = eigenvalues[order]

        m = self.n_components // 2
        picks = list(range(m)) + list(range(n_channels - m, n_channels))
        self.filters_ = V[:, picks]  # (n_channels, n_components)
        self.classes_ = classes
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"X must be (n_epochs, n_channels, n_samples), got {X.shape}")
        # Z = W^T x  ->  log of per-component variance (normalized)
        Z = np.einsum("cf,ecs->efs", self.filters_, X)
        var = Z.var(axis=2)
        var_sum = var.sum(axis=1, keepdims=True)
        var_sum[var_sum == 0] = 1.0
        return np.log(var / var_sum)


class BandCSP(BaseEstimator, TransformerMixin):
    """Band-pass + CSP per frequency band, features concatenated.

    Takes *raw* epochs ``(n_epochs, n_channels, n_samples)`` and does the
    band-pass filtering internally, so the same object works identically in
    training and online inference. Single band ``((8, 30),)`` = classic CSP;
    multiple bands = filter-bank CSP (FBCSP).
    """

    def __init__(self, fs: float = 250.0, bands=((8.0, 30.0),), n_components: int = 4):
        self.fs = fs
        self.bands = bands
        self.n_components = n_components

    def _filter(self, X: np.ndarray, band) -> np.ndarray:
        sos = make_bandpass_sos(self.fs, band[0], band[1])
        return apply_sos(sos, X, axis=-1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.csps_ = []
        for band in self.bands:
            self.csps_.append(CSP(n_components=self.n_components).fit(self._filter(X, band), y))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = [csp.transform(self._filter(X, band)) for csp, band in zip(self.csps_, self.bands)]
        return np.concatenate(feats, axis=1)
