"""CSP unit tests: output shape and separability on synthetic data."""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from flappy_brain.csp import CSP


def _make_two_class_epochs(n_epochs=40, n_channels=6, fs=250.0, dur_s=1.0, seed=0):
    """Class 0: strong 10 Hz on channels 0-2. Class 1: strong 10 Hz on channels 3-5."""
    rng = np.random.default_rng(seed)
    n_samples = int(fs * dur_s)
    t = np.arange(n_samples) / fs
    X, y = [], []
    for i in range(n_epochs):
        label = i % 2
        noise = 0.3 * rng.standard_normal((n_channels, n_samples))
        src = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        ep = noise.copy()
        if label == 0:
            ep[:3] += 2.0 * src
            ep[3:] += 0.2 * src
        else:
            ep[:3] += 0.2 * src
            ep[3:] += 2.0 * src
        X.append(ep)
        y.append(label)
    return np.array(X), np.array(y)


def test_csp_output_shape():
    X, y = _make_two_class_epochs()
    csp = CSP(n_components=4).fit(X, y)
    feats = csp.transform(X)
    assert feats.shape == (len(X), 4)
    assert csp.filters_.shape == (6, 4)
    assert np.all(np.isfinite(feats))


def test_csp_separability():
    X, y = _make_two_class_epochs(n_epochs=60, seed=1)
    pipe = Pipeline([("csp", CSP(n_components=4)), ("lda", LinearDiscriminantAnalysis())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv)
    assert scores.mean() > 0.9, f"CV accuracy too low: {scores}"


def test_csp_rejects_bad_input():
    csp = CSP(n_components=4)
    with pytest.raises(ValueError):
        csp.fit(np.zeros((10, 6)), np.zeros(10))  # not 3-D
    with pytest.raises(ValueError):
        csp.fit(np.zeros((10, 6, 50)), np.arange(10))  # more than 2 classes
