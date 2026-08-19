"""Filter-bank CSP smoke: feature shapes and CV separability on synthetic L/R data."""

import numpy as np

from flappy_brain.csp import BandCSP
from flappy_brain.model import FILTERBANK_BANDS, cross_validate_report

from synth_lr import make_lr_epochs

FS = 250.0


def test_bandcsp_feature_shapes():
    X, y = make_lr_epochs(n_epochs=10, seed=0)
    fbcsp = BandCSP(fs=FS, bands=FILTERBANK_BANDS, n_components=4).fit(X, y)
    feats = fbcsp.transform(X)
    assert feats.shape == (10, 12)  # 3 bands x 4 components
    assert np.all(np.isfinite(feats))


def test_filterbank_cv_separability():
    X, y = make_lr_epochs(n_epochs=60, seed=1)
    scores, _ = cross_validate_report(X, y, FS, bands=FILTERBANK_BANDS, cv=5)
    assert scores.mean() > 0.9, f"filter-bank CV too low: {scores}"


def test_single_band_cv_separability():
    X, y = make_lr_epochs(n_epochs=60, seed=1)
    scores, _ = cross_validate_report(X, y, FS, cv=5)
    assert scores.mean() > 0.9, f"single-band CV too low: {scores}"
