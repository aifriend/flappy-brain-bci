"""Exported arrays + numpy-only inference must match the pickled pipeline."""

import numpy as np

from flappy_brain.inference import ArraysModel
from flappy_brain.model import FILTERBANK_BANDS, export_arrays, train

from synth_lr import make_lr_epochs

FS = 250.0


def _check_equivalence(bands, tmp_path):
    X, y = make_lr_epochs(n_epochs=60, seed=0)
    model = train(X, y, FS, bands=bands)
    path = tmp_path / "model_arrays.npz"
    export_arrays(model, path, FS, bands, X=X, y=y)

    arrays_model = ArraysModel.load(path)
    X_test, _ = make_lr_epochs(n_epochs=20, seed=99)
    p_pickle = model.predict_proba(X_test)
    p_arrays = arrays_model.predict_proba(X_test)

    assert np.array_equal(model.predict(X_test), arrays_model.predict(X_test))
    assert np.abs(p_pickle - p_arrays).max() < 0.05


def test_arrays_inference_matches_pickle_single_band(tmp_path):
    _check_equivalence(((8.0, 30.0),), tmp_path)


def test_arrays_inference_matches_pickle_filterbank(tmp_path):
    _check_equivalence(FILTERBANK_BANDS, tmp_path)


def test_arrays_export_contents(tmp_path):
    X, y = make_lr_epochs(n_epochs=20, seed=1)
    model = train(X, y, FS)
    path = tmp_path / "export.npz"
    export_arrays(model, path, FS, ((8.0, 30.0),), X=X, y=y)
    z = np.load(path)
    for key in ("version", "fs", "bands", "n_components", "lda_coef", "lda_intercept",
                "filters_0", "feat_mean_left", "feat_mean_right"):
        assert key in z, f"missing {key} in export"
    assert z["filters_0"].shape == (8, 4)
    assert z["lda_coef"].shape == (1, 4)
