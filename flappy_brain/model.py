"""CSP + LDA model: build, cross-validate, train, save, load, export.

Class scheme: LEFT (0) vs RIGHT (1) motor imagery; RIGHT is the flap class.
The pipeline takes *raw* epochs/windows — band-pass filtering lives inside
:class:`flappy_brain.csp.BandCSP` so train and online paths can't diverge.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from .csp import BandCSP

LABEL_LEFT = 0
LABEL_RIGHT = 1  # flap class
LABEL_NAMES = {LABEL_LEFT: "LEFT", LABEL_RIGHT: "RIGHT"}

# Per-participant class schemes; label 1 is the flap class in every scheme.
SCHEMES = {
    "leftright": ("LEFT", "RIGHT"),
    "feetrest": ("REST", "FEET"),
}
DEFAULT_SCHEME = "leftright"

DEFAULT_BANDS = ((8.0, 30.0),)
FILTERBANK_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
EXPORT_VERSION = 1


def scheme_label_names(scheme: str) -> tuple[str, str]:
    """(label 0 name, label 1 name) for a scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {tuple(SCHEMES)}")
    return SCHEMES[scheme]


def warn_scheme_mismatch(model_scheme: str | None, requested_scheme: str | None) -> str | None:
    """Warning text when the model's scheme and the requested one differ, else None."""
    if model_scheme and requested_scheme and model_scheme != requested_scheme:
        return (
            f"scheme mismatch: model was trained for {model_scheme!r} "
            f"but --scheme {requested_scheme!r} was given"
        )
    return None


def build_pipeline(fs: float, bands=DEFAULT_BANDS, n_components: int = 4) -> Pipeline:
    """sklearn Pipeline(BandCSP -> LDA) on raw (unfiltered) epochs."""
    return Pipeline(
        [
            ("csp", BandCSP(fs=fs, bands=tuple(bands), n_components=n_components)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )


def _splitter(y: np.ndarray, cv: int, random_state: int) -> StratifiedKFold:
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(cv, counts.min()))
    if n_splits < 2:
        raise ValueError("not enough trials per class for cross-validation")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_report(
    X: np.ndarray,
    y: np.ndarray,
    fs: float,
    bands=DEFAULT_BANDS,
    n_components: int = 4,
    cv: int = 5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified k-fold on a fresh pipeline. Returns (per-fold scores, out-of-fold preds)."""
    skf = _splitter(y, cv, random_state)
    pipe = build_pipeline(fs, bands, n_components)
    scores = cross_val_score(pipe, X, y, cv=skf)
    y_oof = cross_val_predict(pipe, X, y, cv=skf)
    return scores, y_oof


def train(X: np.ndarray, y: np.ndarray, fs: float, bands=DEFAULT_BANDS, n_components: int = 4) -> Pipeline:
    """Fit the final pipeline on all calibration data (raw epochs)."""
    model = build_pipeline(fs, bands, n_components)
    model.fit(X, y)
    return model


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Pickle the pipeline. Requires ``flappy_brain.csp`` importable at load time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str | Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def flap_class_index(model) -> int:
    """Column of P(RIGHT) in ``model.predict_proba`` output."""
    if hasattr(model, "named_steps"):
        classes = model.named_steps["lda"].classes_
    else:
        classes = model.classes_  # numpy-only ArraysModel
    return list(classes).index(LABEL_RIGHT)


def export_arrays(
    model: Pipeline,
    path: str | Path,
    fs: float,
    bands,
    n_components: int = 4,
    scheme: str = DEFAULT_SCHEME,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> Path:
    """Export everything needed for sklearn-free inference to an .npz.

    Stores per-band CSP filter matrices, LDA coef/intercept, fs, band edges,
    n_components, the class scheme and a layout version tag (plus per-class
    feature means when X/y are given, for documentation/retraining). Load with
    :class:`flappy_brain.inference.ArraysModel`.
    """
    csp_step: BandCSP = model.named_steps["csp"]
    lda: LinearDiscriminantAnalysis = model.named_steps["lda"]
    arrays = {
        "version": np.int64(EXPORT_VERSION),
        "fs": np.float64(fs),
        "bands": np.asarray(bands, dtype=float),
        "n_components": np.int64(n_components),
        "scheme": np.str_(scheme),
        "lda_coef": np.asarray(lda.coef_, dtype=float),
        "lda_intercept": np.asarray(lda.intercept_, dtype=float),
    }
    for i, csp in enumerate(csp_step.csps_):
        arrays[f"filters_{i}"] = csp.filters_
    if X is not None and y is not None:
        feats = model.named_steps["csp"].transform(X)
        for cls, name in ((LABEL_LEFT, "left"), (LABEL_RIGHT, "right")):
            arrays[f"feat_mean_{name}"] = feats[y == cls].mean(axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
