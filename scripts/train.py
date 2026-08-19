#!/usr/bin/env python3
"""Train BandCSP + LDA from a calibration .npz -> model pickle + arrays export.

Prints stratified k-fold CV accuracy plus a recall-focused out-of-fold
evaluation (confusion matrix, per-class precision/recall/f1) — a missed
RIGHT (flap) means the bird falls, so watch RIGHT recall.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from flappy_brain.calibration import load_calibration, load_calibration_meta
from flappy_brain.model import (
    DEFAULT_BANDS,
    FILTERBANK_BANDS,
    cross_validate_report,
    export_arrays,
    save_model,
    train,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="calibration .npz from calibrate.py")
    ap.add_argument("--out", default="model.pkl", help="model output path (default model.pkl)")
    ap.add_argument("--export", default=None, help="arrays export .npz (default: alongside --out)")
    ap.add_argument("--components", type=int, default=4, help="CSP components per band (default 4)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    ap.add_argument("--low", type=float, default=8.0, help="single-band low edge Hz (default 8)")
    ap.add_argument("--high", type=float, default=30.0, help="single-band high edge Hz (default 30)")
    ap.add_argument("--filterbank", action="store_true",
                    help=f"filter-bank CSP over {FILTERBANK_BANDS} Hz instead of single band")
    ap.add_argument("--plot", default=None, help="optional: save CSP feature scatter to this PNG")
    args = ap.parse_args()

    epochs, labels, fs = load_calibration(args.data)
    meta = load_calibration_meta(args.data)
    scheme = meta["scheme"]
    bands = FILTERBANK_BANDS if args.filterbank else ((args.low, args.high),)
    print(
        f"[train] {epochs.shape[0]} epochs, {epochs.shape[1]} ch, fs={fs:g} Hz, "
        f"scheme={scheme}, bands={bands} ({'filterbank' if len(bands) > 1 else 'single band'})"
    )

    # BandCSP filters raw epochs internally; X stays unfiltered here.
    scores, y_oof = cross_validate_report(
        epochs, labels, fs, bands=bands, n_components=args.components, cv=args.cv
    )
    print(
        f"[train] {len(scores)}-fold CV accuracy: "
        + " ".join(f"{s:.2f}" for s in scores)
        + f"  (mean {scores.mean():.3f} +/- {scores.std():.3f})"
    )

    names = [meta["label_names"][0], meta["label_names"][1]]
    cm = confusion_matrix(labels, y_oof)
    print("[train] out-of-fold confusion matrix (rows=true, cols=pred):")
    print(f"[train]           pred {names[0]:>5} {names[1]:>5}")
    for i, name in enumerate(names):
        print(f"[train]   true {name:>5} {cm[i, 0]:>7} {cm[i, 1]:>7}")
    report = classification_report(labels, y_oof, target_names=names, digits=3)
    print("[train] per-class metrics (out-of-fold):")
    for line in report.splitlines():
        if line.strip():
            print(f"[train]   {line}")
    print(
        f"[train] hint: a missed {names[1]} = a fallen bird — if {names[1]} recall is low, "
        "lower --threshold in play.py toward recall (and raise --refractory if "
        "false flaps follow)."
    )

    model = train(epochs, labels, fs, bands=bands, n_components=args.components)
    model.scheme_ = scheme  # pickled alongside, for scheme-mismatch checks in play.py
    path = save_model(model, args.out)
    print(f"[train] model saved -> {path} (scheme={scheme})")

    export_path = Path(args.export) if args.export else path.parent / f"{path.stem}_arrays.npz"
    export_arrays(model, export_path, fs, bands, args.components, scheme=scheme, X=epochs, y=labels)
    print(f"[train] arrays exported -> {export_path} (sklearn-free inference via --arrays)")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feats = model.named_steps["csp"].transform(epochs)
        fig, ax = plt.subplots(figsize=(5, 4))
        for cls, name in enumerate(names):
            f = feats[labels == cls]
            ax.scatter(f[:, 0], f[:, -1], label=name, alpha=0.7)
        ax.set_xlabel("CSP feature 1")
        ax.set_ylabel(f"CSP feature {feats.shape[1]}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"[train] feature scatter saved -> {args.plot}")


if __name__ == "__main__":
    main()
