"""Scheme-selection tests: feetrest separability, scripted-cycle flapping,
scheme metadata round-trip, mismatch warning."""

import numpy as np

from flappy_brain.model import (
    cross_validate_report,
    export_arrays,
    load_model,
    save_model,
    train,
    warn_scheme_mismatch,
)
from flappy_brain.inference import ArraysModel
from flappy_brain.online import run_online
from flappy_brain.simulate import SimulatedEEGSource

from synth_lr import make_feetrest_epochs

FS = 250.0


def test_feetrest_cv_separability():
    X, y = make_feetrest_epochs(n_epochs=60, seed=1)
    scores, _ = cross_validate_report(X, y, FS, cv=5)
    assert scores.mean() > 0.9, f"feetrest CV too low: {scores}"


def test_scheme_metadata_roundtrip(tmp_path):
    X, y = make_feetrest_epochs(n_epochs=20, seed=0)
    model = train(X, y, FS)
    model.scheme_ = "feetrest"
    pkl = save_model(model, tmp_path / "m.pkl")
    assert load_model(pkl).scheme_ == "feetrest"

    npz = export_arrays(model, tmp_path / "m.npz", FS, ((8.0, 30.0),), scheme="feetrest")
    assert ArraysModel.load(npz).scheme == "feetrest"


def test_scheme_mismatch_warning():
    assert warn_scheme_mismatch("feetrest", "leftright") is not None
    assert warn_scheme_mismatch("leftright", "leftright") is None
    assert warn_scheme_mismatch(None, "feetrest") is None  # old model without metadata


class _ReplayBuffer:
    """Replays a pregenerated recording, advancing ~step_s per get_last call."""

    fs = FS

    def __init__(self, data, step_s):
        self.data = data
        self.step = int(step_s * FS)
        self.pos = 0

    @property
    def sim_t(self):
        return self.pos / self.fs

    def get_last(self, n_samples):
        self.pos = min(self.pos + self.step, len(self.data))
        return self.data[max(0, self.pos - n_samples): self.pos], None


class _CountSender:
    def __init__(self):
        self.sent = 0

    def send_flap(self):
        self.sent += 1


def test_feetrest_scripted_cycle_flaps_only_in_feet():
    """Train on slices of a scripted feetrest recording, then run the online
    loop against the replay: flaps must land in FEET phases, never in rest."""
    source = SimulatedEEGSource(
        scheme="feetrest", scripted=True, rest_duration=1.0, imagery_duration=1.0, seed=3
    )
    data = source.generate(int(FS * 8))  # cycle: rest [0,1) FEET [1,2), repeat

    # training epochs from well inside each phase of the first 3 cycles
    X, y = [], []
    for k in range(3):
        for label, (a, b) in ((0, (2 * k + 0.25, 2 * k + 0.85)), (1, (2 * k + 1.25, 2 * k + 1.85))):
            seg = data[int(a * FS): int(b * FS)]
            X.append(seg.T)
            y.append(label)
    model = train(np.array(X), np.array(y), FS)

    buffer = _ReplayBuffer(data, step_s=0.05)
    flap_times = []
    stats = run_online(
        buffer,
        model,
        _CountSender(),
        window_s=0.5,
        step_s=0.05,
        threshold=0.7,
        refractory_s=0.0,  # wall-clock refractory would couple to test-machine load
        smoothing=0.4,
        duration=4.5,
        flap_callback=lambda _wall, _p: flap_times.append(buffer.sim_t),
        verbose=False,
    )
    feet_intervals = [(1.0, 2.0), (3.0, 4.0)]
    assert stats["flaps"] >= 1, "no flaps at all"
    assert flap_times, "flap_callback never fired"
    # A classification at sim-t sees data [t-0.5, t], so FEET-phase ERD keeps
    # influencing windows for ~window_s + smoothing tail after the phase ends.
    lag_s = 0.5 + 0.3
    unexplained = [
        t for t in flap_times
        if not any(a - 0.1 <= t <= b + lag_s for a, b in feet_intervals)
    ]
    assert not unexplained, f"flaps not caused by a FEET phase: {unexplained}"
    # and there must be flaps squarely inside each FEET phase
    for a, b in feet_intervals:
        assert any(a + 0.2 <= t <= b for t in flap_times), f"no flaps inside FEET [{a}, {b}]"
