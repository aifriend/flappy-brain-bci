"""Scheme-selection tests: feetrest separability, scripted-cycle flapping,
scheme metadata round-trip, mismatch warning."""

import numpy as np

from flappy_brain.model import (
    LABEL_RIGHT,
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


def test_scripted_cycle_custom_durations():
    """--sim-rest/--sim-left/--sim-right wire through to the scripted cycle."""
    src = SimulatedEEGSource(scripted=True, rest_duration=1.0, left_duration=2.0,
                             right_duration=3.0)
    # cycle: rest [0,1) LEFT [1,3) rest [3,4) RIGHT [4,7), repeat
    assert src.state_at(0.5) == "rest"
    assert src.state_at(1.5) == "left"
    assert src.state_at(3.5) == "rest"
    assert src.state_at(4.5) == "right"
    assert src.state_at(7.5) == "rest"  # wraps around
    # defaults unchanged: symmetric 4 s phases
    default = SimulatedEEGSource(scripted=True)
    assert default.left_duration == default.right_duration == 4.0
    assert default.state_at(5.0) == "left" and default.state_at(13.0) == "right"


class _StaleBuffer:
    """Always returns the same full window, stamped ``age_s`` seconds in the past."""

    fs = FS

    def __init__(self, data, age_s):
        self.data = data
        self.age_s = age_s

    def get_last(self, n_samples):
        n = min(n_samples, len(self.data))
        ts = np.arange(n) / self.fs + (1000.0 - self.age_s - n / self.fs)
        return self.data[-n:], ts


class _AlwaysFlapModel:
    classes_ = np.array([0, LABEL_RIGHT])

    def predict_proba(self, X):
        return np.array([[0.0, 1.0]])


def _run_stale_case(age_s):
    sender = _CountSender()
    stats = run_online(
        _StaleBuffer(np.zeros((int(FS), 2)), age_s),
        _AlwaysFlapModel(),
        sender,
        window_s=0.5,
        step_s=0.02,
        threshold=0.5,
        refractory_s=0.0,
        duration=0.3,
        verbose=False,
        stale_s=1.0,
        clock=lambda: 1000.0,
    )
    return stats, sender


def test_fresh_stream_flaps():
    stats, sender = _run_stale_case(age_s=0.0)
    assert stats["windows"] > 0
    assert sender.sent > 0


def test_stale_stream_does_not_flap():
    """A stalled stream leaves old samples in the buffer; they must not keep flapping."""
    stats, sender = _run_stale_case(age_s=5.0)
    assert stats == {"windows": 0, "flaps": 0}
    assert sender.sent == 0
