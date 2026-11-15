"""Session log tests: header, row counts, integration with run_online (fakes)."""

import numpy as np
import pylsl

from flappy_brain.online import SessionLogger, run_online


class FakeBuffer:
    fs = 250.0

    def get_last(self, n_samples):
        data = np.random.default_rng(0).standard_normal((n_samples, 8))
        # stamped "now" so the stale-stream guard in run_online treats it as live
        return data, pylsl.local_clock() - np.arange(n_samples)[::-1] / self.fs


class FakeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))  # confident RIGHT


class FakeSender:
    def __init__(self):
        self.sent = 0

    def send_flap(self):
        self.sent += 1


def test_session_logger_file(tmp_path):
    path = tmp_path / "logs" / "session.csv"
    with SessionLogger(path, header={"model": "m.pkl", "threshold": 0.7}) as log:
        log.log(0.0, 0.81, 0.81, True)
        log.log(0.25, 0.79, 0.80, False)
    lines = path.read_text().splitlines()
    assert lines[0] == "# flappy-brain session log v1"
    assert "# model=m.pkl" in lines and "# threshold=0.7" in lines
    assert "t_rel,p_right,p_smooth,flapped" in lines
    rows = [l for l in lines if not l.startswith("#")][1:]
    assert len(rows) == 2
    assert rows[0] == "0.000,0.8100,0.8100,1"


def test_run_online_logs_every_window(tmp_path):
    path = tmp_path / "session.csv"
    with SessionLogger(path, header={"model": "fake"}) as log:
        stats = run_online(
            FakeBuffer(),
            FakeModel(),
            FakeSender(),
            window_s=0.25,
            step_s=0.05,
            threshold=0.65,
            refractory_s=0.2,
            smoothing=0.5,
            duration=0.55,
            logger=log,
            verbose=False,
        )
    assert stats["windows"] >= 8
    assert stats["flaps"] >= 1  # confident-RIGHT fake model must flap
    assert log.rows == stats["windows"]
    rows = [l for l in path.read_text().splitlines() if not l.startswith("#")][1:]
    assert len(rows) == stats["windows"]
    flapped = sum(int(r.split(",")[3]) for r in rows)
    assert flapped == stats["flaps"]
