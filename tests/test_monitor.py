"""Monitor tests: data history/formatting logic, plus one headless render.

No display required — matplotlib is forced to Agg before LiveMonitor is built.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from flappy_brain.monitor import LiveMonitor, MonitorState

FS = 250.0
STEP_S = 0.25
WIN = int(FS * 1.0)


def _window(t0, n=WIN, n_ch=6):
    t = (np.arange(n) + t0) / FS
    data = np.stack(
        [np.sin(2 * np.pi * (8 + 2 * k) * t) for k in range(n_ch)], axis=1
    )
    return data


def test_state_eeg_history_continuity():
    st = MonitorState(fs=FS, n_channels_show=4, eeg_seconds=4.0, step_s=STEP_S)
    step = int(STEP_S * FS)
    n_updates = 10
    for i in range(n_updates):
        st.update(i * STEP_S, _window(i * step), p_smooth=0.1, flapped=False)
    times, stacked, labels = st.eeg_view()
    # 1 s first window + 9 steps of 0.25 s = 3.25 s of continuous trace
    expected = WIN + (n_updates - 1) * step
    assert stacked.shape == (4, expected)
    assert len(labels) == 4
    assert times[0] == - (expected - 1) / FS and times[-1] == 0.0


def test_state_eeg_history_capped():
    st = MonitorState(fs=FS, n_channels_show=4, eeg_seconds=4.0, step_s=STEP_S)
    step = int(STEP_S * FS)
    for i in range(40):  # 1 + 39*0.25 = 10.75 s > 4 s cap
        st.update(i * STEP_S, _window(i * step), p_smooth=0.5, flapped=False)
    _, stacked, _ = st.eeg_view()
    assert stacked.shape[1] == int(4.0 * FS)


def test_state_prob_history_and_flaps_pruned():
    st = MonitorState(fs=FS, n_channels_show=2, prob_seconds=15.0, step_s=STEP_S)
    step = int(STEP_S * FS)
    for i in range(100):  # 25 s of updates
        flapped = i in (10, 50, 90)
        st.update(i * STEP_S, _window(i * step), p_smooth=0.2 + 0.01 * i, flapped=flapped)
    t, p = st.prob_view()
    assert len(t) == int(15.0 / STEP_S) + 1  # only the last 15 s kept
    assert t[0] >= 25 * STEP_S - 15.0 - 1e-9
    tf, pf = st.flap_view()
    # last t_rel = 99*0.25 = 24.75 -> cutoff 9.75: flap at t=2.5 has scrolled out,
    # flaps at t=12.5 and t=22.5 remain
    assert len(tf) == 2 and abs(tf[0] - 12.5) < 1e-9 and abs(tf[1] - 22.5) < 1e-9
    assert "2 flap" in st.title_text()


def test_live_monitor_renders_headless(tmp_path):
    mon = LiveMonitor(fs=FS, n_channels_show=4, threshold=0.7, step_s=STEP_S)
    step = int(STEP_S * FS)
    for i in range(30):  # 7.5 s
        p = 0.9 if 12 <= i <= 20 else 0.1
        flapped = i in (14, 18)
        assert mon.update(i * STEP_S, _window(i * step), p_smooth=p, flapped=flapped)
    out = tmp_path / "dashboard.png"
    mon.fig.savefig(out)
    mon.close()
    assert out.exists() and out.stat().st_size > 5000
