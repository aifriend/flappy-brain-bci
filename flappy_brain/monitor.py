"""Live monitor dashboard for the online play loop.

Two layers:

- :class:`MonitorState` — pure data history/formatting, no matplotlib import.
  Unit-testable headless.
- :class:`LiveMonitor` — matplotlib figure (band-passed EEG traces + smoothed
  P(MI) trace with threshold line and flap markers), redrawn from the online
  loop each step (~4 Hz). Drive it by passing ``monitor.update`` as the
  ``observer`` of :func:`flappy_brain.online.run_online`.

If the user closes the dashboard window, ``update`` returns ``False`` and the
online loop stops (same as Ctrl-C) — the classifier is not meant to run
unwatched on game day. Under a headless backend (e.g. ``MPLBACKEND=Agg``) no
window is shown and the loop just runs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

NON_GUI_BACKENDS = ("agg", "svg", "pdf", "ps", "cairo")


class MonitorState:
    """Data history behind the dashboard (display-free, unit-testable)."""

    def __init__(
        self,
        fs: float,
        n_channels_show: int = 4,
        eeg_seconds: float = 4.0,
        prob_seconds: float = 15.0,
        step_s: float = 0.25,
        threshold: float = 0.7,
    ):
        self.fs = float(fs)
        self.n_channels_show = n_channels_show
        self.eeg_seconds = eeg_seconds
        self.prob_seconds = prob_seconds
        self.threshold = threshold
        self._step_samples = max(1, int(round(step_s * self.fs)))
        self._eeg: deque[np.ndarray] = deque(maxlen=int(round(eeg_seconds * self.fs)))
        self._prob: deque[tuple[float, float]] = deque()  # (t_rel, p_smooth)
        self.flaps: list[tuple[float, float]] = []  # (t_rel, p_smooth at flap)

    # ---------------------------------------------------------------- update
    def update(self, t_rel: float, window: np.ndarray, p_smooth: float, flapped: bool) -> None:
        """Append one loop iteration.

        ``window`` is the band-passed classification window ``(n_samples, n_channels)``;
        consecutive windows overlap, so only the newest step's worth of samples
        is appended (the whole window on the first call, to pre-fill the trace).
        """
        window = np.asarray(window, dtype=float)
        seg = window if not self._eeg else window[-self._step_samples :]
        for row in seg:
            self._eeg.append(row)
        self._prob.append((t_rel, p_smooth))
        cutoff = t_rel - self.prob_seconds
        while self._prob and self._prob[0][0] < cutoff:
            self._prob.popleft()
        if flapped:
            self.flaps.append((t_rel, p_smooth))
        self.flaps = [(t, p) for t, p in self.flaps if t >= cutoff]

    # ----------------------------------------------------------------- views
    def eeg_view(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """(times, stacked_traces, labels) for the EEG panel.

        Times run from ``-eeg_seconds`` to ~0 (relative to now). Traces are
        scaled by a global robust amplitude and stacked vertically with offsets.
        """
        data = np.asarray(self._eeg, dtype=float)  # (n, n_channels)
        n = len(data)
        times = (np.arange(n) - max(n - 1, 0)) / self.fs
        n_show = min(self.n_channels_show, data.shape[1]) if n else self.n_channels_show
        labels = [f"ch{k}" for k in range(n_show)]
        if n == 0:
            return times, np.zeros((n_show, 0)), labels
        sel = data[:, :n_show]
        scale = float(np.percentile(np.abs(sel), 99)) or 1.0
        stacked = sel.T / scale + 2.5 * np.arange(n_show)[:, None]
        return times, stacked, labels

    def prob_view(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, smoothed probabilities) for the P(MI) panel."""
        if not self._prob:
            return np.zeros(0), np.zeros(0)
        t, p = zip(*self._prob)
        return np.asarray(t), np.asarray(p)

    def flap_view(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, probabilities) of flap markers inside the visible span."""
        if not self.flaps:
            return np.zeros(0), np.zeros(0)
        t, p = zip(*self.flaps)
        return np.asarray(t), np.asarray(p)

    def title_text(self) -> str:
        return f"Flappy Brain monitor — {len(self.flaps)} flap(s)"


class LiveMonitor:
    """Matplotlib dashboard driven from the online loop's observer hook."""

    def __init__(
        self,
        fs: float,
        n_channels_show: int = 4,
        threshold: float = 0.7,
        step_s: float = 0.25,
        eeg_seconds: float = 4.0,
        prob_seconds: float = 15.0,
    ):
        import matplotlib
        import matplotlib.pyplot as plt

        self._plt = plt
        self.state = MonitorState(
            fs=fs,
            n_channels_show=n_channels_show,
            eeg_seconds=eeg_seconds,
            prob_seconds=prob_seconds,
            step_s=step_s,
            threshold=threshold,
        )
        self.closed = False
        self.fig, (self.ax_eeg, self.ax_prob) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            gridspec_kw={"height_ratios": [3, 2]},
        )
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._has_gui = matplotlib.get_backend().lower() not in NON_GUI_BACKENDS
        if self._has_gui:
            plt.ion()
            self.fig.show()

    # ------------------------------------------------------------- protocol
    def update(self, t_rel: float, window: np.ndarray, p_smooth: float, flapped: bool) -> bool:
        """Observer hook for ``run_online``. Returns False when the window was
        closed by the user (the loop should stop)."""
        self.state.update(t_rel, window, p_smooth, flapped)
        self._redraw(t_rel)
        if self._has_gui:
            self._plt.pause(0.001)
        return not self.closed

    def close(self) -> None:
        self._plt.close(self.fig)

    # ------------------------------------------------------------- internal
    def _on_close(self, _event) -> None:
        self.closed = True

    def _redraw(self, t_rel: float) -> None:
        st = self.state

        ax = self.ax_eeg
        ax.clear()
        times, stacked, labels = st.eeg_view()
        if stacked.shape[1]:
            ax.plot(times, stacked.T, lw=0.6)
        ax.set_yticks(2.5 * np.arange(len(labels)), labels)
        ax.set_xlim(-st.eeg_seconds, 0)
        ax.set_ylim(-1.5, 2.5 * max(len(labels) - 1, 0) + 1.5)
        ax.set_ylabel("band-passed EEG")
        ax.set_xlabel("s (relative to now)")

        ax = self.ax_prob
        ax.clear()
        t, p = st.prob_view()
        ax.plot(t, p, lw=1.2, color="tab:blue", label="smoothed P(MI)")
        ax.axhline(st.threshold, color="tab:red", ls="--", lw=1.0, label=f"threshold {st.threshold:g}")
        tf, pf = st.flap_view()
        if len(tf):
            ax.scatter(tf, pf, marker="v", s=60, color="tab:red", zorder=5, label="flap")
        ax.set_xlim(max(0.0, t_rel - st.prob_seconds), max(st.prob_seconds, t_rel + 0.5))
        ax.set_ylim(0, 1)
        ax.set_xlabel("s (since start)")
        ax.set_ylabel("P(MI)")
        ax.set_title(st.title_text())
        ax.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
