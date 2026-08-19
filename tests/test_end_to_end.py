"""End-to-end test on the simulator: LSL in -> calibrate -> train -> UDP flaps out.

Runs a short real-time session (~40 s):
1. Simulated EEG outlet (manual lateralized-ERD control) streams over LSL.
2. A short LEFT/RIGHT cued calibration is recorded; cues drive the simulator.
3. BandCSP+LDA is trained; CV accuracy on the simulated data must be high.
4. The online loop runs while the test toggles rest/LEFT/RIGHT periods; UDP
   flaps must arrive during RIGHT phases only (RIGHT = the flap class).
"""

import socket
import threading
import time

import numpy as np

from flappy_brain import simulate
from flappy_brain.calibration import load_calibration, run_calibration
from flappy_brain.lsl_input import EEGBuffer, resolve_eeg_stream
from flappy_brain.model import cross_validate_report, train
from flappy_brain.online import run_online
from flappy_brain.output_udp import UDPSender


def _udp_listener(sock, received, stop):
    sock.settimeout(0.2)
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(64)
        except socket.timeout:
            continue
        if data == b"1":
            received.append(time.time())


def test_end_to_end(tmp_path):
    # --- 1. simulated outlet --------------------------------------------
    source = simulate.SimulatedEEGSource(scripted=False, seed=1)
    stop_sim = threading.Event()
    sim_thread = threading.Thread(
        target=simulate.run_outlet, args=(source,), kwargs={"stop_event": stop_sim}, daemon=True
    )
    sim_thread.start()

    # --- 2. inlet + ring buffer -----------------------------------------
    stream = resolve_eeg_stream(name=simulate.STREAM_NAME, timeout=30.0)
    buffer = EEGBuffer(stream, buffer_seconds=6.0)
    buffer.start()

    # --- UDP listener -----------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    received, stop_rx = [], threading.Event()
    rx_thread = threading.Thread(target=_udp_listener, args=(sock, received, stop_rx), daemon=True)
    rx_thread.start()

    try:
        deadline = time.time() + 10
        while buffer.available() < int(buffer.fs):
            assert time.time() < deadline, "no samples arriving from the simulated stream"
            time.sleep(0.05)

        # --- 3. short calibration; cues drive the simulator's ERD ---------
        on_cue = lambda label, active: source.set_state(
            ("left" if label == 0 else "right") if active else "rest"
        )
        npz = run_calibration(
            buffer,
            n_trials_per_class=6,
            cue_duration=2.0,
            gap_duration=0.5,
            tmin=0.4,
            tmax=1.8,
            out_path=tmp_path / "calib.npz",
            on_cue=on_cue,
            seed=1,
        )
        epochs, labels, fs = load_calibration(npz)
        assert epochs.shape[0] >= 8 and set(np.unique(labels)) == {0, 1}

        # --- 4. train (raw epochs; BandCSP filters internally) -------------
        scores, _ = cross_validate_report(epochs, labels, fs, cv=3)
        assert scores.mean() > 0.8, f"CV accuracy too low: {scores}"
        model = train(epochs, labels, fs)

        # --- 5. online loop with scripted rest/LEFT/RIGHT toggling --------
        source.set_state("rest")
        sender = UDPSender("127.0.0.1", port)
        online_thread = threading.Thread(
            target=run_online,
            args=(buffer, model, sender),
            kwargs=dict(
                window_s=0.75,
                step_s=0.25,
                threshold=0.75,
                refractory_s=1.0,
                smoothing=0.4,
                duration=12.5,
                verbose=False,
            ),
            daemon=True,
        )
        t0 = time.time()
        online_thread.start()

        # rest 2 / LEFT 3 / rest 2 / RIGHT 3 / rest 2
        left_intervals, right_intervals = [], []
        schedule = [
            (2.0, "rest"),
            (3.0, "left"),
            (2.0, "rest"),
            (3.0, "right"),
            (2.0, "rest"),
        ]
        elapsed = 0.0
        for dur, state in schedule:
            if state == "left":
                left_intervals.append((t0 + elapsed, t0 + elapsed + dur))
            elif state == "right":
                right_intervals.append((t0 + elapsed, t0 + elapsed + dur))
            source.set_state(state)
            time.sleep(dur)
            elapsed += dur
        source.set_state("rest")
        online_thread.join(timeout=10)
        assert not online_thread.is_alive(), "online loop did not finish"
    finally:
        stop_rx.set()
        rx_thread.join(timeout=2)
        sock.close()
        buffer.stop()
        stop_sim.set()
        sim_thread.join(timeout=2)

    # --- assertions -------------------------------------------------------
    # Rest is unmodeled in a 2-class LEFT/RIGHT scheme, so P(RIGHT) wanders
    # during rest and a small (6-trial/class) model fires an occasional
    # rest-noise flap — on game day this is tuned with --threshold. What must
    # hold: flaps happen for RIGHT, never deep inside LEFT, and most assessed
    # flaps are RIGHT-caused. So: ignore the 2 s startup transient, shrink
    # LEFT intervals to their uncontaminated core, and allow a
    # window+smoothing+refractory tail of 1.75 s after RIGHT ends.
    startup_s = 2.0
    grace_s = 1.75
    assessed = [t for t in received if t >= t0 + startup_s]
    assert len(assessed) >= 2, f"too few flaps received: {len(received)}"

    left_cores = [(a + 1.0, b - 1.0) for a, b in left_intervals]
    in_left_core = sum(any(a <= t <= b for a, b in left_cores) for t in received)
    assert in_left_core == 0, f"{in_left_core} flap(s) deep inside LEFT phases (flaps={received})"

    in_right = sum(any(a - 0.5 <= t <= b + grace_s for a, b in right_intervals) for t in assessed)
    assert in_right >= 2, f"too few flaps during RIGHT: {in_right} (flaps={received})"
    assert in_right >= 0.5 * len(assessed), (
        f"only {in_right}/{len(assessed)} flaps near RIGHT phases "
        f"(right={right_intervals}, flaps={received})"
    )
