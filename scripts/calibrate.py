#!/usr/bin/env python3
"""Record a cued calibration session over LSL -> data/calib_<timestamp>.npz

Schemes: ``leftright`` cues LEFT / RIGHT motor imagery (RIGHT = flap class);
``feetrest`` cues REST / both-feet imagery (FEET = flap class).
With --simulate, a fake EEG source is started in-process and its ERD state is
driven by the cues, for headset-free testing.
"""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flappy_brain import simulate
from flappy_brain.calibration import run_calibration
from flappy_brain.lsl_input import EEGBuffer, resolve_eeg_stream
from flappy_brain.model import SCHEMES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", choices=list(SCHEMES), default="leftright",
                    help="class scheme to record (default leftright)")
    ap.add_argument("--trials", type=int, default=20, help="trials per class (default 20)")
    ap.add_argument("--cue", type=float, default=4.0, help="cue duration in s (default 4)")
    ap.add_argument("--gap", type=float, default=2.0, help="gap between trials in s (default 2)")
    ap.add_argument("--tmin", type=float, default=0.5, help="epoch start, s post-cue (default 0.5)")
    ap.add_argument("--tmax", type=float, default=2.5, help="epoch end, s post-cue (default 2.5)")
    ap.add_argument("--out-dir", default="data", help="output directory (default data/)")
    ap.add_argument("--stream-name", default=None, help="LSL stream name (default: any EEG stream)")
    ap.add_argument("--any-host", action="store_true", help="accept streams from any host")
    ap.add_argument("--simulate", action="store_true", help="use the built-in simulated EEG source")
    args = ap.parse_args()

    stop_sim = None
    on_cue = None
    stream_name = args.stream_name
    if args.simulate:
        source = simulate.SimulatedEEGSource(scripted=False, scheme=args.scheme)
        stop_sim = threading.Event()
        threading.Thread(
            target=simulate.run_outlet, args=(source,), kwargs={"stop_event": stop_sim}, daemon=True
        ).start()
        # cue -> ERD state for that class, rest during gaps
        if args.scheme == "feetrest":
            on_cue = lambda label, active: source.set_state(
                "feet" if (active and label == 1) else "rest"
            )
        else:
            on_cue = lambda label, active: source.set_state(
                ("left" if label == 0 else "right") if active else "rest"
            )
        stream_name = stream_name or simulate.STREAM_NAME

    stream = resolve_eeg_stream(name=stream_name, any_host=args.any_host)
    with EEGBuffer(stream) as buffer:
        try:
            run_calibration(
                buffer,
                n_trials_per_class=args.trials,
                cue_duration=args.cue,
                gap_duration=args.gap,
                tmin=args.tmin,
                tmax=args.tmax,
                out_dir=args.out_dir,
                on_cue=on_cue,
                scheme=args.scheme,
            )
        finally:
            if stop_sim:
                stop_sim.set()


if __name__ == "__main__":
    main()
