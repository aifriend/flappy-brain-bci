#!/usr/bin/env python3
"""Play: live LSL EEG -> BandCSP+LDA -> UDP flap commands for the game.

Flaps when smoothed P(RIGHT motor imagery) crosses --threshold. Default
output is ASCII "1" to 127.0.0.1:5005; point --host at the game PC (or run
scripts/mock_game.py locally). With --simulate, a scripted fake EEG outlet
(rest/LEFT/rest/RIGHT cycle) is started in-process for headset-free testing.
Use --arrays to run from the exported .npz (no sklearn needed on this machine).
"""

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flappy_brain import simulate
from flappy_brain.lsl_input import EEGBuffer, resolve_eeg_stream
from flappy_brain.model import SCHEMES, load_model, warn_scheme_mismatch
from flappy_brain.online import SessionLogger, run_online
from flappy_brain.output_udp import UDPSender


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="model.pkl", help="trained model pickle (default model.pkl)")
    ap.add_argument("--arrays", default=None,
                    help="exported arrays .npz — numpy-only inference instead of the pickle")
    ap.add_argument("--filterbank", action="store_true",
                    help="expect a filter-bank model (consistency check; bands are stored in the model)")
    ap.add_argument("--host", default="127.0.0.1", help="game host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5005, help="game UDP port (default 5005)")
    ap.add_argument("--window", type=float, default=1.0, help="sliding window in s (default 1.0)")
    ap.add_argument("--step", type=float, default=0.25, help="window step in s (default 0.25)")
    ap.add_argument("--threshold", type=float, default=0.7, help="flap threshold on smoothed P(RIGHT)")
    ap.add_argument("--refractory", type=float, default=1.0, help="min s between flaps (default 1.0)")
    ap.add_argument("--smoothing", type=float, default=0.3, help="weight of newest P(RIGHT), 0..1")
    ap.add_argument("--duration", type=float, default=None, help="stop after N s (default: run until Ctrl-C)")
    ap.add_argument("--log", default=None, help="session log CSV (default data/session_<timestamp>.csv)")
    ap.add_argument("--no-log", action="store_true", help="disable session logging")
    ap.add_argument("--stream-name", default=None, help="LSL stream name (default: any EEG stream)")
    ap.add_argument("--any-host", action="store_true", help="accept streams from any host")
    ap.add_argument("--simulate", action="store_true", help="use the built-in simulated EEG source")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="leftright",
                    help="scripted simulator cycle with --simulate (default leftright); "
                         "warns if the model was trained for another scheme")
    ap.add_argument("--sim-rest", type=float, default=4.0,
                    help="scripted cycle: rest phase in s (default 4)")
    ap.add_argument("--sim-left", type=float, default=4.0,
                    help="scripted cycle (leftright): LEFT phase in s (default 4)")
    ap.add_argument("--sim-right", type=float, default=4.0,
                    help="scripted cycle (leftright): RIGHT phase in s (default 4)")
    ap.add_argument("--sim-feet", type=float, default=4.0,
                    help="scripted cycle (feetrest): FEET phase in s (default 4)")
    ap.add_argument("--monitor", action="store_true", help="show live dashboard (EEG + P(RIGHT) + flaps)")
    ap.add_argument("--monitor-channels", type=int, default=4, help="EEG channels shown in the monitor (default 4)")
    args = ap.parse_args()

    stop_sim = None
    stream_name = args.stream_name
    if args.simulate:
        source = simulate.SimulatedEEGSource(
            scripted=True,
            scheme=args.scheme,
            rest_duration=args.sim_rest,
            imagery_duration=args.sim_feet,
            left_duration=args.sim_left,
            right_duration=args.sim_right,
        )
        stop_sim = threading.Event()
        threading.Thread(
            target=simulate.run_outlet, args=(source,), kwargs={"stop_event": stop_sim}, daemon=True
        ).start()
        stream_name = stream_name or simulate.STREAM_NAME
        if args.scheme == "feetrest":
            cycle = f"{source.rest_duration}s rest / {source.imagery_duration}s FEET"
        else:
            cycle = (
                f"{source.rest_duration}s rest / {source.left_duration}s LEFT / "
                f"{source.rest_duration}s rest / {source.right_duration}s RIGHT"
            )
        print(f"[play] simulator cycle: {cycle} (flaps expected only during the imagery phase)")

    if args.arrays:
        from flappy_brain.inference import ArraysModel

        model = ArraysModel.load(args.arrays)
        bands = model.bands
        model_desc = args.arrays
        model_scheme = model.scheme
    else:
        model = load_model(args.model)
        bands = model.named_steps["csp"].bands
        model_desc = args.model
        model_scheme = getattr(model, "scheme_", None)
    bands = tuple(tuple(float(v) for v in b) for b in bands)  # clean repr for logs
    if args.arrays:
        print(f"[play] numpy-only inference from {args.arrays} (bands={bands}, scheme={model_scheme})")
    warning = warn_scheme_mismatch(model_scheme, args.scheme)
    if warning:
        print(f"[play] WARNING: {warning}")
    if args.filterbank and len(bands) < 3:
        print(f"[play] WARNING: --filterbank set but model has bands={bands}")
    if not args.filterbank and len(bands) > 1:
        print(f"[play] note: model is filter-bank (bands={bands}); --filterbank not required")

    stream = resolve_eeg_stream(name=stream_name, any_host=args.any_host)
    with EEGBuffer(stream) as buffer, UDPSender(args.host, args.port) as sender:
        print(f"[play] sending flaps to {args.host}:{args.port}")
        monitor = None
        if args.monitor:
            from flappy_brain.monitor import LiveMonitor

            monitor = LiveMonitor(
                fs=buffer.fs,
                n_channels_show=min(args.monitor_channels, buffer.n_channels),
                threshold=args.threshold,
                step_s=args.step,
            )
        logger = None
        if not args.no_log:
            log_path = args.log or f"data/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            logger = SessionLogger(
                log_path,
                header={
                    "model": model_desc,
                    "bands": bands,
                    "threshold": args.threshold,
                    "refractory": args.refractory,
                    "smoothing": args.smoothing,
                    "window": args.window,
                    "step": args.step,
                },
            )
            print(f"[play] logging session -> {log_path}")
        try:
            run_online(
                buffer,
                model,
                sender,
                window_s=args.window,
                step_s=args.step,
                threshold=args.threshold,
                refractory_s=args.refractory,
                smoothing=args.smoothing,
                duration=args.duration,
                observer=monitor.update if monitor else None,
                logger=logger,
            )
        finally:
            if logger:
                logger.close()
            if monitor:
                monitor.close()
            if stop_sim:
                stop_sim.set()


if __name__ == "__main__":
    main()
