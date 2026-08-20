#!/usr/bin/env python3
"""Simulated pilot — the brain of the closed-loop demo.

Watches the mock game's state broadcast (mock_game.py --state-broadcast) and
drives a simulated EEG source (LSL stream "SimulatedEEG") with the matching
brain state: flap-class imagery when the bird should climb, rest otherwise.
play.py (run WITHOUT --simulate, default EEG resolution finds the stream)
then decodes the stream into UDP flaps — the full pipeline with realistic
classifier latency plays the game.

Three terminals: mock_game.py (game) <- state - pilot.py (brain) -> LSL ->
play.py (decoder) -> UDP flaps -> game.
"""

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flappy_brain import simulate
from flappy_brain.model import SCHEMES
from flappy_brain.pilot import (
    FLAP_STATE,
    SINK_STATE,
    PilotController,
    choose_target,
    parse_state,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-port", type=int, default=5006,
                    help="UDP port of the game's state broadcast (default 5006)")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="leftright",
                    help="brain scheme; the flap class is the climb intent (default leftright)")
    ap.add_argument("--deadband", type=float, default=25.0, help="controller deadband in px")
    ap.add_argument("--lead", type=float, default=0.7, help="velocity prediction lead in s")
    ap.add_argument("--mode", choices=["pulse", "bangbang"], default="pulse",
                    help="controller mode (default pulse: one flap-class burst per dip)")
    ap.add_argument("--pulse", type=float, default=0.9, help="pulse mode: climb burst length in s")
    ap.add_argument("--cooldown", type=float, default=0.4, help="pulse mode: rest after a burst in s")
    ap.add_argument("--duration", type=float, default=None, help="stop after N s (default: Ctrl-C)")
    args = ap.parse_args()

    source = simulate.SimulatedEEGSource(scripted=False, scheme=args.scheme)
    stop_sim = threading.Event()
    threading.Thread(
        target=simulate.run_outlet, args=(source,), kwargs={"stop_event": stop_sim}, daemon=True
    ).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.state_port))
    sock.setblocking(False)

    controller = PilotController(deadband=args.deadband, lead=args.lead, mode=args.mode,
                                 pulse_s=args.pulse, cooldown_s=args.cooldown)
    flap_state = FLAP_STATE[args.scheme]
    sink_state = SINK_STATE[args.scheme]
    print(
        f"[pilot] listening for game state on 127.0.0.1:{args.state_port}, "
        f"brain='{simulate.STREAM_NAME}' ({args.scheme}, climb={flap_state}, sink={sink_state})"
    )

    start = time.monotonic()
    last_rx = None
    n_packets = 0
    n_switches = 0
    climb_time = 0.0
    last_tick = start
    try:
        while True:
            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            if source.state == flap_state:
                climb_time += dt

            latest = None
            while True:  # drain; keep newest
                try:
                    data, _ = sock.recvfrom(128)
                except (BlockingIOError, InterruptedError):
                    break
                parsed = parse_state(data)
                if parsed:
                    latest = parsed
            if latest is not None:
                last_rx = now
                n_packets += 1
                _, bird_y, bird_vy, corridor_y, _, alive, gdx, gdy, rdx, rdy = latest
                if alive:
                    desired = choose_target(bird_y, corridor_y, gdx, gdy, rdx, rdy)
                    climb = controller.decide(bird_y, bird_vy, corridor_y,
                                              now=now, desired_y=desired)
                else:
                    climb = False
                new_state = flap_state if climb else sink_state
                if new_state != source.state:
                    source.set_state(new_state)
                    n_switches += 1
            elif last_rx is not None and now - last_rx > 1.0:
                source.set_state("rest")  # game gone quiet -> neutral idle

            if args.duration is not None and now - start >= args.duration:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[pilot] interrupted")
    finally:
        stop_sim.set()
        sock.close()
    total = time.monotonic() - start
    print(
        f"[pilot] done: {n_packets} state packets, {n_switches} intent switches, "
        f"climb intent {climb_time:.1f}s/{total:.1f}s"
    )


if __name__ == "__main__":
    main()
