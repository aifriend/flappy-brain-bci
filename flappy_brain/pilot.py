"""Closed-loop simulated pilot: controller logic for the mock game.

The pilot watches the mock game's state broadcast (see ``--state-broadcast``)
and decides whether the simulated brain should produce flap-class imagery
(bird too low -> climb) or rest (bird high enough -> sink). The real decoder
(play.py) then turns that brain state into UDP flaps with realistic latency.

Screen coordinates: **y grows downward**, so ``bird_y > corridor_y`` means the
bird is visually BELOW the corridor center and should climb.
"""

from __future__ import annotations

# Flap-class / sink brain state names per scheme. For leftright the sink
# intent uses LEFT imagery — the *opposite trained class* — which holds
# P(flap) near 0 far more reliably than unmodeled rest, whose probability
# wanders and causes false flaps.
FLAP_STATE = {"leftright": "right", "feetrest": "feet"}
SINK_STATE = {"leftright": "left", "feetrest": "rest"}

NEURON_SENTINEL = 9999.0  # dx/dy when no neuron of that kind is on screen


def parse_state(packet: bytes):
    """Parse a state broadcast packet.

    v2: ``t,bird_y,bird_vy,corridor_center_y,score,alive,green_dx,green_dy,red_dx,red_dy``
    v1 (6 fields, no neurons) is accepted too — neuron fields become the
    sentinel. Returns a 10-tuple, or None for malformed packets.
    """
    try:
        parts = packet.decode().strip().split(",")
        if len(parts) == 6:
            parts += ["9999"] * 4
        if len(parts) != 10:
            return None
        t, bird_y, bird_vy, corridor_y, score, alive = parts[:6]
        gdx, gdy, rdx, rdy = (float(v) for v in parts[6:])
        return (float(t), float(bird_y), float(bird_vy), float(corridor_y),
                int(score), int(alive), gdx, gdy, rdx, rdy)
    except (ValueError, UnicodeDecodeError):
        return None


def choose_target(
    bird_y: float,
    corridor_y: float,
    green_dx: float = NEURON_SENTINEL,
    green_dy: float = NEURON_SENTINEL,
    red_dx: float = NEURON_SENTINEL,
    red_dy: float = NEURON_SENTINEL,
    corridor_half: float = 150.0,
    chase_band: float = 90.0,
    red_lookahead: float = 150.0,
    red_path_margin: float = 60.0,
) -> float:
    """Desired altitude for the controller (screen coords: y down).

    - Green neuron ahead and within ``chase_band`` of the corridor center ->
      aim at it. (Off-center greens are deliberately skipped: with ~1 s of
      classifier latency, big altitude excursions end in terrain.)
    - Otherwise -> corridor center.
    - If a red neuron is close ahead (dx < red_lookahead) and within
      red_path_margin of the intended height, bias the target ~80 px to the
      other side of it. Deliberately simple: vertical dodging only.
    - Final target clamped to corridor center +/- (corridor_half - 50).
    """
    target = corridor_y
    if green_dx < NEURON_SENTINEL:
        green_y = bird_y + green_dy
        if abs(green_y - corridor_y) <= chase_band:
            target = green_y
    if red_dx < NEURON_SENTINEL and 0 < red_dx < red_lookahead:
        red_y = bird_y + red_dy
        if abs(red_y - target) < red_path_margin:
            away = target - 80.0 if red_y > target else target + 80.0
            target = away
    lo = corridor_y - corridor_half + 50.0
    hi = corridor_y + corridor_half - 50.0
    return min(max(target, lo), hi)


class PilotController:
    """Altitude controller with deadband + velocity lead, two modes.

    Screen coordinates: y grows downward; ``bird_y > corridor_y`` = too low.

    ``bang-bang`` mode: climb intent while predicted altitude
    (``y + vy * lead``) is below the corridor center by more than
    ``deadband``, rest while above, previous intent inside the deadband
    (hysteresis). Simple, but the classifier's flap *train* keeps firing the
    whole time the intent is climb.

    ``pulse`` mode (default for play): when predicted-low, emit ONE climb
    pulse of ``pulse_s`` seconds — just long enough for the classifier to
    fire ~1 flap — then a ``cooldown_s`` rest to observe the effect. This
    decouples flap count from burst length and matches how the decoder
    behaves (latency ~1 s, one flap per sustained burst).
    """

    def __init__(self, deadband: float = 15.0, lead: float = 0.4, mode: str = "bangbang",
                 pulse_s: float = 1.0, cooldown_s: float = 0.8):
        if mode not in ("bangbang", "pulse"):
            raise ValueError(f"mode must be 'bangbang' or 'pulse', got {mode!r}")
        self.deadband = float(deadband)
        self.lead = float(lead)
        self.mode = mode
        self.pulse_s = float(pulse_s)
        self.cooldown_s = float(cooldown_s)
        self.want_climb = False
        self._pulse_until: float | None = None
        self._cooldown_until = 0.0

    def decide(self, bird_y: float, bird_vy: float, corridor_y: float,
               now: float = 0.0, desired_y: float | None = None) -> bool:
        """True = climb (flap-class imagery), False = rest.

        ``desired_y`` is the altitude to track (e.g. from
        :func:`choose_target`); defaults to the corridor center.
        """
        target = corridor_y if desired_y is None else desired_y
        err = (bird_y + bird_vy * self.lead) - target  # >0 = too low
        if self.mode == "bangbang":
            if err > self.deadband:
                self.want_climb = True
            elif err < -self.deadband:
                self.want_climb = False
            return self.want_climb
        # pulse mode (uses `now`, seconds, monotonically increasing)
        if self._pulse_until is not None:
            if now < self._pulse_until:
                return True
            self._pulse_until = None
            self._cooldown_until = now + self.cooldown_s
            return False
        if now < self._cooldown_until:
            return False
        if err > self.deadband:
            self._pulse_until = now + self.pulse_s
            return True
        return False
