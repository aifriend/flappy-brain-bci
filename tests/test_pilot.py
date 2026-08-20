"""Pilot controller unit tests (screen coords: y grows downward)."""

from flappy_brain.pilot import FLAP_STATE, PilotController, parse_state


def test_parse_state_v1_packet_gets_sentinels():
    assert parse_state(b"1.25,320.5,45.0,300.0,3,1") == (
        1.25, 320.5, 45.0, 300.0, 3, 1, 9999.0, 9999.0, 9999.0, 9999.0)


def test_parse_state_v2_packet():
    assert parse_state(b"1.25,320.5,45.0,300.0,3,1,120,-40,9999,9999") == (
        1.25, 320.5, 45.0, 300.0, 3, 1, 120.0, -40.0, 9999.0, 9999.0)


def test_parse_state_malformed():
    assert parse_state(b"") is None
    assert parse_state(b"1.0,2.0") is None
    assert parse_state(b"a,b,c,d,e,f") is None
    assert parse_state(b"1.0,2.0,3.0,4.0,5,1,extra") is None


def test_low_bird_wants_climb():
    c = PilotController(deadband=15.0, lead=0.0)
    assert c.decide(bird_y=350.0, bird_vy=0.0, corridor_y=300.0) is True  # below center


def test_high_bird_wants_rest():
    c = PilotController(deadband=15.0, lead=0.0)
    assert c.decide(bird_y=250.0, bird_vy=0.0, corridor_y=300.0) is False  # above center


def test_hysteresis_holds_intent_inside_deadband():
    c = PilotController(deadband=15.0, lead=0.0)
    assert c.decide(350.0, 0.0, 300.0) is True  # climb
    assert c.decide(310.0, 0.0, 300.0) is True  # inside band: still climb
    assert c.decide(290.0, 0.0, 300.0) is True  # inside band: still climb
    assert c.decide(270.0, 0.0, 300.0) is False  # above band: rest
    assert c.decide(295.0, 0.0, 300.0) is False  # inside band: still rest


def test_velocity_lead_anticipates_fall():
    c = PilotController(deadband=15.0, lead=0.5)
    # slightly above center but sinking fast: predicted position is well below
    assert c.decide(bird_y=290.0, bird_vy=200.0, corridor_y=300.0) is True


def test_velocity_lead_anticipates_rise():
    c = PilotController(deadband=15.0, lead=0.5)
    c.want_climb = True
    # slightly below center but climbing fast: predicted position well above
    assert c.decide(bird_y=310.0, bird_vy=-200.0, corridor_y=300.0) is False


def test_flap_state_mapping():
    assert FLAP_STATE["leftright"] == "right"
    assert FLAP_STATE["feetrest"] == "feet"


def test_pulse_mode_fires_one_burst_then_cools_down():
    c = PilotController(deadband=15.0, lead=0.0, mode="pulse", pulse_s=1.0, cooldown_s=0.5)
    # bird below center at t=0 -> climb pulse for 1.0 s
    assert c.decide(350.0, 0.0, 300.0, now=0.0) is True
    assert c.decide(350.0, 0.0, 300.0, now=0.5) is True
    assert c.decide(350.0, 0.0, 300.0, now=1.2) is False  # pulse over -> cooldown
    assert c.decide(350.0, 0.0, 300.0, now=1.5) is False  # still cooling down
    assert c.decide(350.0, 0.0, 300.0, now=1.8) is True   # cooldown over, still low -> re-pulse


def test_pulse_mode_stays_idle_when_high():
    c = PilotController(deadband=15.0, lead=0.0, mode="pulse")
    assert c.decide(250.0, 0.0, 300.0, now=0.0) is False
    assert c.decide(250.0, 0.0, 300.0, now=10.0) is False


def test_sink_state_mapping():
    from flappy_brain.pilot import SINK_STATE

    # leftright sinks with LEFT imagery (opposite trained class -> P(flap) ~ 0)
    assert SINK_STATE["leftright"] == "left"
    assert SINK_STATE["feetrest"] == "rest"


# --- neuron-aware target tracking ------------------------------------------

from flappy_brain.pilot import NEURON_SENTINEL, choose_target

S = NEURON_SENTINEL


def test_target_defaults_to_corridor_center():
    assert choose_target(300.0, 310.0) == 310.0
    assert choose_target(300.0, 310.0, S, S, S, S) == 310.0


def test_target_aims_at_green_neuron():
    # green 40 px above the bird, inside the corridor -> aim there
    assert choose_target(300.0, 310.0, green_dx=200, green_dy=-40) == 260.0


def test_target_ignores_unreachable_green():
    # green outside the corridor (would mean hitting terrain) -> corridor center
    assert choose_target(300.0, 310.0, green_dx=200, green_dy=-200) == 310.0


def test_target_avoids_red_on_path():
    # red (at 250) sits just above the green target (260) -> bias target below red
    t = choose_target(300.0, 310.0, green_dx=200, green_dy=-40, red_dx=100, red_dy=-50)
    assert t == 340.0  # 260 + 80, inside corridor
    # red far below the path -> no bias
    assert choose_target(300.0, 310.0, green_dx=200, green_dy=-40,
                         red_dx=100, red_dy=100) == 260.0
    # red beyond lookahead -> ignored
    assert choose_target(300.0, 310.0, green_dx=200, green_dy=-40,
                         red_dx=400, red_dy=-45) == 260.0
    # bias clamped inside the chase corridor (center +/- 100)
    t = choose_target(300.0, 310.0, green_dx=S, green_dy=S, red_dx=80, red_dy=0)
    assert 310.0 - 100 <= t <= 310.0 + 100
