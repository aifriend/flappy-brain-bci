"""Mock game tests — headless (SDL dummy), real UDP on 127.0.0.1:5005."""

import os
import socket
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import mock_game  # noqa: E402
from mock_game import BIRD_X, Game, Neuron  # noqa: E402

DT = 1.0 / 60


def test_udp_flap_packet_causes_flap():
    game = Game()  # binds 127.0.0.1:5005
    try:
        game.y, game.vy = 300.0, 0.0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(b"1", ("127.0.0.1", 5005))
        sock.sendto(b"garbage", ("127.0.0.1", 5005))  # ignored
        sock.close()
        import time

        time.sleep(0.05)  # let the packet land
        assert game.poll_udp() == 1
        assert game.flap_packets == 1
        game.step(DT)
        assert game.vy < -450  # flap impulse applied (flap_force - one gravity tick)
    finally:
        game.close()


def test_green_neuron_scores_point():
    game = Game(seed=1)
    try:
        game.neurons.append(Neuron(BIRD_X + 30, game.y, "green"))
        for _ in range(30):
            game.step(DT)
        assert game.score == 1
        assert not game.neurons  # collected, removed
    finally:
        game.close()


def test_red_neuron_penalizes():
    game = Game(seed=1)
    try:
        game.neurons.append(Neuron(BIRD_X + 30, game.y, "red"))
        for _ in range(12):  # few frames: collect, but no time to fall into terrain
            game.step(DT)
        assert game.score == -1
        assert not game.game_over  # penalty, not death
    finally:
        game.close()


def test_terrain_collision_game_over_then_restart():
    game = Game(seed=1)
    try:
        _, bottom = game.corridor(game.bird_world_x)
        game.y = bottom + 50  # inside the terrain
        game.step(DT)
        assert game.game_over
        game.score = 5
        for _ in range(int(1.5 / DT)):  # auto-restart after ~1.2 s
            game.step(DT)
        assert not game.game_over
        assert game.score == 0 and game.elapsed < 1.0
    finally:
        game.close()


def test_autopilot_survives_and_renders():
    import pygame

    game = Game(auto=True, seed=42)
    try:
        flaps = 0
        last_vy = game.vy
        for _ in range(int(8 / DT)):
            game.step(DT)
            if game.vy < last_vy - 100:
                flaps += 1
            last_vy = game.vy
        assert game.elapsed > 7.0  # never long-dead (elapsed resets on restart)
        assert flaps > 10
        # render one frame offscreen — must not crash
        surface = pygame.Surface((mock_game.WIDTH, mock_game.HEIGHT))
        game.draw(surface)
    finally:
        game.close()
