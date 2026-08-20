#!/usr/bin/env python3
"""Mock Flappy Brain game (pygame) — lets the real classifier fly a bird on macOS.

Listens for UDP flap packets (ASCII "1") on 127.0.0.1:5005 — exactly what
``scripts/play.py`` sends. Start this game first (or alongside), then run
``play.py``; each detected motor-imagery burst flaps the bird.

Controls: UDP "1" = flap (the BCI path), SPACE = manual flap, R = restart,
ESC/close = quit. With --auto a built-in autopilot keeps the bird airborne
(demos/tests without the EEG pipeline).

Headless: SDL_VIDEODRIVER=dummy runs without a window; --frames-dir saves a
PNG every 30 frames for visual checks; --duration exits automatically.

Game logic (class Game) is importable and steppable without opening a window.
"""

import argparse
import math
import os
import random
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- game settings (mirrors the Unity game's settings) ----------------------
WIDTH, HEIGHT = 800, 600
FPS = 60
BIRD_X = 160
BIRD_R = 14
SCROLL_SPEED = 180.0          # px/s
GRAVITY = 1800.0              # px/s^2
FLAP_FORCE = 520.0            # impulse velocity per flap, px/s
SPAWN_INTERVAL = 1.6          # s between neuron spawns
SPAWN_JITTER = 0.4            # ± random jitter on the interval
TIME_LIMIT = None             # s, None = endless
CORRIDOR_HALF = 150.0         # half-height of the terrain gap

SKY_TOP = (168, 213, 226)
SKY_BOTTOM = (230, 244, 241)
TERRAIN = (124, 179, 124)
TERRAIN_EDGE = (86, 140, 90)
BIRD = (250, 214, 90)
BIRD_WING = (240, 170, 60)
GREEN = (70, 190, 110)
RED = (215, 80, 80)
INK = (45, 62, 72)


def corridor_center(world_x: float, scale: float = 1.0) -> float:
    """Middle of the flyable gap at a world x position (fixed terrain shape).

    ``scale`` flattens the hills (< 1.0) — a demo assist for open-loop
    (scripted-classifier) flight, where no one closes the altitude loop.
    """
    return HEIGHT / 2 + scale * (80 * math.sin(world_x * 0.003) + 40 * math.sin(world_x * 0.007 + 1.0))


class Neuron:
    R = 12

    def __init__(self, x: float, y: float, kind: str):
        self.x, self.y, self.kind = x, y, kind  # kind: "green" | "red"


class Game:
    """Headless-friendly game state. Call step(dt) yourself; draw() is optional."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5005,
        auto: bool = False,
        seed: int | None = None,
        flap_force: float = FLAP_FORCE,
        gravity: float = GRAVITY,
        spawn_interval: float = SPAWN_INTERVAL,
        spawn_jitter: float = SPAWN_JITTER,
        time_limit: float | None = TIME_LIMIT,
        terrain_scale: float = 1.0,
    ):
        self.auto = auto
        self.flap_force = flap_force
        self.gravity = gravity
        self.spawn_interval = spawn_interval
        self.spawn_jitter = spawn_jitter
        self.time_limit = time_limit
        self.terrain_scale = terrain_scale
        self.rng = random.Random(seed)

        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.bind((host, port))
        self._udp.setblocking(False)

        self.flap_packets = 0  # total UDP flaps received (all runs)
        self.total_deaths = 0
        self.total_score = 0  # cumulative across auto-restarts
        self.best_stretch = 0.0  # longest single life, seconds
        self._queued_flaps = 0
        self._font = None
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.y = corridor_center(BIRD_X, self.terrain_scale)  # spawn inside the gap
        self.vy = 0.0
        self.world_x = 0.0
        self.elapsed = 0.0
        self.score = 0
        self.neurons: list[Neuron] = []
        self.game_over = False
        self.finished = False  # time limit reached
        self._dead_timer = 0.0
        self._next_spawn = self.spawn_interval

    def close(self) -> None:
        self._udp.close()

    @property
    def bird_world_x(self) -> float:
        return self.world_x + BIRD_X

    def corridor(self, world_x: float) -> tuple[float, float]:
        c = corridor_center(world_x, self.terrain_scale)
        return c - CORRIDOR_HALF, c + CORRIDOR_HALF  # (top, bottom)

    # ---------------------------------------------------------------- input
    def poll_udp(self) -> int:
        """Drain the socket; each "1" packet queues one flap. Returns count."""
        n = 0
        while True:
            try:
                data, _ = self._udp.recvfrom(64)
            except (BlockingIOError, InterruptedError):
                break
            if data.strip() == b"1":
                self._queued_flaps += 1
                self.flap_packets += 1
                n += 1
        return n

    def flap(self) -> None:
        if not self.game_over and not self.finished:
            self.vy = -self.flap_force

    def _autopilot(self) -> None:
        """Flap when below (and sinking past) the corridor center ahead."""
        _, bottom = self.corridor(self.bird_world_x + 120)
        target = bottom - CORRIDOR_HALF  # corridor center
        if self.y > target + 8 and self.vy > -60:
            self._queued_flaps += 1

    # ---------------------------------------------------------------- logic
    def step(self, dt: float) -> None:
        if self.finished:
            return
        if self.game_over:
            self._dead_timer += dt
            if self._dead_timer > 1.2:  # auto-restart
                self.reset()
            return

        if self.auto:
            self._autopilot()
        while self._queued_flaps:
            self._queued_flaps -= 1
            self.flap()

        self.elapsed += dt
        self.world_x += SCROLL_SPEED * dt
        self.vy += self.gravity * dt
        self.y += self.vy * dt

        # neuron spawning (seeded when seed= was given)
        self._next_spawn -= dt
        if self._next_spawn <= 0:
            self._next_spawn = self.spawn_interval + self.rng.uniform(
                -self.spawn_jitter, self.spawn_jitter
            )
            top, bottom = self.corridor(self.bird_world_x + WIDTH)
            ny = self.rng.uniform(top + 40, bottom - 40)
            kind = "green" if self.rng.random() < 0.7 else "red"
            self.neurons.append(Neuron(WIDTH + 20, ny, kind))

        # scroll + collect
        for neuron in list(self.neurons):
            neuron.x -= SCROLL_SPEED * dt
            if neuron.x < -30:
                self.neurons.remove(neuron)
                continue
            dx, dy = neuron.x - BIRD_X, neuron.y - self.y
            if dx * dx + dy * dy < (BIRD_R + Neuron.R) ** 2:
                self.neurons.remove(neuron)
                self.score += 1 if neuron.kind == "green" else -1

        # terrain collision
        top, bottom = self.corridor(self.bird_world_x)
        if self.y - BIRD_R < top or self.y + BIRD_R > bottom:
            self.game_over = True
            self._dead_timer = 0.0
            self.total_deaths += 1
            self.total_score += self.score
            self.best_stretch = max(self.best_stretch, self.elapsed)
            where = "top" if self.y - BIRD_R < top else "bottom"
            print(
                f"[game] game over at t={self.elapsed:.1f}s (hit {where} terrain: "
                f"y={self.y:.0f}, corridor {top:.0f}..{bottom:.0f}, score {self.score})",
                flush=True,
            )

        # time limit
        if self.time_limit is not None and self.elapsed >= self.time_limit:
            self.finished = True

    # ----------------------------------------------------------------- draw
    def state_packet(self, t: float) -> str:
        """State broadcast packet (CSV, v2):

        ``t,bird_y,bird_vy,corridor_center_y,score,alive,green_dx,green_dy,red_dx,red_dy``

        ``*_dx/dy`` are the nearest on-screen green/red neuron AHEAD of the
        bird, relative to the bird; 9999 when none. v1 (6-field) packets are
        still accepted by the pilot.
        """
        top, bottom = self.corridor(self.bird_world_x)
        fields = [
            f"{t:.3f}",
            f"{self.y:.1f}",
            f"{self.vy:.1f}",
            f"{(top + bottom) / 2:.1f}",
            str(self.total_score + self.score),
            str(int(not self.game_over and not self.finished)),
        ]
        for kind in ("green", "red"):
            ahead = [n for n in self.neurons if n.kind == kind and n.x > BIRD_X]
            if ahead:
                n = min(ahead, key=lambda n: n.x)
                fields += [f"{n.x - BIRD_X:.0f}", f"{n.y - self.y:.0f}"]
            else:
                fields += ["9999", "9999"]
        return ",".join(fields)

    def draw(self, surface) -> None:
        import pygame

        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("helvetica", 22)
            self._big_font = pygame.font.SysFont("helvetica", 44, bold=True)

        # sky gradient (horizontal bands)
        for i in range(24):
            t = i / 23
            color = tuple(int(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_BOTTOM))
            pygame.draw.rect(surface, color, (0, i * HEIGHT / 24, WIDTH, HEIGHT / 24 + 1))

        # terrain (top + bottom polygons along the corridor)
        xs = list(range(0, WIDTH + 20, 20))
        top_pts = [(x, self.corridor(self.world_x + x)[0]) for x in xs]
        bot_pts = [(x, self.corridor(self.world_x + x)[1]) for x in xs]
        pygame.draw.polygon(surface, TERRAIN, [(0, 0), *top_pts, (WIDTH, 0)])
        pygame.draw.polygon(surface, TERRAIN, [(0, HEIGHT), *bot_pts, (WIDTH, HEIGHT)])
        pygame.draw.lines(surface, TERRAIN_EDGE, False, top_pts, 3)
        pygame.draw.lines(surface, TERRAIN_EDGE, False, bot_pts, 3)

        # neurons
        for neuron in self.neurons:
            color = GREEN if neuron.kind == "green" else RED
            pygame.draw.circle(surface, color, (neuron.x, neuron.y), Neuron.R)
            pygame.draw.circle(surface, (255, 255, 255), (neuron.x, neuron.y), Neuron.R, 2)
            pygame.draw.circle(surface, (255, 255, 255), (neuron.x - 3, neuron.y - 3), 3)

        # bird
        pygame.draw.circle(surface, BIRD, (BIRD_X, self.y), BIRD_R)
        pygame.draw.ellipse(
            surface, BIRD_WING, (BIRD_X - 16, self.y - 4 - 4 * math.sin(self.elapsed * 12), 16, 10)
        )
        pygame.draw.circle(surface, INK, (BIRD_X + 6, self.y - 4), 2)

        # HUD
        hud = self._font.render(
            f"score {self.score}    time {self.elapsed:5.1f}s    flaps rx {self.flap_packets}",
            True, INK,
        )
        surface.blit(hud, (12, 10))

        if self.game_over or self.finished:
            msg = f"TIME UP — final score {self.score}" if self.finished else "GAME OVER"
            text = self._big_font.render(msg, True, INK)
            surface.blit(text, text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="UDP listen host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5005, help="UDP listen port (default 5005)")
    ap.add_argument("--auto", action="store_true", help="autopilot flaps by itself (demo/test mode)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible neuron layouts")
    ap.add_argument("--duration", type=float, default=None, help="exit after S seconds")
    ap.add_argument("--frames-dir", default=None, help="save a PNG frame every 30 frames to DIR")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="game time limit in s")
    ap.add_argument("--flap-force", type=float, default=FLAP_FORCE)
    ap.add_argument("--gravity", type=float, default=GRAVITY)
    ap.add_argument("--spawn-interval", type=float, default=SPAWN_INTERVAL)
    ap.add_argument("--spawn-jitter", type=float, default=SPAWN_JITTER)
    ap.add_argument("--terrain-scale", type=float, default=1.0,
                    help="flatten the hills (<1.0); demo assist for open-loop scripted flight")
    ap.add_argument("--state-broadcast", type=int, default=None, metavar="PORT",
                    help="send game state each frame via UDP to 127.0.0.1:PORT as "
                         "'t,bird_y,bird_vy,corridor_center_y,score,alive,"
                         "green_dx,green_dy,red_dx,red_dy' (CSV; for pilot.py)")
    args = ap.parse_args()

    import pygame

    pygame.init()
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Brain — mock game")
    clock = pygame.time.Clock()
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)

    game = Game(
        host=args.host,
        port=args.port,
        auto=args.auto,
        seed=args.seed,
        flap_force=args.flap_force,
        gravity=args.gravity,
        spawn_interval=args.spawn_interval,
        spawn_jitter=args.spawn_jitter,
        time_limit=args.time_limit,
        terrain_scale=args.terrain_scale,
    )
    print(
        f"[game] listening for flaps on {args.host}:{args.port}"
        + (" [autopilot]" if args.auto else " (waiting for play.py UDP packets)")
    )

    frame = 0
    elapsed_wall = 0.0
    running = True
    state_sock = None
    if args.state_broadcast:
        state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[game] broadcasting state to 127.0.0.1:{args.state_broadcast} "
              "(t,bird_y,bird_vy,corridor_center_y,score,alive,"
              "green_dx,green_dy,red_dx,red_dy)")
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        elapsed_wall += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.flap()
                elif event.key == pygame.K_r:
                    game.reset()

        game.poll_udp()
        game.step(dt)

        if state_sock is not None:
            state_sock.sendto(game.state_packet(elapsed_wall).encode(),
                              ("127.0.0.1", args.state_broadcast))

        game.draw(surface)
        pygame.display.flip()

        if args.frames_dir and frame % 30 == 0:
            pygame.image.save(surface, os.path.join(args.frames_dir, f"frame_{frame:05d}.png"))
        frame += 1
        if args.duration is not None and elapsed_wall >= args.duration:
            break

    print(
        f"[game] done: score {game.total_score + game.score} "
        f"(best life {max(game.best_stretch, game.elapsed):.1f}s), "
        f"{game.total_deaths} game-over(s), "
        f"{game.flap_packets} UDP flap packets received, {frame} frames"
    )
    game.close()
    pygame.quit()


if __name__ == "__main__":
    main()
