"""Pong for two players at the short ends of the table.

Each player's bouncer lies between their two wands (centred on their midpoint,
at most 60% of the court's width). The ball bounces off it like off a moving
wall: pushing the bouncer into the ball sends it back faster, sweeping it
sideways adds spin-like sideways speed, and its angle steers the ball. Fast
balls slowly relax back to a cruising speed.
"""
import math
import random
import time
from collections import deque

import numpy as np

from .. import ui
from ..projector import pygame
from .players import make_players

WIN_SCORE = 7
BALL_MM = 14
CRUISE_MM = 320  # mm/s the ball relaxes back to
MIN_MM, MAX_MM = 200, 1500  # ball speed limits
RELAX_S = 2.5  # time constant for a fast ball to slow back to cruising speed
RESTITUTION = 0.85  # how much of the push along the bouncer's normal comes back
GRIP = 0.35  # share of the bouncer's sideways motion (relative to the ball) passed to the ball
MIN_FORWARD = 0.35  # after a hit the ball heads to the opponent at least this fast (x cruise)
MAX_PADDLE = 0.6  # bouncer length limit, fraction of the court's width
PADDLE_HALF_THICK = 6


def _perp(v):
    return np.array([-v[1], v[0]])


class Pong:
    title = "Pong"
    min_forward = MIN_FORWARD  # 0 lets a hit send the ball back towards your own goal

    def __init__(self, app):
        self.app, self.field, mm = app, app.field, app.px_per_mm
        self.mm = mm
        self.r = BALL_MM * mm
        self.font = pygame.font.SysFont(None, int(self.field.height * 0.16))
        self.small = pygame.font.SysFont(None, int(self.field.height * 0.07))
        self.players = make_players(app.tags, self.field)
        self.new_match(time.monotonic())

    def new_match(self, now):
        self.score = [0, 0]
        self.winner, self.winner_at = None, 0.0
        self.rally = 0
        self.ball, self.vel = None, None
        self.trail = deque(maxlen=12)
        self.serve_to = random.choice((0, 1))
        self.serve_at = now + 2.0
        self.prev_paddles = {}
        self.flash = None

    def paddle(self, tags, i):
        """(end a, end b, velocity) of player i's bouncer, or None without both wands."""
        w = self.players[i].wands(tags)
        if len(w) != 2:
            return None
        a, b = w[0].pos, w[1].pos
        mid, d = (a + b) / 2, b - a
        length = np.linalg.norm(d)
        half = min(length / 2, MAX_PADDLE * self.field.height / 2)
        d = d / max(length, 1e-6)
        vel = (w[0].vel + w[1].vel) / 2
        return mid - d * half, mid + d * half, vel

    # --- rules ---------------------------------------------------------
    def serve(self):
        p = self.players[self.serve_to]
        ang = random.uniform(-0.45, 0.45)
        d = np.array([-p.dir * math.cos(ang), math.sin(ang)])  # towards the player who conceded
        self.ball = np.array(self.field.center, float)
        self.vel = d * CRUISE_MM * self.mm * 0.9
        self.rally = 0
        self.trail.clear()
        self.app.sound.play("valve")
        self.app.fx.ring(self.ball, (255, 255, 230), self.r * 1.5, n=18, speed=120)

    def bounce(self, i, normal, paddle_vel, now):
        """Collision with a moving wall: reflect the velocity relative to the bouncer."""
        p = self.players[i]
        rel = self.vel - paddle_vel
        vn = np.dot(rel, normal)
        if vn >= 0:
            return
        tangent = rel - vn * normal
        v = self.vel - (1 + RESTITUTION) * vn * normal - GRIP * tangent
        fwd = np.array([p.dir, 0.0])
        min_fwd = self.min_forward * CRUISE_MM * self.mm
        if self.min_forward and np.dot(v, fwd) < min_fwd:  # never back into your own goal
            v = v + (min_fwd - np.dot(v, fwd)) * fwd
        speed = np.linalg.norm(v)
        self.vel = v * np.clip(speed, MIN_MM * self.mm, MAX_MM * self.mm) / max(speed, 1e-6)
        self.rally += 1
        self.flash = (i, now)
        hard = min(1.0, np.linalg.norm(self.vel) / (MAX_MM * self.mm * 0.6))
        self.app.sound.play("hit", 0.5 + 0.5 * hard)
        self.app.fx.burst(self.ball - normal * self.r, p.color, n=int(14 + 30 * hard), speed=150 + 400 * hard,
                          life=0.45, size=4, spread=2.2, direction=math.atan2(normal[1], normal[0]),
                          colors=[p.color, (255, 255, 255)])

    def update(self, dt, now, tags, presses):
        if self.winner is not None:
            if now - self.winner_at > 6:
                self.new_match(now)
            return
        paddles = {i: self.paddle(tags, i) for i in (0, 1)}
        if self.ball is None:
            if now >= self.serve_at:
                self.serve()
            self.prev_paddles = paddles
            return
        f = self.field
        speed = np.linalg.norm(self.vel)
        cruise = CRUISE_MM * self.mm
        if speed > cruise:  # fast balls relax back to cruising speed
            self.vel *= (cruise + (speed - cruise) * math.exp(-dt / RELAX_S)) / speed
        prev = self.ball.copy()
        steps = max(1, math.ceil(np.linalg.norm(self.vel) * dt / (self.r * 0.5)))
        for _ in range(steps):
            self.ball += self.vel * dt / steps
            if self.ball[1] < f.top + self.r and self.vel[1] < 0 or self.ball[1] > f.bottom - self.r and self.vel[1] > 0:
                self.vel[1] = -self.vel[1]
                self.app.sound.play("wall", 0.6)
                self.app.fx.burst(self.ball, (200, 200, 200), n=6, speed=120, life=0.25, size=2.5)
            for i, pad in paddles.items():
                if pad is not None and self._touch(i, pad, now):
                    break
        # fast swipes: the bouncer may have jumped across the ball between camera frames
        for i, pad in paddles.items():
            old = self.prev_paddles.get(i)
            if pad is None or old is None:
                continue
            s0, s1 = self._side(prev, old), self._side(self.ball, pad)
            if s0 * s1 < 0 and self._within(self.ball, pad):
                a, b, u = pad
                n = _perp((b - a) / np.linalg.norm(b - a)) * np.sign(s0)
                self.ball = self._closest(self.ball, a, b) + n * (self.r + PADDLE_HALF_THICK + 1)
                self.bounce(i, n, u, now)
        self.prev_paddles = paddles
        self.trail.append(self.ball.copy())
        if self.ball[0] < f.left - self.r or self.ball[0] > f.right + self.r:
            loser = [p.side for p in self.players].index("left" if self.ball[0] < f.left else "right")
            self.score[1 - loser] += 1
            goal_x = f.left if self.players[loser].side == "left" else f.right
            self.app.fx.burst((goal_x, self.ball[1]), self.players[1 - loser].color, n=60, speed=500, life=0.8, size=5,
                              direction=0.0 if goal_x == f.left else math.pi, spread=2.5,
                              colors=[self.players[1 - loser].color, (255, 255, 255)])
            self.app.sound.play("score")
            self.ball = None
            self.serve_to, self.serve_at = loser, now + 1.2
            if self.score[1 - loser] >= WIN_SCORE:
                self.winner, self.winner_at = 1 - loser, now
                self.app.sound.play("win")
                self.app.fx.confetti(f)

    @staticmethod
    def _closest(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0, 1)
        return a + t * ab

    @staticmethod
    def _side(p, pad):
        a, b = pad[:2]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        return float(np.dot(p - a, _perp(t)))

    def _within(self, p, pad):
        a, b = pad[:2]
        ab = b - a
        L = max(np.linalg.norm(ab), 1e-6)
        u = np.dot(p - a, ab) / L
        return -self.r <= u <= L + self.r

    def _touch(self, i, pad, now):
        a, b, u = pad
        cp = self._closest(self.ball, a, b)
        d = self.ball - cp
        dist = np.linalg.norm(d)
        if dist >= self.r + PADDLE_HALF_THICK or dist < 1e-6:
            return False
        n = d / dist
        if np.dot(self.vel - u, n) >= 0:
            return False  # already moving away from the bouncer
        self.ball = cp + n * (self.r + PADDLE_HALF_THICK + 1)
        self.bounce(i, n, u, now)
        return True

    # --- drawing ---------------------------------------------------------
    def draw(self, s, now):
        f = self.field
        s.fill(ui.BACKGROUND)
        pygame.draw.rect(s, (70, 70, 70), f, 3)
        for y in range(f.top, f.bottom, 40):
            pygame.draw.line(s, (75, 75, 75), (f.centerx, y), (f.centerx, y + 20), 3)
        for i, p in enumerate(self.players):
            x = f.centerx - p.dir * f.width * 0.18
            ui.text(s, self.font, str(self.score[i]), (x, f.centery), tuple(c // 2 for c in p.color), p.facing)
            pad = self.paddle(self.app.tags, i)
            if pad is not None:
                a, b, _ = pad
                lit = self.flash and self.flash[0] == i and now - self.flash[1] < 0.15
                color = (255, 255, 255) if lit else p.color
                pygame.draw.line(s, color, a, b, PADDLE_HALF_THICK * 2)
        if self.ball is not None:
            for j, q in enumerate(self.trail):
                pygame.draw.circle(s, (60 + 12 * j,) * 3, q, self.r * (0.3 + 0.05 * j))
            pygame.draw.circle(s, (255, 255, 230), self.ball, self.r)
        elif self.winner is None:
            left = max(0.0, self.serve_at - now)
            for p in self.players:
                ui.text(s, self.small, f"serve in {left:.0f}", (f.centerx - p.dir * f.width * 0.3, f.centery),
                        (200, 200, 200), p.facing)
        if self.winner is not None:
            w = self.players[self.winner]
            for p in self.players:
                msg = "you win!" if p is w else "you lose"
                ui.text(s, self.font, msg, (f.centerx - p.dir * f.width * 0.3, f.centery), w.color, p.facing)
        for p in self.players:
            for w in p.wands(self.app.tags):
                ui.quiet_wand(s, w)
