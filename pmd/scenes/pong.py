"""Pong for two players at the short ends of the table.

Each player's bouncer is the line between their two wands. Holding the wands
closer together makes a shorter bouncer that sends the ball back faster; its
angle steers the ball.
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
BASE_SPEED_MM = 550  # ball speed off a bouncer of REF_LEN_MM
REF_LEN_MM = 150
SPEED_RANGE = (0.6, 2.5)  # speed multiplier limits from bouncer length
MAX_ANGLE = math.radians(60)  # steepest return angle off the long axis
PADDLE_HALF_THICK = 6


def _perp(v):
    return np.array([-v[1], v[0]])


class Pong:
    title = "Pong"

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
        w = self.players[i].wands(tags)
        return (w[0].pos.copy(), w[1].pos.copy()) if len(w) == 2 else None

    # --- rules ---------------------------------------------------------
    def serve(self):
        p = self.players[self.serve_to]
        ang = random.uniform(-0.45, 0.45)
        d = np.array([-p.dir * math.cos(ang), math.sin(ang)])  # towards the player who conceded
        self.ball = np.array(self.field.center, float)
        self.vel = d * BASE_SPEED_MM * self.mm * 0.8
        self.rally = 0
        self.trail.clear()

    def bounce(self, i, a, b, normal, now):
        p = self.players[i]
        length = np.linalg.norm(b - a) / self.mm
        mult = np.clip(REF_LEN_MM / max(length, 1), *SPEED_RANGE) * (1 + 0.03 * self.rally)
        v = self.vel - 2 * np.dot(self.vel, normal) * normal
        # head towards the opponent, within MAX_ANGLE of the long axis
        ang = math.atan2(v[1], v[0] * p.dir)
        if abs(ang) > math.pi / 2:
            ang = math.copysign(math.pi, ang) - ang
        ang = float(np.clip(ang, -MAX_ANGLE, MAX_ANGLE))
        self.vel = np.array([p.dir * math.cos(ang), math.sin(ang)]) * BASE_SPEED_MM * self.mm * mult
        self.rally += 1
        self.flash = (i, now)

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
        prev = self.ball.copy()
        steps = max(1, math.ceil(np.linalg.norm(self.vel) * dt / (self.r * 0.5)))
        for _ in range(steps):
            self.ball += self.vel * dt / steps
            if self.ball[1] < f.top + self.r and self.vel[1] < 0 or self.ball[1] > f.bottom - self.r and self.vel[1] > 0:
                self.vel[1] = -self.vel[1]
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
                a, b = pad
                n = _perp((b - a) / np.linalg.norm(b - a)) * np.sign(s0)
                self.ball = self._closest(self.ball, a, b) + n * (self.r + PADDLE_HALF_THICK + 1)
                if np.dot(self.vel, n) < 0:
                    self.bounce(i, a, b, n, now)
        self.prev_paddles = paddles
        self.trail.append(self.ball.copy())
        if self.ball[0] < f.left - self.r or self.ball[0] > f.right + self.r:
            loser = [p.side for p in self.players].index("left" if self.ball[0] < f.left else "right")
            self.score[1 - loser] += 1
            self.ball = None
            self.serve_to, self.serve_at = loser, now + 1.2
            if self.score[1 - loser] >= WIN_SCORE:
                self.winner, self.winner_at = 1 - loser, now

    @staticmethod
    def _closest(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0, 1)
        return a + t * ab

    @staticmethod
    def _side(p, pad):
        a, b = pad
        t = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        return float(np.dot(p - a, _perp(t)))

    def _within(self, p, pad):
        a, b = pad
        ab = b - a
        L = max(np.linalg.norm(ab), 1e-6)
        u = np.dot(p - a, ab) / L
        return -self.r <= u <= L + self.r

    def _touch(self, i, pad, now):
        a, b = pad
        cp = self._closest(self.ball, a, b)
        d = self.ball - cp
        dist = np.linalg.norm(d)
        if dist >= self.r + PADDLE_HALF_THICK or dist < 1e-6:
            return False
        n = d / dist
        if np.dot(self.vel, n) >= 0:
            return False  # already moving away
        self.ball = cp + n * (self.r + PADDLE_HALF_THICK + 1)
        self.bounce(i, a, b, n, now)
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
                a, b = pad
                length = np.linalg.norm(b - a) / self.mm
                mult = np.clip(REF_LEN_MM / max(length, 1), *SPEED_RANGE)
                lit = self.flash and self.flash[0] == i and now - self.flash[1] < 0.15
                color = (255, 255, 255) if lit else p.color
                pygame.draw.line(s, color, a, b, int(PADDLE_HALF_THICK * 2 + 4 * (mult - 0.6)))
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
