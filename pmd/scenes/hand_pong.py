"""Pong on the laptop screen, played with your hands in front of the webcam.

Same rules and ball physics as the table Pong (this subclasses it): the screen
is the table seen from above, the left player defends the left edge and the
right player the right edge. Instead of two wands, each player's bouncer lies
between the tip of their thumb and the tip of their index finger (a finger gun
is the natural grip, but any hand pose works). The bouncer is a moving wall:
push into the ball to hit harder, sweep sideways to send it at an angle, tilt
your hand to aim. Unlike the table version, a hit can also send the ball back
the wrong way, towards your own goal.

The leftmost hand on screen belongs to the left player, the rightmost to the
right player (one hand alone: whichever half it is in). Coordinates are
screen pixels; app.to_screen maps the camera's normalised coordinates 1:1 onto
the video drawn behind the game.
"""
import numpy as np

from .. import ui
from ..filters import OneEuro
from ..hands import BONES, WRIST
from ..projector import pygame
from .pong import CRUISE_MM, MAX_MM, MAX_PADDLE, Pong

REQUIRE_GUN = False  # True: only a finger-gun pose makes a bouncer
MIN_SIDEWAYS = 0.25  # after a hit the ball crosses the court at least this fast (x cruise), either way
SEPARATION = 0.4  # after a hit the ball leaves the paddle at least this fast relative to it (x cruise)
HIT_COOLDOWN_S = 0.3  # a paddle can't hit the ball again this soon (contact just pushes it clear)
PADDLE_STRETCH = 1.0  # bouncer length / thumb-to-index distance (1.0: exactly between the fingertips)
GRACE_S = 0.2  # keep a bouncer this long after its hand drops out (tracking blinks)
PREDICT_S = 0.04  # extrapolate the bouncer towards "now" by at most this much (camera + model latency)
JUMP = 0.3  # a hand appearing this far (x screen width) from the last one restarts its smoothing
COLOR_NAMES = {(70, 220, 120): "green", (255, 150, 50): "orange"}
SHADOW = (0, 0, 0)


class _Hand:
    """Smoothed fingertips of one player's hand."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.filter = OneEuro(min_cutoff=1.5, beta=0.004, d_cutoff=3.0)
        self.tips = None  # (thumb, index) in screen px, smoothed
        self.vel = np.zeros(2)  # bouncer velocity, px/s
        self.t = None  # capture time of the last frame used
        self.seen = -1e9  # capture time the hand was last seen


class HandPong(Pong):
    title = "Finger-gun pong"
    min_forward = 0.0  # hits may go either way

    def __init__(self, app):
        super().__init__(app)
        self.hands = [_Hand(), _Hand()]
        self.last_hit = [-1e9, -1e9]  # time of each player's last hit
        self.frame_n = -1
        self.hint_font = pygame.font.SysFont(None, int(self.field.height * 0.05))
        self.thick = max(6, int(self.field.height * 0.012))

    # --- hands -> bouncers ---------------------------------------------
    def _track(self):
        frame = self.app.frame
        if frame is None or frame.n == self.frame_n:
            return
        self.frame_n = frame.n
        tips = [np.array([self.app.to_screen(h.thumb), self.app.to_screen(h.index)]) for h in frame.hands
                if h.gun or not REQUIRE_GUN]
        tips.sort(key=lambda g: g[:, 0].mean())
        by_side = {}
        if len(tips) >= 2:
            by_side = {"left": tips[0], "right": tips[-1]}
        elif tips:
            by_side = {"left" if tips[0][:, 0].mean() < self.field.centerx else "right": tips[0]}
        W = self.field.width
        for p, h in zip(self.players, self.hands):
            g = by_side.get(p.side)
            if g is None:
                if frame.t - h.seen > GRACE_S:
                    h.reset()
                continue
            if h.tips is not None and np.linalg.norm(g.mean(0) - h.tips.mean(0)) > JUMP * W:
                h.reset()
            dt = frame.t - h.t if h.t is not None else 0.0
            h.tips = h.filter(g.reshape(4), dt).reshape(2, 2)
            h.vel = h.filter.dx.reshape(2, 2).mean(0)
            h.t = h.seen = frame.t

    def paddle(self, tags, i):
        """(end a, end b, velocity) of player i's bouncer, or None without a hand."""
        h = self.hands[i]
        if h.tips is None:
            return None
        lead = np.clip(self.app.now - h.t, 0.0, PREDICT_S)
        a, b = h.tips + h.vel * lead
        mid, d = (a + b) / 2, b - a
        length = np.linalg.norm(d)
        half = min(length / 2 * PADDLE_STRETCH, MAX_PADDLE * self.field.height / 2)
        d = d / max(length, 1e-6)
        return mid - d * half, mid + d * half, h.vel

    def bounce(self, i, normal, paddle_vel, now):
        """Table Pong's bounce, without forcing the ball towards the opponent.

        - The ball keeps at least MIN_SIDEWAYS x cruise speed sideways (in whichever
          direction it is going), so it can't end up bouncing between the top and
          bottom walls forever.
        - It always leaves at least SEPARATION x cruise faster than the paddle moves
          along the normal, so a paddle carried along with the ball can't catch it
          again straight away.
        - Within HIT_COOLDOWN_S of a hit, touching the same paddle only pushes the
          ball clear (no new hit, sound or sparks), so jitter can't rattle it.
        """
        if now - self.last_hit[i] < HIT_COOLDOWN_S:
            self._separate(normal, paddle_vel)
            return
        rally = self.rally
        super().bounce(i, normal, paddle_vel, now)
        if self.rally == rally:
            return  # ball was already moving away: not a hit
        self.last_hit[i] = now
        vx, need = self.vel[0], MIN_SIDEWAYS * CRUISE_MM * self.mm
        if abs(vx) < need:
            self.vel[0] = need * (np.sign(vx) or self.players[i].dir)
        self._separate(normal, paddle_vel)

    def _separate(self, normal, paddle_vel):
        """Make the ball move away from the paddle along the normal at least SEPARATION x cruise."""
        need = SEPARATION * CRUISE_MM * self.mm
        away = float(np.dot(self.vel - paddle_vel, normal))
        if away < need:
            self.vel = self.vel + (need - away) * normal
        speed, top = np.linalg.norm(self.vel), MAX_MM * self.mm
        if speed > top:  # same speed limit as the table game
            self.vel *= top / speed

    def update(self, dt, now, tags, presses):
        self._track()
        if self.ball is None and self.winner is None and any(self.paddle(tags, i) is None for i in (0, 1)):
            self.serve_at = max(self.serve_at, now + 1.0)  # serve only once both players are in
        super().update(dt, now, tags, presses)

    # --- drawing -------------------------------------------------------
    def _text(self, s, font, msg, center, color):
        ui.text(s, font, msg, (center[0] + 3, center[1] + 3), SHADOW)
        return ui.text(s, font, msg, center, color)

    def _name(self, p):
        return COLOR_NAMES.get(p.color, p.side)

    def draw(self, s, now):
        f = self.field
        for y in range(f.top, f.bottom, 60):
            pygame.draw.line(s, (235, 235, 235), (f.centerx, y), (f.centerx, y + 30), 4)
        for p in self.players:  # goal lines in the defender's colour
            pygame.draw.line(s, p.color, (p.end_x, f.top), (p.end_x, f.bottom), 8)
        for i, p in enumerate(self.players):
            x = f.centerx - p.dir * f.width * 0.08
            self._text(s, self.font, str(self.score[i]), (x, f.top + f.height * 0.1), p.color)

        frame = self.app.frame
        if frame is not None:
            for h in frame.hands:
                pts = self.app.to_screen(h.pts)
                if self.app.debug:
                    for a, b in BONES:
                        pygame.draw.line(s, (255, 255, 255) if h.gun else (150, 150, 150), pts[a], pts[b], 2)
                if REQUIRE_GUN and not h.gun and self.winner is None:
                    self._text(s, self.hint_font, "finger gun!", pts[WRIST] + (0, f.height * 0.06), (255, 255, 255))

        for i, p in enumerate(self.players):
            pad = self.paddle(None, i)
            if pad is None:
                continue
            a, b, _ = pad
            lit = self.flash and self.flash[0] == i and now - self.flash[1] < 0.15
            color = (255, 255, 255) if lit else p.color
            pygame.draw.line(s, SHADOW, a, b, self.thick * 2 + 6)
            for q in (a, b):
                pygame.draw.circle(s, SHADOW, q, self.thick + 3)
            pygame.draw.line(s, color, a, b, self.thick * 2)
            for q in (a, b):
                pygame.draw.circle(s, color, q, self.thick)

        if self.ball is not None:
            n = len(self.trail)
            for j, q in enumerate(self.trail):
                k = (j + 1) / n
                pygame.draw.circle(s, (int(120 + 135 * k), int(120 + 135 * k), int(100 + 130 * k)), q,
                                   self.r * (0.25 + 0.6 * k))
            pygame.draw.circle(s, SHADOW, self.ball, self.r + 3)
            pygame.draw.circle(s, (255, 255, 230), self.ball, self.r)
        elif self.winner is None:
            waiting = [p for i, p in enumerate(self.players) if self.paddle(None, i) is None]
            for p in waiting:
                msg = "make a finger gun" if REQUIRE_GUN else "show your hand"
                self._text(s, self.small, f"{self._name(p)}: {msg}",
                           (f.centerx - p.dir * f.width * 0.25, f.centery), p.color)
            if not waiting:
                left = max(0.0, self.serve_at - now)
                self._text(s, self.small, f"serve in {left:.0f}", (f.centerx, f.centery), (255, 255, 255))
        if self.winner is not None:
            w = self.players[self.winner]
            self._text(s, self.font, f"{self._name(w)} wins!", f.center, w.color)
