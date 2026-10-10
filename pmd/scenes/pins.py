"""Pin shooter: knock down the opponent's line of 10 pins with bowling balls.

Players stand at the short ends. Each wand holds one ball; cover the wand's
red spot to fire it straight ahead (the tag's printed top). Wands block
balls coming at you: an empty wand catches the ball (now it's yours), a
loaded one deflects it. Balls that get past your pin line roll into your
ball return (on the wall to your right, halfway to the centre); hold an empty
wand in the return to pick one up.
"""
import math
import time

import numpy as np

from .. import ui
from ..projector import pygame
from .players import make_players

PINS = 10
START_BALLS = 3
BALL_MM, PIN_MM = 15, 10
SPEED_MM = 700
MAX_ANGLE = math.radians(65)  # steepest shot off the long axis


class Ball:
    def __init__(self, pos, vel, color):
        self.pos, self.vel, self.color = pos, vel, color


class PinShooter:
    title = "Pin Shooter"

    def __init__(self, app):
        self.app, self.field, self.mm = app, app.field, app.px_per_mm
        self.r, self.pin_r = BALL_MM * self.mm, PIN_MM * self.mm
        self.font = pygame.font.SysFont(None, int(self.field.height * 0.08))
        self.big = pygame.font.SysFont(None, int(self.field.height * 0.14))
        self.players = make_players(app.tags, self.field)
        self.new_match(time.monotonic())

    def new_match(self, now):
        f = self.field
        self.pin_x, self.pins, self.trays = [], [], []
        for p in self.players:
            x = p.end_x + p.dir * f.width * 0.16  # pin line, measured from the player's end
            ys = np.linspace(f.top + f.height * 0.1, f.bottom - f.height * 0.1, PINS)  # clear of the ball returns
            self.pin_x.append(x)
            self.pins.append([[np.array([x, y]), None] for y in ys])  # [pos, knocked down at]
            # ball return against the wall on the player's right-hand side, halfway
            # between their end and the centre line
            w, h = self.r * 2.3 * 6.5, self.r * 3  # room for all 6 balls in a row
            rect = pygame.Rect(0, 0, w, h)
            rect.centerx = int(p.end_x + p.dir * f.width / 4)
            if p.side == "left":
                rect.bottom = f.bottom
            else:
                rect.top = f.top
            self.trays.append(rect)
        self.tray_balls = [START_BALLS, START_BALLS]
        self.loaded = {}
        self.load_ready = {}
        self.balls = []
        self.effects = []
        self.winner, self.winner_at = None, 0.0

    def owner(self, wand_id):
        return next(i for i, p in enumerate(self.players) if wand_id in p.ids)

    def aim(self, direction, p):
        """Clamp a direction to within MAX_ANGLE of p's shooting direction."""
        ang = math.atan2(direction[1], direction[0] * p.dir)
        if abs(ang) > math.pi / 2:
            ang = math.copysign(math.pi, ang) - ang
        ang = float(np.clip(ang, -MAX_ANGLE, MAX_ANGLE))
        return np.array([p.dir * math.cos(ang), math.sin(ang)])

    # --- rules ---------------------------------------------------------
    def update(self, dt, now, tags, presses):
        if self.winner is not None:
            if now - self.winner_at > 6:
                self.new_match(now)
            return
        for i, p in enumerate(self.players):
            for w in p.wands(tags):
                loaded = self.loaded.get(w.id, False)
                if not loaded and self.tray_balls[i] and now >= self.load_ready.get(w.id, 0) \
                        and self.trays[i].inflate(40, 40).collidepoint(w.pos):
                    self.tray_balls[i] -= 1
                    self.loaded[w.id] = True
                    self.effects.append(("load", w.pos.copy(), now, p.color))
                elif loaded and w.id in presses:
                    d = self.aim(w.forward, p)
                    self.balls.append(Ball(w.pos + d * (w.size + self.r + 4), d * SPEED_MM * self.mm, p.color))
                    self.loaded[w.id] = False
                    self.load_ready[w.id] = now + 0.6  # don't instantly reload when firing from the tray
        self._move_balls(dt, now, tags)
        for i in (0, 1):
            if all(down is not None for _, down in self.pins[i]):
                self.winner, self.winner_at = 1 - i, now

    def _move_balls(self, dt, now, tags):
        f = self.field
        for b in list(self.balls):
            steps = max(1, math.ceil(np.linalg.norm(b.vel) * dt / (self.r * 0.5)))
            for _ in range(steps):
                b.pos = b.pos + b.vel * dt / steps
                if b.pos[1] < f.top + self.r and b.vel[1] < 0 or b.pos[1] > f.bottom - self.r and b.vel[1] > 0:
                    b.vel[1] = -b.vel[1]
                if self._hit_wands(b, now, tags):
                    break
                for row in self.pins:
                    for pin in row:
                        if pin[1] is None and np.linalg.norm(b.pos - pin[0]) < self.r + self.pin_r:
                            pin[1] = now
                            b.vel = b.vel * 0.95
                            self.effects.append(("pin", pin[0].copy(), now, (255, 255, 255)))
                if self._returned(b, now):
                    break

    def _hit_wands(self, b, now, tags):
        """Wands only block balls heading towards their own player's end."""
        for i, p in enumerate(self.players):
            if b.vel[0] * p.dir >= 0:
                continue
            for w in p.wands(tags):
                d = b.pos - w.pos
                dist = np.linalg.norm(d)
                if dist >= self.r + w.size * 0.9 or dist < 1e-6:
                    continue
                if not self.loaded.get(w.id, False):
                    self.balls.remove(b)
                    self.loaded[w.id] = True
                    self.effects.append(("catch", w.pos.copy(), now, p.color))
                    return True
                n = d / dist
                v = b.vel - 2 * np.dot(b.vel, n) * n
                b.vel = self.aim(v, p) * np.linalg.norm(b.vel)
                b.pos = w.pos + n * (self.r + w.size * 0.9 + 1)
                b.color = p.color
                self.effects.append(("block", w.pos.copy(), now, p.color))
        return False

    def _returned(self, b, now):
        """Past a pin line, heading for that end: roll into that player's ball return."""
        for i, p in enumerate(self.players):
            behind = (b.pos[0] - self.pin_x[i]) * p.dir < -(self.r + self.pin_r)
            if behind and b.vel[0] * p.dir < 0:
                self.balls.remove(b)
                self.tray_balls[i] += 1
                self.effects.append(("return", np.array(self.trays[i].center, float), now, p.color))
                return True
        return False

    # --- drawing ---------------------------------------------------------
    def draw(self, s, now):
        f = self.field
        s.fill((0, 0, 0))
        pygame.draw.rect(s, (40, 40, 50), f, 3)
        pygame.draw.line(s, (40, 40, 50), (f.centerx, f.top), (f.centerx, f.bottom), 2)
        tags = self.app.tags
        for i, p in enumerate(self.players):
            dim = tuple(c // 4 for c in p.color)
            pygame.draw.line(s, dim, (self.pin_x[i], f.top), (self.pin_x[i], f.bottom), 2)
            for pos, down in self.pins[i]:
                if down is None:
                    pygame.draw.circle(s, (250, 250, 250), pos, self.pin_r)
                    pygame.draw.circle(s, (40, 90, 220), pos, self.pin_r * 0.6, 3)  # not red: red is the wand buttons
                else:
                    fade = max(0.25, 1 - (now - down) / 0.6)
                    off = np.array([p.dir * self.pin_r * 1.5, 0]) * min(1, (now - down) / 0.3)
                    pygame.draw.ellipse(s, tuple(int(90 * fade) for _ in range(3)),
                                        pygame.Rect(pos[0] + off[0] - self.pin_r * 1.6, pos[1] - self.pin_r * 0.6,
                                                    self.pin_r * 3.2, self.pin_r * 1.2))
            tray = self.trays[i]
            pygame.draw.rect(s, p.color, tray, 3, border_radius=8)
            for k in range(self.tray_balls[i]):  # balls lined up along the wall, nearest the player first
                x = tray.centerx - p.dir * (tray.width / 2 - self.r * 1.3 - k * self.r * 2.3)
                pygame.draw.circle(s, p.color, (x, tray.centery), self.r)
            standing = sum(down is None for _, down in self.pins[i])
            ui.text(s, self.font, f"pins {standing}", (f.centerx - p.dir * f.width * 0.25, f.top + f.height * 0.5),
                    tuple(c // 2 for c in p.color), p.facing)
            for w in p.wands(tags):
                if self.loaded.get(w.id):
                    d = self.aim(w.forward, p)
                    for k in range(1, 6):  # aim dots
                        pygame.draw.circle(s, p.color, w.pos + d * (w.size + k * self.r * 2.2), 3)
                    pygame.draw.circle(s, p.color, w.pos + d * (w.size + self.r + 4), self.r)
                    pygame.draw.circle(s, p.color, w.pos, w.size * 1.25, 4)
                else:
                    pygame.draw.circle(s, tuple(c // 3 for c in p.color), w.pos, w.size * 1.25, 2)
        for b in self.balls:
            pygame.draw.circle(s, b.color, b.pos, self.r)
            pygame.draw.circle(s, (255, 255, 255), b.pos - self.r * 0.35, self.r * 0.25)
        self.effects = [e for e in self.effects if now - e[2] < 0.5]
        for kind, pos, t0, color in self.effects:
            age = (now - t0) / 0.5
            pygame.draw.circle(s, color, pos, self.r * (1 + 2 * age), 3)
        if self.winner is not None:
            win = self.players[self.winner]
            for p in self.players:
                msg = "you win!" if p is win else "you lose"
                ui.text(s, self.big, msg, (f.centerx - p.dir * f.width * 0.25, f.centery + f.height * 0.2),
                        win.color, p.facing)
        for p in self.players:
            for w in p.wands(tags):
                ui.quiet_wand(s, w)
