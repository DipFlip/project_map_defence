"""Operation: two surgeons race to remove organs without touching the others.

Each player (at a short end) has a patient on their half of the table. Either
of their wands is the tweezers: the tool tip sits just beyond the tag's printed
top. Touch the highlighted organ to take it out. Touching any other organ
buzzes, lights up the patient's nose and freezes your tweezers for a moment.
First to clear all organs wins.
"""
import math
import random
import time

import numpy as np

from .. import ui
from ..projector import pygame
from .players import make_players

# organs in the patient's frame, mm: u from the feet (0) to the head, v across the body
ORGANS = [
    ("brain", 272, 0, 18, 15, (240, 170, 190)),
    ("left lung", 205, 36, 15, 17, (240, 130, 140)),
    ("right lung", 205, -36, 15, 17, (240, 130, 140)),
    ("heart", 200, 0, 11, 10, (220, 40, 50)),
    ("liver", 150, -28, 15, 22, (150, 60, 40)),
    ("stomach", 150, 30, 14, 18, (230, 160, 120)),
    ("kidney", 105, 32, 9, 8, (170, 70, 70)),
    ("intestines", 62, 0, 22, 26, (230, 170, 140)),
]
ROUND = 7  # organs to remove per game (in random order)
TIP_MM = 4  # tweezer tip radius
FREEZE_S = 1.2
BODY_LEN = 300


class Patient:
    def __init__(self, player, center, scale, now):
        self.player, self.center, self.scale = player, np.asarray(center, float), scale
        self.axis = np.array([player.dir, 0.0])  # feet towards the player, head towards the middle
        self.side = np.array([0.0, player.dir])
        self.organs = {name: (u, v, rx, ry, color) for name, u, v, rx, ry, color in ORGANS}
        self.todo = random.sample([name for name, *_ in ORGANS], ROUND)
        self.removed = set()
        self.frozen_until = 0.0
        self.nose_until = 0.0
        self.mistakes = 0
        self.done_at = None
        self.started = now

    def to_canvas(self, u, v):
        return self.center + (self.axis * (u - BODY_LEN / 2) + self.side * v) * self.scale

    def to_body(self, p):
        d = (np.asarray(p, float) - self.center) / self.scale
        return np.dot(d, self.axis) + BODY_LEN / 2, np.dot(d, self.side)

    def organ_at(self, tip):
        """Name of the (still present) organ the tip touches, or None."""
        u, v = self.to_body(tip)
        for name, (ou, ov, rx, ry, _) in self.organs.items():
            if name in self.removed:
                continue
            if ((u - ou) / (rx + TIP_MM)) ** 2 + ((v - ov) / (ry + TIP_MM)) ** 2 <= 1:
                return name
        return None


class Operation:
    title = "Operation"

    def __init__(self, app):
        self.app, self.field, self.mm = app, app.field, app.px_per_mm
        f = self.field
        self.font = pygame.font.SysFont(None, int(14 * self.mm))
        self.big = pygame.font.SysFont(None, int(24 * self.mm))
        self.players = make_players(app.tags, f)
        self.new_game(time.monotonic())

    def new_game(self, now):
        f = self.field
        self.patients = [Patient(p, (f.centerx - p.dir * f.width / 4, f.centery), self.mm, now) for p in self.players]
        self.winner, self.winner_at = None, 0.0

    def tips(self, tags, p):
        """Tweezer tips of this player's wands (canvas px)."""
        return [(w, w.pos + w.forward * w.size * 0.9) for w in p.wands(tags)]

    def update(self, dt, now, tags, presses):
        if self.winner is not None:
            if now - self.winner_at > 8:
                self.new_game(now)
            return
        for i, pat in enumerate(self.patients):
            if pat.done_at is not None or now < pat.frozen_until:
                continue
            target = pat.todo[0]
            for w, tip in self.tips(tags, pat.player):
                hit = pat.organ_at(tip)
                if hit is None:
                    continue
                u, v, *_ , color = pat.organs[hit]
                if hit == target:
                    pat.removed.add(hit)
                    pat.todo.pop(0)
                    self.app.sound.play("organ")
                    self.app.fx.burst(pat.to_canvas(u, v), color, n=35, speed=250, life=0.6,
                                      colors=[color, (255, 255, 255), (255, 230, 120)])
                    if not pat.todo:
                        pat.done_at = now
                        self.winner, self.winner_at = i, now
                        self.app.sound.play("win")
                        self.app.fx.confetti(self.field)
                else:
                    pat.mistakes += 1
                    pat.frozen_until = now + FREEZE_S
                    pat.nose_until = now + FREEZE_S
                    self.app.sound.play("buzz")
                    self.app.fx.burst(tip, (255, 60, 40), n=25, speed=200, life=0.4)
                break

    # --- drawing -------------------------------------------------------
    def draw(self, s, now):
        s.fill(ui.BACKGROUND)
        pygame.draw.line(s, (70, 70, 70), (self.field.centerx, self.field.top), (self.field.centerx, self.field.bottom), 2)
        for i, pat in enumerate(self.patients):
            self._draw_patient(s, pat, now)
            for w, tip in self.tips(self.app.tags, pat.player):
                frozen = now < pat.frozen_until
                pygame.draw.line(s, (200, 200, 210), w.pos + w.forward * w.size * 0.5, tip, 3)
                pygame.draw.circle(s, (255, 70, 50) if frozen else (255, 255, 255), tip, TIP_MM * self.mm, 2)
                pygame.draw.circle(s, (255, 255, 255), tip, 2)
            p = pat.player
            label_pos = pat.center - pat.axis * (BODY_LEN / 2 + 25) * self.mm
            if self.winner is not None:
                msg = "you win!" if self.winner == i else "you lose"
                ui.text(s, self.big, msg, label_pos, p.color, p.facing)
            elif pat.todo:
                pulse = 0.5 + 0.5 * math.sin(now * 6)
                msg = f"take out the {pat.todo[0]}  ({ROUND - len(pat.todo) + 1}/{ROUND})"
                ui.text(s, self.font, msg, label_pos, tuple(int(150 + 105 * pulse) for _ in range(3)), p.facing)

    def _draw_patient(self, s, pat, now):
        mm = self.mm

        def poly(points_uv, color, width=0):
            pygame.draw.polygon(s, color, [pat.to_canvas(u, v) for u, v in points_uv], width)

        skin, outline = (230, 190, 160), (150, 110, 90)
        # legs, arms, torso, head (rough shapes)
        for sv in (-1, 1):
            poly([(40, sv * 12), (40, sv * 48), (-30, sv * 44), (-30, sv * 18)], skin)
            poly([(225, sv * 62), (225, sv * 80), (120, sv * 98), (110, sv * 84)], skin)
        poly([(40, -66), (240, -66), (250, -40), (250, 40), (240, 66), (40, 66), (30, 40), (30, -40)], skin)
        pygame.draw.circle(s, skin, pat.to_canvas(275, 0), 32 * mm)
        pygame.draw.circle(s, outline, pat.to_canvas(275, 0), 32 * mm, 2)
        # cavities (darker) and organs
        t = now - pat.started
        for name, (u, v, rx, ry, color) in pat.organs.items():
            ang = np.linspace(0, 2 * math.pi, 28)
            cav = [pat.to_canvas(u + (rx + 5) * math.cos(a), v + (ry + 5) * math.sin(a)) for a in ang]
            pygame.draw.polygon(s, (120, 70, 60), cav)
            if name in pat.removed:
                continue
            beat = 1 + 0.06 * math.sin(t * 8) if name == "heart" else 1.0
            pts = [pat.to_canvas(u + rx * beat * math.cos(a), v + ry * beat * math.sin(a)) for a in ang]
            pygame.draw.polygon(s, color, pts)
            if pat.todo and name == pat.todo[0]:
                pulse = 0.5 + 0.5 * math.sin(now * 6)
                pygame.draw.polygon(s, (255, 255, int(120 + 135 * pulse)), pts, max(2, int(3 * mm)))
        # the famous nose
        nose_on = now < pat.nose_until and int(now * 10) % 2 == 0
        pygame.draw.circle(s, (255, 30, 20) if nose_on else (200, 120, 110), pat.to_canvas(300, 0), 6 * mm)
