"""Game select: hold a wand over a game and cover its blue spot to start it."""
import math

import numpy as np

from .. import art, ui
from ..projector import pygame
from ..tracker import WAND_IDS
from .players import COLORS, PAIRS


class Menu:
    title = "Menu"

    def __init__(self, app):
        self.app, self.field = app, app.field
        f = self.field
        self.choices = list(app.games)
        # grid in the middle of the table: 2 x 2 for four games, 3 x 2 for five or six
        n = len(self.choices)
        cols = 2 if n <= 4 else 3
        rows = math.ceil(n / cols)
        gap = f.height * 0.04
        th = (f.height * 0.78 - gap * (rows - 1)) / rows
        tw = min((f.width * 0.86 - gap * (cols - 1)) / cols, th * 1.6)
        x0 = f.centerx - (cols * tw + (cols - 1) * gap) / 2
        y0 = f.centery - (rows * th + (rows - 1) * gap) / 2
        self.tiles = [pygame.Rect(int(x0 + (i % cols) * (tw + gap)), int(y0 + (i // cols) * (th + gap)), int(tw), int(th))
                      for i in range(n)]
        self.small = pygame.font.SysFont(None, int(f.height * 0.045))
        # each title in the largest font that fits inside its tile
        self.title_fonts = {}
        for name, tile in zip(self.choices, self.tiles):
            size = int(tile.height * 0.22)
            while size > 10 and pygame.font.SysFont(None, size).size(app.games[name].title)[0] > tile.width * 0.88:
                size -= 2
            self.title_fonts[name] = pygame.font.SysFont(None, size)
        self.facing = app.facing
        self.clicks = []  # (wand position, button position, time, colour): feedback for every press
        self.up = np.array([0.0, 1.0]) if self.facing == "top" else np.array([0.0, -1.0])

    def update(self, dt, now, tags, presses):
        for k in presses:
            pair = next(i for i, p in enumerate(PAIRS) if k in p)
            self.clicks.append((tags[k].pos.copy(), tags[k].button_pos(), now, COLORS[pair]))
            self.app.sound.play("click")
            self.app.fx.burst(tags[k].pos, COLORS[pair], n=16, speed=250, life=0.4)
        self.clicks = [c for c in self.clicks if now - c[2] < 0.6]
        for k in presses:
            for name, tile in zip(self.choices, self.tiles):
                if tile.collidepoint(tags[k].pos):
                    self.app.switch(name)
                    return

    def draw(self, s, now):
        s.fill(ui.BACKGROUND)
        f = self.field
        tags = self.app.tags
        wands = [tags[k] for k in WAND_IDS if k in tags]
        ui.text(s, self.small, "hold a wand over a game and cover its blue spot", f.center + self.up * f.height * 0.45,
                (170, 170, 170), self.facing)
        for name, tile in zip(self.choices, self.tiles):
            hover = any(tile.collidepoint(w.pos) for w in wands)
            pygame.draw.rect(s, (60, 60, 60), tile, border_radius=16)
            pygame.draw.rect(s, (255, 220, 120) if hover else (140, 140, 140), tile, 6 if hover else 3, border_radius=16)
            ui.text(s, self.title_fonts[name], self.app.games[name].title, tile.center + self.up * tile.height * 0.32,
                    (255, 220, 120) if hover else (200, 200, 220), self.facing)
            self._icon(s, name, tile, now)
        for w in wands:
            pair = next(i for i, p in enumerate(PAIRS) if w.id in p)
            pygame.draw.circle(s, COLORS[pair], w.pos, w.size * 1.3, 4)
            ui.quiet_wand(s, w)
        for pos, button, t0, color in self.clicks:  # click burst around the wand that pressed
            age = (now - t0) / 0.6
            r = self.app.px_per_mm * (14 + 40 * age)
            pygame.draw.circle(s, color, pos, r, max(1, int(6 * (1 - age))))
            for i in range(8):
                a = i * math.pi / 4
                d = np.array([math.cos(a), math.sin(a)])
                pygame.draw.line(s, (255, 255, 255), pos + d * r, pos + d * (r + 8 * self.app.px_per_mm * (1 - age)), 3)

    def _icon(self, s, name, tile, now):
        c = np.array(tile.center, float)
        r = tile.width * 0.12
        if name == "snack":
            art.mouse(s, c + (-r * 1.2, 0), np.array([1.0, 0.0]), r * 0.5, (160, 155, 150), now, 1.0)
            pygame.draw.rect(s, art.CHEESE, (c[0] + r * 0.3, c[1] - r * 0.6, r * 1.2, r * 1.2))
        elif name == "pong":
            x = math.sin(now * 2) * r * 1.5
            pygame.draw.line(s, COLORS[0], c + (-r * 2, -r), c + (-r * 2, r), 8)
            pygame.draw.line(s, COLORS[1], c + (r * 2, -r), c + (r * 2, r), 8)
            pygame.draw.circle(s, (255, 255, 230), c + (x, 0), r * 0.25)
        elif name == "labyrinth":
            pygame.draw.rect(s, (150, 110, 70), (c[0] - r * 1.4, c[1] - r * 1.4, r * 2.8, r * 2.8))
            for k in range(3):
                pygame.draw.line(s, (90, 65, 40), c + (-r * 1.4 + k * r, -r * 0.6), c + (-r * 1.4 + k * r, r * 1.4), 4)
            pygame.draw.circle(s, (40, 200, 90), c + (r * 0.9, -r * 0.9), r * 0.3)
            pygame.draw.circle(s, (200, 205, 220), c + (math.sin(now) * r * 0.6, r * 0.5), r * 0.22)
        elif name == "operation":
            pygame.draw.ellipse(s, (230, 190, 160), (c[0] - r * 0.8, c[1] - r * 1.5, r * 1.6, r * 2.6))
            pygame.draw.circle(s, (220, 40, 50), c + (0, -r * 0.4), r * 0.3)
            pygame.draw.circle(s, (255, 30, 20) if int(now * 2) % 2 else (200, 120, 110), c + (0, -r * 1.55), r * 0.2)
        elif name == "water":
            for k, col in enumerate(((255, 140, 40), (40, 215, 160))):
                x = c[0] + (k - 0.5) * r * 1.6
                for j in range(4):
                    y = c[1] - r * 1.2 + ((now * 1.5 + j / 4) % 1.0) * r * 2.2
                    pygame.draw.circle(s, col, (x, y), r * 0.18)
            pygame.draw.lines(s, (200, 200, 200), False, [c + (-r * 1.4, r * 0.6), c + (-r * 1.4, r * 1.4),
                                                          c + (r * 1.4, r * 1.4), c + (r * 1.4, r * 0.6)], 4)
        elif name == "pins":
            for k in range(5):
                pygame.draw.circle(s, (250, 250, 250), c + (r * 1.8, (k - 2) * r * 0.5), r * 0.18)
            pygame.draw.circle(s, COLORS[0], c + (-r * 1.2 + math.sin(now * 2) * r, 0), r * 0.35)
