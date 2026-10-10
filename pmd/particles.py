"""Simple particle effects drawn on top of every scene (in table-canvas pixels)."""
import math
import random

import numpy as np

from .projector import pygame


class Particles:
    def __init__(self, max_particles=1500):
        self.p = np.zeros((0, 2))
        self.v = np.zeros((0, 2))
        self.life = np.zeros(0)
        self.age = np.zeros(0)
        self.size = np.zeros(0)
        self.color = np.zeros((0, 3))
        self.drag = np.zeros(0)
        self.max = max_particles

    def burst(self, pos, color, n=20, speed=200.0, life=0.6, size=4.0, spread=2 * math.pi, direction=0.0,
              drag=2.0, colors=None):
        """n particles from pos; speed in canvas px/s; direction/spread in radians."""
        ang = direction + (np.random.rand(n) - 0.5) * spread
        sp = speed * (0.3 + 0.7 * np.random.rand(n))
        cols = np.array([random.choice(colors) if colors else color for _ in range(n)], float)
        self._add(np.tile(np.asarray(pos, float), (n, 1)), np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None],
                  life * (0.6 + 0.4 * np.random.rand(n)), size * (0.6 + 0.8 * np.random.rand(n)), cols,
                  np.full(n, drag))

    def ring(self, pos, color, radius, n=24, speed=60.0, life=0.4, size=3.0):
        """Particles spaced on a circle, drifting outwards."""
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        self._add(np.asarray(pos, float) + d * radius, d * speed, np.full(n, life), np.full(n, size),
                  np.tile(np.asarray(color, float), (n, 1)), np.full(n, 1.0))

    def confetti(self, rect, n=150):
        """Celebration: bright particles raining over a rectangle."""
        x = rect.left + np.random.rand(n) * rect.width
        y = rect.top + np.random.rand(n) * rect.height * 0.2
        colors = [(255, 80, 80), (255, 210, 60), (80, 230, 120), (240, 120, 255), (255, 255, 255), (255, 150, 40)]
        cols = np.array([random.choice(colors) for _ in range(n)], float)
        v = np.stack([(np.random.rand(n) - 0.5) * 200, 150 + np.random.rand(n) * 250], 1)
        self._add(np.stack([x, y], 1), v, 1.5 + np.random.rand(n) * 1.5, 3 + np.random.rand(n) * 4, cols, np.full(n, 0.3))

    def _add(self, p, v, life, size, color, drag):
        self.p = np.vstack([self.p, p])[-self.max:]
        self.v = np.vstack([self.v, v])[-self.max:]
        self.life = np.concatenate([self.life, life])[-self.max:]
        self.age = np.concatenate([self.age, np.zeros(len(life))])[-self.max:]
        self.size = np.concatenate([self.size, size])[-self.max:]
        self.color = np.vstack([self.color, color])[-self.max:]
        self.drag = np.concatenate([self.drag, drag])[-self.max:]

    def update(self, dt):
        if not len(self.p):
            return
        self.age += dt
        self.v *= np.exp(-self.drag * dt)[:, None]
        self.p += self.v * dt
        keep = self.age < self.life
        for name in ("p", "v", "life", "age", "size", "color", "drag"):
            setattr(self, name, getattr(self, name)[keep])

    def draw(self, surf):
        fade = 1 - self.age / self.life
        for p, s, c, f in zip(self.p, self.size, self.color, fade):
            pygame.draw.circle(surf, tuple(int(v * min(1.0, f * 1.5)) for v in c), p, max(1.0, s * (0.4 + 0.6 * f)))

    def clear(self):
        self.p = self.p[:0]
        self.v, self.life, self.age = self.v[:0], self.life[:0], self.age[:0]
        self.size, self.color, self.drag = self.size[:0], self.color[:0], self.drag[:0]
