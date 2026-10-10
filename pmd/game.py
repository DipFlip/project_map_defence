"""Minimal tower-defence loop on top of the tracker.

Everything is in projector pixels. Towers are physical tags (ids in
TOWER_IDS); enemies walk a path across the calibrated table area and towers
shoot whatever is in range. This is a skeleton to build the real game on.
"""
import math
import random
import time

import numpy as np

from .projector import pygame

TOWER_IDS = range(10, 20)  # the cut-out tags A-J from the print kit
TOWER_RANGE = 160
TOWER_COOLDOWN = 0.6


class Enemy:
    def __init__(self, path, speed):
        self.path, self.speed = path, speed
        self.seg, self.t = 0, 0.0
        self.pos = np.array(path[0], float)
        self.hp = self.max_hp = 3

    def update(self, dt):
        """Advance along the path; returns True when the end is reached."""
        remaining = self.speed * dt
        while remaining > 0 and self.seg < len(self.path) - 1:
            a, b = np.array(self.path[self.seg]), np.array(self.path[self.seg + 1])
            seg_len = np.linalg.norm(b - a)
            step = min(remaining, (1 - self.t) * seg_len)
            self.t += step / seg_len
            remaining -= step
            if self.t >= 1 - 1e-9:
                self.seg, self.t = self.seg + 1, 0.0
            self.pos = a + (b - a) * self.t if self.t else np.array(self.path[self.seg], float)
        return self.seg >= len(self.path) - 1


class Game:
    def __init__(self, proj, tracker, calib):
        self.proj, self.tracker, self.calib = proj, tracker, calib
        self.font = pygame.font.SysFont(None, 40)
        poly = np.array(calib.table_poly or [(0, 0), (proj.width, 0), (proj.width, proj.height), (0, proj.height)])
        self.table_poly = poly
        self.path = self._make_path(poly)
        self.enemies, self.shots = [], []
        self.cooldowns = {}
        self.lives, self.score = 20, 0
        self.next_spawn = 0.0

    @staticmethod
    def _make_path(poly):
        """A zig-zag path inside the table's bounding box (shrunk a bit)."""
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        mx, my = (x1 - x0) * 0.12, (y1 - y0) * 0.15
        x0, x1, y0, y1 = x0 + mx, x1 - mx, y0 + my, y1 - my
        n = 4
        pts = []
        for i in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            pts += [(x, y0), (x, y1)] if i % 2 == 0 else [(x, y1), (x, y0)]
        return pts

    def update(self, dt, now):
        if now >= self.next_spawn:
            self.enemies.append(Enemy(self.path, random.uniform(60, 90)))
            self.next_spawn = now + 1.5
        for e in list(self.enemies):
            if e.update(dt):
                self.enemies.remove(e)
                self.lives -= 1
        towers = {k: t for k, t in self.tracker.tags().items() if k in TOWER_IDS}
        for k, tower in towers.items():
            if now < self.cooldowns.get(k, 0):
                continue
            in_range = [e for e in self.enemies if np.linalg.norm(e.pos - tower.pos) < TOWER_RANGE]
            if in_range:
                target = min(in_range, key=lambda e: np.linalg.norm(e.pos - tower.pos))
                target.hp -= 1
                self.shots.append((tower.pos.copy(), target.pos.copy(), now))
                self.cooldowns[k] = now + TOWER_COOLDOWN
                if target.hp <= 0:
                    self.enemies.remove(target)
                    self.score += 1
        self.shots = [s for s in self.shots if now - s[2] < 0.12]
        self.towers = towers

    def draw(self):
        s = self.proj.screen
        s.fill((0, 0, 0))
        pygame.draw.polygon(s, (25, 30, 25), self.table_poly)
        pygame.draw.lines(s, (90, 70, 40), False, self.path, 34)
        for tower in self.towers.values():
            x, y = tower.pos
            pygame.draw.circle(s, (40, 60, 110), (x, y), TOWER_RANGE, 2)
            pygame.draw.circle(s, (255, 255, 255), (x, y), max(tower.size * 0.9, 20))  # keep tag lit
        for e in self.enemies:
            pygame.draw.circle(s, (220, 50, 40), e.pos, 16)
            w = 32 * e.hp / e.max_hp
            pygame.draw.rect(s, (80, 220, 80), (e.pos[0] - 16, e.pos[1] - 26, w, 5))
        for a, b, _ in self.shots:
            pygame.draw.line(s, (255, 240, 120), a, b, 4)
        hud = self.font.render(f"lives {self.lives}   score {self.score}", True, (220, 220, 220))
        x0, y0 = self.table_poly.min(axis=0)
        s.blit(hud, (x0 + 20, y0 + 10))

    def snapshot(self, path):
        """Save what the camera currently sees (handy when debugging over ssh)."""
        frame, _, _ = self.tracker.camera.latest()
        if frame is not None:
            import cv2
            cv2.imwrite(path, frame)

    def run(self, duration=math.inf, snapshot_path=None, snapshot_every=5.0):
        t_prev = t_start = last_snap = time.monotonic()
        while self.proj.pump() and self.lives > 0 and time.monotonic() - t_start < duration:
            now = time.monotonic()
            self.update(min(now - t_prev, 0.1), now)
            t_prev = now
            self.draw()
            self.proj.flip()
            self.proj.clock.tick(60)
            if snapshot_path and now - last_snap > snapshot_every:
                self.snapshot(snapshot_path)
                last_snap = now
