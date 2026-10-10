"""Sneak: a single agent creeps through a maze to steal a diamond and get out.

Put any wand on the green start spot: that wand is the agent (its tag). Guards
patrol the maze with flashlight cones (blocked by walls). Stay in a cone for
too long and you're caught and start over. Brushing a wall (or lifting the wand
over one) makes a noise that brings nearby guards to look. Grab the diamond,
then get back to the start.
"""
import math
import random
import time
from collections import deque

import cv2
import numpy as np

from .. import ui
from ..projector import pygame
from ..tracker import WAND_IDS

CELL_MM = 48
WALL_MM = 5
AGENT_MM = 7  # agent radius for wall bumps
GUARDS = 3
GUARD_SPEED_MM = 45  # mm/s on patrol (alerted guards are faster)
VIEW_MM, VIEW_FOV = 150, math.radians(70)
SPOT_S = 0.5  # seconds inside a cone before you're caught
NOISE_MM = 260  # guards closer than this hear a wall bump
RAYS = 28


class Guard:
    def __init__(self, cell, game):
        self.game = game
        self.pos = game.cell_center(cell)
        self.heading = random.uniform(0, 2 * math.pi)
        self.path = deque()
        self.alert_until = 0.0
        self.cone = None

    def plan(self, target):
        self.path = deque(self.game.route(self.game.cell_of(self.pos), target))

    def update(self, dt, now):
        if not self.path:
            self.plan(random.choice(self.game.open_cells))
        speed = GUARD_SPEED_MM * self.game.mm * (1.8 if now < self.alert_until else 1.0)
        target = self.game.cell_center(self.path[0])
        d = target - self.pos
        dist = np.linalg.norm(d)
        if dist < 2:
            self.path.popleft()
            return
        want = math.atan2(d[1], d[0])
        turn = (want - self.heading + math.pi) % (2 * math.pi) - math.pi
        self.heading += float(np.clip(turn, -4 * dt, 4 * dt))  # turn smoothly towards where it walks
        if abs(turn) < 0.6:
            self.pos = self.pos + d / dist * min(speed * dt, dist)


class Sneak:
    title = "Sneak"

    def __init__(self, app):
        self.app, self.mm = app, app.px_per_mm
        self.font = pygame.font.SysFont(None, int(13 * self.mm))
        self.big = pygame.font.SysFont(None, int(26 * self.mm))
        self._build()
        self.new_round(time.monotonic())

    # --- maze ------------------------------------------------------------
    def _build(self):
        f = self.app.field.copy()
        f.top += int(20 * self.mm)  # a strip for the status text (it reads from the table's top edge)
        f.height -= int(20 * self.mm)
        c = CELL_MM * self.mm
        self.cell = c
        self.cols, self.rows = int(f.width // c), int(f.height // c)
        self.origin = np.array([f.centerx - self.cols * c / 2, f.centery - self.rows * c / 2])
        # walls between cells: True = wall. Random depth-first maze, then knock out walls for loops.
        self.h_walls = np.ones((self.rows + 1, self.cols), bool)  # above cell (r, c)
        self.v_walls = np.ones((self.rows, self.cols + 1), bool)  # left of cell (r, c)
        seen = np.zeros((self.rows, self.cols), bool)
        stack = [(self.rows - 1, 0)]
        seen[stack[0]] = True
        while stack:
            r, col = stack[-1]
            nxt = [(r + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= r + dr < self.rows and 0 <= col + dc < self.cols and not seen[r + dr, col + dc]]
            if not nxt:
                stack.pop()
                continue
            nr, nc = random.choice(nxt)
            self._open((r, col), (nr, nc))
            seen[nr, nc] = True
            stack.append((nr, nc))
        for _ in range(int(self.rows * self.cols * 0.18)):  # loops, so guards can be dodged
            r, col = random.randrange(self.rows), random.randrange(self.cols)
            dr, dc = random.choice(((1, 0), (0, 1)))
            if r + dr < self.rows and col + dc < self.cols:
                self._open((r, col), (r + dr, col + dc))
        self.open_cells = [(r, col) for r in range(self.rows) for col in range(self.cols)]
        self.start_cell = (self.rows - 1, 0)
        dist = self._bfs(self.start_cell)
        self.diamond_cell = max(dist, key=dist.get)
        self._segments()

    def _open(self, a, b):
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            self.v_walls[r1, max(c1, c2)] = False
        else:
            self.h_walls[max(r1, r2), c1] = False

    def neighbours(self, cell):
        r, c = cell
        if r > 0 and not self.h_walls[r, c]:
            yield (r - 1, c)
        if r < self.rows - 1 and not self.h_walls[r + 1, c]:
            yield (r + 1, c)
        if c > 0 and not self.v_walls[r, c]:
            yield (r, c - 1)
        if c < self.cols - 1 and not self.v_walls[r, c + 1]:
            yield (r, c + 1)

    def _bfs(self, start):
        dist, q = {start: 0}, deque([start])
        while q:
            cur = q.popleft()
            for n in self.neighbours(cur):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return dist

    def route(self, a, b):
        prev, q = {a: None}, deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for n in self.neighbours(cur):
                if n not in prev:
                    prev[n] = cur
                    q.append(n)
        path, cur = [], b
        while cur is not None and cur in prev:
            path.append(cur)
            cur = prev[cur]
        return path[::-1][1:]

    def _segments(self):
        c, o = self.cell, self.origin
        segs = []
        for r in range(self.rows + 1):
            for col in range(self.cols):
                if self.h_walls[r, col]:
                    segs.append((o + (col * c, r * c), o + ((col + 1) * c, r * c)))
        for r in range(self.rows):
            for col in range(self.cols + 1):
                if self.v_walls[r, col]:
                    segs.append((o + (col * c, r * c), o + (col * c, (r + 1) * c)))
        self.seg_a = np.array([a for a, _ in segs])
        self.seg_b = np.array([b for _, b in segs])

    def cell_center(self, cell):
        r, c = cell
        return self.origin + (np.array([c, r]) + 0.5) * self.cell

    def cell_of(self, p):
        col, r = ((np.asarray(p) - self.origin) // self.cell).astype(int)
        return int(np.clip(r, 0, self.rows - 1)), int(np.clip(col, 0, self.cols - 1))

    # --- round -----------------------------------------------------------
    def new_round(self, now):
        self.agent_id = None
        self.agent = None
        self.last_agent = None
        self.has_diamond = False
        self.spotted = 0.0
        self.caught_at = None
        self.won_at = None
        self.t_start = None
        far = [cell for cell in self.open_cells
               if np.linalg.norm(self.cell_center(cell) - self.cell_center(self.start_cell)) > 5 * self.cell]
        self.guards = [Guard(random.choice(far), self) for _ in range(GUARDS)]

    def update(self, dt, now, tags, presses):
        if self.won_at is not None:
            if now - self.won_at > 8:
                self.new_round(now)
            return
        if self.caught_at is not None:
            if now - self.caught_at > 2.5:
                self.new_round(now)
            return
        start = self.cell_center(self.start_cell)
        if self.agent_id is None:  # waiting for a wand on the start spot
            for k in WAND_IDS:
                if k in tags and np.linalg.norm(tags[k].pos - start) < self.cell * 0.6:
                    self.agent_id, self.t_start = k, now
                    self.app.sound.play("start")
                    self.app.fx.ring(start, (80, 230, 120), self.cell * 0.5, n=24, speed=100)
            for g in self.guards:
                g.update(dt, now)
            return
        t = tags.get(self.agent_id)
        self.agent = None if t is None else t.pos.copy()
        for g in self.guards:
            g.update(dt, now)
        if self.agent is None:
            return  # lifted: wait for it to come back (checked for jumps below when it does)
        if self.last_agent is not None and np.linalg.norm(self.agent - self.last_agent) > self.cell * 1.2:
            self._noise(self.agent, now)  # moved over walls while lifted
        elif self._bumps(self.agent):
            self._noise(self.agent, now)
        self.last_agent = self.agent
        self._vision(now, dt)
        if not self.has_diamond and np.linalg.norm(self.agent - self.cell_center(self.diamond_cell)) < self.cell * 0.4:
            self.has_diamond = True
            self.app.sound.play("deposit")
            self.app.fx.burst(self.agent, (120, 240, 230), n=50, speed=300, life=0.8,
                              colors=[(120, 240, 230), (255, 255, 255)])
        if self.has_diamond and np.linalg.norm(self.agent - start) < self.cell * 0.5:
            self.won_at = now
            self.app.sound.play("win")
            self.app.fx.confetti(self.app.field)

    def _bumps(self, p):
        ab = self.seg_b - self.seg_a
        t = np.clip(np.sum((p - self.seg_a) * ab, axis=1) / np.sum(ab * ab, axis=1), 0, 1)
        d = np.linalg.norm(self.seg_a + ab * t[:, None] - p, axis=1)
        return bool(np.any(d < (AGENT_MM + WALL_MM / 2) * self.mm))

    def _noise(self, p, now):
        if now - getattr(self, "_last_noise", 0) < 1.0:
            return
        self._last_noise = now
        self.app.sound.play("block")
        self.app.fx.ring(p, (255, 200, 80), self.cell * 0.4, n=20, speed=200, life=0.5)
        for g in self.guards:
            if np.linalg.norm(g.pos - p) < NOISE_MM * self.mm:
                g.plan(self.cell_of(p))
                g.alert_until = now + 4.0

    def _cone(self, g):
        """Flashlight polygon: rays from the guard, each stopped at the first wall."""
        ang = g.heading + np.linspace(-VIEW_FOV / 2, VIEW_FOV / 2, RAYS)
        d = np.stack([np.cos(ang), np.sin(ang)], 1)  # rays x 2
        R = VIEW_MM * self.mm
        e = self.seg_b - self.seg_a  # segs x 2
        diff = self.seg_a - g.pos  # segs x 2
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # rays x segs
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (diff[None, :, 0] * e[None, :, 1] - diff[None, :, 1] * e[None, :, 0]) / den
            u = (diff[None, :, 0] * d[:, None, 1] - diff[None, :, 1] * d[:, None, 0]) / den
        hit = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        dist = np.where(hit, t, np.inf).min(axis=1)
        dist = np.minimum(dist, R)
        return np.vstack([g.pos, g.pos + d * dist[:, None]])

    def _vision(self, now, dt):
        seen = False
        for g in self.guards:
            g.cone = self._cone(g)
            if np.linalg.norm(self.agent - g.pos) < VIEW_MM * self.mm:
                if cv2.pointPolygonTest(g.cone.astype(np.float32), (float(self.agent[0]), float(self.agent[1])), False) >= 0:
                    seen = True
        if seen:
            self.spotted += dt
            self.app.sound.play("wall", 0.6, min_gap=0.12)
            if self.spotted >= SPOT_S:
                self.caught_at = now
                self.app.sound.play("buzz")
                self.app.fx.burst(self.agent, (255, 60, 40), n=60, speed=300, life=0.8)
        else:
            self.spotted = max(0.0, self.spotted - dt * 0.5)

    # --- drawing -------------------------------------------------------
    def draw(self, s, now):
        s.fill(ui.BACKGROUND)
        c = self.cell
        f = self.app.field
        pygame.draw.rect(s, (70, 70, 70), (*self.origin, self.cols * c, self.rows * c))
        for g in self.guards:  # flashlight cones under everything else
            cone = g.cone if g.cone is not None else self._cone(g)
            pygame.draw.polygon(s, (190, 175, 90) if now >= g.alert_until else (220, 120, 60), cone)
        start = self.cell_center(self.start_cell)
        pygame.draw.circle(s, (60, 200, 100), start, c * 0.38)
        if not self.has_diamond:
            dpos = self.cell_center(self.diamond_cell)
            r = c * 0.25 * (1 + 0.1 * math.sin(now * 5))
            pygame.draw.polygon(s, (120, 240, 230), [dpos + (0, -r), dpos + (r, 0), dpos + (0, r), dpos + (-r, 0)])
            pygame.draw.polygon(s, (255, 255, 255), [dpos + (0, -r), dpos + (r, 0), dpos + (0, r), dpos + (-r, 0)], 2)
        for a, b in zip(self.seg_a, self.seg_b):
            pygame.draw.line(s, (25, 25, 25), a, b, int(WALL_MM * self.mm))
        for g in self.guards:
            pygame.draw.circle(s, (200, 40, 40), g.pos, c * 0.2)
            pygame.draw.line(s, (255, 255, 255), g.pos, g.pos + np.array([math.cos(g.heading), math.sin(g.heading)]) * c * 0.25, 3)
        if self.agent is not None:
            col = (255, 80, 60) if self.spotted > 0 else (80, 230, 120)
            pygame.draw.circle(s, col, self.agent, c * 0.32, 3)
            if self.has_diamond:
                pygame.draw.circle(s, (120, 240, 230), self.agent, c * 0.12)
            if self.spotted > 0:  # detection meter
                pygame.draw.arc(s, (255, 80, 60), pygame.Rect(self.agent[0] - c * 0.42, self.agent[1] - c * 0.42, c * 0.84, c * 0.84),
                                math.pi / 2 - 2 * math.pi * min(1, self.spotted / SPOT_S), math.pi / 2, 4)
        W, H = self.app.view.size
        face = self.app.facing
        if self.won_at is not None:
            msg = f"diamond stolen in {self.won_at - self.t_start:.0f}s!"
        elif self.caught_at is not None:
            msg = "caught! back to the start..."
        elif self.agent_id is None:
            msg = "put a wand on the green spot to start"
        else:
            msg = "bring the diamond back to the green spot" if self.has_diamond else "steal the diamond - stay out of the lights"
        ui.text(s, self.font, msg, (W / 2, self.app.field.top + 9 * self.mm), (230, 230, 230), face)
