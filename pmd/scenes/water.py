"""Water: guide two coloured liquids into the containers that want them.

Tags A (10) and E (14) are taps on the far edge of the table (the side the
snack castle's text is on): turn a tag from how it lay at the start to open it
(a quarter turn = fully open). A pours orange, E pours green. The liquid flows
towards the players. Every other tag is a tilting platform, and the wands'
whole sticks are platforms too. Containers show the colour they want; fill one
and it disappears, and a new one turns up somewhere else.

Fluid: a few hundred particles that push each other apart (with a little
viscosity), drawn as blurred, threshold-clipped blobs.
Containers reject the wrong liquid (it fizzles away inside them).
"""
import math
import random
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .. import ui
from ..projector import pygame
from ..tracker import WAND_IDS

PIPES = {10: 0, 14: 1}  # tag id -> liquid
LIQUIDS = [(255, 140, 40), (40, 215, 160)]  # orange, green (nothing blue: blue is the wand buttons)
GRAVITY_MM = 900.0  # mm/s^2 towards the players (a slow, readable flow)
AIR_DRAG = 4.0  # 1/s: falling drops settle at ~GRAVITY/AIR_DRAG and stay close enough to look like a stream
DROP_MM = 3.0  # particle radius
MAX_PARTICLES = 700
FLOW_PER_S = 55  # particles per second from a fully open tap
CUP_W_MM, CUP_H_MM = 70, 50
CUP_FILL = 40  # particles of the right colour to complete a container
CUPS = 3
RENDER_SCALE = 4  # fluid is splatted and blurred at 1/RENDER_SCALE resolution


class Cup:
    def __init__(self, rect, want):
        self.rect, self.want = rect, want
        self.level = 0.0

    def walls(self):
        """Three segments; the opening faces the taps (larger y)."""
        r = self.rect
        return [((r.left, r.top), (r.left, r.bottom)), ((r.left, r.top), (r.right, r.top)), ((r.right, r.top), (r.right, r.bottom))]


class Water:
    title = "Water"

    def __init__(self, app):
        self.app, self.field, self.mm = app, app.field, app.px_per_mm
        self.r = DROP_MM * self.mm
        self.h = self.r * 2.4  # interaction distance
        self.g = np.array([0.0, -GRAVITY_MM * self.mm])  # towards canvas top = the players
        self.font = pygame.font.SysFont(None, int(14 * self.mm))
        self.big = pygame.font.SysFont(None, int(24 * self.mm))
        self.p = np.zeros((0, 2))
        self.v = np.zeros((0, 2))
        self.kind = np.zeros(0, int)
        self.born = np.zeros(0)
        self.valve_ref = {}
        self.emit_acc = {k: 0.0 for k in PIPES}
        self.cups = []
        self.score = 0
        self.next_cup = 0.0
        self.t0 = time.monotonic()
        for _ in range(CUPS):
            self._spawn_cup()

    # --- level -----------------------------------------------------------
    def _spawn_cup(self):
        f = self.field
        w, h = CUP_W_MM * self.mm, CUP_H_MM * self.mm
        for _ in range(50):  # find a spot that doesn't overlap another cup
            x = random.uniform(f.left + w, f.right - 2 * w)
            y = random.uniform(f.top + f.height * 0.08, f.top + f.height * 0.35)
            rect = pygame.Rect(int(x), int(y), int(w), int(h))
            if not any(rect.inflate(w * 0.6, h).colliderect(c.rect) for c in self.cups):
                self.cups.append(Cup(rect, random.randrange(len(LIQUIDS))))
                return

    def segments(self, tags):
        """(a, b, half thickness) for every platform: other tags, wand sticks, cup walls."""
        segs = []
        for k, t in tags.items():
            if k in PIPES or not 10 <= k <= 19:
                continue
            if k in WAND_IDS:  # the whole stick, from the tag to the end of the handle
                segs.append((t.pos + t.forward * t.size * 0.5, t.pos - t.forward * t.size * 4.4, t.size * 0.45))
            else:  # a plank across the tag, tilting with it
                across = np.array([-t.forward[1], t.forward[0]])
                segs.append((t.pos - across * t.size * 1.4, t.pos + across * t.size * 1.4, t.size * 0.2))
        for c in self.cups:
            segs += [(np.array(a, float), np.array(b, float), 2.0) for a, b in c.walls()]
        return segs

    # --- simulation ----------------------------------------------------
    def update(self, dt, now, tags, presses):
        self._taps(dt, now, tags)
        segs = self.segments(tags)
        for _ in range(2):
            self._step(dt / 2, segs)
        self._cups(now)
        f = self.field
        keep = ((self.p[:, 1] > f.top - 40) & (self.p[:, 0] > f.left - 40) & (self.p[:, 0] < f.right + 40)
                & (now - self.born < 40))
        self._keep(keep)
        if len(self.cups) < CUPS and now >= self.next_cup:
            self._spawn_cup()

    def _taps(self, dt, now, tags):
        for k, kind in PIPES.items():
            t = tags.get(k)
            if t is None:
                continue
            ref = self.valve_ref.setdefault(k, t.angle)
            turn = abs((t.angle - ref + math.pi) % (2 * math.pi) - math.pi)
            opening = min(1.0, turn / (math.pi / 2))
            was = self.emit_acc.get((k, "open"), 0.0)
            if (opening > 0.1) != (was > 0.1):
                self.app.sound.play("valve")
            self.emit_acc[(k, "open")] = opening
            self.emit_acc[k] += FLOW_PER_S * opening * dt
            n = int(self.emit_acc[k])
            if n <= 0 or len(self.p) >= MAX_PARTICLES:
                continue
            self.emit_acc[k] -= n
            nozzle = t.pos + np.array([0.0, -1.0]) * t.size * 0.9
            jitter = np.random.uniform(-1, 1, (n, 2)) * self.r
            self._add(nozzle + jitter, np.tile([0.0, -60.0 * self.mm], (n, 1)) + jitter * 0.5, kind, now)

    def _add(self, p, v, kind, now):
        self.p = np.vstack([self.p, p])
        self.v = np.vstack([self.v, v])
        self.kind = np.concatenate([self.kind, np.full(len(p), kind)])
        self.born = np.concatenate([self.born, np.full(len(p), now)])

    def _keep(self, keep):
        self.p, self.v, self.kind, self.born = self.p[keep], self.v[keep], self.kind[keep], self.born[keep]

    def _step(self, dt, segs):
        if not len(self.p):
            return
        acc = np.tile(self.g, (len(self.p), 1))
        pairs = cKDTree(self.p).query_pairs(self.h, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = self.p[i] - self.p[j]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-3)
            n = d / dist[:, None]
            push = (1 - dist / self.h) ** 2 * 6000.0 * self.mm  # pressure: keeps the volume
            rel = self.v[i] - self.v[j]
            visc = 4.0 * (1 - dist / self.h)[:, None] * rel  # viscosity: moves neighbours together
            f = n * push[:, None] - visc
            np.add.at(acc, i, f)
            np.add.at(acc, j, -f)
        self.v += acc * dt
        self.v *= math.exp(-AIR_DRAG * dt)
        speed = np.linalg.norm(self.v, axis=1)
        cap = 900.0 * self.mm
        self.v *= np.minimum(1.0, cap / np.maximum(speed, 1e-6))[:, None]
        self.p += self.v * dt
        for a, b, half in segs:
            ab = b - a
            L2 = max(float(np.dot(ab, ab)), 1e-6)
            t = np.clip(((self.p - a) @ ab) / L2, 0, 1)
            closest = a + t[:, None] * ab
            off = self.p - closest
            dist = np.linalg.norm(off, axis=1)
            reach = half + self.r
            hit = dist < reach
            if not hit.any():
                continue
            n = off[hit] / np.maximum(dist[hit], 1e-6)[:, None]
            self.p[hit] = closest[hit] + n * reach
            vn = np.sum(self.v[hit] * n, axis=1)
            into = vn < 0
            vh = self.v[hit]
            vh[into] -= (1.2 * vn[into])[:, None] * n[into]  # stop going into the platform, a little bounce
            vh *= 0.995  # a touch of friction
            self.v[hit] = vh

    def _cups(self, now):
        for c in list(self.cups):
            r = c.rect
            inside = (self.p[:, 0] > r.left) & (self.p[:, 0] < r.right) & (self.p[:, 1] > r.top) & (self.p[:, 1] < r.bottom)
            wrong = inside & (self.kind != c.want)
            if wrong.any():  # a container rejects the wrong liquid: it fizzles away
                for q in self.p[wrong][:3]:
                    self.app.fx.burst(q, (200, 200, 200), n=3, speed=60, life=0.3, size=2)
                self.app.sound.play("wall", 0.3, min_gap=0.15)
                self._keep(~wrong)
                inside = inside[~wrong]
            c.level = int(np.sum(inside)) / CUP_FILL
            if c.level >= 1.0:
                self._keep(~inside)
                self.cups.remove(c)
                self.score += 1
                self.next_cup = now + 1.0
                self.app.sound.play("filled")
                self.app.fx.burst(r.center, LIQUIDS[c.want], n=60, speed=350, life=0.8,
                                  colors=[LIQUIDS[c.want], (255, 255, 255), (255, 230, 120)])

    # --- drawing -------------------------------------------------------
    def draw(self, s, now):
        s.fill(ui.BACKGROUND)
        tags = self.app.tags
        for c in self.cups:  # containers: wanted colour, fill level
            r = c.rect
            col = LIQUIDS[c.want]
            fill = r.copy()
            fill.height = int(r.height * min(1.0, c.level))
            fill.top = r.top
            pygame.draw.rect(s, tuple(v // 3 for v in col), r)
            pygame.draw.rect(s, tuple(v // 2 for v in col), fill)
            for a, b in c.walls():
                pygame.draw.line(s, col, a, b, max(3, int(2.5 * self.mm)))
        for a, b, half in self.segments(tags)[: -3 * len(self.cups) or None]:  # platforms (not cup walls)
            pygame.draw.line(s, (170, 170, 180), a, b, max(2, int(half * 2)))
        self._draw_fluid(s)
        for k, kind in PIPES.items():  # taps: a pipe from the far edge, a valve wheel that turns with the tag
            t = tags.get(k)
            if t is None:
                continue
            col = LIQUIDS[kind]
            top = (t.pos[0], self.app.view.size[1])
            pygame.draw.line(s, (120, 120, 130), top, t.pos, int(t.size * 0.9))
            pygame.draw.circle(s, col, t.pos, t.size * 1.3, max(2, int(2 * self.mm)))
            for i in range(4):
                a = t.angle + i * math.pi / 2
                pygame.draw.line(s, col, t.pos + np.array([math.cos(a), math.sin(a)]) * t.size * 0.8,
                                 t.pos + np.array([math.cos(a), math.sin(a)]) * t.size * 1.3, 3)
        W, H = self.app.view.size
        ui.text(s, self.font, f"containers filled: {self.score}   (turn tag A or E to open a tap)",
                (W / 2, H * 0.03), (200, 200, 200), self.app.facing)

    def _draw_fluid(self, s):
        """Metaballs per liquid: splat drops at low resolution, blur, turn the blurred field
        into a soft-edged alpha mask, smooth-scale it up and alpha-blit it. Only the
        bounding box of each liquid's drops is processed."""
        W, H = self.app.view.size
        k = RENDER_SCALE
        rr = max(2, int(round(self.r * 2.0 / k)))
        pad = rr * 3
        for kind, col in enumerate(LIQUIDS):
            pts = self.p[self.kind == kind] / k
            if not len(pts):
                continue
            x0, y0 = np.maximum(np.floor(pts.min(axis=0)).astype(int) - pad, 0)
            x1, y1 = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + pad, (W // k, H // k))
            if x1 <= x0 or y1 <= y0:
                continue
            splat = np.zeros((y1 - y0, x1 - x0), np.uint8)
            for x, y in (pts - (x0, y0)).astype(int):
                cv2.circle(splat, (int(x), int(y)), rr, 255, -1)
            field = cv2.GaussianBlur(splat, (0, 0), rr * 0.8).astype(np.int16)
            surf = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
            surf.fill(col)
            lighter = tuple(min(255, int(v * 1.25)) for v in col)
            rgb = pygame.surfarray.pixels3d(surf)
            rgb[(field > 210).T] = lighter  # lighter core
            del rgb
            alpha = pygame.surfarray.pixels_alpha(surf)
            alpha[:] = np.clip((field - 90) * 8, 0, 255).astype(np.uint8).T  # steep but soft edge
            del alpha
            s.blit(pygame.transform.smoothscale(surf, ((x1 - x0) * k, (y1 - y0) * k)), (x0 * k, y0 * k))
