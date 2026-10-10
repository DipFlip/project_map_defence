"""Labyrinth: four players tilt a shared board with their wands to roll a ball to the goal.

The board sits in the middle of the table, half the table's shorter side wide.
Each wand controls the height of one corner (held about 5 to 40 cm above the
table, see pmd/heights.py). The ball starts in one corner, winds through a small
maze with holes, climbs a ramp onto a ledge, and has to jump a gap to reach the
goal. Falling into a hole puts it back at the last checkpoint.

Everything is simulated in board millimetres (x right, y towards the projector's
bottom, z up) and drawn in perspective from an eye above the board's centre.
"""
import itertools
import math
import time

import numpy as np

from .. import ui
from ..heights import WandHeights
from ..projector import pygame
from ..tracker import WAND_IDS
from .players import COLORS, PAIRS

G = 2000.0  # mm/s^2: a slower-than-real world so a tilt game stays playable
ROLL = 5 / 7  # rolling solid sphere
DAMPING = 0.5  # 1/s
RESTITUTION = 0.35
BALL_R = 8.0
TILT_MM = 35.0  # corner height range around the middle, from the wands' full range
EYE_MM = 500.0  # perspective eye height above the board (exaggerates depth)
HEIGHT_VIS = 3.0  # drawn height of things on the board vs. their simulated height
WALL_T, WALL_H = 6.0, 7.0  # wall height is only drawn (x HEIGHT_VIS); walls always block
LEDGE_H, LANDING_H = 12.0, 0.0
RESTART_AFTER_S = 6.0

# --- level, in mm for a board of half-size 168 (scaled to the real board) ---
REF = 168.0
WALLS = [
    ((-168, 90), (40, 90)),        # middle / south corridor divider
    ((80, -110), (80, 90)),        # ramp lane rail and ledge edge
    ((-30, -110), (80, -110)),     # ledge rail along the north strip
    ((-60, -110), (-30, -110)),    # south side of the gap
    ((-60, -110), (-60, -90)),
    ((-168, -90), (-60, -90)),     # south side of the landing
    ((-90, -90), (-90, 40)),       # maze walls
    ((-20, -30), (-20, 90)),
    ((40, -110), (40, 40)),
]
HOLES = [(-65, 0, 12), (25, -15, 10), (125, 135, 12), (-80, 130, 14), (60, -50, 12)]
GAP = (-60, -168, -30, -110)  # x0, y0, x1, y1: the jump
LANDING = (-168, -168, -60, -90)
LEDGE = [(-30, -168, 168, -110), (80, -110, 168, -10)]
RAMP = (80, -10, 168, 90)  # rises from y=90 (floor) to y=-10 (ledge height)
START, GOAL, GOAL_R = (-140, -60), (-125, -130), 18
CHECKPOINT, CHECKPOINT_ZONE = (100, 105), (40, 90, 168, 168)  # reached once the ball is in the corridor


def _in(rect, x, y):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


class Labyrinth:
    title = "Labyrinth"

    def __init__(self, app):
        self.app = app
        mm = app.px_per_mm
        self.mm = mm
        W, H = app.view.size
        self.center = np.array([W / 2, H / 2])
        self.half = min(W, H) / mm / 4  # board half-size in mm: board side = half the table's short side
        k = self.half / REF
        self.k = k
        self.walls = [(np.array(a, float) * k, np.array(b, float) * k) for a, b in
                      [((-168, -168), (168, -168)), ((168, -168), (168, 168)), ((168, 168), (-168, 168)),
                       ((-168, 168), (-168, -168))] + WALLS]
        self.holes = [(np.array((x, y), float) * k, r * k) for x, y, r in HOLES]
        self.gap = tuple(v * k for v in GAP)
        self.landing = tuple(v * k for v in LANDING)
        self.ledge = [tuple(v * k for v in r) for r in LEDGE]
        self.ramp = tuple(v * k for v in RAMP)
        self.start, self.goal, self.goal_r = np.array(START) * k, np.array(GOAL) * k, GOAL_R * k
        self.checkpoint_pos, self.checkpoint_zone = np.array(CHECKPOINT) * k, tuple(v * k for v in CHECKPOINT_ZONE)
        self.heights = WandHeights(app.calib, app.view)
        self.font = pygame.font.SysFont(None, int(12 * mm))
        self.big = pygame.font.SysFont(None, int(26 * mm))
        self.corner_of = self._assign_corners(app.tags)
        self.u = {k: 0.5 for k in WAND_IDS}
        self.new_round(time.monotonic())

    # --- setup ---------------------------------------------------------
    def corners_mm(self):
        h = self.half
        return np.array([(-h, -h), (h, -h), (h, h), (-h, h)])  # NW, NE, SE, SW

    def _assign_corners(self, tags):
        """Each wand controls the board corner it is nearest to now (best overall assignment)."""
        corners = self.center + self.corners_mm() * self.mm
        best, best_cost = list(WAND_IDS), math.inf
        for perm in itertools.permutations(WAND_IDS):
            cost = sum(np.linalg.norm(tags[k].pos - corners[i]) for i, k in enumerate(perm) if k in tags)
            if cost < best_cost:
                best, best_cost = list(perm), cost
        return {k: i for i, k in enumerate(best)}

    def new_round(self, now):
        self.pos, self.vel = self.start.astype(float).copy(), np.zeros(2)
        self.z, self.vz, self.air = 0.0, 0.0, False
        self.respawn = self.start.copy()
        self.falls = 0
        self.t_start = now
        self.won_at = None
        self.flash = None

    # --- level geometry ----------------------------------------------------
    def floor(self, x, y):
        """(height, is_hole) of the board surface at (x, y) in board mm."""
        if _in(self.gap, x, y):
            return 0.0, True
        for c, r in self.holes:
            if (x - c[0]) ** 2 + (y - c[1]) ** 2 < (r - BALL_R * 0.4) ** 2:
                return 0.0, True
        if _in(self.landing, x, y):
            return LANDING_H, False
        for r in self.ledge:
            if _in(r, x, y):
                return LEDGE_H, False
        if _in(self.ramp, x, y):
            x0, y0, x1, y1 = self.ramp
            return LEDGE_H * (y1 - y) / (y1 - y0), False
        return 0.0, False

    def floor_slope(self, x, y):
        """(dF/dx, dF/dy) of the floor (only the ramp slopes)."""
        if _in(self.ramp, x, y) and not any(_in(r, x, y) for r in self.ledge):
            x0, y0, x1, y1 = self.ramp
            return 0.0, -LEDGE_H / (y1 - y0)
        return 0.0, 0.0

    def tilt(self):
        """Board plane z = a x + b y + c from the four corner heights."""
        z = np.zeros(4)
        for k, i in self.corner_of.items():
            z[i] = (self.u.get(k, 0.5) - 0.5) * 2 * TILT_MM
        nw, ne, se, sw = z
        h = self.half
        a = (ne + se - nw - sw) / (4 * h)
        b = (sw + se - nw - ne) / (4 * h)
        return a, b, z.mean(), z

    # --- rules ---------------------------------------------------------
    def update(self, dt, now, tags, presses):
        for k in WAND_IDS:
            if k in tags:
                self.u[k] = self.heights.control(tags[k])
        if self.won_at is not None:
            if now - self.won_at > RESTART_AFTER_S:
                self.new_round(now)
            return
        a, b, _, _ = self.tilt()
        steps = max(1, math.ceil((np.linalg.norm(self.vel) * dt + abs(self.vz) * dt) / 1.5))
        for _ in range(steps):
            if self._step(dt / steps, a, b, now):
                break
        if _in(self.checkpoint_zone, *self.pos) and not self.air and not np.array_equal(self.respawn, self.checkpoint_pos):
            self.respawn = self.checkpoint_pos.copy()
            self.app.sound.play("checkpoint")
            self.app.fx.ring(self.canvas_xy(self.checkpoint_pos), (230, 230, 90), 12 * self.mm, n=24, speed=80)
        if not self.air and np.linalg.norm(self.pos - self.goal) < self.goal_r:
            self.won_at = now
            self.app.sound.play("win")
            self.app.fx.confetti(self.app.field)
            self.app.fx.burst(self.canvas_xy(self.goal), (40, 200, 90), n=60, speed=350, life=0.9,
                              colors=[(40, 200, 90), (230, 255, 230), (255, 220, 80)])

    def canvas_xy(self, p):
        """Board mm -> canvas px, ignoring tilt and height (good enough for effects)."""
        return self.center + np.asarray(p, float) * self.mm

    def _step(self, dt, a, b, now):
        """One physics substep; returns True when the ball fell and was reset."""
        if not self.air:
            fx, fy = self.floor_slope(*self.pos)
            acc = -G * ROLL * np.array([a + fx, b + fy])
            self.vel = (self.vel + acc * dt) * (1 - DAMPING * dt)
            new = self.pos + self.vel * dt
            h, hole = self.floor(*new)
            if hole or h < self.z - 1.0:  # rolled off an edge or into a hole
                self.air, self.vz = True, 0.0
                self.pos = new
                if self.z > 1.0:  # off a ledge: a jump
                    self.app.sound.play("jump", 0.7)
            elif h > self.z + 2.5:  # bumped into the side of something higher
                self.vel = -self.vel * RESTITUTION
            else:
                self.pos, self.z = new, h
        else:
            self.vz -= G * dt
            self.pos = self.pos + self.vel * dt
            self.z += self.vz * dt
            h, hole = self.floor(*self.pos)
            if not hole and self.z <= h:
                if self.z >= h - 6:  # landed on top
                    if self.vz < -150:
                        self.app.sound.play("block", 0.5)
                        self.app.fx.burst(self.canvas_xy(self.pos), (200, 170, 120), n=10, speed=100, life=0.3, size=2.5)
                    self.z, self.vz, self.air = h, 0.0, False
                else:  # hit the side of something higher (e.g. a short jump): drop back
                    self.pos = self.pos - self.vel * dt
                    self.vel = -self.vel * RESTITUTION
            elif hole and self.z < -25:
                self.falls += 1
                self.flash = (self.pos.copy(), now)
                self.app.sound.play("fall")
                self.app.fx.burst(self.canvas_xy(self.pos), (120, 120, 130), n=25, speed=120, life=0.6, size=4)
                self.pos, self.vel = self.respawn.copy(), np.zeros(2)
                self.z, self.vz, self.air = self.floor(*self.pos)[0], 0.0, False
                return True
        self._walls()
        return False

    def _walls(self):
        for p, q in self.walls:
            d = q - p
            t = np.clip(np.dot(self.pos - p, d) / np.dot(d, d), 0, 1)
            closest = p + t * d
            off = self.pos - closest
            dist = np.linalg.norm(off)
            reach = BALL_R + WALL_T / 2
            if 1e-6 < dist < reach:
                n = off / dist
                self.pos = closest + n * reach
                vn = np.dot(self.vel, n)
                if vn < 0:
                    self.vel = self.vel - (1 + RESTITUTION) * vn * n
                    if vn < -60:
                        self.app.sound.play("wall", min(1.0, -vn / 300), min_gap=0.08)

    # --- drawing -------------------------------------------------------
    def project(self, pts, z_local, a, b, c):
        """Board mm points (N,2) at local heights z (scalar or N) -> canvas px, with perspective."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        # heights of things on the board (ledge, ramp, walls, ball) are drawn exaggerated
        zw = a * pts[:, 0] + b * pts[:, 1] + c + np.broadcast_to(z_local, len(pts)) * HEIGHT_VIS
        s = EYE_MM / (EYE_MM - zw)
        return self.center + pts * s[:, None] * self.mm

    def _quad(self, surf, rect, z, color, a, b, c):
        x0, y0, x1, y1 = rect
        pygame.draw.polygon(surf, color, self.project([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z, a, b, c))

    def _shade(self, color, a, b, nx=0.0, ny=0.0):
        """Simple lighting: surfaces tilted towards a light from the top-left are brighter."""
        light = np.array([-0.4, -0.6, 1.0])
        n = np.array([-(a + nx), -(b + ny), 1.0])
        f = np.clip(np.dot(n / np.linalg.norm(n), light / np.linalg.norm(light)) * 1.25, 0.4, 1.3)
        return tuple(int(min(255, v * f)) for v in color)

    def draw(self, s, now):
        s.fill(ui.BACKGROUND)
        a, b, c, zc = self.tilt()
        h = self.half
        # board edge (thickness) then the board top
        top = self.project(self.corners_mm(), 0.0, a, b, c)
        bottom = self.project(self.corners_mm(), -10.0, a, b, c)
        for i in range(4):
            j = (i + 1) % 4
            pygame.draw.polygon(s, (60, 40, 25), [top[i], top[j], bottom[j], bottom[i]])
        pygame.draw.polygon(s, self._shade((150, 110, 70), a, b), top)
        # holes and the gap
        for cc, r in self.holes:
            ring = cc + r * np.stack([np.cos(np.linspace(0, 2 * np.pi, 24)), np.sin(np.linspace(0, 2 * np.pi, 24))], 1)
            pygame.draw.polygon(s, (12, 8, 5), self.project(ring, 0.0, a, b, c))
        self._quad(s, self.gap, 0.0, (12, 8, 5), a, b, c)
        # ramp, ledge (with front faces), landing
        self._draw_ramp(s, a, b, c, now)
        for r in self.ledge:
            rx0, ry0, rx1, ry1 = r
            for e0, e1 in (((rx0, ry1), (rx1, ry1)), ((rx0, ry0), (rx0, ry1))):  # visible-ish side faces
                pygame.draw.polygon(s, (90, 65, 40), np.vstack([self.project([e0, e1], LEDGE_H, a, b, c),
                                                                 self.project([e1, e0], 0.0, a, b, c)]))
        for r in self.ledge:
            self._quad(s, r, LEDGE_H, self._shade((200, 160, 105), a, b), a, b, c)
        self._quad(s, self.landing, LANDING_H, self._shade((170, 200, 140), a, b), a, b, c)
        # goal, start, checkpoint marks
        ring = lambda cc, r, z: self.project(cc + r * np.stack([np.cos(np.linspace(0, 2 * np.pi, 24)),
                                                                np.sin(np.linspace(0, 2 * np.pi, 24))], 1), z, a, b, c)
        pygame.draw.polygon(s, (40, 200, 90), ring(self.goal, self.goal_r, LANDING_H))
        pygame.draw.polygon(s, (230, 255, 230), ring(self.goal, self.goal_r * 0.45, LANDING_H))
        pygame.draw.lines(s, (200, 200, 200), True, ring(self.start, 12 * self.k, 0.0), 2)
        pygame.draw.lines(s, (200, 200, 90), True, ring(self.checkpoint_pos, 10 * self.k, 0.0), 2)
        # walls as boxes
        for p, q in self.walls:
            d = (q - p) / np.linalg.norm(q - p)
            n = np.array([-d[1], d[0]]) * WALL_T / 2
            base = np.array([p - d * WALL_T / 2 + n, q + d * WALL_T / 2 + n, q + d * WALL_T / 2 - n, p - d * WALL_T / 2 - n])
            mid = (p + q) / 2
            ztop = max(self.floor(*(mid + n * 2))[0], self.floor(*(mid - n * 2))[0]) + WALL_H
            t = self.project(base, ztop, a, b, c)
            bt = self.project(base, 0.0, a, b, c)
            for i in range(4):
                j = (i + 1) % 4
                pygame.draw.polygon(s, (70, 50, 35), [t[i], t[j], bt[j], bt[i]])
            pygame.draw.polygon(s, self._shade((120, 90, 60), a, b), t)
        # ball: shadow on the surface below, then the sphere
        fh, hole = self.floor(*self.pos)
        if not hole:
            sh = self.project([self.pos + (2, 2)], fh, a, b, c)[0]
            pygame.draw.circle(s, (40, 28, 18), sh, BALL_R * self.mm * 0.95)
        pc = self.project([self.pos], self.z + BALL_R, a, b, c)[0]
        scale = EYE_MM / (EYE_MM - (a * self.pos[0] + b * self.pos[1] + c + self.z + BALL_R))
        r = BALL_R * self.mm * scale
        pygame.draw.circle(s, (170, 175, 190), pc, r)
        pygame.draw.circle(s, (215, 220, 235), pc - (r * 0.25, r * 0.25), r * 0.6)
        pygame.draw.circle(s, (255, 255, 255), pc - (r * 0.4, r * 0.4), r * 0.22)
        if self.flash and now - self.flash[1] < 0.6:
            pygame.draw.circle(s, (255, 80, 60), self.project([self.flash[0]], 0.0, a, b, c)[0],
                               BALL_R * self.mm * (1 + 4 * (now - self.flash[1])), 3)
        self._draw_players(s, zc)
        self._draw_status(s, now)

    def _draw_ramp(self, s, a, b, c, now):
        """The ramp: side face, a dark-to-light gradient up the slope, and moving uphill chevrons."""
        x0, y0, x1, y1 = self.ramp  # rises from y1 (floor) to y0 (ledge)
        # side face along the open (west) edge: a wedge from the floor up to the ledge
        pygame.draw.polygon(s, (80, 55, 30), np.vstack([self.project([(x0, y1), (x0, y0)], np.array([0.0, LEDGE_H]), a, b, c),
                                                         self.project([(x0, y0)], 0.0, a, b, c)]))
        n = 10
        for i in range(n):
            ya, yb = y1 + (y0 - y1) * i / n, y1 + (y0 - y1) * (i + 1) / n
            za, zb = LEDGE_H * i / n, LEDGE_H * (i + 1) / n
            f = 0.55 + 0.45 * (i + 0.5) / n
            color = self._shade(tuple(int(v * f) for v in (235, 185, 90)), a, b, 0, -LEDGE_H / (y1 - y0))
            pygame.draw.polygon(s, color, self.project([(x0, ya), (x1, ya), (x1, yb), (x0, yb)], np.array([za, za, zb, zb]),
                                                       a, b, c))
        # chevrons pointing uphill, drifting up the slope
        w = x1 - x0
        for k in range(3):
            t = ((now * 0.4 + k / 3) % 1.0)
            y = y1 + (y0 - y1) * (0.15 + 0.7 * t)
            z = LEDGE_H * (y1 - y) / (y1 - y0)
            pts = [(x0 + w * 0.2, y + 10 * self.k), (x0 + w * 0.5, y - 6 * self.k), (x0 + w * 0.8, y + 10 * self.k)]
            pygame.draw.lines(s, (90, 50, 20), False, self.project(pts, z, a, b, c), max(2, int(2.5 * self.mm)))

    def _draw_players(self, s, zc):
        """At each board corner: the controlling wand's height as a bar in its player colour."""
        for k, i in self.corner_of.items():
            corner = self.corners_mm()[i]
            out = corner / np.linalg.norm(corner)
            p = self.center + (corner + out * 30) * self.mm
            color = COLORS[next(j for j, pair in enumerate(PAIRS) if k in pair)]
            bar_h = 50 * self.mm
            rect = pygame.Rect(0, 0, 8 * self.mm, bar_h)
            rect.center = (int(p[0]), int(p[1]))
            pygame.draw.rect(s, (50, 50, 50), rect, 2)
            fill = rect.copy()
            fill.height = int(bar_h * self.u.get(k, 0.5))
            fill.bottom = rect.bottom
            pygame.draw.rect(s, color, fill)
            seen = k in self.app.tags
            if not seen:
                pygame.draw.line(s, (200, 60, 60), rect.topleft, rect.bottomright, 2)

    def _draw_status(self, s, now):
        W, H = self.app.view.size
        if self.won_at is not None:
            msg = f"made it in {self.won_at - self.t_start:.1f}s, {self.falls} falls!"
            for facing, y in (("top", H * 0.12), ("bottom", H * 0.88)):
                ui.text(s, self.big, msg, (W / 2, y), (120, 255, 150), facing)
        else:
            msg = f"time {now - self.t_start:.0f}s   falls {self.falls}"
            for facing, y in (("top", H * 0.06), ("bottom", H * 0.94)):
                ui.text(s, self.font, msg, (W / 2, y), (200, 200, 200), facing)
