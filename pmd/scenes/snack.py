"""Snack castle: mice vs. the snacks, a tower defence played with tags on the table.

Everything is in table-canvas pixels. The playfield is a grid whose cell is
one field-tag spacing (40 mm).

Pieces (tag ids from the ar-of-empires print kit, A-J = 10-19; the wands are
G, B, D, C = 16, 11, 13, 12):
  11 (wand B)       coin collector: move it over coins, bring them to the chest
  16 (wand G)       chest: holds coins; park an attacker next to it to upgrade it
  14, 19            cats: swipe every mouse around them
  12, 13 (wands) and the other tags: slingshots, shoot straight ahead in a narrow cone
"""
import math
import time

import cv2
import numpy as np

from .. import art, ui
from ..projector import pygame

GRID_MM = 40.0
COLLECTOR_IDS = {11}
CHEST_IDS = {16}
CAT_IDS = {14, 19}
SLINGSHOT_IDS = set(range(10, 20)) - COLLECTOR_IDS - CHEST_IDS - CAT_IDS
ATTACKER_IDS = CAT_IDS | SLINGSHOT_IDS

MAX_SNACKS = 20
MAX_LEVEL = 4
UPGRADE_HOLD_S = 1.5
RESTART_AFTER_S = 8.0


def upgrade_cost(level):
    return 5 * (level + 1)


def slingshot_stats(level):
    """(damage, cooldown s, range in cells, half-angle rad)."""
    return 1 + level, 0.6 * 0.85 ** level, 2.0 + 0.5 * level, math.radians(7.5 + 1.5 * level)


def cat_stats(level):
    """(damage, cooldown s, radius in cells)."""
    return 2 + level, 2.0 * 0.88 ** level, 1.2 + 0.15 * level


class Mouse:
    def __init__(self, path, speed, hp, radius, color, reward, steals):
        self.path, self.speed = path, speed
        self.seg, self.t = 0, 0.0
        self.pos = np.array(path[0], float)
        self.dir = np.array([1.0, 0.0])
        self.hp = self.max_hp = hp
        self.radius, self.color, self.reward, self.steals = radius, color, reward, steals

    @property
    def progress(self):
        return self.seg + self.t

    def update(self, dt):
        """Advance along the path; returns True when the castle is reached."""
        remaining = self.speed * dt
        while remaining > 0 and self.seg < len(self.path) - 1:
            a, b = np.array(self.path[self.seg]), np.array(self.path[self.seg + 1])
            seg_len = np.linalg.norm(b - a)
            self.dir = (b - a) / seg_len
            step = min(remaining, (1 - self.t) * seg_len)
            self.t += step / seg_len
            remaining -= step
            self.pos = a + (b - a) * self.t
            if self.t >= 1 - 1e-9:
                self.seg, self.t = self.seg + 1, 0.0
        return self.seg >= len(self.path) - 1


class SnackCastle:
    title = "Snack Castle"

    def __init__(self, app):
        self.app, self.proj, self.calib = app, app.proj, app.calib
        self.cell = GRID_MM * app.px_per_mm
        # the player stands at app.facing; text and the castle are drawn upright for them
        self.facing = app.facing
        self.vdown = np.array([0.0, -1.0]) if self.facing == "top" else np.array([0.0, 1.0])
        self.font = pygame.font.SysFont(None, int(self.cell * 0.55))
        self.small = pygame.font.SysFont(None, int(self.cell * 0.4))
        self.big = pygame.font.SysFont(None, int(self.cell * 1.2))
        self._build_board()
        self.reset()

    # --- setup ---------------------------------------------------------
    def _build_board(self):
        W, H = self.app.view.size  # canvas: the whole table, in table millimetres
        r = self.app.table_rect  # minus the dark border
        poly = np.array([r.topleft, r.topright, r.bottomright, r.bottomleft], np.int32)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        margin = int(self.cell * 0.3)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        c = self.cell
        cols, rows = int((x1 - x0) // c), int((y1 - y0) // c)
        self.origin = np.array([x0 + ((x1 - x0) - cols * c) / 2, y0 + ((y1 - y0) - rows * c) / 2])
        self.cols, self.rows = cols, rows

        def inside(r, col):
            px = self.origin + np.array([col, r]) * c
            return all(0 <= x < W and 0 <= y < H and mask[int(y), int(x)]
                       for x, y in (px, px + (c - 1, 0), px + (0, c - 1), px + (c - 1, c - 1)))

        self.valid = np.array([[inside(r, col) for col in range(cols)] for r in range(rows)])
        self.mask = mask

        # serpentine path: lanes every 4 rows; the first and last rows stay free (HUD)
        lanes = list(range(1, rows - 1, 4))
        lo, hi = 0, cols - 1
        for r in range(lanes[0], lanes[-1] + 1):
            run = np.nonzero(self.valid[r])[0]
            lo, hi = max(lo, run.min()), min(hi, run.max())
        lo, hi = lo + 1, hi - 1  # room for the mouse hole and the castle
        pts = []
        for i, r in enumerate(lanes):
            a, b = (lo, hi) if i % 2 == 0 else (hi, lo)
            pts += [self.cell_center(r, a), self.cell_center(r, b)]
        self.path = pts
        self.path_cells = set()
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            n = int(round(max(abs(bx - ax), abs(by - ay)) / c))
            for k in range(n + 1):
                self.path_cells.add(self.cell_of((ax + (bx - ax) * k / n, ay + (by - ay) * k / n)))
        self.castle_pos = np.array(pts[-1])
        # HUD on the far edge as seen by the player, beside the castle when it's on that edge
        hud_row = rows - 1 if self.facing == "top" else 0
        castle_col = self.cell_of(self.castle_pos)[1]
        hud_col = cols // 2
        if abs(self.cell_of(self.castle_pos)[0] - hud_row) <= 2:
            hud_col = (castle_col - 2) // 2 if castle_col > cols // 2 else (castle_col + 2 + cols) // 2
        self.hud_pos = self.cell_center(hud_row, hud_col)
        self._background = self._render_background()

    def cell_center(self, r, col):
        return tuple(self.origin + (np.array([col, r]) + 0.5) * self.cell)

    def cell_of(self, p):
        col, r = ((np.asarray(p) - self.origin) // self.cell).astype(int)
        return int(r), int(col)

    def _render_background(self):
        s = pygame.Surface(self.app.view.size)
        s.fill((80, 100, 80))  # green base light (tags stay lit)
        c = self.cell
        for r in range(self.rows):
            for col in range(self.cols):
                if self.valid[r, col]:
                    x, y = self.origin + np.array([col, r]) * c
                    pygame.draw.rect(s, (24, 38, 24), (x, y, c, c), 1)
        pygame.draw.lines(s, art.PATH_EDGE, False, self.path, int(c * 0.95))
        pygame.draw.lines(s, art.PATH, False, self.path, int(c * 0.8))
        for p in self.path:
            pygame.draw.circle(s, art.PATH, p, c * 0.4)
        art.mouse_hole(s, self.path[0], c)
        return s

    def reset(self):
        self.mice, self.coins, self.effects = [], [], []
        self.snacks = MAX_SNACKS
        self.chest_coins = 0
        self.carried = 0
        self.score = 0
        self.levels = {}
        self.cooldowns = {}
        self.upgrade_progress = {}
        self.upgrade_locked = set()
        self.t_start = time.monotonic()
        self.next_spawn = self.t_start + 2.0
        self.spawned = 0
        self.game_over_at = None
        self.units = {}

    # --- rules ----------------------------------------------------------
    def difficulty(self, now):
        """Grows by 1 every 30 s."""
        return 1 + (now - self.t_start) / 30

    def spawn(self, now):
        lvl = self.difficulty(now)
        self.spawned += 1
        tier = int(lvl) - 1
        hp = max(1, round(2 * 1.3 ** (lvl - 1)))
        speed = self.cell * min(1.2 + 0.1 * (lvl - 1), 2.5)
        shade = max(60, 170 - tier * 18)
        color = (shade, shade - 5, shade - 10 + min(tier * 6, 30))
        r = self.cell * 0.2 * (1 + 0.06 * min(tier, 6))
        reward, steals = 1 + tier // 2, 1
        if self.spawned % 12 == 0:  # big rat
            hp, speed, r, color, reward, steals = hp * 5, speed * 0.7, r * 1.6, (110, 80, 60), reward * 4, 3
        self.mice.append(Mouse(self.path, speed, hp, r, color, reward, steals))
        self.next_spawn = now + max(0.45, 2.0 / (1 + 0.2 * (lvl - 1)))

    def kill(self, m):
        if m in self.mice:
            self.mice.remove(m)
            self.score += 1
            self.app.sound.play("squeak", 0.7)
            self.app.fx.burst(m.pos, m.color, n=18, speed=160, life=0.5, size=3.5,
                              colors=[m.color, (240, 160, 175), (255, 255, 255)])
            for _ in range(m.reward):
                jitter = np.random.uniform(-0.25, 0.25, 2) * self.cell
                self.coins.append([m.pos + jitter, time.monotonic()])

    def hit(self, m, dmg):
        m.hp -= dmg
        self.app.fx.burst(m.pos, (255, 240, 150), n=5, speed=120, life=0.2, size=2)
        if m.hp <= 0:
            self.kill(m)

    def update(self, dt, now, tags, presses):
        self.units = {k: t for k, t in tags.items() if 10 <= k <= 19}
        if self.game_over_at is not None:
            if now - self.game_over_at > RESTART_AFTER_S:
                self.reset()
            return
        if now >= self.next_spawn:
            self.spawn(now)
        for m in list(self.mice):
            if m.update(dt):
                self.mice.remove(m)
                self.snacks = max(0, self.snacks - m.steals)
                self.effects.append(("steal", self.castle_pos.copy(), now))
                self.app.sound.play("steal")
                self.app.fx.burst(self.castle_pos, (255, 80, 60), n=30, speed=220, life=0.6,
                                  colors=[(200, 140, 70), (230, 80, 120), (250, 220, 120)])
        if self.snacks == 0:
            self.game_over_at = now
            self.app.sound.play("game_over")
            return
        self._attack(now)
        self._collect(now)
        self._upgrades(dt, now)
        self.coins = [c for c in self.coins if now - c[1] < 45]
        self.effects = [e for e in self.effects if now - e[2] < 0.6]

    def _attack(self, now):
        c = self.cell
        for k, u in self.units.items():
            if k not in ATTACKER_IDS or now < self.cooldowns.get(k, 0):
                continue
            lvl = self.levels.get(k, 0)
            if k in SLINGSHOT_IDS:
                dmg, cd, rng, half = slingshot_stats(lvl)
                targets = [m for m in self.mice if self._in_cone(u, m, rng * c, half)]
                if targets:
                    m = max(targets, key=lambda m: m.progress)  # the one closest to the snacks
                    self.effects.append(("shot", (u.pos.copy(), m.pos.copy()), now))
                    self.app.sound.play("shoot", 0.5, min_gap=0.06)
                    self.hit(m, dmg)
                    self.cooldowns[k] = now + cd
            else:
                dmg, cd, rad = cat_stats(lvl)
                targets = [m for m in self.mice if np.linalg.norm(m.pos - u.pos) < rad * c + m.radius]
                if targets:
                    self.effects.append(("swipe", (u.pos.copy(), rad * c), now))
                    self.app.sound.play("swipe", 0.7)
                    for m in targets:
                        self.hit(m, dmg)
                    self.cooldowns[k] = now + cd

    @staticmethod
    def _in_cone(u, m, rng, half):
        d = m.pos - u.pos
        dist = np.linalg.norm(d)
        if dist > rng + m.radius or dist < 1e-6:
            return False
        off = abs((math.atan2(d[1], d[0]) - u.angle + math.pi) % (2 * math.pi) - math.pi)
        return off <= half or dist * math.sin(min(off - half, math.pi / 2)) <= m.radius

    def _chest(self):
        return next((u for k, u in self.units.items() if k in CHEST_IDS), None)

    def _collect(self, now):
        collector = next((u for k, u in self.units.items() if k in COLLECTOR_IDS), None)
        if collector is not None:
            keep = []
            for coin in self.coins:
                if np.linalg.norm(coin[0] - collector.pos) < self.cell * 0.9:
                    self.carried += 1
                    self.effects.append(("pickup", coin[0].copy(), now))
                    self.app.sound.play("coin", 0.6, min_gap=0.05)
                    self.app.fx.burst(coin[0], art.GOLD, n=8, speed=120, life=0.35, size=3)
                else:
                    keep.append(coin)
            self.coins = keep
            chest = self._chest()
            if chest is not None and self.carried and np.linalg.norm(chest.pos - collector.pos) < self.cell * 1.8:
                self.chest_coins += self.carried
                self.carried = 0
                self.effects.append(("deposit", chest.pos.copy(), now))
                self.app.sound.play("deposit")
                self.app.fx.burst(chest.pos, art.GOLD, n=35, speed=260, life=0.7, size=4,
                                  colors=[art.GOLD, (255, 255, 200)])

    def _upgrades(self, dt, now):
        chest = self._chest()
        for k in ATTACKER_IDS:
            u = self.units.get(k)
            near = chest is not None and u is not None and np.linalg.norm(chest.pos - u.pos) < self.cell * 1.8
            if not near:
                self.upgrade_progress.pop(k, None)
                if u is not None or chest is None:
                    self.upgrade_locked.discard(k)  # moved away: may upgrade again next visit
                continue
            lvl = self.levels.get(k, 0)
            if k in self.upgrade_locked or lvl >= MAX_LEVEL or self.chest_coins < upgrade_cost(lvl):
                continue
            self.upgrade_progress[k] = self.upgrade_progress.get(k, 0) + dt
            if self.upgrade_progress[k] >= UPGRADE_HOLD_S:
                self.chest_coins -= upgrade_cost(lvl)
                self.levels[k] = lvl + 1
                self.upgrade_progress.pop(k)
                self.upgrade_locked.add(k)
                self.effects.append(("upgrade", u.pos.copy(), now))
                self.app.sound.play("upgrade")
                self.app.fx.ring(u.pos, (120, 255, 160), u.size, n=30, speed=150, life=0.6, size=4)
                self.app.fx.burst(u.pos, (120, 255, 160), n=30, speed=300, life=0.6, colors=[(120, 255, 160), (255, 255, 255)])

    # --- drawing ---------------------------------------------------------
    def draw(self, s, now):
        s.blit(self._background, (0, 0))
        c = self.cell
        self._draw_castle(s)

        # weapon reach under everything else
        for k, u in self.units.items():
            lvl = self.levels.get(k, 0)
            if k in SLINGSHOT_IDS:
                _, _, rng, half = slingshot_stats(lvl)
                art.cone(s, u.pos, u.angle, half, rng * c, (40, 170, 170))
            elif k in CAT_IDS:
                pygame.draw.circle(s, (110, 60, 20), u.pos, cat_stats(lvl)[2] * c, 2)

        for pos, t0 in self.coins:
            fade = 1.0 if now - t0 < 35 else max(0.2, 1 - (now - t0 - 35) / 10)
            art.coin(s, pos, c * 0.13, fade)
        for m in self.mice:
            art.mouse(s, m.pos, m.dir, m.radius, m.color, now, m.hp / m.max_hp)

        for k, u in self.units.items():
            self._draw_unit(s, k, u, now)

        for kind, data, t0 in self.effects:
            age = (now - t0) / 0.6
            if kind == "shot" and age < 0.25:
                a, b = data
                pygame.draw.line(s, (255, 240, 150), a, b, 3)
                pygame.draw.circle(s, (255, 255, 200), b, 6)
            elif kind == "swipe":
                pos, rad = data
                pygame.draw.circle(s, (255, 150, 50), pos, rad * (0.5 + age * 0.5), 4)
                for i in (-1, 0, 1):
                    off = np.array([i * 10, 0])
                    pygame.draw.line(s, (255, 220, 200), pos + off + (-rad * 0.3, -rad * 0.4),
                                     pos + off + (rad * 0.3, rad * 0.4), 3)
            elif kind in ("pickup", "deposit", "upgrade"):
                color = {"pickup": art.GOLD, "deposit": art.GOLD, "upgrade": (120, 255, 160)}[kind]
                pygame.draw.circle(s, color, data, c * (0.3 + age * 0.9), 3)
            elif kind == "steal":
                pygame.draw.circle(s, (255, 60, 60), data, c * (0.5 + age * 1.5), 5)

        self._draw_hud(s, now)

    def _draw_castle(self, s):
        size = int(self.cell * 6)
        tmp = pygame.Surface((size, size), pygame.SRCALPHA)
        art.castle(tmp, (size / 2, size / 2), self.cell, self.snacks, MAX_SNACKS, self.small)
        if self.facing == "top":
            tmp = pygame.transform.rotate(tmp, 180)
        s.blit(tmp, tmp.get_rect(center=(int(self.castle_pos[0]), int(self.castle_pos[1]))))

    def _draw_unit(self, s, k, u, now):
        c = self.cell
        lit = u.size * 0.85
        pygame.draw.circle(s, (255, 255, 255), u.pos, lit)  # keep the tag readable for the camera
        lvl = self.levels.get(k, 0)
        if k in SLINGSHOT_IDS:
            pygame.draw.circle(s, (60, 220, 220), u.pos, lit + 5, 4)
            tip = u.pos + u.forward * (lit + 14)
            pygame.draw.line(s, (60, 220, 220), u.pos + u.forward * (lit + 5), tip, 5)
        elif k in CAT_IDS:
            pygame.draw.circle(s, (255, 150, 50), u.pos, lit + 5, 4)
            for sgn in (-1, 1):  # cat ears on the tag's printed top
                base = u.pos + art.rot(u.forward, sgn * 0.6) * (lit + 3)
                tip = u.pos + art.rot(u.forward, sgn * 0.45) * (lit + 22)
                side = art.rot(u.forward, sgn * 0.6 + sgn * 0.5) * (lit + 3)
                pygame.draw.polygon(s, (255, 150, 50), [base, tip, u.pos + side])
        elif k in COLLECTOR_IDS:
            pygame.draw.circle(s, art.GOLD, u.pos, lit + 5, 4)
            pygame.draw.circle(s, art.GOLD, u.pos, c * 0.9, 1)
            if self.carried:
                ui.text(s, self.font, f"{self.carried}", u.pos - self.vdown * (lit + 22), art.GOLD, self.facing)
        elif k in CHEST_IDS:
            box = pygame.Rect(0, 0, lit * 2 + 16, lit * 2 + 16)
            box.center = (int(u.pos[0]), int(u.pos[1]))
            pygame.draw.rect(s, (150, 95, 40), box, 6, border_radius=6)
            pygame.draw.circle(s, (120, 80, 30), u.pos, c * 1.8, 1)
            ui.text(s, self.font, f"{self.chest_coins}", u.pos + self.vdown * (lit + 30), art.GOLD, self.facing)
        if k in ATTACKER_IDS:
            art.level_pips(s, u.pos, lit + 16, lvl, (120, 255, 160))
            prog = self.upgrade_progress.get(k, 0)
            art.arc_progress(s, u.pos, lit + 26, prog / UPGRADE_HOLD_S, (120, 255, 160))
            chest = self._chest()
            if chest is not None and np.linalg.norm(chest.pos - u.pos) < c * 1.8 and lvl < MAX_LEVEL \
                    and k not in self.upgrade_locked:
                cost = upgrade_cost(lvl)
                col = (120, 255, 160) if self.chest_coins >= cost else (255, 110, 90)
                ui.text(s, self.small, f"upgrade {cost}", u.pos + self.vdown * (lit + 40), col, self.facing)

    def _draw_hud(self, s, now):
        lvl = int(self.difficulty(now if self.game_over_at is None else self.game_over_at))
        text = f"snacks {self.snacks}    chest {self.chest_coins}    mice level {lvl}    caught {self.score}"
        ui.text(s, self.font, text, self.hud_pos, (220, 220, 200), self.facing)
        if self.game_over_at is not None:
            mid = (self.origin + np.array([self.cols, self.rows]) * self.cell / 2)
            t = self.big.render("The mice got the snacks!", True, (255, 90, 90))
            pygame.draw.rect(s, (0, 0, 0), t.get_rect(center=(int(mid[0]), int(mid[1]))).inflate(40, 30))
            ui.text(s, self.big, "The mice got the snacks!", mid, (255, 90, 90), self.facing)
            left = RESTART_AFTER_S - (now - self.game_over_at)
            ui.text(s, self.font, f"caught {self.score} mice - new game in {left:.0f}",
                    mid + self.vdown * self.cell * 1.1, (220, 220, 200), self.facing)
