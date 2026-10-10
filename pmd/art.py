"""Procedural drawing of the game's pieces (all sizes in projector px)."""
import math

import numpy as np

from .projector import pygame

CHEESE = (245, 200, 60)
CHEESE_DARK = (200, 150, 30)
PATH = (70, 52, 32)
PATH_EDGE = (95, 72, 45)
GOLD = (255, 205, 40)


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def ellipse_poly(center, rx, ry, angle, n=20):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
    c, s = math.cos(angle), math.sin(angle)
    return pts @ np.array([[c, s], [-s, c]]) + center


def mouse(surf, pos, direction, r, body, now, hp_frac):
    a = math.atan2(direction[1], direction[0])
    fwd = np.array([math.cos(a), math.sin(a)])
    side = np.array([-fwd[1], fwd[0]])
    # tail: wiggling polyline behind the body
    wig = math.sin(now * 12 + pos[0] * 0.05)
    tail = [pos - fwd * (r * (1.1 + 0.35 * i)) + side * (r * 0.3 * math.sin(i * 1.3 + wig)) for i in range(7)]
    pygame.draw.lines(surf, (230, 150, 160), False, tail, max(2, int(r * 0.18)))
    pygame.draw.polygon(surf, body, ellipse_poly(pos, r * 1.25, r * 0.85, a))
    head = pos + fwd * r * 1.05
    for sgn in (-1, 1):
        ear = head - fwd * r * 0.2 + side * sgn * r * 0.55
        pygame.draw.circle(surf, body, ear, r * 0.42)
        pygame.draw.circle(surf, (240, 160, 175), ear, r * 0.26)
    pygame.draw.circle(surf, body, head, r * 0.62)
    pygame.draw.circle(surf, (250, 140, 160), head + fwd * r * 0.6, max(2, r * 0.14))
    for sgn in (-1, 1):
        pygame.draw.circle(surf, (10, 10, 10), head + fwd * r * 0.2 + side * sgn * r * 0.25, max(1.5, r * 0.11))
    if hp_frac < 1:
        w = r * 2.2
        x, y = pos[0] - w / 2, pos[1] - r * 1.9
        pygame.draw.rect(surf, (90, 20, 20), (x, y, w, 5))
        pygame.draw.rect(surf, (90, 230, 90), (x, y, w * hp_frac, 5))


def mouse_hole(surf, pos, cell):
    pygame.draw.circle(surf, (30, 20, 12), pos, cell * 0.55)
    pygame.draw.circle(surf, (5, 5, 5), pos, cell * 0.42)


def castle(surf, center, cell, snacks, max_snacks, font):
    """Snack castle: cheese walls with holes, towers and a pile of snacks."""
    cx, cy = center
    w, h = cell * 2.2, cell * 1.3
    base = pygame.Rect(0, 0, w, h)
    base.center = (cx, cy + cell * 0.2)
    tw, th = cell * 0.55, cell * 1.75
    towers = [pygame.Rect(base.left - tw * 0.3, base.bottom - th, tw, th),
              pygame.Rect(base.right - tw * 0.7, base.bottom - th, tw, th),
              pygame.Rect(cx - tw / 2, base.bottom - th - cell * 0.35, tw, th + cell * 0.35)]
    pygame.draw.rect(surf, CHEESE, base)
    for t in towers:
        pygame.draw.rect(surf, CHEESE, t)
        n = 3
        for i in range(n):  # crenellations
            bw = t.width / (2 * n - 1)
            pygame.draw.rect(surf, CHEESE, (t.left + i * 2 * bw, t.top - bw, bw, bw))
        top = (t.centerx, t.top - cell * 0.15)
        pygame.draw.line(surf, (120, 90, 60), top, (top[0], top[1] - cell * 0.5), 2)
        pygame.draw.polygon(surf, (220, 50, 50), [(top[0], top[1] - cell * 0.5), (top[0] + cell * 0.3, top[1] - cell * 0.4),
                                                  (top[0], top[1] - cell * 0.3)])
    rng = np.random.default_rng(7)  # cheese holes, stable between frames
    for _ in range(9):
        x = rng.uniform(base.left + 6, base.right - 6)
        y = rng.uniform(base.top + 6, base.bottom - 6)
        pygame.draw.circle(surf, CHEESE_DARK, (x, y), rng.uniform(cell * 0.05, cell * 0.12))
    door = pygame.Rect(0, 0, cell * 0.5, cell * 0.6)
    door.midbottom = base.midbottom
    pygame.draw.rect(surf, (90, 60, 30), door, border_top_left_radius=int(cell * 0.25),
                     border_top_right_radius=int(cell * 0.25))
    # snack pile in front of the gate: cookies, candies, crisps
    colors = [(200, 140, 70), (230, 80, 120), (250, 220, 120), (120, 220, 200)]
    for i in range(snacks):
        ang = i * 2.4
        rad = cell * 0.12 * math.sqrt(i + 1)
        p = (cx + math.cos(ang) * rad * 1.6, base.bottom + cell * 0.45 + math.sin(ang) * rad * 0.6)
        pygame.draw.circle(surf, colors[i % 4], p, cell * 0.11)
    label = font.render(f"{snacks}/{max_snacks} snacks", True, CHEESE)
    surf.blit(label, label.get_rect(midtop=(cx, base.bottom + cell * 1.0)))


def coin(surf, pos, r, alpha=1.0):
    c = tuple(int(v * alpha) for v in GOLD)
    pygame.draw.circle(surf, c, pos, r)
    pygame.draw.circle(surf, tuple(int(v * alpha * 0.7) for v in GOLD), pos, r * 0.6, 2)


def cone(surf, pos, angle, half_angle, rng, color):
    pts = [pos] + [pos + rng * np.array([math.cos(a), math.sin(a)])
                   for a in np.linspace(angle - half_angle, angle + half_angle, 8)]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, tuple(min(255, v + 70) for v in color), pts, 2)


def level_pips(surf, pos, r, level, color):
    for i in range(level):
        a = -math.pi / 2 + (i - (level - 1) / 2) * 0.35
        pygame.draw.circle(surf, color, pos + r * np.array([math.cos(a), math.sin(a)]), 5)


def arc_progress(surf, pos, r, frac, color, width=6):
    if frac <= 0:
        return
    rect = pygame.Rect(0, 0, 2 * r, 2 * r)
    rect.center = (int(pos[0]), int(pos[1]))
    pygame.draw.arc(surf, color, rect, math.pi / 2 - 2 * math.pi * frac, math.pi / 2, width)
