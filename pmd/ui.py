"""Drawing helpers shared by the scenes."""
import cv2
import numpy as np

from .projector import pygame

# Which table edge a reader stands at -> rotation (degrees, pygame CCW) that makes
# text upright for them. "bottom" is the projector's own bottom edge.
FACING = {"bottom": 0, "top": 180, "left": -90, "right": 90}

BACKGROUND = (90, 90, 90)  # base light everywhere, so tags and spots stay lit as the room gets dark
# Wand tags and spots get the same steady light (with the camera exposure locked to
# the projector's colour cycle, its light no longer upsets the blue spots); quiet_wand
# just keeps game graphics off them.
QUIET = BACKGROUND


def text(surf, font, msg, center, color=(220, 220, 220), facing="bottom"):
    t = font.render(msg, True, color)
    if FACING[facing]:
        t = pygame.transform.rotate(t, FACING[facing])
    rect = t.get_rect(center=(int(center[0]), int(center[1])))
    surf.blit(t, rect)
    return rect


def table_rect(calib, size, margin=20):
    """Largest centred axis-aligned rectangle inside the calibrated table area."""
    W, H = size
    poly = np.array(calib.table_poly or [(0, 0), (W, 0), (W, H), (0, H)], np.float32)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    c, half = (lo + hi) / 2, (hi - lo) / 2

    a, b = 0.1, 1.0
    for _ in range(20):
        m = (a + b) / 2
        a, b = (m, b) if _fits(c, half, (m, m), poly, margin) else (a, m)
    # then let width and height grow separately while they still fit
    sx = sy = a
    for _ in range(40):
        grown = False
        for axis in (0, 1):
            trial = [sx, sy]
            trial[axis] += 0.01
            if _fits(c, half, trial, poly, margin):
                sx, sy = trial
                grown = True
        if not grown:
            break
    x0, y0 = c - half * (sx, sy)
    x1, y1 = c + half * (sx, sy)
    return pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _fits(c, half, s, poly, margin):
    corners = [c + half * np.array(s) * np.array(v) for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= margin for x, y in corners)


def quiet_wand(surf, tag, color=QUIET):
    """Steady base light over a wand's tag, stick and button spot.

    Nothing the game draws should land there: changing light on the tag or the
    spot makes detection and button presses unreliable.
    """
    pygame.draw.circle(surf, color, tag.pos, tag.size * 0.95)
    pygame.draw.line(surf, color, tag.pos, tag.button_pos(), int(tag.size * 1.4))
    pygame.draw.circle(surf, color, tag.button_pos(), tag.size * 1.2)
