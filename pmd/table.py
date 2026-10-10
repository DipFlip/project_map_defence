"""Metric table frame: draw in table millimetres, warp to projector pixels.

Projector pixels are skewed on the table (keystone, the projector's angle), so
a rectangle drawn in projector pixels isn't a rectangle on the table. We find
a homography projector px -> table mm from tags lying flat on the table (each
is a square of known size), align it with the table's outline (where the
projector's light lands on the table plane), and render every scene onto a
canvas in table coordinates that is warped to the projector each frame.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

CANVAS_PX_PER_MM = 1.6  # canvas resolution; about the projector's own density on this table


def _apply(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return q[:, :2] / q[:, 2:]


# tag corners in OpenCV order are printed BR, BL, TL, TR (see tracker.py)
_SQUARE = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def fit_metric(tags, px_per_mm_guess):
    """Homography projector px -> table mm from flat tags of known size.

    tags: list of (4x2 corners in projector px, black square size in mm).
    Each tag must map to a square of its size; the fit finds the perspective
    that makes all of them square at once. The result's rotation and origin
    are arbitrary (see align_to_outline).
    """
    if len(tags) < 4:
        raise ValueError(f"need at least 4 flat tags spread over the table, got {len(tags)}")
    H0 = np.diag([1 / px_per_mm_guess, 1 / px_per_mm_guess, 1.0])
    poses0 = []
    for corners, _ in tags:
        q = _apply(H0, corners)
        e = q[3] - q[0]  # printed bottom-right -> top-right: printed "up"
        poses0.append([np.arctan2(e[1], e[0]) + np.pi / 2, *q.mean(axis=0)])
    poses0 = np.array(poses0)
    fixed = poses0[0]  # first tag's pose fixes the frame's rotation and origin

    def model(pose, size):
        th, tx, ty = pose
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return _SQUARE * size @ R.T + (tx, ty)

    def residual(p):
        H = np.append(p[:8], 1).reshape(3, 3)
        poses = np.vstack([fixed, p[8:].reshape(-1, 3)])
        return np.concatenate([(_apply(H, c) - model(pose, size)).ravel()
                               for (c, size), pose in zip(tags, poses)])

    p0 = np.concatenate([H0.ravel()[:8], poses0[1:].ravel()])
    sol = least_squares(residual, p0, x_scale="jac")
    H = np.append(sol.x[:8], 1).reshape(3, 3)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return H, rms


def align_to_outline(H, outline_proj):
    """Rotate/translate the mm frame so the table's long edge is +x and its corner is (0, 0).

    Picks, of the two rotations putting the long edge on x, the one closest to
    the projector's own orientation (so "left" stays left). Returns the new
    homography and the table size in mm.
    """
    pts = _apply(H, outline_proj).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    ang = np.radians(ang if w >= h else ang + 90)  # direction of the long side
    # projector +x in the mm frame, to choose between ang and ang + 180
    px = _apply(H, [[0, 0], [100, 0]])
    ref = np.arctan2(*(px[1] - px[0])[::-1])
    if np.cos(ang - ref) < 0:
        ang += np.pi
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = _apply(R @ H, outline_proj)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # use the rotated rectangle's size, not the outline's bounding box (outline is ragged)
    L, S = max(w, h), min(w, h)
    mid = (lo + hi) / 2
    T = np.array([[1, 0, -(mid[0] - L / 2)], [0, 1, -(mid[1] - S / 2)], [0, 0, 1]])
    return T @ R @ H, (float(L), float(S))


def lit_outline(lit, cam_to_proj, frac=0.5):
    """Outline (projector px) of the table plane from a white-minus-black capture.

    The table is the biggest bright region; the floor further away is far
    dimmer. The projector's flicker leaves dark horizontal stripes at short
    exposures, so the mask is closed over small gaps and the hull is taken.
    Returns a polygon in projector px, or None.
    """
    b = cv2.GaussianBlur(lit.astype(np.float32), (31, 31), 0)
    mask = (b > frac * np.percentile(b, 90)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2)
    return cam_to_proj(hull.astype(np.float64))


class TableView:
    """The canvas every scene draws on, in table millimetres.

    1 canvas px = 1 / CANVAS_PX_PER_MM mm. render() warps it to the projector.
    Without a table frame in the calibration it falls back to plain projector
    pixels over the calibrated table area.
    """

    def __init__(self, calib, proj_size, surface_like):
        import pygame

        if calib.table_H is not None:
            S = np.diag([CANVAS_PX_PER_MM, CANVAS_PX_PER_MM, 1.0])
            self.proj_to_canvas = S @ np.array(calib.table_H)
            w, h = (np.array(calib.table_size_mm) * CANVAS_PX_PER_MM).astype(int)
            self.px_per_mm = CANVAS_PX_PER_MM
        else:
            poly = np.array(calib.table_poly or [(0, 0), (proj_size[0], proj_size[1])])
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            self.proj_to_canvas = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], float)
            w, h = (hi - lo).astype(int)
            self.px_per_mm = None  # unknown: estimate from tags
        self.canvas_to_proj = np.linalg.inv(self.proj_to_canvas)
        self.size = (int(w), int(h))
        self.proj_size = proj_size
        self.canvas = pygame.Surface(self.size, 0, surface_like)

    def to_canvas(self, pts):
        return _apply(self.proj_to_canvas, pts)

    def to_proj(self, pts):
        return _apply(self.canvas_to_proj, pts)

    def render(self, screen):
        """Warp the canvas onto the projector surface (zero-copy, ~4 ms at 1080p)."""
        cw, ch = self.size
        src = np.frombuffer(self.canvas.get_view("1"), np.uint8).reshape(ch, self.canvas.get_pitch() // 4, 4)
        W, H = self.proj_size
        dst = np.frombuffer(screen.get_view("1"), np.uint8).reshape(H, screen.get_pitch() // 4, 4)
        cv2.warpPerspective(src[:, :cw], self.canvas_to_proj, (W, H), dst=dst[:, :W],
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
