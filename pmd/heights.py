"""How high a wand is held above the table, from how big its tag looks.

A tag raised towards the camera looks bigger. For the camera ray through the
tag's centre we compute how big a tag of the same size would look lying on the
table where that ray meets it (through the calibration, so lens distortion and
perspective are exact). The ratio q = observed / on-table size gives the height:
q = D / (D - h), D = camera distance along the ray, so h = D * (1 - 1/q).

tools/wand_height.py records each wand swept between its lowest and highest
position and stores the q range plus an estimate of D in data/wands.json;
control() maps a wand to 0 (lowest) .. 1 (highest).
"""
import json

import numpy as np

from .tracker import TAG_MM, WANDS_PATH

DEFAULT_RANGE_MM = (50.0, 400.0)  # what the calibration sweep is asked to cover


def _load():
    try:
        with open(WANDS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class WandHeights:
    def __init__(self, calib, view, smooth=0.35):
        """calib: Calibration; view: TableView (projector px <-> table mm)."""
        self.calib, self.view = calib, view
        data = _load()
        self.camera_mm = data.get("camera_mm", 1100.0)
        self.range_mm = tuple(data.get("height_range_mm", DEFAULT_RANGE_MM))
        self.q_range = {int(k): tuple(v) for k, v in data.get("q_range", {}).items()}
        self.smooth = smooth
        self._u = {}

    def table_side_px(self, cam_point, size_mm):
        """Camera px side of a size_mm tag lying flat on the table where the ray through cam_point hits it."""
        proj = self.calib.cam_to_proj(cam_point)
        mm = self.view.to_canvas(proj)[0] / self.view.px_per_mm
        sq = mm + np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * size_mm
        cam = self.calib.proj_to_cam(self.view.to_proj(sq * self.view.px_per_mm))
        return float(np.mean(np.linalg.norm(cam - np.roll(cam, 1, axis=0), axis=1)))

    def ratio(self, tag):
        """q = observed tag size / its size if it lay on the table (1 on the table, bigger when raised)."""
        c = tag.cam_corners
        sides = np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)
        observed = float(max(sides[0] + sides[2], sides[1] + sides[3]) / 2)  # longer pair: tilt shortens the other
        return observed / self.table_side_px(c.mean(axis=0), TAG_MM.get(tag.id, 24.0))

    def height_mm(self, q):
        return self.camera_mm * (1 - 1 / max(q, 1e-3))

    def control(self, tag):
        """0 at the wand's lowest calibrated position, 1 at its highest; smoothed."""
        q = self.ratio(tag)
        lo, hi = self.q_range.get(tag.id, (None, None))
        if lo is None:  # uncalibrated wand: use the nominal range and camera distance
            h0, h1 = self.range_mm
        else:
            h0, h1 = self.height_mm(lo), self.height_mm(hi)
        u = float(np.clip((self.height_mm(q) - h0) / max(h1 - h0, 1.0), 0, 1))
        prev = self._u.get(tag.id, u)
        u = prev + self.smooth * (u - prev)
        self._u[tag.id] = u
        return u

    def last(self, wand_id, default=0.5):
        return self._u.get(wand_id, default)


def camera_distance(q_lo, q_hi, range_mm=DEFAULT_RANGE_MM):
    """Camera distance D from the q seen at the known lowest and highest heights (least squares).

    q = D / (D - h)  <=>  D (q - 1) = q h, one equation per height.
    """
    q = np.array([q_lo, q_hi], float)
    h = np.array(range_mm, float)
    a, b = q - 1, q * h
    return float(np.dot(a, b) / np.dot(a, a))
