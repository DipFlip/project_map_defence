"""Measure where each wand's button spot is relative to its tag.

    .venv/bin/python tools/measure_wands.py

Lay the wands flat on the table with their spots uncovered. Writes
data/wands.json: per wand, the spot centre in tag heights printed-down and
towards printed-left, used by the tracker and for drawing.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pmd import CALIB_PATH  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.camera import Camera  # noqa: E402
from pmd.projector import Projector  # noqa: E402
from pmd.tags import TagDetector  # noqa: E402
from pmd.tracker import BUTTON_OFFSET, WAND_IDS, WANDS_PATH, spot_mask  # noqa: E402


def spot_centre(color, c):
    """Centre of the spot pixels near the expected spot, in (down, left) tag heights, or None."""
    center = c.mean(axis=0)
    down, left = c[0] - c[3], c[1] - c[0]
    side = float(np.linalg.norm(down))
    x, y = (center + down * BUTTON_OFFSET).astype(int)
    r = int(side * 1.5)
    h, w = color.shape[:2]
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return None
    box = color[y - r:y + r, x - r:x + r].astype(np.float32)
    b, g, rr = np.moveaxis(box / 255.0, -1, 0)
    ys, xs = np.nonzero(spot_mask(b, g, rr))
    if len(xs) < 10:
        return None
    pts = np.stack([xs + x - r, ys + y - r], 1) - center
    return np.linalg.solve(np.stack([down, left], 1), pts.T).T.mean(axis=0)


def main():
    calib = Calibration.load(CALIB_PATH)
    proj = Projector()
    proj.fill((0, 0, 0))  # otherwise the text console is projected onto the wands
    cam = Camera()
    cam.manual_exposure(calib.exposure)
    time.sleep(1.5)
    det = TagDetector()
    found = {k: [] for k in WAND_IDS}
    _, _, seq = cam.latest()
    for _ in range(15):
        color, _, seq = cam.wait_newer(seq)
        for t in det.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)):
            if t.id in found:
                s = spot_centre(color, t.corners)
                if s is not None:
                    found[t.id].append(s)
    cam.auto_exposure()
    cam.close()
    proj.close()
    spots = {}
    for k, v in found.items():
        if len(v) < 5:
            print(f"wand {k}: spot not found (is it on the table, uncovered?)")
            continue
        down, left = np.median(v, axis=0)
        spots[str(k)] = [round(float(down), 3), round(float(left), 3)]
        print(f"wand {k}: spot {down:.2f} tag heights down, {left:+.2f} towards printed left")
    try:
        with open(WANDS_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if "spots" not in data:  # older files held only the spots
        data = {"spots": {k: v for k, v in data.items() if k.isdigit()}}
    data["spots"].update(spots)
    with open(WANDS_PATH, "w") as f:
        json.dump(data, f, indent=1)
    print("saved", WANDS_PATH)


if __name__ == "__main__":
    main()
