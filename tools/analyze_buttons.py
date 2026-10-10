"""Extract per-frame wand data from a recorded button dataset (tools/record_buttons.py).

    .venv/bin/python tools/analyze_buttons.py data/datasets/buttons_<time>

For every frame and wand: distance to tag A in table mm (the label source), the
current detector's button value, and a crop of the camera image around the
wand's spot, resampled into the wand's own frame (so crops line up across
frames). Saved to <dataset>/features.npz for evaluating detector ideas.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pmd.calibration import Calibration  # noqa: E402
from pmd.table import _apply  # noqa: E402
from pmd.tags import TagDetector  # noqa: E402
from pmd.tracker import BUTTON_SPOTS, _Core  # noqa: E402

CROP_PX = 16  # crop resolution: px per tag height
CROP_DOWN = (1.0, 4.6)  # crop range along the wand, tag heights printed-down from the tag centre
CROP_ACROSS = 1.6  # +- tag heights across the wand


def crop(color, c, spot):
    """Resample the region around the spot into the wand's frame: rows = along, cols = across."""
    center = c.mean(axis=0)
    down, left = c[0] - c[3], c[1] - c[0]
    rows = int((CROP_DOWN[1] - CROP_DOWN[0]) * CROP_PX)
    cols = int(2 * CROP_ACROSS * CROP_PX)
    ys, xs = np.mgrid[0:rows, 0:cols]
    along = CROP_DOWN[0] + ys / CROP_PX
    across = -CROP_ACROSS + xs / CROP_PX + spot[1]
    pts = center + along[..., None] * down + across[..., None] * left
    return cv2.remap(color, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def main(directory):
    meta = json.load(open(os.path.join(directory, "meta.json")))
    cal = meta["calibration"]
    calib = Calibration(tuple(cal["cam_size"]), tuple(cal["proj_size"]), np.array(cal["K"]), np.array(cal["dist"]),
                        np.array(cal["H"]), cal["exposure"], cal["rms_px"], cal.get("footprint", []),
                        cal.get("table_poly", []), cal.get("table_H"), cal.get("table_size_mm"))
    table_H = np.array(calib.table_H)
    anchor, wands = meta["anchor_id"], meta["wand_ids"]
    spots = {int(k): tuple(v) for k, v in meta["button_spots"].items()}
    stamps = [line.strip().split(",") for line in open(os.path.join(directory, "stamps.csv")) if line.strip()]
    det = TagDetector()
    rows = []  # frame index, stamp, wand id, distance mm, detector value, side px
    crops = []
    anchor_mm = None
    for i, (name, stamp) in enumerate(stamps):
        raw = np.fromfile(os.path.join(directory, name), np.uint8)
        color = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if color is None:
            continue
        tags = {t.id: t for t in det.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))}
        if anchor in tags:
            anchor_mm = _apply(table_H, calib.cam_to_proj(tags[anchor].corners.mean(axis=0)))[0]
        for k in wands:
            t = tags.get(k)
            if t is None or anchor_mm is None:
                continue
            pos_mm = _apply(table_H, calib.cam_to_proj(t.corners.mean(axis=0)))[0]
            value = _Core._sample_button(color, t.corners, spots.get(k, BUTTON_SPOTS[k]))
            side = float(np.linalg.norm(t.corners[0] - t.corners[3]))
            rows.append((i, float(stamp), k, float(np.linalg.norm(pos_mm - anchor_mm)),
                         np.nan if value is None else value, side))
            crops.append(crop(color, t.corners, spots.get(k, BUTTON_SPOTS[k])))
        if i % 500 == 0:
            print(f"frame {i}/{len(stamps)}: {len(rows)} wand samples")
    rows = np.array(rows)
    np.savez_compressed(os.path.join(directory, "features.npz"), rows=rows, crops=np.array(crops))
    print(f"saved {len(rows)} wand samples to {directory}/features.npz")


if __name__ == "__main__":
    main(sys.argv[1])
