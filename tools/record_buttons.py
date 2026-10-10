"""Record a labelled dataset for the wand buttons.

    .venv/bin/python tools/record_buttons.py [seconds]

Protocol: with tag A (id 10) lying on the table, bring a wand close to it
(within the projected 5 cm ring), cover its red spot, move away, uncover.
Do that 5 times per wand. The "pressed" label comes from the distance to tag A,
not from the button detector, so the detector can be evaluated against it
(tools/analyze_buttons.py).

Frames are recorded to RAM (/dev/shm: the SD card is too slow to write live)
and copied to data/datasets/buttons_<time>/ afterwards. Stops after `seconds`
or as soon as the file data/datasets/STOP exists.
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pmd import CALIB_PATH, DATA_DIR, ui  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.projector import Projector, pygame  # noqa: E402
from pmd.system import performance_governor  # noqa: E402
from pmd.table import TableView  # noqa: E402
from pmd.tracker import BUTTON_SPOTS, WAND_IDS, Tracker  # noqa: E402

ANCHOR_ID = 10
RADIUS_MM = 50


def main(seconds):
    calib = Calibration.load(CALIB_PATH)
    tracker = Tracker(CALIB_PATH)
    proj = Projector()
    view = TableView(calib, proj.size, proj.screen)
    mm = view.px_per_mm
    font = pygame.font.SysFont(None, int(16 * mm))
    name = time.strftime("buttons_%Y%m%d_%H%M%S")
    final = os.path.join(DATA_DIR, "datasets", name)
    stop_file = os.path.join(DATA_DIR, "datasets", "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    out = os.path.join("/dev/shm", "pmd_" + name)
    os.makedirs(out)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"anchor_id": ANCHOR_ID, "radius_mm": RADIUS_MM, "wand_ids": list(WAND_IDS),
                   "button_spots": {str(k): v for k, v in BUTTON_SPOTS.items()},
                   "calibration": json.load(open(CALIB_PATH))}, f)
    time.sleep(1.0)
    tracker.record(out)
    t_start = time.monotonic()
    t_end = t_start + seconds
    try:
        while proj.pump() and time.monotonic() < t_end and not os.path.exists(stop_file):
            s = view.canvas
            s.fill((0, 0, 0))
            tags = {k: t for k, t in tracker.tags().items()}
            anchor = tags.get(ANCHOR_ID)
            near = False
            if anchor is not None:
                a = view.to_canvas(anchor.pos)[0]
                for k in WAND_IDS:
                    if k in tags and np.linalg.norm(view.to_canvas(tags[k].pos)[0] - a) < RADIUS_MM * mm:
                        near = True
                pygame.draw.circle(s, (60, 220, 120) if near else (90, 90, 120), a, RADIUS_MM * mm, 4)
            for k in WAND_IDS:
                t = tags.get(k)
                if t is not None:
                    p, b = view.to_canvas([t.pos, t.button_pos()])
                    pygame.draw.circle(s, ui.QUIET, p, 18 * mm)
                    pygame.draw.circle(s, ui.QUIET, b, 14 * mm)
            w, h = view.size
            ui.text(s, font, f"recording {time.monotonic() - t_start:.0f}s - press only inside the ring",
                    (w / 2, h * 0.08), (200, 200, 200), "top")
            view.render(proj.screen)
            proj.flip()
            proj.clock.tick(60)
    finally:
        tracker.record(None)
        time.sleep(0.3)
        tracker.close()
        proj.close()
    n = len(open(os.path.join(out, "stamps.csv")).readlines())
    print(f"recorded {n} frames in RAM, copying to {final} ...", flush=True)
    shutil.copytree(out, final)
    shutil.rmtree(out)
    print(f"saved {n} frames to {final}")


if __name__ == "__main__":
    with performance_governor():
        main(float(sys.argv[1]) if len(sys.argv) > 1 else 600)
