"""Calibrate wand heights: sweep each wand between its lowest and highest position.

    .venv/bin/python tools/wand_height.py [seconds]

Hold each wand flat-ish (tag facing up) and move it slowly between about 5 cm
and 40 cm above the table a few times. Saves each wand's size-ratio range and
the camera distance to data/wands.json (see pmd/heights.py).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pmd import CALIB_PATH, ui  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.heights import DEFAULT_RANGE_MM, WandHeights, camera_distance  # noqa: E402
from pmd.projector import Projector, pygame  # noqa: E402
from pmd.system import performance_governor  # noqa: E402
from pmd.table import TableView  # noqa: E402
from pmd.tracker import WAND_IDS, WANDS_PATH, Tracker  # noqa: E402
from pmd.scenes.players import COLORS, PAIRS  # noqa: E402


def main(seconds):
    calib = Calibration.load(CALIB_PATH)
    tracker = Tracker(CALIB_PATH)
    proj = Projector()
    view = TableView(calib, proj.size, proj.screen)
    heights = WandHeights(calib, view)
    mm = view.px_per_mm
    font = pygame.font.SysFont(None, int(14 * mm))
    samples = {k: {} for k in WAND_IDS}
    W, H = view.size
    t_end = time.monotonic() + seconds
    try:
        while proj.pump() and time.monotonic() < t_end:
            s = view.canvas
            s.fill((0, 0, 0))
            tags = tracker.tags(predict=False)
            for i, k in enumerate(WAND_IDS):
                x = W * (0.2 + 0.2 * i)
                color = COLORS[next(j for j, p in enumerate(PAIRS) if k in p)]
                bar = pygame.Rect(0, 0, 20 * mm, 160 * mm)
                bar.midbottom = (int(x), int(H * 0.85))
                pygame.draw.rect(s, (60, 60, 60), bar, 2)
                t = tags.get(k)
                label = f"{k}: --"
                if t is not None:
                    q = heights.ratio(t)
                    samples[k][t.last_seen] = q
                    h = heights.height_mm(q)
                    frac = np.clip((h - DEFAULT_RANGE_MM[0]) / (DEFAULT_RANGE_MM[1] - DEFAULT_RANGE_MM[0]), 0, 1)
                    fill = bar.copy()
                    fill.height = int(bar.height * frac)
                    fill.bottom = bar.bottom
                    pygame.draw.rect(s, color, fill)
                    label = f"{k}: ~{h / 10:.0f} cm"
                ui.text(s, font, label, (x, H * 0.9), color, "top")
            ui.text(s, font, f"move each wand slowly from 5 cm to 40 cm a few times - {t_end - time.monotonic():.0f}s",
                    (W / 2, H * 0.1), (200, 200, 200), "top")
            view.render(proj.screen)
            proj.flip()
            proj.clock.tick(60)
    finally:
        tracker.close()
        proj.close()

    try:
        with open(WANDS_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    q_range, distances = data.get("q_range", {}), []
    for k in WAND_IDS:
        q = np.array(list(samples[k].values()))
        if len(q) < 30:
            print(f"wand {k}: only {len(q)} readings, skipped")
            continue
        lo, hi = np.percentile(q, [3, 97])
        d = camera_distance(lo, hi)
        distances.append(d)
        q_range[str(k)] = [round(float(lo), 4), round(float(hi), 4)]
        print(f"wand {k}: {len(q)} readings, size ratio {lo:.3f} .. {hi:.3f}, implies camera at {d:.0f} mm")
    if distances:
        data["q_range"] = q_range
        data["camera_mm"] = round(float(np.median(distances)), 1)
        data["height_range_mm"] = list(DEFAULT_RANGE_MM)
        with open(WANDS_PATH, "w") as f:
            json.dump(data, f, indent=1)
        print(f"camera distance {data['camera_mm']:.0f} mm; saved {WANDS_PATH}")


if __name__ == "__main__":
    with performance_governor():
        main(float(sys.argv[1]) if len(sys.argv) > 1 else 40)
