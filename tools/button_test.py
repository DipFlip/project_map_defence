"""Check the wand buttons: shows each wand's button value live and whether it counts as pressed.

    .venv/bin/python tools/button_test.py [seconds]

Cover and uncover each wand's blue spot a few times, also while moving and tilting. At the end it prints the
value distribution per wand, to pick PRESS_BELOW / RELEASE_ABOVE in pmd/tracker.py.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pmd import CALIB_PATH  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.projector import Projector, pygame  # noqa: E402
from pmd.system import performance_governor  # noqa: E402
from pmd import ui  # noqa: E402
from pmd.tracker import PRESS_BELOW, RELEASE_ABOVE, WAND_IDS, Tracker  # noqa: E402


def main(seconds):
    center = np.array(Calibration.load(CALIB_PATH).table_poly).mean(axis=0)
    tracker = Tracker(CALIB_PATH)
    proj = Projector()
    font = pygame.font.SysFont(None, 40)
    values = {k: [] for k in WAND_IDS}
    presses = {k: 0 for k in WAND_IDS}
    was = {}
    t_end = time.monotonic() + seconds
    try:
        while proj.pump() and time.monotonic() < t_end:
            s = proj.screen
            s.fill((0, 0, 0))
            tags = tracker.tags()
            for k in WAND_IDS:
                t = tags.get(k)
                if t is None:
                    continue
                if t.button is not None:
                    values[k].append((t.last_seen, t.button))
                if t.pressed and not was.get(k):
                    presses[k] += 1
                was[k] = t.pressed
                color = (255, 60, 60) if t.pressed else (60, 220, 120)
                pygame.draw.circle(s, color, t.pos, t.size * 1.4, 5)
                pygame.draw.circle(s, color, t.button_pos(), t.size * 0.9, 4)
                ui.quiet_wand(s, t)
                label = "--" if t.button is None else f"{t.button:.2f}"
                ui.text(s, font, f"{k}: {label}", t.pos + t.forward * t.size * 2.4, color)
            ui.text(s, font, f"cover the blue spots - {t_end - time.monotonic():.0f}s",
                    center, (200, 200, 200))
            proj.flip()
            proj.clock.tick(60)
    finally:
        tracker.close()
        proj.close()
    print(f"thresholds: pressed below {PRESS_BELOW}, released above {RELEASE_ABOVE}")
    for k in WAND_IDS:
        v = np.array(list(dict(values[k]).values()))  # one value per camera frame
        if not len(v):
            print(f"{k}: no readings")
            continue
        edges = np.arange(0, 1.01, 0.05)
        hist, _ = np.histogram(np.clip(v, 0, 0.999), bins=edges)
        print(f"{k}: {len(v)} readings, presses {presses[k]}, median {np.median(v):.2f}")
        print("    " + " ".join(f"{b:.2f}:{n}" for b, n in zip(edges, hist) if n))


if __name__ == "__main__":
    with performance_governor():
        main(float(sys.argv[1]) if len(sys.argv) > 1 else 30)
