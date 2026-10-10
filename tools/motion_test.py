"""Measure real tracking lag by projecting a tag that moves in a circle.

    .venv/bin/python tools/motion_test.py [speed px/s ...]

For each speed it reports how far (in ms) the tracked position trails the
drawn position, with and without prediction, and how often the tag was lost.
The lag includes projector + camera + processing, i.e. about what a moving
physical piece experiences (minus one render frame).
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pmd import CALIB_PATH  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.projector import Projector, pygame  # noqa: E402
from pmd.system import performance_governor  # noqa: E402
from pmd.tracker import Tracker  # noqa: E402

TAG_ID = 50
TAG_PX = 60


def main(speeds):
    calib = Calibration.load(CALIB_PATH)
    tracker = Tracker(CALIB_PATH)
    proj = Projector()
    poly = np.array(calib.table_poly)
    center = poly.mean(axis=0)
    radius = 0.3 * min(np.ptp(poly, axis=0))
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                           TAG_ID, TAG_PX)
    marker = np.pad(marker, TAG_PX // 4, constant_values=255)
    surf = pygame.surfarray.make_surface(np.dstack([marker] * 3).swapaxes(0, 1))
    try:
        for speed in speeds:
            omega = speed / radius
            rows = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < 8 and proj.pump():
                now = time.monotonic()
                theta = omega * (now - t0)
                pos = center + radius * np.array([math.cos(theta), math.sin(theta)])
                s = proj.screen
                s.fill((0, 0, 0))
                pygame.draw.polygon(s, (60, 60, 60), calib.table_poly)
                s.blit(surf, surf.get_rect(center=(int(pos[0]), int(pos[1]))))
                proj.flip()
                shown = time.monotonic()  # flip returned: frame handed to the display
                proj.clock.tick(60)
                raw = tracker.tags(predict=False).get(TAG_ID)
                pred = tracker.tags().get(TAG_ID)
                rows.append((shown, theta, raw, pred))
            report(speed, omega, center, rows[60:])  # skip the start-up second
    finally:
        tracker.close()
        proj.close()


def lag_ms(rows, omega, center, which):
    lags = []
    for _, theta, raw, pred in rows:
        t = raw if which == "raw" else pred
        if t is None:
            continue
        d = t.pos - center
        tracked = math.atan2(d[1], d[0])
        diff = (theta - tracked + math.pi) % (2 * math.pi) - math.pi
        lags.append(diff / omega * 1000)
    return np.median(lags) if lags else float("nan")


def report(speed, omega, center, rows):
    lost = np.mean([raw is None or rows_t - raw.last_seen > 0.1 for rows_t, _, raw, _ in rows])
    print(f"speed {speed:4.0f} px/s: lag raw {lag_ms(rows, omega, center, 'raw'):4.0f} ms, "
          f"with prediction {lag_ms(rows, omega, center, 'pred'):4.0f} ms, tag stale/lost {lost * 100:.0f}% of frames")


if __name__ == "__main__":
    with performance_governor():
        main([float(a) for a in sys.argv[1:]] or [150, 300, 600])
