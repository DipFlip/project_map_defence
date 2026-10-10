"""Live check of the whole pipeline: projects a ring + id around every tag the camera sees.

    .venv/bin/python tools/preview.py [seconds]

Also writes data/preview.jpg (camera view with detections) every few seconds.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pmd import CALIB_PATH, DATA_DIR  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.projector import Projector, pygame  # noqa: E402
from pmd.tracker import Tracker  # noqa: E402


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    calib = Calibration.load(CALIB_PATH)
    tracker = Tracker(CALIB_PATH)
    proj = Projector()
    font = pygame.font.SysFont(None, 36)
    t_end = time.monotonic() + duration
    last_snap = 0
    try:
        while proj.pump() and time.monotonic() < t_end:
            s = proj.screen
            s.fill((0, 0, 0))
            if calib.table_poly:
                pygame.draw.polygon(s, (40, 40, 60), calib.table_poly)
                pygame.draw.polygon(s, (90, 90, 160), calib.table_poly, 3)
            tags = tracker.tags()
            for tag in tags.values():
                x, y = tag.pos
                r = max(tag.size * 0.9, 20)
                # light the tag itself so the camera keeps seeing it
                pygame.draw.circle(s, (255, 255, 255), (x, y), r)
                pygame.draw.circle(s, (0, 220, 120), (x, y), r + 8, 4)
                end = (x + np.cos(tag.angle) * (r + 30), y + np.sin(tag.angle) * (r + 30))
                pygame.draw.line(s, (255, 200, 0), (x, y), end, 3)
                s.blit(font.render(str(tag.id), True, (255, 200, 0)), (x + r + 10, y - r - 10))
            hud = f"{len(tags)} tags  det {tracker.fps:.0f} Hz  draw {proj.clock.get_fps():.0f} fps"
            s.blit(font.render(hud, True, (200, 200, 200)), (20, 20))
            proj.flip()
            proj.clock.tick(60)
            if time.monotonic() - last_snap > 3:
                last_snap = time.monotonic()
                tracker.snapshot(os.path.join(DATA_DIR, "preview.jpg"), annotate=True)
    finally:
        tracker.close()
        proj.close()


if __name__ == "__main__":
    main()
