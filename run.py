"""Start the games (menu first). Run tools/calibrate.py first (and again whenever the camera or projector moves).

    .venv/bin/python run.py [seconds]
"""
import os
import sys
import time

from pmd import CALIB_PATH, DATA_DIR
from pmd.calibration import Calibration
from pmd.app import App
from pmd.projector import Projector, pygame
from pmd.system import performance_governor
from pmd.tracker import Tracker


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    calib = Calibration.load(CALIB_PATH)
    tracker = Tracker(CALIB_PATH, predict=0.05)
    proj = Projector()
    app = App(proj, tracker, calib)
    t0 = time.monotonic()
    try:
        app.run(duration, snapshot_path=os.path.join(DATA_DIR, "game_cam.jpg"))
    finally:
        print(f"ran {time.monotonic() - t0:.0f}s, draw {proj.clock.get_fps():.0f} fps, "
              f"detect {tracker.fps:.1f} Hz, frame->pose roi {tracker.age_ms['roi']:.0f} ms / full {tracker.age_ms['full']:.0f} ms")
        tracker.close()
        pygame.mixer.quit()  # closing the HDMI audio stream cleanly avoids an ALSA error on exit
        proj.close()


if __name__ == "__main__":
    with performance_governor():
        main()
