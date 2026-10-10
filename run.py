"""Start the game. Run tools/calibrate.py first (and again whenever the camera or projector moves).

    .venv/bin/python run.py [seconds]
"""
import os
import sys
import time

from pmd import CALIB_PATH, DATA_DIR
from pmd.calibration import Calibration
from pmd.camera import Camera
from pmd.game import Game
from pmd.projector import Projector
from pmd.tracker import Tracker


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    calib = Calibration.load(CALIB_PATH)
    cam = Camera()
    cam.manual_exposure(calib.exposure * 2)
    tracker = Tracker(cam, calib)
    proj = Projector()
    game = Game(proj, tracker, calib)
    t0 = time.monotonic()
    try:
        game.run(duration, snapshot_path=os.path.join(DATA_DIR, "game_cam.jpg"))
    finally:
        print(f"ran {time.monotonic() - t0:.0f}s, draw {proj.clock.get_fps():.0f} fps, "
              f"detect {tracker.fps:.1f} Hz, score {game.score}, lives {game.lives}")
        tracker.close()
        cam.auto_exposure()
        cam.close()
        proj.close()


if __name__ == "__main__":
    main()
