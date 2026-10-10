"""Measure only the metric table frame (keeps the camera<->projector calibration).

    .venv/bin/python tools/table_frame.py

Lay at least 4 tags flat and spread over the table, and keep out of the
projector's light while it runs (about 5 s).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmd import CALIB_PATH, DATA_DIR, calibration  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.camera import Camera  # noqa: E402
from pmd.projector import Projector  # noqa: E402


def main():
    calib = Calibration.load(CALIB_PATH)
    cam = Camera()
    proj = Projector()
    try:
        calibration.table_frame(cam, proj, calib, debug_dir=DATA_DIR)
    finally:
        cam.auto_exposure()
        cam.close()
        proj.close()
    calib.save(CALIB_PATH)
    print("saved", CALIB_PATH)


if __name__ == "__main__":
    main()
