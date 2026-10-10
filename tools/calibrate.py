"""Project calibration patterns, fit camera->projector mapping, save data/calibration.json.

Keep the table clear of people. Lay at least 4 tags flat and spread out on the
table so the metric table frame can be measured too (tools/table_frame.py
redoes just that step).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmd import CALIB_PATH, DATA_DIR, calibration  # noqa: E402
from pmd.camera import Camera  # noqa: E402
from pmd.projector import Projector  # noqa: E402


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cam = Camera()
    proj = Projector()
    try:
        calib = calibration.run(cam, proj, debug_dir=DATA_DIR)
        try:
            calibration.table_frame(cam, proj, calib, debug_dir=DATA_DIR)
        except ValueError as e:
            print(f"table frame skipped ({e}); run tools/table_frame.py later")
    finally:
        cam.auto_exposure()
        cam.close()
        proj.close()
    calib.save(CALIB_PATH)
    print("saved", CALIB_PATH)


if __name__ == "__main__":
    main()
