"""Measure display->camera loop latency by flashing the projector black/white.

Prints how long after pygame's flip() the camera thread first delivers a frame
showing the change. That is projector input lag + camera exposure/USB/decode
latency, i.e. what tag tracking pays on top of detection time.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pmd.camera import Camera  # noqa: E402
from pmd.projector import Projector  # noqa: E402


def brightness(frame):
    return float(cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY).mean())


def main(n=12):
    cam = Camera()
    cam.manual_exposure(50)
    proj = Projector()
    proj.fill((0, 0, 0))
    time.sleep(1)
    lo = brightness(cam.latest()[0])
    proj.fill((255, 255, 255))
    time.sleep(1)
    hi = brightness(cam.latest()[0])
    mid = (lo + hi) / 2
    print(f"black {lo:.0f}  white {hi:.0f}")
    lat = []
    white = True
    for _ in range(n):
        white = not white
        time.sleep(0.4)
        _, _, seq = cam.latest()
        proj.fill((255, 255, 255) if white else (0, 0, 0))
        t0 = time.monotonic()
        while True:
            frame, stamp, seq = cam.wait_newer(seq)
            if (brightness(frame) > mid) == white:
                lat.append((stamp - t0) * 1000)
                break
    proj.close()
    cam.auto_exposure()
    cam.close()
    lat = np.array(lat)
    print(f"loop latency ms: median {np.median(lat):.0f}  min {lat.min():.0f}  max {lat.max():.0f}")


if __name__ == "__main__":
    main()
