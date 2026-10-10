"""Background tag tracking: camera frames -> tag poses in projector pixels."""
import threading
import time
from dataclasses import dataclass

import numpy as np

from .tags import TagDetector


@dataclass
class TrackedTag:
    id: int
    pos: np.ndarray  # projector px
    angle: float  # radians, direction of the tag's printed "right" edge in projector space
    size: float  # tag side length in projector px
    corners: np.ndarray  # 4x2 projector px
    cam_corners: np.ndarray  # 4x2 camera px
    last_seen: float


class Tracker:
    """Runs detection on its own thread; read results with tags().

    hold: seconds a tag stays alive after it was last seen (hides brief
    occlusions by hands). smooth: exponential smoothing factor for position.
    """

    def __init__(self, camera, calib, hold=0.4, smooth=0.5):
        self.camera, self.calib = camera, calib
        self.hold, self.smooth = hold, smooth
        self.detector = TagDetector()
        self.fps = 0.0
        self._tags = {}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def tags(self):
        """Dict id -> TrackedTag of currently present tags."""
        with self._lock:
            return dict(self._tags)

    def _loop(self):
        seq = 0
        t_prev = time.monotonic()
        while self._running:
            frame, stamp, seq = self.camera.wait_newer(seq, timeout=1.0)
            if frame is None:
                continue
            found = self.detector.detect(frame)
            now = time.monotonic()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            with self._lock:
                for tag in found:
                    corners = self.calib.cam_to_proj(tag.corners)
                    pos = corners.mean(axis=0)
                    edge = corners[1] - corners[0]
                    angle = float(np.arctan2(edge[1], edge[0]))
                    size = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, 0), axis=1)))
                    old = self._tags.get(tag.id)
                    if old is not None and np.linalg.norm(old.pos - pos) < 40:
                        a = self.smooth
                        pos = a * old.pos + (1 - a) * pos
                    self._tags[tag.id] = TrackedTag(tag.id, pos, angle, size, corners, tag.corners, stamp)
                for k in [k for k, t in self._tags.items() if now - t.last_seen > self.hold]:
                    del self._tags[k]

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
