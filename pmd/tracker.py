"""Background tag tracking: camera frames -> tag poses in projector pixels."""
import threading
import time
from dataclasses import dataclass

import numpy as np

from .tags import TagDetector

# physical size of the black square per tag id, for estimating mm <-> px
TAG_MM = {**{i: 24.0 for i in range(10, 20)}, **{i: 30.0 for i in range(100, 135)}}


@dataclass
class TrackedTag:
    id: int
    pos: np.ndarray  # projector px
    angle: float  # radians, direction of the tag's printed "up" in projector space
    size: float  # tag side length in projector px
    corners: np.ndarray  # 4x2 projector px
    cam_corners: np.ndarray  # 4x2 camera px
    last_seen: float

    @property
    def forward(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])


class Tracker:
    """Runs detection on worker threads; read results with tags().

    hold: seconds a tag stays alive after it was last seen (hides brief
    occlusions by hands). Small movements are smoothed to remove jitter,
    larger ones are applied immediately so moving pieces don't lag.
    """

    def __init__(self, camera, calib, hold=0.4, workers=2, jitter_px=3.0):
        self.camera, self.calib = camera, calib
        self.hold, self.jitter_px = hold, jitter_px
        self.fps = 0.0
        self._tags = {}
        self._lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._claimed = 0
        self._last_applied = 0.0
        self._t_prev = time.monotonic()
        self._running = True
        self._threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    def tags(self):
        """Dict id -> TrackedTag of currently present tags."""
        with self._lock:
            return dict(self._tags)

    def px_per_mm(self, default=1.6):
        """Projector pixels per table millimetre, estimated from the visible tags."""
        scales = [t.size / TAG_MM[t.id] for t in self.tags().values() if t.id in TAG_MM]
        return float(np.median(scales)) if scales else default

    def _next_frame(self):
        # each worker takes a different, newest frame
        with self._claim_lock:
            frame, stamp, seq = self.camera.wait_newer(self._claimed, timeout=1.0)
            self._claimed = max(self._claimed, seq)
            return frame, stamp

    def _loop(self):
        detector = TagDetector()
        while self._running:
            frame, stamp = self._next_frame()
            if frame is None:
                continue
            found = detector.detect(frame)
            self._apply(found, stamp)

    def _apply(self, found, stamp):
        now = time.monotonic()
        with self._lock:
            if stamp <= self._last_applied:
                return  # a newer frame already finished on the other worker
            self._last_applied = stamp
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._t_prev, 1e-3)
            self._t_prev = now
            for tag in found:
                corners = self.calib.cam_to_proj(tag.corners)
                pos = corners.mean(axis=0)
                up = corners[0] - corners[3]  # printed top-left minus bottom-left
                angle = float(np.arctan2(up[1], up[0]))
                size = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, 0), axis=1)))
                old = self._tags.get(tag.id)
                if old is not None and np.linalg.norm(old.pos - pos) < self.jitter_px:
                    pos = 0.6 * old.pos + 0.4 * pos
                    da = (angle - old.angle + np.pi) % (2 * np.pi) - np.pi
                    angle = old.angle + 0.4 * da
                self._tags[tag.id] = TrackedTag(tag.id, pos, angle, size, corners, tag.corners, stamp)
            for k in [k for k, t in self._tags.items() if now - t.last_seen > self.hold]:
                del self._tags[k]

    def close(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2)
