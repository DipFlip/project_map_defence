"""Background tag tracking: camera frames -> tag poses in projector pixels.

Latency matters more than anything here, so:
- camera capture and detection run in their own process: in-process threads
  fight the render loop for Python's GIL, which cost 100+ ms;
- tags we already know are re-found by cutting small crops around their
  expected position, packing the crops into one mosaic image and running the
  detector once on that (~2 ms instead of ~25 ms for the full frame);
- a full-frame scan still runs a few times a second to pick up new tags;
- positions are extrapolated along each tag's velocity to hide the
  camera + processing + projector delay (~150 ms in total on this rig).
"""
import multiprocessing as mp
import sys
import threading
import time
from dataclasses import dataclass, replace

import cv2
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
    last_seen: float  # capture time (time.monotonic) of the frame it was last seen in
    vel: np.ndarray = None  # projector px / s
    cam_vel: np.ndarray = None  # camera px / s

    @property
    def forward(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])


class Tracker:
    """Handle to the tracking process; read results with tags().

    predict: extra seconds of extrapolation on top of the frame's age,
    covering camera exposure/transfer and projector lag. Other keyword
    arguments go to the tracking core (hold, jitter_px, full_scan_every).
    """

    def __init__(self, calib_path, exposure=None, predict=0.10, **core_args):
        self.predict = predict
        self.fps = 0.0
        self.age_ms = {"roi": 0.0, "full": 0.0}
        self._tags = {}
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_child_main, args=(child, calib_path, exposure, core_args), daemon=True)
        self._proc.start()
        msg = self._conn.recv()  # wait until the camera is running
        if msg[0] == "error":
            raise RuntimeError(msg[1])

    def _drain(self):
        while self._conn.poll():
            kind, data = self._conn.recv()
            if kind == "tags":
                self._tags, self.fps, self.age_ms = data

    def tags(self, predict=True):
        """Dict id -> TrackedTag of present tags, positions extrapolated to now."""
        self._drain()
        if not predict:
            return dict(self._tags)
        now = time.monotonic()
        out = {}
        for k, t in self._tags.items():
            speed = float(np.linalg.norm(t.vel))
            # deadband so detection jitter doesn't get amplified on still tags
            gain = np.clip((speed - 40) / 80, 0, 1)
            if gain > 0:
                horizon = min(now - t.last_seen + self.predict, 0.3)
                delta = t.vel * gain * horizon
                n = np.linalg.norm(delta)
                if n > 2 * t.size:
                    delta *= 2 * t.size / n
                t = replace(t, pos=t.pos + delta, corners=t.corners + delta)
            out[k] = t
        return out

    def px_per_mm(self, default=1.6):
        """Projector pixels per table millimetre, estimated from the visible tags."""
        scales = [t.size / TAG_MM[t.id] for t in self.tags(predict=False).values() if t.id in TAG_MM]
        return float(np.median(scales)) if scales else default

    def snapshot(self, path, annotate=False):
        """Ask the tracking process to save the current camera frame."""
        self._conn.send(("snapshot", (path, annotate)))

    def close(self):
        try:
            self._conn.send(("close", None))
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()


# --- tracking process ------------------------------------------------------

def _child_main(conn, calib_path, exposure, core_args):
    from .calibration import Calibration
    from .camera import Camera

    sys.setswitchinterval(0.001)
    cv2.setNumThreads(4)  # the little cores on this SoC slow OpenCV's pool down; 4 is fastest
    try:
        calib = Calibration.load(calib_path)
        cam = Camera()
        cam.manual_exposure(exposure or calib.exposure * 2)
    except Exception as e:  # report instead of leaving the parent waiting
        conn.send(("error", repr(e)))
        return
    core = _Core(cam, calib, **core_args)
    conn.send(("ready", None))
    sent = 0
    try:
        while True:
            version = core.wait_update(sent, timeout=0.1)
            if version != sent:
                sent = version
                conn.send(("tags", core.state()))
            while conn.poll():
                kind, arg = conn.recv()
                if kind == "close":
                    return
                if kind == "snapshot":
                    core.snapshot(*arg)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        core.close()
        cam.auto_exposure()
        cam.close()


def _mosaic(gray, boxes, pad=8):
    """Pack crops into one white image; returns image and per-crop offsets."""
    cell = max(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes) + pad
    cols = int(np.ceil(np.sqrt(len(boxes))))
    rows = int(np.ceil(len(boxes) / cols))
    img = np.full((rows * cell + pad, cols * cell + pad), 255, np.uint8)
    offsets = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        oy, ox = pad + (i // cols) * cell, pad + (i % cols) * cell
        img[oy:oy + y1 - y0, ox:ox + x1 - x0] = gray[y0:y1, x0:x1]
        offsets.append((ox - x0, oy - y0, ox, oy, ox + x1 - x0, oy + y1 - y0))
    return img, offsets


class _Core:
    """Detection workers inside the tracking process.

    hold: seconds a tag stays alive after it was last seen (hides brief
    occlusions by hands). Small movements are smoothed to remove jitter.
    """

    def __init__(self, camera, calib, hold=0.4, workers=2, jitter_px=3.0, full_scan_every=0.2):
        self.camera, self.calib = camera, calib
        self.hold, self.jitter_px, self.full_scan_every = hold, jitter_px, full_scan_every
        self.fps = 0.0  # frames processed per second
        self.age_ms = {"roi": 0.0, "full": 0.0}  # frame capture -> pose available
        self._n_frames, self._fps_t0 = 0, time.monotonic()
        self._tags = {}
        self._version = 0
        self._cond = threading.Condition()
        self._claim_lock = threading.Lock()
        self._claimed = 0
        self._full_running = False
        self._last_full = 0.0
        self._running = True
        self._threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    def state(self):
        with self._cond:
            return dict(self._tags), self.fps, dict(self.age_ms)

    def wait_update(self, version, timeout):
        with self._cond:
            self._cond.wait_for(lambda: self._version != version, timeout)
            return self._version

    def snapshot(self, path, annotate):
        frame, _, _ = self.camera.latest()
        if frame is None:
            return
        if annotate:
            frame = frame.copy()
            for t in self.state()[0].values():
                cv2.polylines(frame, [t.cam_corners.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(frame, [np.array(self.calib.footprint, np.int32)], True, (255, 0, 255), 2)
        cv2.imwrite(path, frame)

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
            with self._cond:
                known = list(self._tags.values())
                full = not known or (not self._full_running and stamp - self._last_full > self.full_scan_every)
                if full:
                    self._full_running = True
                    self._last_full = stamp
            try:
                found = detector.detect(frame) if full else self._detect_roi(detector, frame, known, stamp)
            finally:
                if full:
                    self._full_running = False
            self._apply(found, stamp, "full" if full else "roi")

    @staticmethod
    def _detect_roi(detector, frame, known, stamp):
        h, w = frame.shape[:2]
        boxes = []
        for t in known:
            c = t.cam_corners + t.cam_vel * (stamp - t.last_seen)
            side = float(np.linalg.norm(c[1] - c[0]))
            margin = side + np.linalg.norm(t.cam_vel) * 0.04  # one frame of motion
            x0, y0 = np.maximum(c.min(axis=0) - margin, 0).astype(int)
            x1, y1 = np.minimum(c.max(axis=0) + margin, (w, h)).astype(int)
            if x1 - x0 > 8 and y1 - y0 > 8:
                boxes.append((x0, y0, x1, y1))
        if not boxes:
            return []
        gray = np.empty(frame.shape[:2], np.uint8)
        for x0, y0, x1, y1 in boxes:  # only convert what we look at
            gray[y0:y1, x0:x1] = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        img, offsets = _mosaic(gray, boxes)
        out, seen = [], set()
        for tag in detector.detect(img):
            cx, cy = tag.center
            for dx, dy, ax, ay, bx, by in offsets:
                # only accept tags fully inside their own crop (crops may overlap)
                if ax <= cx < bx and ay <= cy < by and tag.id not in seen:
                    tag.corners = tag.corners - (dx, dy)
                    out.append(tag)
                    seen.add(tag.id)
                    break
        return out

    def _apply(self, found, stamp, kind):
        now = time.monotonic()
        with self._cond:
            self._n_frames += 1
            if now - self._fps_t0 > 1.0:
                self.fps = self._n_frames / (now - self._fps_t0)
                self._n_frames, self._fps_t0 = 0, now
            self.age_ms[kind] = 0.9 * self.age_ms[kind] + 0.1 * (now - stamp) * 1000
            for tag in found:
                old = self._tags.get(tag.id)
                if old is not None and stamp <= old.last_seen:
                    continue  # a newer frame already updated this tag
                corners = self.calib.cam_to_proj(tag.corners)
                pos = corners.mean(axis=0)
                up = corners[0] - corners[3]  # printed top-left minus bottom-left
                angle = float(np.arctan2(up[1], up[0]))
                size = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, 0), axis=1)))
                vel, cam_vel = np.zeros(2), np.zeros(2)
                if old is not None:
                    dt = stamp - old.last_seen
                    if dt < 0.25:
                        a = 0.6
                        vel = a * (pos - old.pos) / dt + (1 - a) * old.vel
                        cam_vel = a * (tag.center - old.cam_corners.mean(axis=0)) / dt + (1 - a) * old.cam_vel
                    if np.linalg.norm(old.pos - pos) < self.jitter_px:
                        pos = 0.6 * old.pos + 0.4 * pos
                        da = (angle - old.angle + np.pi) % (2 * np.pi) - np.pi
                        angle = old.angle + 0.4 * da
                self._tags[tag.id] = TrackedTag(tag.id, pos, angle, size, corners, tag.corners, stamp, vel, cam_vel)
            for k in [k for k, t in self._tags.items() if now - t.last_seen > self.hold]:
                del self._tags[k]
            self._version += 1
            self._cond.notify_all()

    def close(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2)
