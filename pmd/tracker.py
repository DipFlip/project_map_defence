"""Background tag tracking: camera frames -> tag poses in projector pixels.

Latency matters more than anything here, so:
- camera capture and detection run in their own process: in-process threads
  fight the render loop for Python's GIL, which cost 100+ ms;
- tags we already know are re-found by cutting small crops around their
  expected position, packing the crops into one mosaic image and running the
  detector once on that (~2 ms instead of ~25 ms for the full frame);
- a full-frame scan still runs a few times a second to pick up new tags;
- positions are smoothed with a One Euro filter (strong at rest, light when
  moving); optional extrapolation along velocity can hide some of the
  camera + projector delay, but amplifies jitter, so it is off by default.
"""
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from . import DATA_DIR
from .filters import OneEuro
from .tags import TagDetector

# handheld wands: a blue spot about BUTTON_OFFSET tag heights below the tag (printed
# down, from the tag centre) is their button; each wand's exact spot position is
# measured by tools/measure_wands.py into data/wands.json.
# Covering it is a press. The button value is the spot's area found around the
# expected spot position, in tag areas: ~0.6 visible, ~0 covered.
WAND_IDS = (16, 11, 13, 12)
BUTTON_OFFSET = 2.9
WANDS_PATH = os.path.join(DATA_DIR, "wands.json")


def load_button_spots():
    """id -> (tag heights printed-down, tag heights towards printed-left) of each wand's spot."""
    spots = {k: (BUTTON_OFFSET, 0.0) for k in WAND_IDS}
    try:
        with open(WANDS_PATH) as f:
            data = json.load(f)
        data = data.get("spots", data)  # older files held only the spots
        spots.update({int(k): tuple(v) for k, v in data.items() if k.isdigit()})
    except (OSError, ValueError):
        pass
    return spots


BUTTON_SPOTS = load_button_spots()
BUTTON_SEARCH = 2.6  # +- tag heights of image searched around the expected spot
SPOT_ALONG, SPOT_ACROSS = 2.2, 1.2  # a blob counts if its centre is this close to the expected spot (tag heights)
# a pixel belongs to the (blue) spot when blue > BLUE_PER_RED * red and
# blue > BLUE_PER_GREEN * green (raw camera colours; ratios don't change when a
# hand shades the spot). Measured: spots blue/red ~2.2, blue/green ~2.1; brown
# stick 0.65 / 0.87; white table <= 1.02 / 1.05; skin ~0.63 / ~1.05. (Red spots
# were too close to skin in dim light: tools/record_buttons.py + analyze_buttons.py.)
BLUE_PER_RED, BLUE_PER_GREEN = 1.4, 1.3
SPOT_MAX_AREA = 1.5  # tag areas: a visible spot is ~0.3-0.5; anything much bigger isn't a spot


def spot_mask(b, g, r):
    """Pixels of a wand's button spot, from float channels scaled 0..1."""
    return (b > BLUE_PER_RED * r) & (b > BLUE_PER_GREEN * g) & (r + g + b > MIN_LIGHT)


MIN_LIGHT = 0.1  # ignore pixels darker than this (r+g+b, white = 3): colour is noise there
BUTTON_MIN_LEVEL = 0.15  # visible level never learned below this (a wand that starts covered)
# Thresholds as a fraction of each wand's own learned "visible" level (wands differ:
# 0.9 to 1.8 when visible), with hysteresis. Covered reads ~0.
PRESS_BELOW, RELEASE_ABOVE = 0.4, 0.6

# physical size of the black square per tag id, for estimating mm <-> px
TAG_MM = {**{i: 24.0 for i in range(10, 20)}, **{i: 30.0 for i in range(100, 135)}}


@dataclass
class TrackedTag:
    id: int
    pos: np.ndarray  # projector px
    angle: float  # radians, direction of the tag's printed top (label text upright) in projector space
    size: float  # tag side length in projector px
    corners: np.ndarray  # 4x2 projector px
    cam_corners: np.ndarray  # 4x2 camera px
    last_seen: float  # capture time (time.monotonic) of the frame it was last seen in
    vel: np.ndarray = None  # projector px / s (filtered)
    cam_vel: np.ndarray = None  # camera px / s
    raw_pos: np.ndarray = None  # unfiltered projector px of the last detection
    button: float = None  # button spot area in tag areas, ~0.6 visible, ~0 covered (wands only)
    pressed: bool = False  # button patch covered (debounced)

    @property
    def raw_cam_center(self):
        return self.cam_corners.mean(axis=0)

    @property
    def forward(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    def button_pos(self):
        """Position of this wand's button spot, in the same coordinates as pos."""
        down, left = BUTTON_SPOTS.get(self.id, (BUTTON_OFFSET, 0.0))
        c = self.corners  # printed BR, BL, TL, TR
        return self.pos + (c[0] - c[3]) * down + (c[1] - c[0]) * left


def extrapolation(vel, age, predict, size, deadband=(40, 120), max_horizon=0.3):
    """Offset to add to a filtered position: velocity times (frame age + predict).

    Below deadband[0] px/s nothing is added (so jitter on still tags isn't
    amplified), ramping to full strength at deadband[1]. Capped at 2 tag sizes.
    """
    speed = float(np.linalg.norm(vel))
    gain = np.clip((speed - deadband[0]) / (deadband[1] - deadband[0]), 0, 1)
    if gain <= 0:
        return np.zeros(2)
    delta = vel * gain * min(age + predict, max_horizon)
    n = np.linalg.norm(delta)
    if n > 2 * size:
        delta *= 2 * size / n
    return delta


class Tracker:
    """Handle to the tracking process; read results with tags().

    predict: extra seconds of extrapolation on top of the frame's age,
    covering camera exposure/transfer and projector lag. Other keyword
    arguments go to the tracking core (hold, full_scan_every, pos_filter,
    angle_filter).
    """

    def __init__(self, calib_path, exposure=None, predict=0.0, **core_args):
        self.predict = predict
        self.fps = 0.0
        self.age_ms = {"roi": 0.0, "full": 0.0}
        self._tags = {}
        self.log = None  # set to a list to record every tracker update as (receive time, tags)
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
                if self.log is not None:
                    self.log.append((time.monotonic(), self._tags))

    def tags(self, predict=True):
        """Dict id -> TrackedTag of present tags, positions extrapolated to now."""
        self._drain()
        if not predict or not self.predict:
            return dict(self._tags)
        now = time.monotonic()
        out = {}
        for k, t in self._tags.items():
            delta = extrapolation(t.vel, now - t.last_seen, self.predict, t.size)
            out[k] = replace(t, pos=t.pos + delta, corners=t.corners + delta)
        return out

    def px_per_mm(self, default=1.6):
        """Projector pixels per table millimetre, estimated from the visible tags."""
        scales = [t.size / TAG_MM[t.id] for t in self.tags(predict=False).values() if t.id in TAG_MM]
        return float(np.median(scales)) if scales else default

    def snapshot(self, path, annotate=False):
        """Ask the tracking process to save the current camera frame."""
        self._conn.send(("snapshot", (path, annotate)))

    def record(self, directory):
        """Save every raw camera frame to directory (None stops). For building datasets."""
        self._conn.send(("record", directory))

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
        cam = Camera()  # colour: the wand buttons are coloured spots
        cam.manual_exposure(exposure or FLICKER_FREE)
    except Exception as e:  # report instead of leaving the parent waiting
        conn.send(("error", repr(e)))
        return
    core = _Core(cam, calib, **core_args)
    conn.send(("ready", None))
    sent = 0
    ae = _AutoExposure(cam, core.table_box, FLICKER_FREE) if not exposure else None
    try:
        while True:
            if ae:
                ae.step()
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
                elif kind == "record":
                    cam.stop_recording()
                    if arg:
                        cam.start_recording(arg)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        core.close()
        cam.stop_recording()
        cam.auto_exposure()
        cam.close()


# camera exposure (0.1 ms units). The projector shows red, green and blue in turn,
# 120 times a second: an exposure of exactly 1/120 s sees whole cycles, so its
# light looks steady (measured colour flicker 3.6 at 8.3 ms vs 45-160 at 2-7 ms;
# blue button spots stay readable even when lit by the projector). Auto exposure
# keeps FLICKER_FREE whenever the room is dim enough, and only goes shorter
# (EXPOSURE_RANGE) when the table would be too bright at it, e.g. in daylight.
FLICKER_FREE = 83
EXPOSURE_RANGE = (12, FLICKER_FREE)


class _AutoExposure:
    """Stay at the flicker-free exposure unless the table nearly saturates there;
    below it, keep the table at a medium brightness (75th percentile around `target`).

    Metering on the brightest pixels let bright projected graphics push the
    exposure down until the (unlit) wand spots were too dark to see.
    """

    def __init__(self, cam, box, exposure, target=130, every=0.5):
        self.cam, self.box, self.exposure = cam, box, float(exposure)
        self.target, self.every = target, every
        self._next = time.monotonic() + 1.0

    def step(self):
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + self.every
        frame, _, _ = self.cam.latest()
        if frame is None:
            return
        x0, y0, x1, y1 = self.box
        # typical table brightness, not the brightest projected graphics
        p = float(np.percentile(frame[y0:y1:4, x0:x1:4].max(axis=-1) if frame.ndim == 3 else frame[y0:y1:4, x0:x1:4], 75))
        if round(self.exposure) == FLICKER_FREE:
            if p < 235:
                return  # stay locked to the projector's colour cycle unless the table nearly saturates
            ratio = 0.7
        else:
            ratio = self.target / max(p, 1.0)
            if p >= 254:
                ratio = 0.6  # saturated: the percentile says nothing, back off hard
            if abs(ratio - 1) < 0.08:
                return
        new = float(np.clip(self.exposure * np.clip(ratio, 0.6, 1.5), *EXPOSURE_RANGE))
        if new > FLICKER_FREE * 0.75:  # dim enough: lock to the projector's colour cycle
            new = FLICKER_FREE
        if round(new) != round(self.exposure):
            self.exposure = new
            self.cam.manual_exposure(round(new))


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
    occlusions by hands). Positions and angles go through One Euro filters:
    heavy smoothing at rest, little when moving fast.
    """

    def __init__(self, camera, calib, hold=0.4, workers=2, full_scan_every=0.2,
                 pos_filter=None, angle_filter=None, button_ids=WAND_IDS):
        self.camera, self.calib = camera, calib
        self.button_ids = set(button_ids)
        self._button_state = {}  # id -> pressed state, frames disagreeing, recent visible readings
        self.hold, self.full_scan_every = hold, full_scan_every
        # One Euro settings; position in projector px, angle in radians
        self.pos_filter = pos_filter or dict(min_cutoff=1.0, beta=0.08, d_cutoff=4.0)  # tools/tune_prediction.py
        self.angle_filter = angle_filter or dict(min_cutoff=1.0, beta=0.5, d_cutoff=2.0)
        self._filters = {}
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
        self._missed = False  # a known tag fell out of its crop: rescan the whole table
        self.table_box = self._table_box()
        # wands held up in the air appear shifted outwards (parallax): scan wider
        x0, y0, x1, y1 = self.table_box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = self.calib.cam_size
        self.scan_box = (max(0, int(cx - (cx - x0) * 1.5)), max(0, int(cy - (cy - y0) * 1.5)),
                         min(w, int(cx + (x1 - cx) * 1.5)), min(h, int(cy + (y1 - cy) * 1.5)))
        self._running = True
        self._threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    def _table_box(self, margin=40):
        """Camera-pixel bounding box of the calibrated table area; full scans look only here."""
        w, h = self.calib.cam_size
        if not self.calib.table_poly:
            return 0, 0, w, h
        poly = np.array(self.calib.table_poly, float)
        edge = np.concatenate([np.linspace(poly[i], poly[(i + 1) % len(poly)], 20) for i in range(len(poly))])
        cam = self.calib.proj_to_cam(edge)
        x0, y0 = np.maximum(cam.min(axis=0) - margin, 0).astype(int)
        x1, y1 = np.minimum(cam.max(axis=0) + margin, (w, h)).astype(int)
        return x0, y0, x1, y1

    def _detect_full(self, detector, frame):
        x0, y0, x1, y1 = self.scan_box
        found = detector.detect(frame[y0:y1, x0:x1])
        for tag in found:
            tag.corners = tag.corners + (x0, y0)
        return found

    def state(self):
        with self._cond:
            return dict(self._tags), self.fps, dict(self.age_ms)

    def wait_update(self, version, timeout):
        with self._cond:
            self._cond.wait_for(lambda: self._version != version, timeout)
            return self._version

    def snapshot(self, path, annotate):
        frame = self.camera.latest_color()
        if frame is None:
            return
        if annotate:
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
            color, stamp = self._next_frame()
            if color is None:
                continue
            frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            with self._cond:
                known = list(self._tags.values())
                due = self._missed or stamp - self._last_full > self.full_scan_every
                full = not known or (not self._full_running and due)
                if full:
                    self._full_running = True
                    self._last_full = stamp
                    self._missed = False
            try:
                found = self._detect_full(detector, frame) if full else self._detect_roi(detector, frame, known, stamp)
                if not full:
                    # tags seen in the previous frame but not in their crop moved too fast for it
                    got = {t.id for t in found}
                    if any(t.id not in got and stamp - t.last_seen < 0.07 for t in known):
                        self._missed = True
            finally:
                if full:
                    self._full_running = False
            # the tag and its button come from the same frame, so no speed limit is needed
            buttons = {t.id: self._sample_button(color, t.corners, BUTTON_SPOTS[t.id])
                       for t in found if t.id in self.button_ids}
            self._apply(found, stamp, "full" if full else "roi", buttons)

    @staticmethod
    def _sample_button(color, c, spot=(BUTTON_OFFSET, 0.0)):
        """Area of the button spot near where it should be, in tag areas (~0.5 visible, ~0 covered).

        The box is generous (BUTTON_SEARCH tag heights each way) because nothing
        else near a wand has the spot's colour: tilt, angle or position errors don't matter.
        Only the spot-coloured blob closest to the expected position counts, so another wand's spot in
        the box can't hide a press: it must sit on this wand's own stick axis.
        """
        h, w = color.shape[:2]
        down = c[0] - c[3]  # printed down, one tag height long
        side = float(np.linalg.norm(down))
        x, y = (c.mean(axis=0) + down * spot[0] + (c[1] - c[0]) * spot[1]).astype(int)
        r = int(side * BUTTON_SEARCH)
        if r < 4:
            return None
        x0, y0, x1, y1 = max(x - r, 0), max(y - r, 0), min(x + r, w), min(y + r, h)
        if x1 - x0 < r or y1 - y0 < r:
            return None  # mostly outside the image
        # raw camera colours: white-balancing against the box failed when a hand
        # filled it (skin became "white"), and the raw shares separate fine
        b, g, rr = np.moveaxis(color[y0:y1, x0:x1].astype(np.float32) / 255.0, -1, 0)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(spot_mask(b, g, rr).astype(np.uint8))
        if n < 2:
            return 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        # blob centres relative to the expected spot, in tag heights along / across this wand
        rel = centroids[1:] + (x0, y0) - (x, y)
        along, across = np.linalg.solve(np.stack([down, c[1] - c[0]], 1), rel.T)
        ok = ((areas >= max(3, 0.03 * side * side)) & (areas <= SPOT_MAX_AREA * side * side)  # spot-sized only
              & (np.abs(along) < SPOT_ALONG) & (np.abs(across) < SPOT_ACROSS))
        if not ok.any():
            return 0.0  # spot covered (another wand's spot beside this stick doesn't count)
        best = np.argmin(np.where(ok, np.hypot(along, across), np.inf))
        return float(areas[best]) / (side * side)

    @staticmethod
    def _detect_roi(detector, frame, known, stamp):
        h, w = frame.shape[:2]
        boxes = []
        for t in known:
            c = t.cam_corners + t.cam_vel * (stamp - t.last_seen)
            side = float(np.linalg.norm(c[1] - c[0]))
            # search window: the crop around where we expect the tag. A piece moved
            # at 1.5 m/s travels ~50 camera px between frames, so besides a fixed
            # border we add 1.5 frames of motion at its current speed.
            margin = max(1.5 * side, 40) + np.linalg.norm(t.cam_vel) * 0.05
            x0, y0 = np.maximum(c.min(axis=0) - margin, 0).astype(int)
            x1, y1 = np.minimum(c.max(axis=0) + margin, (w, h)).astype(int)
            if x1 - x0 > 8 and y1 - y0 > 8:
                boxes.append((x0, y0, x1, y1))
        if not boxes:
            return []
        img, offsets = _mosaic(frame, boxes)
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

    def _apply(self, found, stamp, kind, buttons=None):
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
                # OpenCV's corner 0 is the printed bottom-right of the print kit's
                # tags (checked on the kit PDF), so corner 3 -> 0 points printed-down
                up = corners[3] - corners[0]
                angle = float(np.arctan2(up[1], up[0]))
                size = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, 0), axis=1)))
                cam_vel = np.zeros(2)
                dt = stamp - old.last_seen if old is not None else 0.0
                if old is None or dt > 0.25:  # new or reappeared: start the filters fresh
                    self._filters[tag.id] = [OneEuro(**self.pos_filter), OneEuro(**self.angle_filter)]
                    dt = 0.0
                else:
                    # raw camera-space velocity, only used to place the ROI crop
                    cam_vel = 0.6 * (tag.center - old.raw_cam_center) / dt + 0.4 * old.cam_vel
                    angle = old.angle + (angle - old.angle + np.pi) % (2 * np.pi) - np.pi  # unwrap
                f_pos, f_angle = self._filters[tag.id]
                raw_pos = pos
                pos = f_pos(raw_pos, dt).copy()
                angle = float(f_angle(angle, dt))
                vel = f_pos.dx.copy()
                corners = corners + (pos - raw_pos)
                ratio, pressed = (buttons or {}).get(tag.id), False
                if tag.id in self.button_ids:
                    pressed = self._debounce(tag.id, ratio)
                self._tags[tag.id] = TrackedTag(tag.id, pos, angle, size, corners, tag.corners, stamp, vel, cam_vel,
                                                raw_pos, ratio, pressed)
            for k in [k for k, t in self._tags.items() if now - t.last_seen > self.hold]:
                del self._tags[k]
            self._version += 1
            self._cond.notify_all()

    def _debounce(self, tag_id, value, press_frames=2, release_frames=4):
        """Pressed state from button values, relative to this wand's visible level.

        The level is the median of the wand's recent visible readings (robust to
        single spikes). (Ignoring presses from wands lying still removed arm-over-
        wand presses in recordings, but blocked real presses on wands lying on the
        table, so it's gone.)
        """
        st = self._button_state.setdefault(tag_id, {"pressed": False, "count": 0, "hist": deque(maxlen=90)})
        if value is None:
            return st["pressed"]
        hist = st["hist"]
        level = max(BUTTON_MIN_LEVEL, float(np.median(hist))) if len(hist) >= 5 else max(value, BUTTON_MIN_LEVEL)
        pressed = st["pressed"]
        flip = value < PRESS_BELOW * level if not pressed else value > RELEASE_ABOVE * level
        st["count"] = st["count"] + 1 if flip else 0
        if st["count"] >= (release_frames if pressed else press_frames):
            st["pressed"], st["count"] = not pressed, 0  # slower release: no double presses from a flicker
        if not st["pressed"] and value > RELEASE_ABOVE * level:
            hist.append(value)
        return st["pressed"]

    def close(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2)
