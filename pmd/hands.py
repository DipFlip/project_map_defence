"""Webcam hand tracking (MediaPipe hand landmarker) for the laptop games.

A background thread grabs webcam frames, mirrors them (so the screen acts like
a mirror), runs MediaPipe's hand landmarker and publishes each frame together
with the hands found in it. Each hand says whether it is making a finger gun:
index finger out, thumb up, the other three fingers curled.

The landmark model (about 8 MB) is downloaded to data/ on first use.
"""
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import DATA_DIR

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(DATA_DIR, "hand_landmarker.task")

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))

STRAIGHT = 0.85  # finger counts as extended above this straightness (knuckle-to-tip / bone length sum)
CURLED = 0.70  # ... and as curled below this
THUMB_OUT = 0.52  # thumb tip at least this far from the index knuckle (x wrist-to-middle-knuckle length)


def ensure_model(path=MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading the hand model to {path} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, path)
    return path


def _straightness(w, idx):
    a, b, c, d = (w[i] for i in idx)
    bones = np.linalg.norm(b - a) + np.linalg.norm(c - b) + np.linalg.norm(d - c)
    return float(np.linalg.norm(d - a) / max(bones, 1e-6))


def finger_gun(world):
    """True for a finger gun, from the 21 metric 3D landmarks (independent of where the hand points)."""
    scale = np.linalg.norm(world[MIDDLE_MCP] - world[WRIST])
    index_out = _straightness(world, FINGERS["index"]) > STRAIGHT
    middle_in = _straightness(world, FINGERS["middle"]) < CURLED  # not a peace sign
    others_in = any(_straightness(world, FINGERS[f]) < CURLED for f in ("ring", "pinky"))
    thumb_out = np.linalg.norm(world[THUMB_TIP] - world[INDEX_MCP]) > THUMB_OUT * scale
    return bool(index_out and thumb_out and middle_in and others_in)


@dataclass
class Hand:
    pts: np.ndarray  # 21x2 landmark positions in the (mirrored) frame, normalised 0..1
    gun: bool

    @property
    def thumb(self):
        return self.pts[THUMB_TIP]

    @property
    def index(self):
        return self.pts[INDEX_TIP]


@dataclass
class Frame:
    image: np.ndarray  # HxWx3 RGB, mirrored
    hands: list = field(default_factory=list)
    t: float = 0.0  # time.monotonic() when the frame was grabbed
    n: int = 0  # frame counter


class HandTracker:
    def __init__(self, camera=0, size=(1280, 720), max_hands=4, mirror=True):
        from mediapipe.tasks.python import BaseOptions, vision

        self.mirror = mirror
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {camera} (on macOS, allow camera access for your terminal "
                               "in System Settings > Privacy & Security > Camera)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._frame = None
        self.fps = 0.0
        self.error = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._frame

    def _run(self):
        import mediapipe as mp

        t0 = time.monotonic()
        last_ts, n, fps_t, fps_n = -1, 0, t0, 0
        while self._running:
            ok, bgr = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            if self.mirror:
                bgr = cv2.flip(bgr, 1)
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = max(int((t - t0) * 1000), last_ts + 1)  # VIDEO mode wants strictly increasing timestamps
            last_ts = ts
            try:
                res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            except Exception as e:  # keep the camera running; show the error
                self.error = str(e)
                continue
            hands = []
            for lm, wl in zip(res.hand_landmarks, res.hand_world_landmarks):
                pts = np.array([(p.x, p.y) for p in lm], float)
                world = np.array([(p.x, p.y, p.z) for p in wl], float)
                hands.append(Hand(pts, finger_gun(world)))
            n += 1
            with self._lock:
                self._frame = Frame(rgb, hands, t, n)
            fps_n += 1
            if t - fps_t > 1.0:
                self.fps, fps_t, fps_n = fps_n / (t - fps_t), t, 0

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
        self.landmarker.close()
