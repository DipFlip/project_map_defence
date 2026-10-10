"""Threaded USB camera capture that always exposes the newest frame."""
import glob
import os
import subprocess
import threading
import time

import cv2


def find_camera(name_hint="U20CAM"):
    """Return the /dev/videoN capture node for the USB camera."""
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = open(os.path.join(node, "name")).read()
            index = int(open(os.path.join(node, "index")).read())
        except OSError:
            continue
        if name_hint in name and index == 0:
            return "/dev/" + os.path.basename(node)
    raise RuntimeError(f"no camera matching {name_hint!r}")


class Camera:
    def __init__(self, device=None, width=1920, height=1080, buffers=2, gray=False):
        """gray=True decodes only the luma channel (about half the decode time);
        latest_color() still gives a colour image of the newest frame."""
        self.device = device or find_camera()
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # 2 buffers keeps latency low; 1 buffer makes the camera drop to 15 fps
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffers)
        self.gray = gray
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # hand us the raw JPEG, we decode ourselves
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.device}")
        self._cond = threading.Condition()
        self._frame = None
        self._raw = None
        self._stamp = 0.0
        self._count = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, raw = self.cap.read()
            frame = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR) if ok else None
            if frame is None:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._raw, self._stamp = frame, raw, time.monotonic()
                self._count += 1
                self._cond.notify_all()

    def latest(self):
        """(frame, timestamp, sequence number) of the newest frame, frame may be None."""
        with self._cond:
            return self._frame, self._stamp, self._count

    def latest_color(self):
        with self._cond:
            raw = self._raw
        return None if raw is None else cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def wait_newer(self, seq, timeout=2.0):
        with self._cond:
            self._cond.wait_for(lambda: self._count > seq, timeout)
            return self._frame, self._stamp, self._count

    def frame_after(self, t, timeout=3.0):
        """First frame whose capture started after monotonic time t."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._stamp <= t:
                if not self._cond.wait(deadline - time.monotonic()):
                    raise TimeoutError("camera stalled")
            return self._frame

    def set_controls(self, **controls):
        """Set v4l2 controls, e.g. auto_exposure=1, exposure_time_absolute=60."""
        args = ["v4l2-ctl", "-d", self.device]
        for k, v in controls.items():
            args += ["-c", f"{k}={int(v)}"]
        subprocess.run(args, check=True)

    def manual_exposure(self, exposure):
        self.set_controls(auto_exposure=1, exposure_time_absolute=exposure, exposure_dynamic_framerate=0)

    def auto_exposure(self):
        self.set_controls(auto_exposure=3)

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()
