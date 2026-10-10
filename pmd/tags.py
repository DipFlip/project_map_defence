"""AprilTag (tag36h11) detection on the table."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tag:
    id: int
    corners: np.ndarray  # 4x2, camera pixels, order TL TR BR BL as printed

    @property
    def center(self):
        return self.corners.mean(axis=0)

    @property
    def side_px(self):
        return float(np.linalg.norm(self.corners[1] - self.corners[0]))


class TagDetector:
    def __init__(self, family=cv2.aruco.DICT_APRILTAG_36h11):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(family), params)

    def detect(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        return [Tag(int(i), c.reshape(4, 2)) for c, i in zip(corners, ids.ravel())]
