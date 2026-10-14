"""Camera <-> projector calibration for a flat table.

The table is a plane, so after removing the camera's lens distortion the
mapping camera -> projector pixels is a single homography. We find it by
projecting ArUco markers (plus their inverse, and subtracting the two
captures to cancel ambient light and paper texture), collecting matching
corner points, and fitting radial distortion + homography jointly.

Projector pixels are the canonical "world" coordinates of the game: anything
drawn at (x, y) on screen lands on the table where the camera sees
proj_to_cam((x, y)).
"""
import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

PATTERN_DICT = cv2.aruco.DICT_4X4_250


@dataclass
class Calibration:
    cam_size: tuple
    proj_size: tuple
    K: np.ndarray  # nominal camera matrix (focal is a fixed guess; only dist is fit)
    dist: np.ndarray  # k1, k2, p1, p2, k3
    H: np.ndarray  # undistorted camera px -> projector px
    exposure: int = 60
    rms_px: float = 0.0
    footprint: list = field(default_factory=list)  # projector outline in camera px
    table_poly: list = field(default_factory=list)  # calibrated table area, projector px
    table_H: list = None  # projector px -> table mm, table edges along the axes (pmd/table.py)
    table_size_mm: list = None  # table rectangle (long, short) in mm

    # --- point mapping -------------------------------------------------
    def undistort(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def distort(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        norm = (pts - self.K[:2, 2]) / [self.K[0, 0], self.K[1, 1]]
        obj = np.hstack([norm, np.ones((len(norm), 1))])
        out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)

    def cam_to_proj(self, pts):
        und = self.undistort(pts).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(und, self.H).reshape(-1, 2)

    def proj_to_cam(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)
        return self.distort(und)

    def warp_camera_to_projector(self, frame):
        """Resample a camera frame into projector pixel space (for debug views)."""
        if not hasattr(self, "_maps"):
            w, h = self.proj_size
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            cam = self.proj_to_cam(np.stack([xs.ravel(), ys.ravel()], 1)).astype(np.float32)
            self._maps = cam[:, 0].reshape(h, w), cam[:, 1].reshape(h, w)
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR)

    # --- persistence ---------------------------------------------------
    def save(self, path):
        d = {
            "cam_size": list(self.cam_size), "proj_size": list(self.proj_size),
            "K": self.K.tolist(), "dist": self.dist.ravel().tolist(), "H": self.H.tolist(),
            "exposure": self.exposure, "rms_px": self.rms_px, "footprint": self.footprint, "table_poly": self.table_poly,
            "table_H": None if self.table_H is None else np.asarray(self.table_H).tolist(),
            "table_size_mm": self.table_size_mm,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(tuple(d["cam_size"]), tuple(d["proj_size"]), np.array(d["K"]),
                   np.array(d["dist"]), np.array(d["H"]), d["exposure"], d["rms_px"],
                   d.get("footprint", []), d.get("table_poly", []), d.get("table_H"), d.get("table_size_mm"))


def nominal_K(cam_size, focal=None):
    w, h = cam_size
    f = focal or 0.55 * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def fit(groups, cam_size, table=0):
    """Jointly fit radial distortion (k1, k2, k3) and one homography per flat surface.

    groups: [(camera points, projector points)] per surface (table, floor, ...).
    The lens is the same for all of them, so every surface helps pin down its
    distortion even when the table covers only part of the image. Returns the
    table's homography, its rms error and the table points that were kept.
    """
    K = nominal_K(cam_size)
    groups = [(np.asarray(c, np.float64), np.asarray(q, np.float64)) for c, q in groups]
    keep = [np.ones(len(c), bool) for c, _ in groups]

    def homs(p):
        return [np.append(p[3 + 8 * i:11 + 8 * i], 1).reshape(3, 3) for i in range(len(groups))]

    def errors(p):
        dist = np.array([p[0], p[1], 0, 0, p[2]])
        out = []
        for (c, q), H, k in zip(groups, homs(p), keep):
            und = cv2.undistortPoints(c[k].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            h = np.hstack([und, np.ones((len(und), 1))]) @ H.T
            out.append(h[:, :2] / h[:, 2:] - q[k])
        return out

    def residual(p):
        return np.concatenate([e.ravel() for e in errors(p)])

    p0 = [0.0, 0.0, 0.0]
    for c, q in groups:
        H0, _ = cv2.findHomography(c, q, 0)
        p0 += list((H0 / H0[2, 2]).ravel()[:8])
    sol = least_squares(residual, np.array(p0), loss="soft_l1", f_scale=3.0, x_scale="jac")
    # drop gross outliers (misdetected corners) per surface and refit
    for i, e in enumerate(errors(sol.x)):
        err = np.linalg.norm(e, axis=1)
        idx = np.nonzero(keep[i])[0]
        keep[i][idx[err >= max(5.0, 4 * np.median(err))]] = False
    sol = least_squares(residual, sol.x, x_scale="jac")
    err = np.linalg.norm(errors(sol.x)[table], axis=1)
    dist = np.array([sol.x[0], sol.x[1], 0, 0, sol.x[2]])
    return K, dist, homs(sol.x)[table], float(np.sqrt(np.mean(err ** 2))), groups[table][1][keep[table]]


# --- capture procedure ---------------------------------------------------

def capture(cam, proj, img, settle=0.25, n=2):
    """Show img on the projector and return the mean of n fresh camera frames (float32 gray)."""
    proj.show_gray(img)
    t = time.monotonic() + settle
    frame = cam.frame_after(t)
    acc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    _, _, seq = cam.latest()
    for _ in range(n - 1):
        frame, _, seq = cam.wait_newer(seq)
        acc += cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return acc / n


def choose_exposure(cam, proj, candidates=(200, 150, 110, 80, 60, 45, 33, 25, 18), log=print):
    W, H = proj.size
    white = np.full((H, W), 255, np.uint8)
    black = np.zeros((H, W), np.uint8)
    chosen = candidates[-1]
    for exp in candidates:
        cam.manual_exposure(exp)
        w = capture(cam, proj, white, settle=0.4)
        b = capture(cam, proj, black, settle=0.3)
        lit = (w - b) > 25
        if lit.sum() < 1000:
            continue
        p99 = np.percentile(w[lit], 99)
        log(f"  exposure {exp}: lit white p99={p99:.0f}")
        if p99 < 235:
            chosen = exp
            break
    cam.manual_exposure(chosen)
    return chosen


def marker_pattern(size, marker_px, spacing, offset, dictionary, first_id=0):
    """White image with a grid of markers; returns image and {id: 4x2 corners}."""
    W, H = size
    img = np.full((H, W), 255, np.uint8)
    corners = {}
    k = first_id
    y = offset[1]
    while y + marker_px <= H:
        x = offset[0]
        while x + marker_px <= W:
            if k >= dictionary.bytesList.shape[0]:
                return img, corners
            img[y:y + marker_px, x:x + marker_px] = cv2.aruco.generateImageMarker(dictionary, k, marker_px)
            # outer corners of the black border, matching detectMarkers ordering
            corners[k] = np.array([[x, y], [x + marker_px, y], [x + marker_px, y + marker_px], [x, y + marker_px]],
                                  np.float64) - 0.5
            k += 1
            x += spacing
        y += spacing
    return img, corners


def detect_projected(cam, proj, img, dictionary):
    """Capture pattern and inverse, return detected markers in the difference image."""
    a = capture(cam, proj, img)
    b = capture(cam, proj, 255 - img)
    diff = a - b
    scale = 110.0 / max(np.percentile(np.abs(diff), 99), 1)
    norm = np.clip(128 + diff * scale, 0, 255).astype(np.uint8)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = det.detectMarkers(norm)
    found = {} if ids is None else {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}
    return found, norm


def run(cam, proj, log=print, debug_dir=None, marker_px=90, spacing=150):
    W, H = proj.size
    dictionary = cv2.aruco.getPredefinedDictionary(PATTERN_DICT)
    log("choosing exposure...")
    exposure = choose_exposure(cam, proj, log=log)
    log(f"exposure = {exposure}")

    cam_pts, proj_pts = [], []
    offsets = [(30, 30), (30 + spacing // 2, 30 + spacing // 2)]
    first_id = 0
    for i, off in enumerate(offsets):
        img, truth = marker_pattern((W, H), marker_px, spacing, off, dictionary, first_id)
        found, norm = detect_projected(cam, proj, img, dictionary)
        good = [k for k in found if k in truth]
        log(f"pass {i}: {len(good)}/{len(truth)} markers seen")
        for k in good:
            cam_pts.extend(found[k])
            proj_pts.extend(truth[k])
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/calib_pass{i}.jpg", norm)
        first_id = 0  # same ids per pass is fine, each pass is matched separately

    if len(cam_pts) < 40:
        raise RuntimeError(f"only {len(cam_pts)} correspondences; check focus/exposure/aim")

    # projector footprint outline in the camera (white minus black)
    w = capture(cam, proj, np.full((H, W), 255, np.uint8))
    b = capture(cam, proj, np.zeros((H, W), np.uint8))
    proj.fill((0, 0, 0))

    cam_size = (int(w.shape[1]), int(w.shape[0]))
    cam_pts, proj_pts, lit = np.array(cam_pts), np.array(proj_pts), np.clip(w - b, 0, 255)
    groups, table = surfaces(cam_pts, proj_pts, lit, log)
    K, dist, Hm, rms, used = fit(groups, cam_size, table)
    # split again on lens-corrected points with a tight tolerance: a low table and the
    # floor right next to it are nearly the same plane under the loose first split
    und = cv2.undistortPoints(cam_pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    groups, table = surfaces(cam_pts, proj_pts, lit, log, min_points=20, thr=8.0, fit_pts=und)
    K, dist, Hm, rms, used = fit(groups, cam_size, table)
    log(f"fit: {len(used)}/{len(groups[table][0])} table points, rms {rms:.2f} projector px, "
        f"dist {np.round(dist[[0, 1, 4]], 4)}")
    calib = Calibration(cam_size, (W, H), K, dist, Hm, exposure, rms)
    outline = [(0, 0), (W, 0), (W, H), (0, H)]
    edge = np.concatenate([np.linspace(outline[i], outline[(i + 1) % 4], 50) for i in range(4)])
    calib.footprint = calib.proj_to_cam(edge).round(1).tolist()
    calib.table_poly = cv2.convexHull(used.astype(np.float32)).reshape(-1, 2).round(1).tolist()
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/calib_lit.jpg", np.clip(w - b, 0, 255).astype(np.uint8))
    return calib


def surfaces(cam_pts, proj_pts, lit, log=print, min_points=40, thr=25.0, fit_pts=None):
    """Split marker points by flat surface; returns (groups, index of the table).

    Each flat surface (table, floor, a wall) is a separate homography. The table
    is taken to be the brightest under projector white (a light table over a
    darker floor; light paper or cloth on the table helps).
    """
    fit_pts = cam_pts if fit_pts is None else fit_pts  # e.g. undistorted points for a tighter split
    rest = np.arange(len(cam_pts))
    planes = []
    while len(rest) >= min_points:
        _, inl = cv2.findHomography(fit_pts[rest], proj_pts[rest], cv2.RANSAC, thr)
        if inl is None or inl.sum() < min_points:
            break
        inl = inl.ravel().astype(bool)
        planes.append(rest[inl])
        rest = rest[~inl]
    if not planes:
        return [(cam_pts, proj_pts)], 0
    h, w = lit.shape[:2]

    def brightness(idx):
        xy = np.clip(cam_pts[idx].round().astype(int), 0, (w - 1, h - 1))
        return float(np.median(lit[xy[:, 1], xy[:, 0]]))

    scores = [brightness(p) for p in planes]
    best = int(np.argmax(scores))
    if len(planes) > 1:
        log("surfaces: " + ", ".join(f"{len(p)} points, brightness {sc:.0f}" for p, sc in zip(planes, scores))
            + f" -> the table is the brightest ({len(planes[best])} points)")
    return [(cam_pts[p], proj_pts[p]) for p in planes], best


def table_frame(cam, proj, calib, log=print, debug_dir=None):
    """Measure the metric table frame (see pmd/table.py) and store it in calib.

    Needs at least 4 tags lying flat and spread over the table, and nobody
    leaning over it (shadows change the table outline).
    """
    from .table import align_to_outline, fit_metric, lit_outline
    from .tags import TagDetector
    from .tracker import TAG_MM

    W, H = proj.size
    cam.manual_exposure(calib.exposure)
    gray = capture(cam, proj, np.full((H, W), 110, np.uint8), settle=0.5)
    table = np.array(calib.table_poly, np.float32)
    found = [t for t in TagDetector().detect(gray.astype(np.uint8)) if t.id in TAG_MM
             and cv2.pointPolygonTest(table, tuple(float(v) for v in calib.cam_to_proj(t.corners.mean(axis=0))[0]),
                                      True) > -40]  # only tags lying on the table (not the floor)
    tags = [(calib.cam_to_proj(t.corners), TAG_MM[t.id]) for t in found]
    log(f"table frame: {len(tags)} tags of known size: {sorted(t.id for t in found)}")
    px_per_mm = np.median([np.linalg.norm(c[1] - c[0]) / mm for c, mm in tags]) if tags else 1.7
    Hm, rms = fit_metric(tags, px_per_mm)
    log(f"table frame: tags square to {rms:.2f} mm rms")
    # more frames average out the projector's flicker stripes (a longer exposure
    # would saturate the white table even with the projector off)
    w = capture(cam, proj, np.full((H, W), 255, np.uint8), settle=0.6, n=8)
    b = capture(cam, proj, np.zeros((H, W), np.uint8), settle=0.6, n=8)
    proj.fill((0, 0, 0))
    outline = lit_outline(np.clip(w - b, 0, 255), calib.cam_to_proj)
    markers = np.array(calib.table_poly, np.float32)
    if outline is None or cv2.contourArea(np.float32(outline)) > 1.6 * cv2.contourArea(markers):
        # no lit area, or the lit area spans more than the table (floor, walls): use the
        # area covered by the table's calibration markers instead
        log("table frame: using the calibration markers' area as the table outline")
        outline = markers
    Ht, size = align_to_outline(Hm, outline)
    calib.table_H, calib.table_size_mm = Ht.tolist(), [round(size[0], 1), round(size[1], 1)]
    log(f"table frame: table {size[0]:.0f} x {size[1]:.0f} mm")
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/table_lit.jpg", np.clip(w - b, 0, 255).astype(np.uint8))
    return calib
