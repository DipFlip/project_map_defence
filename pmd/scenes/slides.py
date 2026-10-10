"""Slides about the project, shown on the table. Any wand press goes to the next
slide; after the last one it's back to the game select screen.

1. the title slide (slides/title_slide.png)
2. what the camera sees, live, with what the tracker found drawn on it
3. what the camera saw during calibration (projected markers, data/calib_pass0.jpg)
4. how a game frame is pre-distorted before it is sent to the projector
5. the last slide (slides/last_slide.png)
"""
import os
import time

import cv2
import numpy as np

from .. import ROOT, DATA_DIR, ui
from ..projector import pygame
from ..tracker import WAND_IDS

SLIDE_DIR = os.path.join(ROOT, "slides")
DIM = 0.75  # slides are shown a bit dimmed: lots of bright white makes the camera shorten its exposure


def _surface(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))


def _load(path):
    img = cv2.imread(path)
    return None if img is None else _surface((img * DIM).astype(np.uint8))


class Slides:
    title = "Slides"

    def __init__(self, app):
        self.app, self.mm = app, app.px_per_mm
        self.font = pygame.font.SysFont(None, int(24 * self.mm))
        self.small = pygame.font.SysFont(None, int(13 * self.mm))
        self.title_img = _load(os.path.join(SLIDE_DIR, "title_slide.png"))
        self.last_img = _load(os.path.join(SLIDE_DIR, "last_slide.png"))
        self.calib_img = None
        for name in ("calib_pass0.jpg", "calib_lit.jpg"):
            img = cv2.imread(os.path.join(DATA_DIR, name))
            if img is not None:
                self.calib_img = _surface(img)
                break
        self.pages = [self._title, self._camera, self._calibration, self._warp, self._last]
        self.page = 0
        self.camera_img = None
        self._next_frame_request = 0.0
        self.warp_imgs = None
        self.changed = time.monotonic()

    # --- navigation ------------------------------------------------------
    def update(self, dt, now, tags, presses):
        if presses and now - self.changed > 0.4:
            k = next(iter(presses))
            self.app.fx.burst(tags[k].pos, (255, 255, 255), n=14, speed=200, life=0.35)
            self.app.sound.play("click")
            self.page += 1
            self.changed = now
            if self.page >= len(self.pages):
                self.app.switch("menu")
                return
        if self.pages[self.page] == self._camera and now >= self._next_frame_request:
            self.app.tracker.request_frame()
            self._next_frame_request = now + 0.3
        jpeg = getattr(self.app.tracker, "frame_jpeg", None)
        if jpeg is not None:
            self.app.tracker.frame_jpeg = None
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.camera_img = _surface(self._annotate(frame))

    # --- layout ------------------------------------------------------------
    def area(self):
        """Slide area: the table minus a margin, 16:9-ish, centred."""
        f = self.app.field
        w = min(f.width * 0.94, f.height * 0.94 * 16 / 9)
        h = w * 9 / 16
        return pygame.Rect(int(f.centerx - w / 2), int(f.centery - h / 2), int(w), int(h))

    def blit_fit(self, s, img, rect):
        """Scale img into rect keeping its aspect, rotated to read from the player's side."""
        if img is None:
            return
        iw, ih = img.get_size()
        k = min(rect.width / iw, rect.height / ih)
        scaled = pygame.transform.smoothscale(img, (int(iw * k), int(ih * k)))
        scaled = pygame.transform.rotate(scaled, ui.FACING[self.app.facing])
        s.blit(scaled, scaled.get_rect(center=rect.center))

    def heading(self, s, text, rect):
        """Slide heading on the reader's top side of rect."""
        up = np.array([0.0, 1.0]) if self.app.facing == "top" else np.array([0.0, -1.0])
        ui.text(s, self.font, text, np.array(rect.center) + up * rect.height * 0.43, (230, 230, 230), self.app.facing)

    def caption(self, s, text, rect):
        down = np.array([0.0, -1.0]) if self.app.facing == "top" else np.array([0.0, 1.0])
        ui.text(s, self.small, text, np.array(rect.center) + down * rect.height * 0.45, (190, 190, 190), self.app.facing)

    def inner(self, rect):
        return rect.inflate(-rect.width * 0.06, -rect.height * 0.28)

    # --- pages -------------------------------------------------------------
    def _title(self, s, now, a):
        self.blit_fit(s, self.title_img, a)

    def _last(self, s, now, a):
        self.blit_fit(s, self.last_img, a)

    def _camera(self, s, now, a):
        self.heading(s, "What the camera sees", a)
        self.blit_fit(s, self.camera_img, self.inner(a))
        self.caption(s, "live: tags (green), wands (orange), table (yellow), projector light (magenta)", a)

    def _calibration(self, s, now, a):
        self.heading(s, "Calibration: the camera watches projected markers", a)
        self.blit_fit(s, self.calib_img, self.inner(a))
        self.caption(s, "pattern minus its inverse cancels room light; marker corners link camera and projector", a)

    def _warp(self, s, now, a):
        self.heading(s, "What the game draws  ->  what we send to the projector", a)
        if self.warp_imgs is None:
            self.warp_imgs = self._make_warp_images()
        inner = self.inner(a)
        half_w = inner.width // 2 - int(6 * self.mm)
        left = pygame.Rect(inner.left, inner.top, half_w, inner.height)
        right = pygame.Rect(inner.right - half_w, inner.top, half_w, inner.height)
        if self.app.facing == "top":  # reading from the far side: left and right swap
            left, right = right, left
        self.blit_fit(s, self.warp_imgs[0], left)
        self.blit_fit(s, self.warp_imgs[1], right)
        self.caption(s, "the projector hits the table at an angle, so each frame is pre-skewed to land straight", a)

    def _make_warp_images(self):
        """A game frame on the table canvas, and the same frame warped as it goes over HDMI."""
        from .labyrinth import Labyrinth

        view = self.app.view
        canvas = pygame.Surface(view.size, 0, self.app.proj.screen)
        game = Labyrinth(self.app)
        game.draw(canvas, time.monotonic())
        sent = pygame.Surface(view.proj_size, 0, self.app.proj.screen)
        sent.fill((0, 0, 0))
        view.warp(canvas, sent)
        pygame.draw.rect(canvas, (255, 255, 255), canvas.get_rect(), 6)
        pygame.draw.rect(sent, (255, 255, 255), sent.get_rect(), 6)
        return canvas, sent

    def _annotate(self, frame):
        """Draw what the tracker knows onto a (downscaled) camera frame."""
        calib = self.app.calib
        k = frame.shape[1] / calib.cam_size[0]
        if calib.footprint:
            cv2.polylines(frame, [(np.array(calib.footprint) * k).astype(np.int32)], True, (255, 0, 255), 2)
        if calib.table_poly:
            outline = calib.proj_to_cam(np.array(calib.table_poly, float))
            cv2.polylines(frame, [(outline * k).astype(np.int32)], True, (0, 220, 255), 2)
        for t in self.app.tracker.tags(predict=False).values():
            color = (255, 160, 60) if t.id in WAND_IDS else (60, 230, 60)
            cv2.polylines(frame, [(t.cam_corners * k).astype(np.int32)], True, color, 2)
        return frame

    def draw(self, s, now):
        s.fill(ui.BACKGROUND)
        a = self.area()
        pygame.draw.rect(s, (40, 40, 40), a, border_radius=int(6 * self.mm))
        self.pages[self.page](s, now, a)
        down = np.array([0.0, -1.0]) if self.app.facing == "top" else np.array([0.0, 1.0])
        ui.text(s, self.small, f"{self.page + 1} / {len(self.pages)}   press any wand for the next slide",
                np.array(self.app.field.center) + down * (self.app.field.height / 2 - 6 * self.mm),
                (170, 170, 170), self.app.facing)
