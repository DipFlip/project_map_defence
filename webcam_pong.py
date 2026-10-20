"""Finger-gun Pong on a laptop: the webcam view fills the screen and the game is drawn over it.

    python webcam_pong.py [--camera N] [--windowed] [--no-mirror]

Two players stand side by side in front of the laptop. Make a finger gun (index
finger out, thumb up, other fingers curled): your bouncer is the line between
your thumb tip and index tip. Left player defends the left edge, right player
the right edge. First to 7.

Keys: Esc/q quit, r new match, v video brighter/darker, d hand skeletons, f fullscreen.
"""
import argparse
import math
import os
import time

os.environ["PMD_WINDOWED"] = "1"  # never the Radxa's KMS output here

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pmd.hands import HandTracker  # noqa: E402
from pmd.particles import Particles  # noqa: E402
from pmd.projector import pygame  # noqa: E402
from pmd.scenes.hand_pong import HandPong  # noqa: E402
from pmd.sound import Sound  # noqa: E402

VIRTUAL_TABLE_MM = 1100  # the screen's width stands for this much table, so the table game's speeds carry over
DIM_LEVELS = (0.35, 0.6, 1.0)  # video brightness under the game


class CamApp:
    """What the Pong scene needs from the table App, for a screen showing the webcam."""

    def __init__(self, screen, tracker):
        self.screen, self.tracker = screen, tracker
        W, H = screen.get_size()
        self.px_per_mm = W / VIRTUAL_TABLE_MM
        self.field = pygame.Rect(0, 0, W, H).inflate(-8, -8)
        self.tags = {}
        self.sound = Sound()
        self.fx = Particles()
        self.overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        self.small = pygame.font.SysFont(None, max(18, H // 40))
        self.frame, self.now = None, time.monotonic()
        self.debug, self.dim = False, 1
        self._video, self._video_n = None, -1
        self.scale, self.offset = 1.0, np.zeros(2)
        self.scene = HandPong(self)

    def _fit(self, image):
        """Scale the camera image to cover the screen (cropping the overflow), keeping its proportions."""
        fh, fw = image.shape[:2]
        W, H = self.screen.get_size()
        self.scale = max(W / fw, H / fh)
        self.offset = (np.array([W, H]) - np.array([fw, fh]) * self.scale) / 2
        self.frame_size = np.array([fw, fh], float)

    def to_screen(self, pts):
        """Normalised camera coordinates (0..1) -> screen px, the same mapping the video is drawn with."""
        return np.asarray(pts, float) * self.frame_size * self.scale + self.offset

    def _video_surface(self):
        if self.frame.n != self._video_n:
            self._video_n = self.frame.n
            img = self.frame.image
            w, h = (np.array(img.shape[1::-1]) * self.scale + 0.5).astype(int)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            if DIM_LEVELS[self.dim] < 1:
                img = cv2.convertScaleAbs(img, alpha=DIM_LEVELS[self.dim])
            self._video = pygame.image.frombuffer(img.tobytes(), (w, h), "RGB")
        return self._video

    def _draw_fx(self):
        """Particles fade out with transparency (the table version fades them into a flat background)."""
        fx, s = self.fx, self.overlay
        s.fill((0, 0, 0, 0))
        if len(fx.p):
            fade = np.minimum(1.0, (1 - fx.age / fx.life) * 1.5)
            for p, r, c, f in zip(fx.p, fx.size, fx.color, fade):
                pygame.draw.circle(s, (int(c[0]), int(c[1]), int(c[2]), int(255 * f)), p, max(1.0, r * (0.4 + 0.6 * f)))
        self.screen.blit(s, (0, 0))

    def step(self, dt, now):
        self.now = now
        frame = self.tracker.latest()
        if frame is not None and self.frame is None:
            self._fit(frame.image)
        self.frame = frame
        s = self.screen
        s.fill((0, 0, 0))
        if frame is None:
            ui_msg = self.tracker.error or "starting the camera..."
            t = self.scene.small.render(ui_msg, True, (220, 220, 220))
            s.blit(t, t.get_rect(center=s.get_rect().center))
            return
        s.blit(self._video_surface(), self.offset)
        self.scene.update(dt, now, self.tags, set())
        self.fx.update(dt)
        self.scene.draw(s, now)
        self._draw_fx()
        hud = f"hands {self.tracker.fps:.0f} Hz   draw {self.clock_fps:.0f} fps"
        s.blit(self.small.render(hud, True, (200, 200, 200)), (12, s.get_height() - self.small.get_height() - 8))

    clock_fps = 0.0


def open_screen(windowed):
    if windowed:
        return pygame.display.set_mode((1280, 720))
    return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--windowed", action="store_true", help="1280x720 window instead of fullscreen")
    ap.add_argument("--no-mirror", action="store_true", help="show the camera unmirrored")
    ap.add_argument("--seconds", type=float, default=math.inf, help=argparse.SUPPRESS)
    args = ap.parse_args()

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Finger-gun pong")
    screen = open_screen(args.windowed)
    tracker = HandTracker(args.camera, mirror=not args.no_mirror)
    app = CamApp(screen, tracker)
    clock = pygame.time.Clock()
    fullscreen = not args.windowed
    t_prev = t_start = time.monotonic()
    try:
        while time.monotonic() - t_start < args.seconds:
            quit_ = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_ = True
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                    app.fx.clear()
                    app.scene.new_match(time.monotonic())
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_v:
                    app.dim = (app.dim + 1) % len(DIM_LEVELS)
                    app._video_n = -1
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_d:
                    app.debug = not app.debug
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_f:
                    fullscreen = not fullscreen
                    app = CamApp(open_screen(not fullscreen), tracker)  # new size: refit the video, new field
            if quit_:
                break
            now = time.monotonic()
            app.step(min(now - t_prev, 0.1), now)
            t_prev = now
            pygame.display.flip()
            clock.tick(60)
            app.clock_fps = clock.get_fps()
    finally:
        tracker.close()
        pygame.mixer.quit()
        pygame.display.quit()


if __name__ == "__main__":
    main()
