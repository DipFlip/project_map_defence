"""Fullscreen projector output.

On the Radxa there is no desktop, so we render straight to DRM/KMS through
SDL's kmsdrm backend (needs the distro pygame; the pip wheel lacks kmsdrm).
Set PMD_WINDOWED=1 to get a normal window instead, e.g. over ssh -X.
"""
import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if not os.environ.get("PMD_WINDOWED"):
    os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")

import pygame  # noqa: E402


class Projector:
    def __init__(self, size=None):
        pygame.display.init()
        pygame.font.init()
        if os.environ.get("PMD_WINDOWED"):
            self.screen = pygame.display.set_mode(size or (1280, 720))
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        self.size = self.screen.get_size()
        self.clock = pygame.time.Clock()

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def show_gray(self, img):
        """Display a HxW uint8 (or HxWx3 RGB) numpy image covering the screen."""
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        self.screen.blit(surf, (0, 0))
        self.flip()

    def fill(self, color):
        self.screen.fill(color)
        self.flip()

    def flip(self):
        pygame.display.flip()

    def pump(self):
        """Process window events; returns False when the user asked to quit."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
        return True

    def close(self):
        pygame.display.quit()
