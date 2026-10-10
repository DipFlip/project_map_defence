"""Scene manager: menu, games, wand button edges and the return-to-menu rule."""
import math
import time
from dataclasses import replace

import numpy as np

from .particles import Particles
from .projector import pygame
from .table import TableView
from .scenes.labyrinth import Labyrinth
from .scenes.menu import Menu
from .scenes.operation import Operation
from .scenes.pins import PinShooter
from .scenes.pong import Pong
from .scenes.snack import SnackCastle
from .scenes.water import Water
from .sound import Sound
from .tracker import WAND_IDS

GAMES = {"snack": SnackCastle, "pong": Pong, "pins": PinShooter, "labyrinth": Labyrinth, "operation": Operation,
         "water": Water}
MENU_AFTER_S = 3.0  # all wands off the table this long -> back to the menu


class App:
    def __init__(self, proj, tracker, calib, facing="top"):
        """facing: table edge the single-player view (menu, snack castle) is read from.

        Scenes draw on self.view.canvas, which is in table millimetres
        (self.px_per_mm canvas px per mm, table edges along the axes), and get
        tag poses in the same canvas coordinates.
        """
        self.proj, self.tracker, self.calib = proj, tracker, calib
        self.facing = facing
        self.games = GAMES
        self.view = TableView(calib, proj.size, proj.screen)
        self.sound = Sound()  # self.sound.play("name"): effects in pmd/sound.py
        self.fx = Particles()  # particle effects drawn over every scene
        self.px_per_mm = self.view.px_per_mm or self._measure_scale()
        margin = int(10 * self.px_per_mm)
        self.field = pygame.Rect(0, 0, *self.view.size).inflate(-2 * margin, -2 * margin)
        self.tags = self._canvas_tags()
        self._was_pressed = {}
        self._wands_seen = time.monotonic()
        self.scene = Menu(self)

    def _measure_scale(self):
        """Projector px per table mm from the tags' known size (waits briefly for tags)."""
        deadline = time.monotonic() + 2.0
        while not self.tracker.tags() and time.monotonic() < deadline:
            self.proj.pump()
            time.sleep(0.05)
        return self.tracker.px_per_mm()

    def _canvas_tags(self):
        """Tracker poses (projector px) converted to canvas coordinates."""
        out = {}
        for k, t in self.tracker.tags().items():
            corners = self.view.to_canvas(t.corners)
            pos = self.view.to_canvas(t.pos)[0]
            up = corners[3] - corners[0]  # printed top (see tracker.py)
            ahead = self.view.to_canvas(t.pos + t.vel * 0.1)[0]
            out[k] = replace(t, pos=pos, corners=corners, angle=float(np.arctan2(up[1], up[0])),
                             size=float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, 0), axis=1))),
                             vel=(ahead - pos) / 0.1)
        return out

    def switch(self, name):
        self.fx.clear()
        self.sound.play("start" if name != "menu" else "click")
        self.scene = Menu(self) if name == "menu" else self.games[name](self)

    def _presses(self):
        """Wand ids whose button went from released to pressed since the last frame."""
        presses = set()
        for k in WAND_IDS:
            t = self.tags.get(k)
            if t is None:
                continue  # unseen: keep the last state so a reappearing wand doesn't fire
            if t.pressed and not self._was_pressed.get(k, False):
                presses.add(k)
            self._was_pressed[k] = t.pressed
        return presses

    def step(self, dt, now):
        self.tags = self._canvas_tags()
        presses = self._presses()
        if any(k in self.tags for k in WAND_IDS):
            self._wands_seen = now
        elif not isinstance(self.scene, Menu) and now - self._wands_seen > MENU_AFTER_S:
            self.switch("menu")
        self.scene.update(dt, now, self.tags, presses)
        self.fx.update(dt)
        self.scene.draw(self.view.canvas, now)
        self.fx.draw(self.view.canvas)
        self.view.render(self.proj.screen)

    def run(self, duration=math.inf, snapshot_path=None, snapshot_every=5.0):
        t_prev = t_start = last_snap = time.monotonic()
        while self.proj.pump() and time.monotonic() - t_start < duration:
            now = time.monotonic()
            self.step(min(now - t_prev, 0.1), now)
            t_prev = now
            self.proj.flip()
            self.proj.clock.tick(60)
            if snapshot_path and now - last_snap > snapshot_every:
                self.tracker.snapshot(snapshot_path)
                last_snap = now
