"""Scene manager: menu, games, wand button edges and the return-to-menu rule."""
import math
import time

from . import ui
from .scenes.menu import Menu
from .scenes.pins import PinShooter
from .scenes.pong import Pong
from .scenes.snack import SnackCastle
from .tracker import WAND_IDS

GAMES = {"snack": SnackCastle, "pong": Pong, "pins": PinShooter}
MENU_AFTER_S = 3.0  # all wands off the table this long -> back to the menu


class App:
    def __init__(self, proj, tracker, calib, facing="top"):
        """facing: table edge the single-player view (menu, snack castle) is read from."""
        self.proj, self.tracker, self.calib = proj, tracker, calib
        self.facing = facing
        self.games = GAMES
        self.px_per_mm = self._measure_scale()
        self.field = ui.table_rect(calib, proj.size, margin=int(8 * self.px_per_mm))
        self.tags = tracker.tags()
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

    def switch(self, name):
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
        self.tags = self.tracker.tags()
        presses = self._presses()
        if any(k in self.tags for k in WAND_IDS):
            self._wands_seen = now
        elif not isinstance(self.scene, Menu) and now - self._wands_seen > MENU_AFTER_S:
            self.switch("menu")
        self.scene.update(dt, now, self.tags, presses)
        self.scene.draw(self.proj.screen, now)

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
