"""Measure how much still tags shake, for a few filter settings.

    .venv/bin/python tools/jitter.py

Leave the pieces still on the table. Prints the per-tag position spread
(projector px, RMS around its mean) and angle spread (degrees).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pmd import CALIB_PATH  # noqa: E402
from pmd.system import performance_governor  # noqa: E402
from pmd.tracker import Tracker  # noqa: E402

SETTINGS = {
    "raw (no filter)": dict(pos_filter=dict(min_cutoff=1e6, beta=0), angle_filter=dict(min_cutoff=1e6, beta=0)),
    "default": {},
    "stronger (0.5 Hz)": dict(pos_filter=dict(min_cutoff=0.5, beta=0.04), angle_filter=dict(min_cutoff=0.5, beta=0.5)),
}


def measure(seconds=6, **kw):
    tr = Tracker(CALIB_PATH, **kw)
    time.sleep(3)
    samples = {}
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for k, t in tr.tags().items():
            samples.setdefault(k, {})[t.last_seen] = (*t.pos, t.angle)
        time.sleep(0.01)
    tr.close()
    pos, ang = [], []
    for k, s in samples.items():
        a = np.array(list(s.values()))
        if len(a) < 20 or not 10 <= k <= 19:
            continue
        pos.append(np.sqrt(((a[:, :2] - a[:, :2].mean(0)) ** 2).sum(1).mean()))
        ang.append(np.degrees(a[:, 2].std()))
    return np.mean(pos), np.max(pos), np.mean(ang), np.max(ang), len(pos)


if __name__ == "__main__":
    with performance_governor():
        for name, kw in SETTINGS.items():
            p, pmax, a, amax, n = measure(**kw)
            print(f"{name:18s}: position {p:.2f} px (worst {pmax:.2f}), angle {a:.2f} deg (worst {amax:.2f}), {n} tags")
