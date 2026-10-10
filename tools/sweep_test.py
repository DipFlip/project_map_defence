"""Move real pieces around fast and see how well tracking keeps up.

    .venv/bin/python tools/sweep_test.py [seconds]

Projects a thin ring where each piece is tracked (nothing bright on the
tags). Sweep pieces at different speeds; at the end it prints, per speed
range, the share of camera frames in which the moving piece was detected.
Every missed frame shows up as roughly 33 ms of extra lag.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle  # noqa: E402

import numpy as np  # noqa: E402

from pmd import CALIB_PATH, DATA_DIR  # noqa: E402
from pmd.calibration import Calibration  # noqa: E402
from pmd.projector import Projector, pygame  # noqa: E402
from pmd.system import performance_governor  # noqa: E402
from pmd.tracker import Tracker  # noqa: E402

FRAME = 1 / 30
BINS = [0, 100, 300, 600, 1000, 1500, 3000]  # mm/s


def main(seconds):
    calib = Calibration.load(CALIB_PATH)
    tracker = Tracker(CALIB_PATH, predict=0.05)
    proj = Projector()
    font = pygame.font.SysFont(None, 48)
    time.sleep(2)
    px_per_mm = tracker.px_per_mm()
    center = np.array(calib.table_poly).mean(axis=0)
    t_go = time.monotonic() + 5
    while time.monotonic() < t_go and proj.pump():  # countdown so the player is ready
        proj.screen.fill((0, 0, 0))
        msg = font.render(f"get ready... {t_go - time.monotonic():.0f}", True, (255, 200, 0))
        proj.screen.blit(msg, msg.get_rect(center=center))
        proj.flip()
        proj.clock.tick(30)
    tracker.log = []
    seen = {}  # id -> {last_seen: pos}
    t_end = time.monotonic() + seconds
    try:
        while proj.pump() and time.monotonic() < t_end:
            s = proj.screen
            s.fill((0, 0, 0))
            pygame.draw.polygon(s, (30, 30, 30), calib.table_poly)
            for k, t in tracker.tags(predict=False).items():
                seen.setdefault(k, {})[t.last_seen] = t.pos.copy()
            for t in tracker.tags().values():
                pygame.draw.circle(s, (0, 220, 120), t.pos, t.size * 1.1 + 12, 3)
            left = t_end - time.monotonic()
            msg = font.render(f"sweep the pieces around - {left:.0f}s", True, (200, 200, 200))
            s.blit(msg, msg.get_rect(center=np.array(calib.table_poly).mean(axis=0)))
            proj.flip()
            proj.clock.tick(60)
    finally:
        tracker.close()
        proj.close()
    # raw detections for tuning the filter/prediction offline (tools/tune_prediction.py)
    raw = {}
    for recv, tags in tracker.log:
        for k, t in tags.items():
            raw.setdefault(k, {}).setdefault(t.last_seen, (recv, *t.raw_pos))  # first time we knew it
    with open(os.path.join(DATA_DIR, "sweep.pkl"), "wb") as f:
        pickle.dump({"px_per_mm": px_per_mm, "tracks": {k: sorted((s, *v) for s, v in d.items()) for k, d in raw.items()}}, f)

    det = np.zeros(len(BINS) - 1)
    frames = np.zeros(len(BINS) - 1)
    for k, track in seen.items():
        if not 10 <= k <= 19:
            continue
        ts = sorted(track)
        for a, b in zip(ts, ts[1:]):
            gap = b - a
            if gap > 0.5:
                continue  # gone (lifted / covered), not a tracking miss
            speed = np.linalg.norm(track[b] - track[a]) / gap / px_per_mm
            i = np.searchsorted(BINS, speed, side="right") - 1
            if 0 <= i < len(det):
                det[i] += 1
                frames[i] += max(1.0, round(gap / FRAME))
    print("speed (cm/s)   detected in   samples")
    for i in range(len(det)):
        if frames[i]:
            print(f"{BINS[i] / 10:5.0f}-{BINS[i + 1] / 10:<5.0f}     {det[i] / frames[i] * 100:5.0f}%      {int(det[i])}")


if __name__ == "__main__":
    with performance_governor():
        main(float(sys.argv[1]) if len(sys.argv) > 1 else 20)
