"""Tune the position filter + prediction on a recorded sweep (tools/sweep_test.py).

    .venv/bin/python tools/tune_prediction.py [latency_s]

Replays the recorded raw detections exactly as the game would receive them,
rendering at 60 Hz. For each rendered estimate it measures the distance to
where the piece really was when that frame reached the table.

latency_s: time from rendering a frame until the camera timestamps what was
physically there then = projector lag + camera lag (tools/latency.py measures
the flip->camera loop) + about half a render frame. Default 0.105 s.
"""
import itertools
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pmd import DATA_DIR  # noqa: E402
from pmd.filters import OneEuro  # noqa: E402
from pmd.tracker import extrapolation  # noqa: E402

TAG_SIZE_PX = 42.0  # 24 mm tower tag on this rig; only used for the prediction cap


def load():
    with open(os.path.join(DATA_DIR, "sweep.pkl"), "rb") as f:
        d = pickle.load(f)
    tracks = {k: np.array(v) for k, v in d["tracks"].items() if 10 <= k <= 19 and len(v) > 30}
    return d["px_per_mm"], tracks  # rows: stamp, recv, x, y


def simulate(track, latency, min_cutoff, beta, d_cutoff, predict):
    stamp, recv, xy = track[:, 0], track[:, 1], track[:, 2:4]
    errors, speeds = [], []
    f = None
    i, last = 0, None
    for T in np.arange(recv[0], recv[-1] - latency, 1 / 60):
        while i < len(stamp) and recv[i] <= T:
            dt = stamp[i] - last if last is not None else 0.0
            if f is None or dt > 0.25:
                f, dt = OneEuro(min_cutoff, beta, d_cutoff), 0.0
            f(xy[i], dt)
            last = stamp[i]
            i += 1
        if f is None or T - last > 0.25:
            continue
        est = f.x + extrapolation(f.dx, T - last, predict, TAG_SIZE_PX)
        target_t = T + latency
        j = np.searchsorted(stamp, target_t)
        if j == 0 or j >= len(stamp) or stamp[j] - stamp[j - 1] > 0.1:
            continue
        a = (target_t - stamp[j - 1]) / (stamp[j] - stamp[j - 1])
        truth = xy[j - 1] + a * (xy[j] - xy[j - 1])
        errors.append(np.linalg.norm(est - truth))
        speeds.append(np.linalg.norm(xy[j] - xy[j - 1]) / (stamp[j] - stamp[j - 1]))
    return errors, speeds


def evaluate(tracks, px_per_mm, latency, params):
    errs, spd = [], []
    for track in tracks.values():
        e, s = simulate(track, latency, *params)
        errs += e
        spd += s
    errs = np.array(errs) / px_per_mm  # mm
    spd = np.array(spd) / px_per_mm  # mm/s
    moving = spd > 100
    return {
        "rms": float(np.sqrt(np.mean(errs ** 2))),
        "moving_rms": float(np.sqrt(np.mean(errs[moving] ** 2))),
        "moving_p90": float(np.percentile(errs[moving], 90)),
        "still_rms": float(np.sqrt(np.mean(errs[~moving] ** 2))),
    }


def fmt(name, r):
    return (f"{name:42s} all {r['rms']:5.1f} mm | moving rms {r['moving_rms']:5.1f} p90 {r['moving_p90']:5.1f} mm"
            f" | still {r['still_rms']:4.2f} mm")


def main(latency):
    px_per_mm, tracks = load()
    current = (1.0, 0.08, 4.0, 0.05)
    print(fmt("current (mc 1.0, beta .08, dc 4, pred 50ms)", evaluate(tracks, px_per_mm, latency, current)))
    print(fmt("no filter, no prediction", evaluate(tracks, px_per_mm, latency, (1e6, 0, 2.0, 0.0))))
    grid = itertools.product([0.5, 1.0, 2.0], [0.01, 0.02, 0.04, 0.08], [1.0, 2.0, 4.0],
                             [0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
    results = [(evaluate(tracks, px_per_mm, latency, p), p) for p in grid]
    results.sort(key=lambda r: r[0]["rms"])
    print("best:")
    for r, p in results[:8]:
        print(fmt(f"mc {p[0]}, beta {p[1]}, dc {p[2]}, pred {p[3] * 1000:.0f}ms", r))
    best = results[0][1]
    print("prediction sweep at the best filter:")
    for pred in [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]:
        r = evaluate(tracks, px_per_mm, latency, (*best[:3], pred))
        print(fmt(f"pred {pred * 1000:.0f}ms", r))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.105)
