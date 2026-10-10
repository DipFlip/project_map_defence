"""One Euro filter (Casiez et al. 2012) for jittery tracked positions.

A low-pass filter whose cutoff rises with speed: when a piece is still or
slow it smooths heavily (no jitter), when it moves fast it barely smooths
(little lag). min_cutoff (Hz) sets the smoothing at rest, beta how quickly
the cutoff opens up with speed, d_cutoff (Hz) smooths the speed estimate.
"""
import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=2.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None  # filtered value
        self.dx = None  # filtered derivative (units per second)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.x is None or dt <= 0:
            self.x, self.dx = x, np.zeros_like(x)
            return self.x
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + _alpha(cutoff, dt) * (x - self.x)
        return self.x
