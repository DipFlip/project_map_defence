"""Synthesised sound effects (no audio files).

Every effect is generated once at start-up from tones, noise and envelopes.
If there is no audio output the game simply stays silent. (The Radxa's own
audio needs a topology firmware that isn't packaged yet; a USB or Bluetooth
speaker works.)
"""
import numpy as np

from .projector import pygame

RATE = 44100


def _t(dur):
    return np.arange(int(RATE * dur)) / RATE


def _env(n, attack=0.005, release=0.08):
    """Linear attack, exponential-ish release envelope over n samples."""
    e = np.ones(n)
    a, r = int(RATE * attack), int(RATE * release)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r and r < n:
        e[-r:] *= np.linspace(1, 0, r) ** 2
    return e


def tone(freq, dur, wave="sine", vol=0.5, slide=0.0, attack=0.005, release=0.08):
    """A note; slide: frequency multiplier reached at the end (pitch sweep)."""
    t = _t(dur)
    f = freq * (1 + (slide - 1) * t / dur) if slide else np.full_like(t, freq)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    if wave == "square":
        y = np.sign(np.sin(phase)) * 0.6
    elif wave == "saw":
        y = 2 * ((phase / (2 * np.pi)) % 1) - 1
    elif wave == "tri":
        y = 2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1
    else:
        y = np.sin(phase)
    return y * _env(len(t), attack, release) * vol


def noise(dur, vol=0.4, lowpass=0.2, release=0.1):
    y = np.random.default_rng(1).uniform(-1, 1, int(RATE * dur))
    if lowpass < 1:  # one-pole low-pass for a softer noise
        out = np.empty_like(y)
        acc = 0.0
        for i, v in enumerate(y):
            acc += lowpass * (v - acc)
            out[i] = acc
        y = out / max(np.abs(out).max(), 1e-6)
    return y * _env(len(y), 0.002, release) * vol


def seq(*parts, gap=0.0):
    """Concatenate parts with optional silence between them."""
    pad = np.zeros(int(RATE * gap))
    out = []
    for p in parts:
        out += [p, pad]
    return np.concatenate(out[:-1]) if out else np.zeros(1)


def mix(*parts):
    n = max(len(p) for p in parts)
    return sum(np.pad(p, (0, n - len(p))) for p in parts)


def _library():
    n = {"C4": 261.6, "E4": 329.6, "G4": 392.0, "A4": 440.0, "C5": 523.3, "E5": 659.3, "G5": 784.0, "C6": 1046.5}
    return {
        "click": tone(1400, 0.05, "square", 0.25, release=0.04),
        "start": seq(tone(n["C5"], 0.08, "square", 0.3), tone(n["G5"], 0.14, "square", 0.3)),
        "hit": mix(tone(520, 0.08, "square", 0.35, slide=0.7), noise(0.05, 0.2, 0.5)),
        "wall": tone(300, 0.05, "tri", 0.3, release=0.04),
        "score": seq(tone(n["E5"], 0.09, "square", 0.3), tone(n["C6"], 0.2, "square", 0.3)),
        "lose_point": tone(300, 0.35, "saw", 0.25, slide=0.5),
        "win": seq(*[tone(n[k], 0.12, "square", 0.3) for k in ("C5", "E5", "G5")], tone(n["C6"], 0.45, "square", 0.3)),
        "shoot": mix(tone(900, 0.07, "square", 0.18, slide=0.5), noise(0.05, 0.12, 0.6)),
        "swipe": noise(0.18, 0.35, 0.15, release=0.12),
        "squeak": tone(1800, 0.09, "sine", 0.25, slide=1.6),
        "coin": seq(tone(988, 0.05, "square", 0.22), tone(1319, 0.14, "square", 0.22)),
        "deposit": seq(*[tone(f, 0.06, "square", 0.2) for f in (784, 988, 1175, 1568)]),
        "upgrade": tone(300, 0.45, "square", 0.25, slide=3.0),
        "steal": seq(tone(220, 0.15, "saw", 0.3), tone(165, 0.3, "saw", 0.3)),
        "game_over": seq(*[tone(f, 0.18, "saw", 0.28) for f in (392, 330, 262)], tone(196, 0.6, "saw", 0.28)),
        "fire": mix(noise(0.25, 0.3, 0.08, release=0.2), tone(140, 0.2, "sine", 0.4, slide=0.6)),
        "pin": mix(tone(1200, 0.06, "tri", 0.3), noise(0.08, 0.3, 0.7)),
        "catch": tone(600, 0.12, "sine", 0.4, slide=1.5),
        "block": mix(tone(200, 0.08, "square", 0.3), noise(0.06, 0.2, 0.4)),
        "roll": noise(0.35, 0.25, 0.05, release=0.25),
        "fall": tone(900, 0.6, "sine", 0.3, slide=0.2),
        "checkpoint": seq(tone(n["G5"], 0.07, "tri", 0.3), tone(n["C6"], 0.12, "tri", 0.3)),
        "jump": tone(250, 0.25, "sine", 0.35, slide=2.5),
        "buzz": tone(110, 0.5, "saw", 0.35),
        "organ": seq(tone(n["C5"], 0.06, "sine", 0.35), tone(n["G5"], 0.06, "sine", 0.35), tone(n["C6"], 0.16, "sine", 0.35)),
        "splash": noise(0.3, 0.25, 0.3, release=0.25),
        "valve": tone(400, 0.08, "tri", 0.2, slide=1.3),
        "filled": seq(*[tone(f, 0.08, "sine", 0.3) for f in (523, 659, 784, 1046)]),
    }


class Sound:
    def __init__(self, volume=0.8):
        self.ok = False
        self.sounds = {}
        self.volume = volume
        self._last = {}
        try:
            pygame.mixer.pre_init(RATE, -16, 2, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(24)
        except pygame.error:
            return
        for name, y in _library().items():
            pcm = (np.clip(y, -1, 1) * 32767 * volume).astype(np.int16)
            self.sounds[name] = pygame.sndarray.make_sound(np.ascontiguousarray(np.stack([pcm, pcm], axis=1)))
        self.ok = True

    def play(self, name, volume=1.0, min_gap=0.03):
        """Play an effect; the same effect is not restarted within min_gap seconds (no machine-gun spam)."""
        if not self.ok or name not in self.sounds:
            return
        now = pygame.time.get_ticks() / 1000
        if now - self._last.get(name, -1) < min_gap:
            return
        self._last[name] = now
        ch = self.sounds[name].play()
        if ch is not None:
            ch.set_volume(volume)
