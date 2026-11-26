"""Synthetic welding-arc light on a camera frame (for testing line tracking under arc glare).

The arc sits at the torch point (here: the laser dot). Effects, each 0..1 strength:
  intensity : brightness of the arc (core saturates to white)
  radius    : size of the saturated core (px)
  glow      : how far the bloom/halo spreads (px)
  flicker   : frame-to-frame brightness change, incl. occasional full flashes
  spatter   : hot particles flying out of the arc (motion streaks, gravity, fade)
  smoke     : drifting haze that lowers contrast
  darken    : the rest of the scene gets darker (auto exposure reacting to the arc)
  streaks   : lens-flare star through the arc

    sim = ArcSim(w, h); out = sim.apply(frame, (x, y))
"""
import time

import cv2
import numpy as np

PRESETS = {
    'off':    dict(intensity=0.0, radius=0, glow=0, flicker=0.0, spatter=0.0, smoke=0.0, darken=0.0, streaks=0.0),
    'weak':   dict(intensity=0.6, radius=10, glow=60, flicker=0.2, spatter=0.2, smoke=0.1, darken=0.2, streaks=0.2),
    'medium': dict(intensity=0.85, radius=22, glow=110, flicker=0.4, spatter=0.5, smoke=0.25, darken=0.45, streaks=0.4),
    'strong': dict(intensity=1.0, radius=40, glow=180, flicker=0.6, spatter=0.9, smoke=0.4, darken=0.7, streaks=0.6),
}

CORE_BGR = np.array([255, 245, 240], np.float32)      # bluish white core
HALO_BGR = np.array([255, 190, 150], np.float32)      # blue-violet halo (TIG/MIG arc light)
SPARK_BGR = np.array([80, 200, 255], np.float32)      # orange-yellow spatter


class ArcSim:
    def __init__(self, w, h, seed=0, **params):
        self.w, self.h = w, h
        self.rng = np.random.default_rng(seed)
        self.p = dict(PRESETS['medium'])
        self.p.update(params)
        self._key = None
        self.sparks = np.zeros((0, 6), np.float32)       # x, y, vx, vy, age, life
        self.t_last = time.time()
        sm = self.rng.random((h // 16 + 4, w // 16 + 4)).astype(np.float32)
        sm = cv2.GaussianBlur(sm, (0, 0), 2)
        sm = (sm - sm.min()) / max(float(np.ptp(sm)), 1e-6)
        self.smoke_tex = cv2.resize(sm, ((w // 16 + 4) * 32, (h // 16 + 4) * 32), interpolation=cv2.INTER_CUBIC)
        self.flash = 0.0

    def set(self, **params):
        self.p.update(params)

    def _maps(self):
        """Light maps (2h x 2w, arc at the centre) rebuilt only when radius/glow/streaks change."""
        key = (int(self.p['radius']), int(self.p['glow']), round(self.p['streaks'], 2))
        if key == self._key:
            return
        self._key = key
        H, W = 2 * self.h, 2 * self.w
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        x -= W / 2; y -= H / 2
        d = np.sqrt(x * x + y * y)
        r = max(key[0], 1)
        core = np.clip(1.5 - d / r, 0, 1) ** 0.5                       # flat saturated disc
        g = max(key[1], 1)
        halo = 1.0 / (1.0 + (d / (0.35 * g)) ** 2)                     # bloom, slow falloff
        halo += 0.5 * np.exp(-(d / g) ** 2)
        star = np.zeros_like(d)
        if key[2] > 0:
            for ang in (0, 60, 120):                                   # 6-point flare
                a = np.deg2rad(ang + 15)
                across = np.abs(-np.sin(a) * x + np.cos(a) * y)
                star += np.exp(-(across / 2.5) ** 2) * np.exp(-d / (2.5 * g))
            star *= key[2]
        self.core_map = np.ascontiguousarray(core, np.float32)
        self.halo1 = np.ascontiguousarray(halo, np.float32)
        self.light_map = np.ascontiguousarray(
            halo[..., None] * 260.0 * (HALO_BGR / 255.0) + star[..., None] * 255.0 * (CORE_BGR / 255.0),
            np.float32)
        self.core_img = np.empty((self.h, self.w, 3), np.uint8); self.core_img[:] = np.clip(CORE_BGR * 1.2, 0, 255)
        self.grey = np.full((self.h, self.w, 3), 170, np.uint8)

    def _crop(self, m, cx, cy):
        x0 = int(round(self.w - cx)); y0 = int(round(self.h - cy))
        x0 = min(max(x0, 0), self.w); y0 = min(max(y0, 0), self.h)
        return m[y0:y0 + self.h, x0:x0 + self.w]

    def _update_sparks(self, cx, cy, dt, k):
        s = self.sparks
        if len(s):
            s[:, 0] += s[:, 2] * dt; s[:, 1] += s[:, 3] * dt
            s[:, 3] += 900 * dt                                        # gravity (px/s^2)
            s[:, 4] += dt
            s = s[(s[:, 4] < s[:, 5]) & (s[:, 0] > -50) & (s[:, 0] < self.w + 50) & (s[:, 1] < self.h + 50)]
        n_new = self.rng.poisson(120 * k * dt) if k > 0 else 0
        if n_new:
            ang = self.rng.uniform(0, 2 * np.pi, n_new)
            sp = self.rng.uniform(150, 700, n_new)
            new = np.stack([np.full(n_new, cx), np.full(n_new, cy), sp * np.cos(ang),
                            sp * np.sin(ang) - 200, np.zeros(n_new), self.rng.uniform(0.15, 0.6, n_new)], 1)
            s = np.vstack([s, new.astype(np.float32)])
        self.sparks = s

    def apply(self, frame, center):
        """All per-frame work is OpenCV uint8/saturating ops (numpy float was ~100 ms on the Jetson)."""
        p = self.p
        now = time.time(); dt = min(now - self.t_last, 0.1); self.t_last = now
        if p['intensity'] <= 0 or center is None:
            self.sparks = self.sparks[:0]
            return frame.copy()
        self._maps()
        if not hasattr(self, 'noise_bank'):
            self.noise_bank = [np.abs(self.rng.normal(0, 1, (self.h, self.w, 3))).astype(np.float32)
                               for _ in range(6)]
        cx, cy = float(center[0]), float(center[1])

        # flicker: random brightness each frame + rare short full flashes
        fl = p['flicker']
        if self.rng.random() < 0.04 * fl:
            self.flash = 1.0
        b = p['intensity'] * (1.0 + fl * self.rng.normal(0, 0.25)) + 0.8 * fl * self.flash
        self.flash *= 0.5
        b = max(b, 0.05)

        img = cv2.convertScaleAbs(frame, alpha=1.0 - p['darken'] * min(b, 1.2) * 0.75)   # exposure drops

        if p['smoke'] > 0:                                             # drifting haze, denser near the arc
            ox = int(now * 25) % (self.smoke_tex.shape[1] - self.w)
            oy = int(now * 12) % (self.smoke_tex.shape[0] - self.h)
            tex = self.smoke_tex[oy:oy + self.h, ox:ox + self.w]
            near = self._crop(self.halo1, cx, cy)
            a = cv2.multiply(cv2.add(0.35, 0.65 * tex), cv2.add(0.4, 0.6 * near)) * p['smoke']
            img = cv2.blendLinear(img, self.grey, np.ascontiguousarray(1.0 - a), np.ascontiguousarray(a))

        light = cv2.convertScaleAbs(self._crop(self.light_map, cx, cy), alpha=b)
        img = cv2.add(img, light)
        c = self._crop(self.core_map, cx, cy) * min(b, 1.0)
        img = cv2.blendLinear(img, self.core_img, np.ascontiguousarray(1.0 - c), np.ascontiguousarray(c))

        self._update_sparks(cx, cy, dt, p['spatter'])
        if len(self.sparks):
            sp = np.zeros_like(img)
            for x, y, vx, vy, age, life in self.sparks:
                f = 1.0 - age / life
                x2, y2 = x - vx * 0.02, y - vy * 0.02                  # motion streak
                cv2.line(sp, (int(x), int(y)), (int(x2), int(y2)), tuple(float(v) * f for v in SPARK_BGR),
                         2, cv2.LINE_AA)
            img = cv2.add(img, cv2.GaussianBlur(sp, (3, 3), 0))

        k = self.rng.integers(len(self.noise_bank))                    # sensor noise
        sd = 3 + 6 * p['darken']
        img = cv2.add(img, cv2.convertScaleAbs(self.noise_bank[k], alpha=sd))
        img = cv2.subtract(img, cv2.convertScaleAbs(self.noise_bank[(k + 3) % len(self.noise_bank)], alpha=sd))
        return img
