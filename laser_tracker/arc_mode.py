"""Welding-arc mode for follow5.py (synthetic arc light from arc_filter.ArcSim).

Like a real weld: the arc is struck at the start mark and put out at the end mark, so it is
ON only while travelling along the line. While it is on, the TRACKER sees only the filtered
frames:
  * torch position = the saturated arc core, found in the filtered frame every frame
    (the red laser dot is invisible under the arc). A frame that is too saturated to show
    a core (a flash) gives no position: the arm holds, as for any unusable frame.
  * line mask: lighting flattened (line_detect.FLAT), limited to the sheet region
    (sheet.FLAT), the rim of the core ignored.
  * the part of the line hidden under the arc is bridged: where the line enters and leaves
    the hidden disc it is joined by a straight chord (only what is seen around the disc).
The SIMULATOR (ArcCamera) places the arc at the real laser dot in the raw frame; the
tracker never gets that raw frame or that position.
"""
import cv2
import numpy as np

import config as C
import line_detect
import sheet
from arc_filter import ArcSim, PRESETS
from laser_spot import find_laser

HOLE_MARGIN = 34          # full-res px beyond the saturated core where the line is not trusted
ROI_PX = 200              # full-res px: window around the torch where the line is searched
RING_PX = 90              # full-res px beyond the core searched for the line pieces to bridge
ACTIVE = False
_state = {'radius': 20.0}


class ArcCamera:
    """Wraps vision.Camera: frames come out with the arc light added while `on`."""

    def __init__(self, cam, preset='medium'):
        self.cam, self.on, self.torch = cam, False, None
        self.sim = ArcSim(C.FRAME_W, C.FRAME_H, **PRESETS[preset])

    def _f(self, f):
        if not self.on:
            return f
        p, _ = find_laser(f)                 # simulator only: the torch is where the laser is
        if p is not None:
            self.torch = p
        return self.sim.apply(f, self.torch)

    def read(self, newer_than=None, timeout=1.0):
        seq, f = self.cam.read(newer_than=newer_than, timeout=timeout)
        return seq, self._f(f)

    def fresh(self, skip=3):
        seq, _ = self.cam.read()
        return self._f(self.cam.read(newer_than=seq + skip - 1, timeout=2.0)[1])

    def close(self):
        self.cam.close()


def set_active(on):
    global ACTIVE
    ACTIVE = bool(on)
    line_detect.FLAT = ACTIVE
    sheet.FLAT = ACTIVE


def find_arc(frame):
    """Arc core in the filtered frame: largest saturated blob (thin flare streaks and sparks
    removed). Returns (centre_full, radius_full) or (None, 0)."""
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lo = cv2.min(cv2.min(small[..., 0], small[..., 1]), small[..., 2])
    sat = (lo >= 245).astype(np.uint8)
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(sat, 8)
    if n < 2:
        return None, 0.0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = float(st[k, cv2.CC_STAT_AREA])
    if area < 6 or area > 0.15 * sat.size:          # nothing / whole view washed out (flash)
        return None, 0.0
    return cen[k] * 2.0, 2.0 * np.sqrt(area / np.pi)


def find_torch_near(frame, laser, half=60):
    p, r = find_arc(frame)
    if p is None or np.max(np.abs(p - np.asarray(laser))) > max(half, 2 * r):
        return None, 0.0
    _state['radius'] = 0.7 * _state['radius'] + 0.3 * r
    _state['torch'] = p
    return p, r


def hole_radius():
    """Full-res radius around the torch where the line cannot be seen."""
    return _state['radius'] + HOLE_MARGIN


def line_mask(frame):
    """Line only in a window around the torch seen in this frame (a seam tracker's view)."""
    t = _state.get('torch')
    line_detect.ROI = None if t is None else (t[0], t[1], ROI_PX)
    return line_detect.line_mask(frame)


def bridge_hole(mask, torch):
    """Join the line across the part hidden by the arc: line pieces seen in a wide ring around
    the core are joined by a straight chord between their INNER ends (2 pieces = the line
    passes under the arc; 1 piece = the line ends under it)."""
    s = C.PROC_SCALE
    c = np.asarray(torch, float) * s
    r_core = _state['radius'] * s
    r_out = (_state['radius'] + RING_PX) * s
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
    mask[d2 < r_core ** 2] = 0                       # nothing inside the core is real ink
    ring = ((d2 <= r_out ** 2) & (mask > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ring, 8)
    ends = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 6:
            continue
        ys, xs = np.nonzero(lab == i)
        k = int(np.argmin((xs - c[0]) ** 2 + (ys - c[1]) ** 2))
        ends.append(((xs[k], ys[k]), st[i, cv2.CC_STAT_AREA]))
    # the line passes UNDER the arc: two pieces on opposite sides of the torch (>= ~120 deg
    # apart). Spatter/flare false bits are rarely placed like that; take the most opposite pair.
    best, best_cos = None, -0.5
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            u = np.subtract(ends[i][0], c); v = np.subtract(ends[j][0], c)
            cs = float(u @ v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-6)
            if cs < best_cos:
                best, best_cos = (ends[i][0], ends[j][0]), cs
    if best is not None:
        a, b = best
        cv2.line(mask, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, 3)
    return mask
