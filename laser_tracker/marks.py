"""Detect the start (blue) and end (red) marker dots drawn on the sheet.

The red mark is ink (saturated colour, not over-exposed); the laser dot is a tiny
over-exposed spot, so pixels near the laser and very bright pixels are excluded.
"""
import cv2
import numpy as np

MIN_AREA = 12      # px at half scale (blue)
RED_MIN_AREA = 30  # px at FULL res: a small red ink dot is ~50 px; reddish pen line bits are < 25 px


NEAR_LINE = 12     # px at half scale: a start/end mark sits ON the line (at its end)


def _line_dist(line_mask):
    """Distance (half-res px) to the nearest line pixel, or None when no line mask is given."""
    if line_mask is None:
        return None
    return cv2.distanceTransform((line_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _near_line(dist, c_half):
    if dist is None:
        return True
    x = int(np.clip(round(c_half[0]), 0, dist.shape[1] - 1)); y = int(np.clip(round(c_half[1]), 0, dist.shape[0] - 1))
    return dist[y, x] <= NEAR_LINE


def _blob(mask, laser_px, exclude_r, dist=None):
    if laser_px is not None:
        cv2.circle(mask, (int(laser_px[0] / 2), int(laser_px[1] / 2)), int(exclude_r / 2), 0, -1)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= MIN_AREA and _near_line(dist, cen[i]) and (best is None or a > st[best, cv2.CC_STAT_AREA]):
            best = i
    return None if best is None else cen[best] * 2.0


def find_marks(frame, laser_px=None, exclude_r=30, region=None, exclude_blue=None, line_mask=None):
    """Return (blue_center, red_center) in full-res pixels (None if not seen).
    line_mask (half res): only blobs on/next to the black line count (ignores red pens, cups...)."""
    dist = _line_dist(line_mask)
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = ((h >= 100) & (h <= 130) & (s > 90) & (v > 40)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k)
    if region is not None:
        r = cv2.resize(region, (blue.shape[1], blue.shape[0]), interpolation=cv2.INTER_NEAREST)
        blue[r == 0] = 0
    eb = exclude_r if exclude_blue is None else exclude_blue
    return (_blob(blue, laser_px if eb > 0 else None, eb, dist),
            _red_full(frame, laser_px, exclude_r, region, dist))


def _red_full(frame, laser_px, exclude_r, region, dist=None):
    """Red end mark at full resolution (it can be only a few pixels across)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h <= 8) | (h >= 170)) & (s > 120) & (v > 60) & (v < 250)).astype(np.uint8)
    if region is not None:
        red[region == 0] = 0
    if laser_px is not None:
        cv2.circle(red, (int(laser_px[0]), int(laser_px[1])), int(exclude_r), 0, -1)
    n, lab, st, cen = cv2.connectedComponentsWithStats(red, 8)
    ok = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= RED_MIN_AREA and _near_line(dist, cen[i] / 2.0)]
    if not ok:
        return None
    i = max(ok, key=lambda k: st[k, cv2.CC_STAT_AREA])
    return cen[i].astype(float)
