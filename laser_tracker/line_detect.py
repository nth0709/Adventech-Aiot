"""Thin dark line on a bright sheet: robust to sheet brightness, ignores big dark areas.

Black-hat picks structures thinner than KERNEL that are darker than their
surroundings; the local background (grey closing) must be bright relative to the
frame, so cables/edges on a dark desk are rejected.
"""
import cv2
import numpy as np

import config as C

KERNEL = 21          # px at processed scale; must be wider than the line
BG_FRAC = 0.75       # local background must be >= this * bright level of frame
MIN_CONTRAST = 18    # minimum black-hat response (grey levels)
WEAK_FRAC = 0.5      # hysteresis: faint pixels connected to the line count down to this * threshold


def line_mask(frame):
    s = C.PROC_SCALE
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)            # line filled in = paper level
    bh = cv2.subtract(bg, gray)                                # black-hat
    bright = float(np.percentile(bg, 95))
    sheet = bg >= BG_FRAC * bright
    vals = bh[sheet]
    thr = MIN_CONTRAST if vals.size == 0 else max(MIN_CONTRAST, 0.3 * float(np.percentile(vals, 99.5)))   # 0.5 dropped faint ink next to a dark mark
    # a real line has bright paper on BOTH sides; a sheet edge only on one side
    # thin dark pixels (this line or another arm of it at a sharp corner) are left out of
    # the count, so only WIDE dark areas (desk beyond a sheet edge) lower the score
    thin = (bh > thr) & sheet
    bright_px = ((gray >= BG_FRAC * bright) & ~thin).astype(np.float32)
    other = (~thin).astype(np.float32)
    around = cv2.blur(bright_px, (KERNEL, KERNEL)) / np.maximum(cv2.blur(other, (KERNEL, KERNEL)), 1e-3)
    ok = sheet & (around > 0.7)
    strong = (bh > thr) & ok
    # hysteresis: faint ink (>= WEAK_FRAC of the strong threshold) is kept only where it is
    # connected to strong line pixels. A line seen at a grazing angle gets thin where it runs
    # across the view (e.g. the top of an S curve) and loses contrast at half resolution.
    weak = (bh > max(MIN_CONTRAST, WEAK_FRAC * thr)) & ok
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[strong])] = True; keep[0] = False
    mask = keep[lab].astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = C.ROI_MARGIN
    if m > 0:
        mask[:m, :] = 0; mask[-m:, :] = 0; mask[:, :m] = 0; mask[:, -m:] = 0
    return mask
