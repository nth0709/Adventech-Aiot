"""Region of the white work sheet, so laser/marks are only searched ON the paper
(red/blue packaging, skin, cables elsewhere in the view are ignored)."""
import cv2
import numpy as np

FLAT = False      # arc mode (see line_detect.FLAT)


def sheet_region(frame, grow=12):
    """uint8 mask (full res) of the largest bright, unsaturated area, hull-filled; None if absent."""
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    if FLAT:
        # arc mode: the glow is the brightest, bluish thing in view. Take the paper level from
        # outside the glow, and count the glow itself as paper (the torch is on the sheet).
        vb = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 6)
        glow = vb > 1.25 * float(np.median(vb))
        bright = float(np.percentile(v[~glow], 97)) if np.any(~glow) else float(np.percentile(v, 97))
        white = (((v > 0.72 * bright) & (s < 50)) | glow).astype(np.uint8) * 255
    else:
        bright = float(np.percentile(v, 97))
        white = ((v > 0.72 * bright) & (s < 50)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 0.03 * white.size:
        return None
    cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(white)
    cv2.drawContours(region, [cv2.convexHull(max(cs, key=cv2.contourArea))], -1, 255, -1)
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow // 4 + 1,) * 2))
    return cv2.resize(region, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
