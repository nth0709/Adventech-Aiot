"""Find an always-on red laser dot by colour.

The dot is SMALL and much brighter/redder than its immediate surroundings, so the
score is a local top-hat: large reddish areas (skin, wood, cables) are suppressed.
"""
import cv2
import numpy as np


def find_laser(frame, min_score=35, region=None):
    """region: optional full-res uint8 mask; only peaks inside it count."""
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    red = np.clip(r - np.maximum(g, b), 0, 255)                 # red halo
    hot = ((r > 230) & (r - b > 20)).astype(np.float32) * 60      # bright, reddish core
    score = cv2.GaussianBlur(red + hot, (5, 5), 0)
    local = cv2.GaussianBlur(score, (0, 0), 10)                   # surroundings
    spot = score - 1.5 * local                                    # compact peaks only
    if region is not None:
        reg = cv2.resize(region, (spot.shape[1], spot.shape[0]), interpolation=cv2.INTER_NEAREST)
        spot[reg == 0] = 0
    core = ((r > 230) & (g > 150)).astype(np.uint8)              # over-exposed core: laser, not ink
    core = cv2.dilate(core, np.ones((7, 7), np.uint8))
    spot[core == 0] = 0
    _, mx, _, loc = cv2.minMaxLoc(spot)
    if mx < min_score:
        return None, mx
    return np.array([loc[0] * 2.0, loc[1] * 2.0]), mx
