"""Camera capture + black-line detection + laser-spot detection."""
import threading

import cv2
import numpy as np

import config as C


class Camera:
    """Grabs frames in a background thread so read() always returns the newest frame."""

    def __init__(self, index=C.CAMERA_INDEX, w=C.FRAME_W, h=C.FRAME_H, fps=C.FPS):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open camera %s" % index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.seq = None, 0
        self.cond = threading.Condition()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.cond:
                    self.frame, self.seq = f, self.seq + 1
                    self.cond.notify_all()

    def read(self, newer_than=None, timeout=1.0):
        """Return (seq, frame). With newer_than, block until a frame after that seq arrives."""
        with self.cond:
            ok = self.cond.wait_for(
                lambda: self.frame is not None and (newer_than is None or self.seq > newer_than),
                timeout)
            if not ok:
                raise TimeoutError("camera timeout")
            return self.seq, self.frame.copy()

    def fresh(self, skip=3):
        """Frame captured after at least `skip` new frames (use after the arm moved)."""
        seq, _ = self.read()
        return self.read(newer_than=seq + skip - 1, timeout=2.0)[1]

    def close(self):
        self.running = False
        self.cap.release()


def line_mask(frame):
    """Binary mask (processed scale) of dark line pixels."""
    s = C.PROC_SCALE
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(otsu, (gray < C.DARK_MAX).astype(np.uint8) * 255)
    # no opening (it erases thin lines); small specks are dropped by MIN_LINE_AREA instead.
    # closing bridges small gaps, e.g. laser glare on the line
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    m = C.ROI_MARGIN
    if m > 0:
        mask[:m, :] = 0; mask[-m:, :] = 0; mask[:, :m] = 0; mask[:, -m:] = 0
    return mask


class LineTracker:
    """Finds the aim point on the black line relative to the laser pixel."""

    def __init__(self, lookahead=C.LOOKAHEAD):
        self.lookahead = lookahead
        self.prev_dir = None

    def _ahead(self, pts, closest, direction):
        ahead = closest + self.lookahead * direction
        return pts[int(np.argmin(np.sum((pts - ahead) ** 2, 1)))]   # snap back onto the line

    def update(self, frame, laser_px):
        """Return dict(target, closest, dir, mask) in FULL-res pixels, or None if no line."""
        s = C.PROC_SCALE
        mask = line_mask(frame)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        lx, ly = laser_px[0] * s, laser_px[1] * s

        best, best_d = None, 1e18          # blob closest to the laser -> keeps tracking the same line
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < C.MIN_LINE_AREA:
                continue
            ys, xs = np.nonzero(labels == i)
            d = (xs - lx) ** 2 + (ys - ly) ** 2
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_d, best = d[j], (xs.astype(np.float32), ys.astype(np.float32), j)
        if best is None:
            self.prev_dir = None
            return None

        xs, ys, j = best
        pts = np.stack([xs, ys], 1)
        closest = pts[j]

        # local line direction via PCA
        near = pts[np.sum((pts - closest) ** 2, 1) < C.LOCAL_RADIUS ** 2]
        direction = np.zeros(2, np.float32)
        if len(near) >= 5:
            cov = np.cov((near - near.mean(0)).T)
            w, v = np.linalg.eigh(cov)
            direction = v[:, np.argmax(w)].astype(np.float32)
            if self.prev_dir is not None:
                flip = float(np.dot(direction, self.prev_dir)) < 0     # keep travel direction consistent
            else:
                proj = (pts - closest) @ direction                     # start toward the longer part of the line
                flip = np.count_nonzero(proj < 0) > np.count_nonzero(proj > 0)
            if flip:
                direction = -direction
            self.prev_dir = direction

        target = closest
        if self.lookahead > 0 and np.any(direction):
            target = self._ahead(pts, closest, direction)
            if np.linalg.norm(target - closest) < 0.4 * self.lookahead:   # reached a line end -> turn back
                direction = -direction
                self.prev_dir = direction
                target = self._ahead(pts, closest, direction)
        target = self._center(pts, target, direction)

        return {"target": target / s, "closest": closest / s, "dir": direction, "mask": mask}

    @staticmethod
    def _center(pts, p, direction):
        """Move an edge pixel to the middle of the line's cross-section (matters for thick tape)."""
        if not np.any(direction):
            return p
        normal = np.array([-direction[1], direction[0]], np.float32)
        rel = pts - p
        across = rel @ normal
        sel = (np.abs(rel @ direction) < 1.5) & (np.abs(across) < C.LOCAL_RADIUS)
        if not np.any(sel):
            return p
        return p + normal * float(np.median(across[sel]))


def detect_laser_spot(frame_on, frame_off):
    """Laser pixel from a laser-ON vs laser-OFF frame pair (static scene). Returns ((x,y), strength)."""
    diff = cv2.absdiff(frame_on, frame_off).astype(np.int16)
    b, g, r = diff[..., 0], diff[..., 1], diff[..., 2]
    score = np.clip(r * 2 - b - g, 0, None) + 0.5 * (r + g + b) / 3.0   # favour red, accept saturated core
    score = cv2.GaussianBlur(score.astype(np.float32), (9, 9), 0)
    _, mx, _, loc = cv2.minMaxLoc(score)
    return (float(loc[0]), float(loc[1])), float(mx)


def image_shift(img_a, img_b):
    """Translation (dx, dy) of scene content from img_a to img_b, plus confidence."""
    ga = np.float32(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY))
    gb = np.float32(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return np.array([dx, dy], np.float64), resp
