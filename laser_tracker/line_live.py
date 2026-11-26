"""Live camera view with black-line detection overlay (no arm motion).

q/ESC: quit.  Saves the latest overlay every 0.5 s to $LINE_LIVE_SNAP (if set).
Aim reference = image centre (stand-in for the laser pixel until calibration).
"""
import os, time
import cv2
import numpy as np
import config as C
import vision
import line_detect
from laser_spot import find_laser
vision.line_mask = line_detect.line_mask   # robust thin-line detector

SNAP = os.environ.get('LINE_LIVE_SNAP')
HEADLESS = os.environ.get('HEADLESS') == '1'
cam = vision.Camera()
tracker = vision.LineTracker()
last_snap, t0, n, fps = 0.0, time.time(), 0, 0.0
seq = None
try:
    while True:
        seq, f = cam.read(newer_than=seq)
        h, w = f.shape[:2]
        lp, lscore = find_laser(f)
        ref = (w / 2, h / 2) if lp is None else (float(lp[0]), float(lp[1]))
        laser_txt = 'LASER ?' if lp is None else 'LASER ok'
        r = tracker.update(f, ref)
        ov = f.copy()
        if r is not None:
            m = cv2.resize(r['mask'], (w, h), interpolation=cv2.INTER_NEAREST) > 0
            ov[m] = (0.5 * ov[m] + [0, 0, 127]).astype(np.uint8)
            c = tuple(int(v) for v in r['closest']); t = tuple(int(v) for v in r['target'])
            cv2.circle(ov, c, 9, (255, 0, 0), 2)
            cv2.circle(ov, t, 9, (0, 255, 0), 2)
            cv2.line(ov, (int(ref[0]), int(ref[1])), t, (0, 255, 0), 1)
            err = np.hypot(t[0] - ref[0], t[1] - ref[1])
            status = f'{laser_txt}  LINE  err={err:.0f}px'
        else:
            status = f'{laser_txt}  NO LINE'
        cv2.drawMarker(ov, (int(ref[0]), int(ref[1])), (0, 0, 255) if lp is not None else (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.rectangle(ov, (w//2 - 160, h//2 - 110), (w//2 + 160, h//2 + 110), (255, 255, 0), 1)
        n += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps, n, t0 = n / (now - t0), 0, now
        cv2.putText(ov, f'{status}  {fps:.1f} fps', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if (r and lp is not None) else (0, 0, 255), 2)
        if SNAP and now - last_snap >= 0.5:
            cv2.imwrite(SNAP + '.tmp.jpg', ov); os.replace(SNAP + '.tmp.jpg', SNAP)
            last_snap = now
        if HEADLESS:
            continue
        cv2.imshow('Line detection', ov)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
finally:
    cam.close()
    cv2.destroyAllWindows()
