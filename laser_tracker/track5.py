"""Laser black-line tracking for the 5-servo arm on the CH343 board (eye-in-hand).

    python3 track5.py                 # calibrate (small jogs) then follow the line for 60 s
    python3 track5.py --duration 120 --hold

The red laser is always on and fixed to the camera, so its pixel is found by colour.
At start each servo is jogged once by a few degrees to measure the image Jacobian
(scene shift per tick); then damped-least-squares visual servoing moves the arm so
the black line comes under the laser dot.

Safety: joints limited to start pose +/- RANGE ticks (ID 3 never above start),
per-frame step limit, load check every second, auto stop after --duration.
On exit the arm returns slowly to the start pose and torque is switched off.
Keys: q/ESC quit, space pause, f follow/hold, r reverse direction.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser('~/so101_arm'))
from servo_bus import ServoBus, GOAL_POSITION, PRESENT_LOAD  # noqa: E402

import config as C  # noqa: E402
import vision  # noqa: E402
import line_detect  # noqa: E402
from laser_spot import find_laser  # noqa: E402

vision.line_mask = line_detect.line_mask

IDS = (1, 2, 3, 4, 5)
CALIB_SIGN = {1: 1, 2: 1, 3: -1, 4: 1, 5: 1}   # ID 3 jammed going +, so jog it -
NO_UP = {3}                                     # never command above start pose
LOAD_LIMIT = 60.0                               # % - abort above this


def high_pass(img):
    """Keep fine detail; drop soft shading/vignetting that moves WITH the camera."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 4)


def patch(img, center, w=480, h=320):
    H, W = img.shape[:2]
    x0 = int(np.clip(center[0] - w / 2, 0, W - w)); y0 = int(np.clip(center[1] - h / 2, 0, H - h))
    return img[y0:y0 + h, x0:x0 + w]


_orb = cv2.ORB_create(1500)


def shift(a, b, center):
    """How the scene point under `center` (the laser pixel) moves from image a to b.

    ORB features + RANSAC similarity transform, so rotation of the view (base yaw,
    wrist roll) is handled; the laser dot itself is masked out. Returns (shift, inlier ratio).
    """
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    mask = np.full(ga.shape, 255, np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), 40, 0, -1)
    ka, da = _orb.detectAndCompute(ga, mask); kb, db = _orb.detectAndCompute(gb, mask)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return np.zeros(2), 0.0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 10:
        return np.zeros(2), 0.0
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return np.zeros(2), 0.0
    c = np.array([center[0], center[1], 1.0])
    return M @ c - c[:2], float(inl.mean())


def goal(bus, q):
    bus.sync_write(GOAL_POSITION, {i: [int(v) & 0xFF, int(v) >> 8] for i, v in zip(IDS, q)})


def loads(bus, ids):
    out = {}
    for i in ids:
        b = bus.read(i, PRESENT_LOAD, 2)
        v = b[0] | (b[1] << 8)
        out[i] = (v & 0x3FF) / 10.0
    return out


def P(v):
    return tuple(int(round(float(x))) for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--range', type=int, default=200, help='ticks each way from start pose')
    ap.add_argument('--delta', type=int, default=40, help='calibration jog, ticks')
    ap.add_argument('--gain', type=float, default=0.2)
    ap.add_argument('--max-step', type=int, default=12)
    ap.add_argument('--hold', action='store_true', help='stay on nearest line point, no travel')
    ap.add_argument('--snap', default=None, help='save latest overlay here every 0.5 s')
    ap.add_argument('--log', default=None, help='CSV log of errors/joints')
    ap.add_argument('--mode', choices=('auto', 'fixed', 'arm'), default='auto',
                    help='fixed = camera fixed, laser dot moves; arm = camera on the arm')
    ap.add_argument('--release', action='store_true', help='switch torque off at exit (arm may drop!)')
    ap.add_argument('--calib-only', action='store_true', help='measure the Jacobian, do not track')
    args = ap.parse_args()

    cam = vision.Camera()
    bus = ServoBus(ids=IDS).connect()
    print(f'port {bus.port} (opens={bus.open_attempts})', flush=True)
    home = np.array([bus.position(i) for i in IDS], float)
    print('start pose:', home.astype(int).tolist(), flush=True)
    qmin = np.maximum(home - args.range, 10)
    qmax = np.minimum(home + args.range, 4085)
    for i in NO_UP:
        qmax[IDS.index(i)] = home[IDS.index(i)]
    writer = csv.writer(open(args.log, 'w', newline='')) if args.log else None
    gui = bool(os.environ.get('DISPLAY'))
    q = home.copy()
    try:
        bus.set_speed(speed=800, accel=50)
        bus.torque(True)
        time.sleep(0.3)

        # ---- laser pixel ----
        pts = [p for p, _ in (find_laser(cam.fresh()) for _ in range(5)) if p is not None]
        if len(pts) < 3:
            raise RuntimeError('laser dot not found - is the laser on and visible in the camera?')
        laser_px = np.median(pts, 0)
        print('laser pixel:', np.round(laser_px, 1).tolist(), flush=True)

        # ---- image Jacobian: one-sided jog per joint ----
        # Measure both how the scene shifts (camera on the arm) and how the laser dot
        # moves (camera fixed); whichever dominates decides the mode.
        Js, Jl = np.zeros((2, len(IDS))), np.zeros((2, len(IDS)))
        base = cam.fresh(skip=5)
        for k, sid in enumerate(IDS):
            d = CALIB_SIGN[sid] * args.delta
            qj = home.copy(); qj[k] += d
            goal(bus, qj); time.sleep(0.7)
            moved = cam.fresh(skip=5)
            goal(bus, home); time.sleep(0.7)
            s, resp = shift(base, moved, laser_px)
            lb, _ = find_laser(base); lm, _ = find_laser(moved)
            ls = (lm - lb) if (lb is not None and lm is not None) else np.zeros(2)
            Js[:, k] = s / d
            Jl[:, k] = ls / d
            print(f'ID {sid}: jog {d:+d} -> scene shift ({s[0]:+.1f},{s[1]:+.1f}) conf={resp:.2f}'
                  f' | laser moved ({ls[0]:+.1f},{ls[1]:+.1f}) px', flush=True)
            base = cam.fresh(skip=5)
        laser_move_px = np.median(np.linalg.norm(Jl, axis=0)) * args.delta
        fixed_cam = {'fixed': True, 'arm': False}.get(args.mode, laser_move_px > 8.0)
        J = Jl if fixed_cam else Js
        sign = 1.0 if fixed_cam else -1.0     # fixed cam: move the dot by +err; on-arm: move scene by -err
        print('mode:', 'FIXED camera (laser dot moves)' if fixed_cam else 'camera ON ARM (scene moves)', flush=True)
        mag = np.linalg.norm(J, axis=0)
        use = mag > 0.15 * mag.max()
        W = np.diag(np.where(use, 1.0, 0.0))
        print('Jacobian px/tick:\n', np.round(J, 3), '\nactive IDs:', [i for i, u in zip(IDS, use) if u], flush=True)
        sv = np.linalg.svd(J[:, use], compute_uv=False)
        print('singular values:', np.round(sv, 3), flush=True)
        if len(sv) < 2 or sv[-1] < 0.05 * sv[0] or sv[0] < 0.3:
            raise RuntimeError('Jacobian too small or singular - calibration not trustworthy; not tracking')
        if args.calib_only:
            return
        lam = 0.1
        Jinv = W @ J.T @ np.linalg.inv(J @ W @ J.T + lam ** 2 * np.eye(2))

        # ---- tracking ----
        tracker = vision.LineTracker(lookahead=0 if args.hold else C.LOOKAHEAD)
        t_end = time.time() + args.duration
        seq, n, last_snap, last_load, paused = None, 0, 0.0, time.time(), False
        fps, t_prev, lost = 0.0, time.time(), 0
        while time.time() < t_end:
            seq, frame = cam.read(newer_than=seq)
            p, _ = find_laser(frame)
            if fixed_cam:
                if p is not None and np.linalg.norm(p - laser_px) < 120:
                    laser_px = 0.5 * laser_px + 0.5 * p
            elif p is not None and np.linalg.norm(p - laser_px) < 15:
                laser_px = 0.95 * laser_px + 0.05 * p   # camera on arm: dot is ~fixed in image
            res = tracker.update(frame, laser_px)
            err = None
            if res is None:
                lost += 1
                state = 'NO LINE (holding)'
            else:
                lost = 0
                err = res['target'] - laser_px
                if not paused and np.linalg.norm(err) > 3:
                    dq = np.clip(args.gain * (Jinv @ (sign * err)), -args.max_step, args.max_step)
                    q_new = np.clip(q + dq, qmin, qmax)
                    if fixed_cam and p is None:
                        laser_px = laser_px + J @ (q_new - q)   # dot hidden (e.g. on the line): predict
                    q = q_new
                    goal(bus, q)
                mode = 'PAUSED' if paused else ('HOLD' if tracker.lookahead == 0 else 'FOLLOW')
                state = f'{mode} err=({err[0]:+.0f},{err[1]:+.0f})'
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3); t_prev = now
            n += 1
            if now - last_load > 1.0:
                ld = loads(bus, [i for i, u in zip(IDS, use) if u])
                last_load = time.time()
                if max(ld.values()) > LOAD_LIMIT:
                    print('ABORT: load too high', ld, flush=True)
                    break
            if writer:
                writer.writerow([round(now, 3)] + ([] if err is None else np.round(err, 1).tolist())
                                + q.astype(int).tolist())
            if n % 30 == 0:
                print(f'{state}  q={q.astype(int).tolist()}  {fps:.0f}fps', flush=True)

            vis = frame.copy()
            if res is not None:
                m = cv2.resize(res['mask'], (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                vis[m] = (0.5 * vis[m] + (0, 90, 0)).astype(np.uint8)
                cv2.circle(vis, P(res['target']), 9, (0, 255, 0), 2)
                cv2.line(vis, P(laser_px), P(res['target']), (0, 255, 255), 1)
            cv2.drawMarker(vis, P(laser_px), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            left = max(0, t_end - now)
            cv2.putText(vis, f'{state}  {fps:.0f}fps  {left:.0f}s', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if args.snap and now - last_snap > 0.5:
                cv2.imwrite(args.snap + '.tmp.jpg', vis); os.replace(args.snap + '.tmp.jpg', args.snap)
                last_snap = now
            if gui:
                cv2.imshow('5-axis laser tracker', vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == ord(' '):
                    paused = not paused
                if k == ord('f'):
                    tracker.lookahead = C.LOOKAHEAD if tracker.lookahead == 0 else 0
                if k == ord('r') and tracker.prev_dir is not None:
                    tracker.prev_dir = -tracker.prev_dir
    except KeyboardInterrupt:
        pass
    finally:
        try:
            print('returning to start pose...', flush=True)
            bus.set_speed(speed=200, accel=20)
            goal(bus, home)
            time.sleep(2.0)
            if args.release:
                bus.torque(False)
            print('final pose:', [bus.position(i) for i in IDS],
                  'torque OFF' if args.release else 'torque ON (holding pose)', flush=True)
        finally:
            bus.close()
            cam.close()
            cv2.destroyAllWindows()


def _graceful(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)
    main()
