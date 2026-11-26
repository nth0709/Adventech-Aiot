"""One straight pass with the laser: blue start mark -> along the black line -> red end mark.

    python3 path5.py                 # calibrate, go to blue, travel to red, hold 2 s, return
    python3 path5.py --speed 2       # slower travel (px per frame along the path)

Camera and laser are both on the arm (laser dot ~fixed in the image). The marks are
re-detected every frame; while hidden (e.g. under the laser) their image position is
predicted from the joint motion via the calibrated image Jacobian.
"""
import argparse
import os
import time

import cv2
import numpy as np

import track5 as T
import vision
from laser_spot import find_laser
from marks import find_marks

IDS = T.IDS


def calibrate(bus, cam, home, laser_px, delta):
    J = np.zeros((2, len(IDS)))
    base = cam.fresh(skip=5)
    for k, sid in enumerate(IDS):
        d = T.CALIB_SIGN[sid] * delta
        qj = home.copy(); qj[k] += d
        T.goal(bus, qj); time.sleep(0.7)
        moved = cam.fresh(skip=5)
        T.goal(bus, home); time.sleep(0.7)
        s, conf = T.shift(base, moved, laser_px)
        J[:, k] = s / d
        print(f'ID {sid}: jog {d:+d} -> scene shift ({s[0]:+.1f},{s[1]:+.1f}) conf={conf:.2f}', flush=True)
        base = cam.fresh(skip=5)
    mag = np.linalg.norm(J, axis=0)
    use = mag > 0.15 * mag.max()
    sv = np.linalg.svd(J[:, use], compute_uv=False)
    print('active IDs:', [i for i, u in zip(IDS, use) if u], 'singular values:', np.round(sv, 3), flush=True)
    if len(sv) < 2 or sv[-1] < 0.05 * sv[0] or sv[0] < 0.3:
        raise RuntimeError('calibration not trustworthy; not moving')
    W = np.diag(np.where(use, 1.0, 0.0))
    Jinv = W @ J.T @ np.linalg.inv(J @ W @ J.T + 0.1 ** 2 * np.eye(2))
    return J, Jinv, use


def nearest_on_mask(mask, pt, radius=25):
    """Closest line pixel (full res) to pt within radius, else None."""
    s = 0.5
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    d = (xs - pt[0] * s) ** 2 + (ys - pt[1] * s) ** 2
    j = int(np.argmin(d))
    return np.array([xs[j], ys[j]], float) / s if d[j] < (radius * s) ** 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=3.0, help='carrot advance along path, px per frame')
    ap.add_argument('--gain', type=float, default=0.25)
    ap.add_argument('--max-step', type=int, default=10)
    ap.add_argument('--range', type=int, default=300)
    ap.add_argument('--base-range', type=int, default=750, help='range for ID 1 (base yaw)')
    ap.add_argument('--lead', type=float, default=25.0, help='max px the target runs ahead of the laser')
    ap.add_argument('--delta', type=int, default=40)
    ap.add_argument('--timeout', type=float, default=60.0)
    ap.add_argument('--snap', default=None)
    ap.add_argument('--log', default=None)
    ap.add_argument('--record', default=None, help='save overlay video (.avi)')
    args = ap.parse_args()

    cam = vision.Camera()
    bus = T.ServoBus(ids=IDS).connect()
    home = np.array([bus.position(i) for i in IDS], float)
    print('start pose:', home.astype(int).tolist(), flush=True)
    qmin = np.maximum(home - args.range, 10); qmax = np.minimum(home + args.range, 4085)
    for i in T.NO_UP:
        qmax[IDS.index(i)] = home[IDS.index(i)]
    qmin[0] = max(home[0] - args.base_range, 10); qmax[0] = min(home[0] + args.base_range, 4085)
    gui = bool(os.environ.get('DISPLAY'))
    log = open(args.log, 'w') if args.log else None
    writer = None
    q = home.copy()
    try:
        bus.set_speed(speed=800, accel=50)
        bus.torque(True)
        time.sleep(0.3)
        pts = [p for p, _ in (find_laser(cam.fresh()) for _ in range(5)) if p is not None]
        if len(pts) < 3:
            raise RuntimeError('laser dot not found')
        laser = np.median(pts, 0)
        f0 = cam.fresh()
        blue, red = find_marks(f0, laser)
        if blue is None or red is None:
            raise RuntimeError(f'marks not visible (blue={blue}, red={red})')
        print('laser', laser.round().tolist(), 'blue', blue.round().tolist(), 'red', red.round().tolist(), flush=True)

        J, Jinv, use = calibrate(bus, cam, home, laser, args.delta)

        phase, progress, t_done, t0 = 'TO START', 0.0, None, time.time()
        seq, last_snap, last_load, settle, n = None, 0.0, time.time(), 0, 0
        blind, stuck, best_err = 0, 0, -1.0
        while time.time() - t0 < args.timeout:
            seq, frame = cam.read(newer_than=seq)
            b, r = find_marks(frame, laser)
            if b is not None and np.linalg.norm(b - blue) < 60:
                blue = b
            if r is not None and np.linalg.norm(r - red) < 60:
                red = r
            path = red - blue
            L = float(np.linalg.norm(path))
            u = path / max(L, 1e-6)
            mask = vision.line_mask(frame)
            near_line = nearest_on_mask(mask, laser, radius=80) is not None
            blind = blind + 1 if (b is None and r is None and not near_line) else 0
            if blind > 20:
                print('ABORT: lost the sheet (no marks, no line near the laser)', flush=True)
                break

            if phase == 'TO START':
                target = blue.copy()
                if np.linalg.norm(target - laser) < 8:
                    settle += 1
                    if settle > 15:
                        phase, progress = 'TRAVEL', 0.0
                        print('at start (blue) -> travelling', flush=True)
                else:
                    settle = 0
            elif phase == 'TRAVEL':
                # how far along blue->red the laser currently is, then a carrot ahead of it
                along = float(np.dot(laser - blue, u))
                progress = max(progress, min(progress + args.speed, along + args.lead, L))
                carrot = blue + progress * u
                snap = nearest_on_mask(mask, carrot)
                target = snap if (snap is not None and progress < L - 15) else carrot
                if progress >= L and np.linalg.norm(red - laser) < 8:
                    phase, t_done = 'ARRIVED', time.time()
                    print('arrived at end (red)', flush=True)
            else:
                target = red.copy()
                if time.time() - t_done > 2.0:
                    break

            err = target - laser
            at_limit = np.any(((q <= qmin + 1) | (q >= qmax - 1)) & use)
            if phase == 'TRAVEL':
                if progress > best_err + 1:          # best_err = best progress so far
                    best_err, stuck = progress, 0
                else:
                    stuck += 1
                if at_limit and stuck > 45:
                    lim = [i for i, a, b_, k in zip(IDS, q <= qmin + 1, q >= qmax - 1, use) if k and (a or b_)]
                    print(f'ABORT: no progress with joints at range limit {lim}', flush=True)
                    break
            if np.linalg.norm(err) > 2:
                dq = np.clip(args.gain * (Jinv @ (-err)), -args.max_step, args.max_step)
                q_new = np.clip(q + dq, qmin, qmax)
                moved = J @ (q_new - q)                 # predicted scene shift for hidden marks
                if b is None:
                    blue = blue + moved
                if r is None:
                    red = red + moved
                q = q_new
                T.goal(bus, q)

            n += 1
            now = time.time()
            if now - last_load > 1.0:
                ld = T.loads(bus, [i for i, k in zip(IDS, use) if k])
                last_load = time.time()
                if max(ld.values()) > T.LOAD_LIMIT:
                    print('ABORT: load too high', ld, flush=True)
                    break
            if log:
                log.write(f'{now:.3f},{phase},{laser[0]:.1f},{laser[1]:.1f},{blue[0]:.1f},{blue[1]:.1f},'
                          f'{red[0]:.1f},{red[1]:.1f},{err[0]:.1f},{err[1]:.1f},{progress:.1f},{L:.1f}\n')
            if n % 15 == 0:
                print(f'{phase} progress={progress:.0f}/{L:.0f}px err=({err[0]:+.0f},{err[1]:+.0f}) '
                      f'q={q.astype(int).tolist()}', flush=True)

            vis = frame.copy()
            m = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            vis[m] = (0.5 * vis[m] + (0, 90, 0)).astype(np.uint8)
            cv2.line(vis, T.P(blue), T.P(red), (255, 255, 0), 1)
            cv2.circle(vis, T.P(blue), 12, (255, 0, 0), 2)
            cv2.circle(vis, T.P(red), 12, (0, 0, 255), 2)
            cv2.circle(vis, T.P(target), 7, (0, 255, 0), 2)
            cv2.drawMarker(vis, T.P(laser), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            pct = 100 * progress / max(L, 1)
            cv2.putText(vis, f'{phase}  {pct:.0f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'MJPG'), 30,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.snap and now - last_snap > 0.5:
                cv2.imwrite(args.snap + '.tmp.jpg', vis); os.replace(args.snap + '.tmp.jpg', args.snap)
                last_snap = now
            if gui:
                cv2.imshow('laser path: blue -> red', vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        else:
            print('TIMEOUT', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            print('returning to start pose...', flush=True)
            bus.set_speed(speed=200, accel=20)
            T.goal(bus, home)
            time.sleep(2.0)
            print('final pose:', [bus.position(i) for i in IDS], 'torque ON (holding pose)', flush=True)
        finally:
            if writer is not None:
                writer.release()
            if log:
                log.close()
            bus.close()
            cam.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGTERM, T._graceful)
    signal.signal(signal.SIGINT, T._graceful)
    main()
