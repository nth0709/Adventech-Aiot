"""Closed-loop laser line following: blue start mark -> along the black line -> red end mark.

Camera and laser are both on the arm (the laser dot is ~fixed in the image).

    python3 follow5.py --save-rest   # arm resting, torque off: remember the rest pose (read only)
    python3 follow5.py --teach       # hold the arm where the camera sees the sheet: remember it,
                                     # then torque on so the arm holds that pose by itself
    python3 follow5.py               # rest -> observe pose -> follow line -> observe -> rest -> torque off

Start check (no hand-held pose, no fixed laser pixel):
  1. laser: the red dot that stays put in the image while the base turns a little.
  2. line: if not in view, raise the head (ID 4) step by step, then swing the base.
  3. approach: closed loop until the laser sits on the line seen in the current frame.
  Calibration: image Jacobian measured once on a grid of poses (base x vertical) around the
  observe pose, saved to calib_grid.json and reused by every later run/task
  (re-measured only if the observe pose or the laser pixel / table height changed, or --recalibrate).
Path: walked along the line in the current frame (geodesic), so sharp corners (Z, zig-zag) are
  followed; travel direction = continuity ("not the branch we came from").

Every frame uses only what is observed NOW:
  * laser dot, black line (component nearest the laser), blue and red marks.
  * No predicted/remembered mark positions are ever used for control.
  * Nothing usable in view -> hold still; after 1 s -> abort.
Path = the black line itself (local direction from PCA around the point under the laser).
Travel direction: toward the red mark when it is visible, otherwise keep the previous direction.
Start: go to the blue mark if visible; otherwise search along the line (reversing at line ends).
End: red mark detected under the laser.
The image Jacobian is re-measured whenever the arm has moved far from where it was measured.
"""
import argparse
import csv
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

import track5 as T          # also installs line_detect.line_mask into vision
import path5
import vision
import config as C
from laser_spot import find_laser
from marks import find_marks
from sheet import sheet_region

IDS = T.IDS
HERE = os.path.dirname(os.path.abspath(__file__))
OBSERVE_FILE = os.path.join(HERE, 'observe_pose.json')
REST_FILE = os.path.join(HERE, 'rest_pose.json')
LASER_FILE = os.path.join(HERE, 'laser_pixel.json')
CALIB_FILE = os.path.join(HERE, 'calib_grid.json')
CAL_IDS = (1, 2, 3, 4)          # joints that move the view at the laser (ID 5 ~ roll, no effect)
S = C.PROC_SCALE
LOAD_LIMIT = 50.0


# ---------------- poses ----------------
def save_pose(path, q):
    with open(path, 'w') as f:
        json.dump({'q': [int(v) for v in q], 'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f)


def load_pose(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return np.array(json.load(f)['q'], float)


def read_q(bus):
    return np.array([bus.position(i) for i in IDS], float)


def move_slow(bus, q, timeout=10.0):
    """Joint move at low speed with load monitoring; returns reached pose."""
    bus.set_speed(speed=250, accel=20)
    T.goal(bus, q)
    t0, cur = time.time(), read_q(bus)
    while time.time() - t0 < timeout:
        time.sleep(0.2)
        prev, cur = cur, read_q(bus)
        if np.max(np.abs(cur - prev)) > 3:      # still moving: never treat as reached
            continue
        ld = T.loads(bus, IDS)
        if max(ld.values()) > LOAD_LIMIT + 15:
            T.goal(bus, cur)
            raise RuntimeError(f'load too high while moving: {ld}')
        if np.max(np.abs(cur - q)) < 50:          # weak joints sag a little under gravity at 5 V
            if np.max(np.abs(cur - q)) >= 15:
                print('  (sag vs target:', (cur - q).astype(int).tolist(), ')', flush=True)
            return cur
    raise RuntimeError(f'pose not reached in {timeout}s: now {cur.astype(int).tolist()}')


# ---------------- perception ----------------
def line_at_laser(mask, laser):
    """Line component nearest the laser: dict(pts[half-res], closest[full], dist[full px], dir) or None."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    lx, ly = laser[0] * S, laser[1] * S
    best = None
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 60:
            continue
        ys, xs = np.nonzero(lab == i)
        d2 = (xs - lx) ** 2 + (ys - ly) ** 2
        j = int(np.argmin(d2))
        if best is None or d2[j] < best[0]:
            best = (d2[j], xs, ys, j)
    if best is None:
        return None
    d2, xs, ys, j = best
    pts = np.stack([xs, ys], 1).astype(np.float32)
    closest = pts[j]
    near = pts[np.sum((pts - closest) ** 2, 1) < C.LOCAL_RADIUS ** 2]
    direction = None
    if len(near) >= 5:
        w, v = np.linalg.eigh(np.cov((near - near.mean(0)).T))
        direction = v[:, np.argmax(w)].astype(np.float64)
    return {'pts': pts, 'closest_h': closest, 'closest': closest / S, 'dist': float(np.sqrt(d2)) / S,
            'dir': direction}


def bridge_laser_gap(mask, laser, r=24, k=15):
    """The bright laser dot breaks the dark line under it; close that gap locally only."""
    x, y = int(laser[0] * S), int(laser[1] * S)
    h, w = mask.shape
    x0, x1, y0, y1 = max(x - r, 0), min(x + r, w), max(y - r, 0), min(y + r, h)
    roi = mask[y0:y1, x0:x1]
    if roi.size:
        mask[y0:y1, x0:x1] = cv2.morphologyEx(
            roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def ahead_on_line(line, direction, lookahead_px, mask=None):
    """Point `lookahead_px` ahead along the line (full res) and whether the line ends before it.

    If this line piece ends (e.g. the laser glare cuts the line), line pixels in a narrow
    corridor further ahead (other pieces of the same line) are used to continue.
    """
    pts, c = line['pts'], line['closest_h']
    la = lookahead_px * S
    want = c + la * direction
    p = pts[int(np.argmin(np.sum((pts - want) ** 2, 1)))]
    at_end = float(np.linalg.norm(p - c)) < 0.4 * la
    if at_end and mask is not None:
        ys, xs = np.nonzero(mask)
        allp = np.stack([xs, ys], 1).astype(np.float32)
        rel = allp - c
        along = rel @ direction
        across = np.abs(rel @ np.array([-direction[1], direction[0]]))
        sel = (along > 0.5 * la) & (along < 8 * la) & (across < 8)
        if np.any(sel):
            cand = allp[sel]
            p = cand[int(np.argmin(np.abs((cand - c) @ direction - la)))]
            pts = allp
            at_end = False
    p = vision.LineTracker._center(pts, p, direction.astype(np.float32))
    return p / S, at_end


def _geo_dist(conn, seed, maxd):
    """8-neighbour geodesic distance inside `conn` from `seed` (bool masks), up to maxd; -1 = unreached."""
    dist = np.full(conn.shape, -1, np.int16)
    reached = seed & conn
    dist[reached] = 0
    k = np.ones((3, 3), np.uint8)
    for d in range(1, maxd + 1):
        grown = cv2.dilate(reached.astype(np.uint8), k).astype(bool) & conn
        new = grown & ~reached
        if not new.any():
            break
        dist[new] = d
        reached = grown
    return dist


GAP_K = 7          # half-res px: line breaks up to ~6 px (12 px full res) count as connected
JUMP_R = 20        # half-res px: a separate line piece this close (not behind) continues the path


def _jump_gap(mask, c, d_local, wx0, wy0, heading, back_dir):
    """Nearest line pixel within JUMP_R of c that is not part of what we can reach and does
    not lie back the way we came. Returns its half-res position or None."""
    h, w = mask.shape
    cx, cy = int(c[0]), int(c[1])
    x0, y0 = max(cx - JUMP_R, 0), max(cy - JUMP_R, 0)
    sub = mask[y0:min(cy + JUMP_R + 1, h), x0:min(cx + JUMP_R + 1, w)] > 0
    ys, xs = np.nonzero(sub)
    if xs.size == 0:
        return None
    pts = np.stack([xs + x0, ys + y0], 1).astype(float)
    v = pts - c
    r = np.linalg.norm(v, axis=1)
    u = v / np.maximum(r[:, None], 1e-6)
    # reachable part (within the walked window) is excluded via its distance map footprint
    ok = (r > 6) & (r <= JUMP_R) & (u @ back_dir < 0.3) & (u @ heading > -0.9)
    lx, ly = (pts[:, 0] - wx0).astype(int), (pts[:, 1] - wy0).astype(int)
    inside = (lx >= 0) & (ly >= 0) & (lx < d_local.shape[1]) & (ly < d_local.shape[0])
    reach = np.zeros(len(pts), bool)
    reach[inside] = d_local[ly[inside], lx[inside]] >= 0
    ok &= ~reach
    if not np.any(ok):
        return None
    i = int(np.argmin(np.where(ok, r, np.inf)))
    return pts[i]


def _branches_at(mask, cx, cy, R):
    """Walk the line R px (geodesic, gaps bridged) from (cx, cy); the ways out at distance R.
    Branches are split on the THIN line, so the two arms of a very sharp corner stay apart."""
    h, w = mask.shape
    W = R + 4
    x0, y0, x1, y1 = max(cx - W, 0), max(cy - W, 0), min(cx + W + 1, w), min(cy + W + 1, h)
    win = mask[y0:y1, x0:x1]
    conn = cv2.dilate(win, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_K, GAP_K))) > 0
    conn[cy - y0, cx - x0] = True
    seed = np.zeros_like(conn); seed[cy - y0, cx - x0] = True
    d = _geo_dist(conn, seed, R + 1)
    thin = cv2.dilate(win, np.ones((3, 3), np.uint8)) > 0
    ring = (((d >= R - 1) & (d <= R + 1)) & thin).astype(np.uint8)
    n, lab = cv2.connectedComponents(ring, connectivity=8)
    c = np.array([cx, cy], float)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if xs.size < 2:
            continue
        p = np.array([xs.mean() + x0, ys.mean() + y0])
        v = p - c
        if np.linalg.norm(v) >= 0.4 * R:
            out.append((p, v / np.linalg.norm(v), lab == i))
    return out, conn, d, x0, y0


def path_ahead(mask, c_h, prev_dir, la_h, R=12, away_from=None):
    """Next target ALONG the line (follows sharp corners), all in the current frame.

    From the line pixel under the laser (c_h, half res) the line is walked R px each way
    (geodesic, so a corner is walked around, not cut). Each branch found at distance R is a
    way to go; the one we came from is the one pointing against prev_dir (recent heading).
    Returns (target_full, branch_dir, at_end)."""
    R = int(max(R, la_h + 3))
    cx, cy = int(round(c_h[0])), int(round(c_h[1]))
    c = np.array([cx, cy], float)
    branches, conn, d, x0, y0 = _branches_at(mask, cx, cy, R)
    path_ahead.n_branches = len(branches)          # 1 = the laser is at an end of the line
    if not branches:
        return None, prev_dir, True
    if prev_dir is not None:
        b = max(branches, key=lambda x: float(x[1] @ prev_dir))
        # a lone branch is followed only if it really continues ahead; otherwise it may be the
        # way back with the way forward momentarily invisible (faint ink, blur at a corner)
        if len(branches) == 1 and float(b[1] @ prev_dir) < 0.3:
            # only the way back seen: a very sharp corner looks like one branch up close,
            # so look again twice as far, then for a separate line piece nearby (faint ink)
            far, conn2, d2_, x02, y02 = _branches_at(mask, cx, cy, 2 * R)
            fwd = [x for x in far if float(x[1] @ b[1]) < 0.9 and float(x[1] @ prev_dir) > -0.98]
            if len(far) >= 2 and fwd:
                bf = max(fwd, key=lambda x: float(x[1] @ prev_dir))
                dd = _geo_dist(conn2, bf[2], 2 * R + 2)
                sel = (np.abs(d2_ - la_h) <= 1) & (d2_ >= 0) & (dd >= 0) & (np.abs(dd - (2 * R - la_h)) <= 2)
                if np.any(sel):
                    ys, xs = np.nonzero(sel)
                    return np.array([xs.mean() + x02, ys.mean() + y02]) / S, bf[1], False
                return (c + bf[1] * la_h) / S, bf[1], False
            jump = _jump_gap(mask, c, d, x0, y0, prev_dir, b[1])
            if jump is not None:
                return jump / S, prev_dir, False
            return None, prev_dir, True                 # really the end of the line
    elif away_from is not None:
        b = max(branches, key=lambda x: float(np.linalg.norm(x[0] - away_from)))
    else:
        b = max(branches, key=lambda x: int(x[2].sum()))
    # target: the pixels of the chosen branch that lie la_h along the line from c
    d2 = _geo_dist(conn, b[2], R + 2)
    sel = (np.abs(d - la_h) <= 1) & (d >= 0) & (d2 >= 0) & (np.abs(d2 - (R - la_h)) <= 2)
    if np.any(sel):
        ys, xs = np.nonzero(sel)
        t = np.array([xs.mean() + x0, ys.mean() + y0])
    else:
        t = c + b[1] * la_h
    return t / S, b[1], False


# ---------------- calibration grid (measured once, saved, reused) ----------------
def build_grid(bus, cam, center, laser, delta, base_offs, vert_offs, J_v, qmin, qmax):
    """Measure the image Jacobian on a grid of poses around `center`:
    base (ID1) offsets x vertical view offsets (IDs 2-4 together; v>0 = laser goes up the image).
    Poses where the view has too little texture are skipped. Saved to CALIB_FILE."""
    entries, total = [], len(base_offs) * len(vert_offs)
    k234 = [IDS.index(i) for i in (2, 3, 4)]
    for row, v in enumerate(vert_offs):
        for b in (base_offs if row % 2 == 0 else base_offs[::-1]):      # serpentine: short moves
            qc = center.copy(); qc[0] += b
            for k in k234:                       # scene moves down v px -> laser goes up the image
                qc[k] += np.clip(v / (3 * J_v[k]), -300, 300)
            qc = np.clip(qc, qmin, qmax)
            try:
                qc = move_slow(bus, qc)
            except RuntimeError as e:
                if 'load' in str(e):
                    raise
                print(f'  grid: skip base {b:+d} v {v:+d}: {e}', flush=True)
                continue
            bus.set_speed(speed=800, accel=50); time.sleep(0.3)
            try:
                J = measure_J(bus, cam, qc, laser, delta)
            except RuntimeError as e:
                print(f'  grid: skip base {b:+d} v {v:+d}: {e}', flush=True)
                continue
            entries.append({'q': qc[:4].tolist(), 'J': J.tolist(), 'base': b, 'v': v})
            print(f'  grid {len(entries):2d}/{total}: base {b:+4d} v {v:+4d}  '
                  f'J=[{" ".join(f"{x:+.2f}" for x in J[0, :4])} | {" ".join(f"{x:+.2f}" for x in J[1, :4])}]',
                  flush=True)
    move_slow(bus, center)
    if len(entries) < 0.5 * total:
        raise RuntimeError(f'calibration: only {len(entries)}/{total} poses measured - not saved')
    grid = {'center': center.tolist(), 'laser': laser.tolist(), 'delta': delta, 'entries': entries,
            'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(CALIB_FILE, 'w') as f:
        json.dump(grid, f, indent=1)
    print(f'calibration saved: {len(entries)}/{total} poses -> {CALIB_FILE}', flush=True)
    return grid


def load_grid(center, laser):
    """Saved calibration, if it was made from this observe pose with the laser at ~this pixel
    (a different table height moves the laser pixel and changes the Jacobian)."""
    if not os.path.exists(CALIB_FILE):
        return None, 'no calibration file'
    with open(CALIB_FILE) as f:
        g = json.load(f)
    if np.max(np.abs(np.array(g['center']) - center)) > 15:
        return None, 'observe pose changed'
    if np.linalg.norm(np.array(g['laser']) - laser) > 25:
        return None, f'laser pixel moved ({np.round(g["laser"]).tolist()} -> {laser.round().tolist()}): table height changed?'
    return g, 'ok'


def grid_J_fn(grid, k_near=4):
    """J(q): inverse-distance blend of the nearest calibrated poses (IDs 1-4)."""
    Q = np.array([e['q'] for e in grid['entries']], float)
    Js = np.array([e['J'] for e in grid['entries']], float)

    def J_of(q):
        dist = np.linalg.norm(Q - np.asarray(q, float)[:4], axis=1)
        idx = np.argsort(dist)[:k_near]
        w = 1.0 / (dist[idx] + 10.0) ** 2
        return np.tensordot(w / w.sum(), Js[idx], axes=1)
    return J_of


# ---------------- Jacobian measurement ----------------
def measure_J(bus, cam, qc, laser, delta):
    """Image Jacobian at pose qc: jog each CAL_IDS joint once, ORB shift at the laser pixel."""
    J = np.zeros((2, len(IDS)))
    for sid in CAL_IDS:
        k = IDS.index(sid)
        imgs = []
        for sign in (-1, 1):                     # central difference: -delta .. +delta
            qj = qc.copy(); qj[k] += sign * delta
            T.goal(bus, qj); time.sleep(0.6)
            imgs.append(cam.fresh(skip=4))
        T.goal(bus, qc); time.sleep(0.5)
        sh, conf = T.shift(imgs[0], imgs[1], laser)
        if conf < 0.25:
            raise RuntimeError(f'Jacobian measurement unreliable at ID {sid} (conf {conf:.2f})')
        J[:, k] = sh / (2 * delta)
    return J


def servo_step(J, err, gain, max_px, max_step):
    """Joint step that moves the view by gain*err, but at most max_px in the IMAGE per cycle.
    (Several joints move the view up/down together, so a per-joint cap alone lets vertical
    motion run ~3x faster than horizontal and overshoot with the camera delay.)"""
    dq = dls_inverse(J) @ (-gain * np.asarray(err, float))
    px = float(np.linalg.norm(J @ dq))
    if px > max_px:
        dq *= max_px / px
    return np.clip(dq, -max_step, max_step)


def dls_inverse(J, lam=0.1):
    W = np.diag([1.0 if i in CAL_IDS else 0.0 for i in IDS])
    return W @ J.T @ np.linalg.inv(J @ W @ J.T + lam ** 2 * np.eye(2))


def goal_with_speed(bus, q, speed):
    """One sync-write per cycle: goal position + per-joint speed (STS regs 42..47) so each
    servo arrives just as the next command comes -> continuous motion, no stop-and-go."""
    data = {}
    for i, v, sp in zip(IDS, q, speed):
        v, sp = int(round(v)), int(round(sp))
        data[i] = [v & 0xFF, v >> 8, 0, 0, sp & 0xFF, sp >> 8]
    bus.sync_write(T.GOAL_POSITION, data)


LASER_WIN = 80          # px: the dot moves in the image when the table distance changes (parallax)


COAST_PX = 30           # px of view motion allowed on the last observed laser pixel


class LaserTrack:
    """Laser pixel observed every frame (searched around the last observation).

    The dot vanishes on blue ink (absorbs red) and can fade on the black line. It is fixed to
    the camera except for slow parallax, so the LAST OBSERVED pixel stays valid while the view
    has moved < COAST_PX since then. Beyond that the arm holds; unusable for 1 s -> stop."""

    def __init__(self, laser):
        self.p, self.seen, self.coast, self.bad = np.asarray(laser, float), True, 0.0, 0

    def update(self, frame):
        p, _ = find_laser_near(frame, self.p, half=LASER_WIN)
        self.seen = p is not None
        if self.seen:
            self.p, self.coast = 0.3 * self.p + 0.7 * p, 0.0
        self.bad = 0 if self.usable else self.bad + 1
        if self.bad > 30:
            raise RuntimeError('laser dot not seen for 1 s - stopped')
        return self.p

    @property
    def usable(self):
        return self.seen or self.coast < COAST_PX

    def moved(self, J, dq):
        if not self.seen:
            self.coast += float(np.linalg.norm(J @ dq))


def find_laser_near(frame, laser, half=60):
    """Laser is ~fixed in the image (on the arm): search only a small window around it."""
    x0 = int(max(laser[0] - half, 0)); y0 = int(max(laser[1] - half, 0))
    crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    p, sc = find_laser(crop)
    return (None, sc) if p is None else (p + [x0, y0], sc)


LINE_MIN_PX = 150             # line-mask pixels (half res) that count as "the line is in view"
HEAD_ID = 4                    # wrist pitch: "raise the head" to see further along the table


def look_laser(cam, near, half, n=5):
    """Median laser position over n fresh frames, searching only within `half` px of `near`."""
    pts, fr = [], None
    for _ in range(n):
        fr = cam.fresh()
        p, _ = find_laser_near(fr, near, half=half)
        if p is not None:
            pts.append(p)
    return (np.median(pts, 0) if len(pts) >= max(3, n // 2 + 1) else None), fr


def verify_laser(bus, cam, q, known, tol=150, jog=60):
    """Confirm the red dot is the laser: it stays put in the image while the arm turns
    (camera and laser are on the arm), whereas red ink / reflections move with the scene.
    Accepts a different pixel than last time (the dot shifts with table height)."""
    p0, f0 = look_laser(cam, known, tol)
    if p0 is None:
        raise RuntimeError(f'laser dot not visible within {tol}px of {known.round().tolist()}')
    qj = q.copy(); qj[0] += jog
    T.goal(bus, qj); time.sleep(0.7)
    p1, f1 = look_laser(cam, p0, 60)
    T.goal(bus, q); time.sleep(0.7)
    if p1 is None:
        raise RuntimeError('laser candidate vanished when the base turned - not the laser')
    moved = float(np.linalg.norm(p1 - p0))
    scene, conf = T.shift(f0, f1, p0)
    print(f'laser check: dot moved {moved:.1f}px while the view moved {np.linalg.norm(scene):.0f}px '
          f'(conf {conf:.2f})', flush=True)
    # the laser shifts a little with table height/tilt (parallax); scene-fixed red ink moves
    # about as much as the view does. Reject only what follows the scene.
    lim = max(8.0, 0.3 * float(np.linalg.norm(scene))) if conf >= 0.25 else 8.0
    if moved > lim:
        raise RuntimeError(f'red dot moved {moved:.0f}px with the scene (limit {lim:.0f}px) - not the laser')
    if conf >= 0.25 and np.linalg.norm(scene) < 15:
        raise RuntimeError('view did not move during the laser check - cannot confirm the laser')
    return (p0 + p1) / 2


def line_in_view(frame):
    return int(np.count_nonzero(vision.line_mask(frame))) >= LINE_MIN_PX


def search_line(bus, cam, q_obs, head_sign):
    """Line not seen: raise the head step by step, then swing the base left/right.
    Each stop only reports what the camera sees there; nothing is remembered between stops."""
    plan = [(0, 0)] + [(0, h) for h in (60, 120, 180, 240)]
    plan += [(b, h) for b in (-250, 250) for h in (0, 120, 240)]
    for db, dh in plan:
        qt = q_obs.copy(); qt[0] += db; qt[IDS.index(HEAD_ID)] += head_sign * dh
        if db or dh:
            print(f'  looking: base {db:+d}, head up {dh}', flush=True)
            qt = move_slow(bus, qt)
            bus.set_speed(speed=800, accel=50)
            time.sleep(0.3)
        if line_in_view(cam.fresh()) and line_in_view(cam.fresh()):
            print('  black line in view', flush=True)
            return qt
    raise RuntimeError('black line not found (looked up and left/right) - stopped')


def approach_line(bus, cam, q, laser, J_fn, qmin, qmax, args, video=None, gui=False):
    """Closed loop: bring the laser onto the nearest point of the line seen in THIS frame."""
    q_cmd, dq_f = q.copy(), np.zeros(len(IDS))
    seq, lost, settle, period, t_last = None, 0, 0, 1 / 30, time.time()
    lt = LaserTrack(laser)
    t0 = time.time()
    while time.time() - t0 < 25:
        seq, frame = cam.read(newer_than=seq)
        laser = lt.update(frame)
        mask = bridge_laser_gap(vision.line_mask(frame), laser)
        line = line_at_laser(mask, laser)
        dq = np.zeros(len(IDS))
        if not lt.usable:
            settle = 0; dq_f[:] = 0
            note = 'laser not seen - holding'
        elif line is None:
            lost += 1; settle = 0; dq_f[:] = 0
            note = 'line not in view - holding'
            if lost > 30:
                raise RuntimeError('line lost for 1 s while approaching - stopped (no blind moves)')
        else:
            lost = 0
            err = line['closest'] - laser
            settle = settle + 1 if line['dist'] < 15 else 0
            if settle >= 8:
                print(f'laser on the line (dist {line["dist"]:.0f}px) at q={q_cmd.astype(int).tolist()}',
                      flush=True)
                return q_cmd, laser
            dq = servo_step(J_fn(q_cmd), err, args.gain, args.max_px, args.max_step)
            note = f'approach line {line["dist"]:.0f}px'
        now = time.time()
        period = 0.8 * period + 0.2 * min(max(now - t_last, 0.01), 0.2); t_last = now
        dq_f = args.smooth * dq_f + (1 - args.smooth) * dq
        q_new = np.clip(q_cmd + dq_f, qmin, qmax)
        speed = np.clip(np.abs(q_new - q_cmd) / period * 1.3, 20, 1500)
        lt.moved(J_fn(q_cmd), q_new - q_cmd)
        q_cmd = q_new
        goal_with_speed(bus, q_cmd, speed)
        if video is not None or gui:
            vis = frame.copy()
            m = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            vis[m] = (0.5 * vis[m] + (0, 90, 0)).astype(np.uint8)
            if line is not None:
                cv2.circle(vis, T.P(line['closest']), 7, (0, 255, 0), 2)
            cv2.circle(vis, T.P(laser), 16, (0, 255, 255), 1)   # hollow: never hides the dot
            cv2.putText(vis, f'APPROACH  {note}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if video is not None:
                video.put(vis)
            if gui:
                cv2.imshow('follow line: blue -> red', vis); cv2.waitKey(1)
    raise RuntimeError('could not bring the laser onto the line in 25 s - stopped')


class VideoSaver:
    """Writes frames on a background thread (MJPG encode costs ~33 ms per frame)."""

    def __init__(self, path):
        self.path, self.q, self.w = path, queue.Queue(maxsize=8), None
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def put(self, img):
        try:
            self.q.put_nowait(img)
        except queue.Full:
            pass

    def _run(self):
        while True:
            img = self.q.get()
            if img is None:
                break
            if self.w is None:
                self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (img.shape[1], img.shape[0]))
            self.w.write(img)

    def close(self):
        self.q.put(None); self.t.join(timeout=5)
        if self.w is not None:
            self.w.release()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teach', action='store_true')
    ap.add_argument('--save-rest', action='store_true')
    ap.add_argument('--lookahead', type=float, default=15.0, help='px ahead on the line (sets speed)')
    ap.add_argument('--gain', type=float, default=0.25)
    ap.add_argument('--max-step', type=int, default=10)
    ap.add_argument('--max-px', type=float, default=8.0, help='max laser motion in the image per cycle (px)')
    ap.add_argument('--range', type=int, default=300)
    ap.add_argument('--base-range', type=int, default=750)
    ap.add_argument('--recalibrate', action='store_true', help='measure the calibration grid again')
    ap.add_argument('--grid-base', default='-450,-300,-150,0,150,300,450',
                    help='calibration: ID1 offsets from the observe pose')
    ap.add_argument('--grid-vert', default='-50,50,150,250,350',
                    help='calibration: vertical view offsets in image px (+ = laser goes up the image)')
    ap.add_argument('--smooth', type=float, default=0.5, help='0..1 low-pass on commands/target (higher=smoother)')
    ap.add_argument('--delta', type=int, default=40)
    ap.add_argument('--timeout', type=float, default=90.0)
    ap.add_argument('--snap', default=None)
    ap.add_argument('--log', default=None)
    ap.add_argument('--record', default=None)
    args = ap.parse_args()

    bus = T.ServoBus(ids=IDS).connect()
    if args.save_rest or args.teach:
        try:
            q = read_q(bus)
            if args.save_rest:
                save_pose(REST_FILE, q)
                print('rest pose saved:', q.astype(int).tolist())
            else:
                save_pose(OBSERVE_FILE, q)
                bus.set_speed(speed=250, accel=20)
                bus.torque(True)
                print('observe pose saved:', q.astype(int).tolist(), '- torque ON, arm holds it now')
        finally:
            bus.close()
        return

    observe = load_pose(OBSERVE_FILE)
    rest = load_pose(REST_FILE)
    if observe is None:
        bus.close()
        raise SystemExit('no observe pose yet: run  python3 follow5.py --teach')

    cam = vision.Camera()
    gui = bool(os.environ.get('DISPLAY'))
    log = csv.writer(open(args.log, 'w', newline='')) if args.log else None
    writer = None
    qmin = np.maximum(observe - args.range, 10); qmax = np.minimum(observe + args.range, 4085)
    qmin[0] = max(observe[0] - args.base_range, 10); qmax[0] = min(observe[0] + args.base_range, 4085)
    result = 'aborted'
    try:
        bus.torque(True)                          # hold wherever it is now (goal = current)
        print('moving to observe pose', observe.astype(int).tolist(), flush=True)
        q = move_slow(bus, observe)
        bus.set_speed(speed=800, accel=50)
        time.sleep(0.5)

        video = VideoSaver(args.record) if args.record else None

        # ---- start check 1: which red dot is the laser (wiggle test, no fixed pixel) ----
        known = load_pose(LASER_FILE)
        if known is None:
            known = np.array([C.FRAME_W / 2, C.FRAME_H / 2])
        laser = verify_laser(bus, cam, q, known)
        print('laser pixel', laser.round().tolist(), flush=True)
        save_pose(LASER_FILE, laser)

        # ---- calibration: load the saved grid, or measure it once (then reused by later tasks) ----
        grid, why = (None, 'recalibrate requested') if args.recalibrate else load_grid(observe, laser)
        if grid is None:
            print(f'calibration needed ({why})', flush=True)
            J_obs = measure_J(bus, cam, q, laser, args.delta)          # for the grid's vertical moves
            base_offs = [int(v) for v in args.grid_base.split(',')]
            vert_offs = [int(v) for v in args.grid_vert.split(',')]
            print(f'measuring calibration grid: base {base_offs} x vertical {vert_offs} '
                  f'({len(base_offs) * len(vert_offs)} poses, a few minutes)', flush=True)
            grid = build_grid(bus, cam, observe, laser, args.delta, base_offs, vert_offs,
                              J_obs[1], qmin, qmax)
            q = read_q(bus)
        else:
            print(f'using saved calibration {CALIB_FILE} ({len(grid["entries"])} poses, {grid["time"]})',
                  flush=True)
        J_fn = grid_J_fn(grid)

        # ---- start check 2: find the black line (raise head / swing base) and put the laser on it ----
        bus.set_speed(speed=800, accel=50)
        head_sign = float(np.sign(J_fn(q)[1, IDS.index(HEAD_ID)]) or -1)   # head up = view moves down
        q = search_line(bus, cam, q, head_sign)
        q, laser = approach_line(bus, cam, q, laser, J_fn, qmin, qmax, args, video, gui)
        bus.set_speed(speed=800, accel=50)
        q_cmd = q.copy(); dq_f = np.zeros(len(IDS)); target_f = None; period = 1 / 30
        load_idx = 0

        phase, prev_dir, settle, lost, offline, reversals = 'SEEK_START', None, 0, 0, 0, 0
        blue_missing = end_frames = 0
        red_near_t = blue_near_t = -1e9
        lt = LaserTrack(laser)
        t0, t_done, seq, n, last_snap, last_load = time.time(), None, None, 0, 0.0, time.time()
        while time.time() - t0 < args.timeout:
            seq, frame = cam.read(newer_than=seq)
            region = sheet_region(frame)                  # only look ON the paper
            if phase == 'ARRIVED':
                lt.bad = 0                     # after arrival the dot may sit on the red ink
            laser = lt.update(frame)           # observed NOW (or last observed, within COAST_PX)
            mask = bridge_laser_gap(vision.line_mask(frame), laser)
            if region is None:
                blue = red = None
            else:                                         # marks ON the line only, observed NOW
                blue, red = find_marks(frame, laser, exclude_r=14, region=region, exclude_blue=0, line_mask=mask)
            line = line_at_laser(mask, laser)
            on_line = line is not None and line['dist'] < 120 and line['dir'] is not None
            target, note, slow = None, '', 1.0

            if phase == 'SEEK_START':
                blue_missing = 0 if blue is not None else blue_missing + 1
                if blue is not None and np.linalg.norm(blue - laser) < 40:
                    blue_near_t = time.time()             # blue really seen next to the laser
                at_line_end = False
                if on_line:
                    path_ahead(mask, line['closest_h'], None, args.lookahead * S)
                    at_line_end = path_ahead.n_branches == 1
                if blue is None and at_line_end and time.time() - blue_near_t < 0.7:
                    # the small blue dot is hidden under the laser at the end of the line: start
                    target, note = laser, 'on blue (hidden by the laser)'
                    settle += 1
                    if settle >= 10:
                        phase, prev_dir, settle = 'TRAVEL', None, 0
                        print('at START (blue, under the laser) -> travelling along the line', flush=True)
                elif blue is not None:
                    target, note = blue, 'to blue'
                    # the laser glare hides part of the small dot and shifts its visible centre,
                    # so "at the start" = within the dot + glare size, counted with hysteresis
                    if np.linalg.norm(blue - laser) < 25:
                        settle += 1
                        if settle >= 10:
                            phase, prev_dir, settle = 'TRAVEL', None, 0
                            print('at START (blue) -> travelling along the line', flush=True)
                    else:
                        settle = max(settle - 1, 0)
                elif blue_missing <= 10:
                    note = 'blue just missed - waiting'      # hold still (target None), no blind move
                    lost = 0
                elif on_line:                             # blue not in view: search along the line
                    away = red * S if (prev_dir is None and red is not None) else None   # not toward the end
                    tgt, d, at_end = path_ahead(mask, line['closest_h'], prev_dir, args.lookahead * S,
                                                away_from=away)
                    if at_end:
                        reversals += 1
                        if reversals > 2:
                            raise RuntimeError('start mark (blue) not found along the line')
                        prev_dir = None if prev_dir is None else -prev_dir
                        tgt, d, _ = path_ahead(mask, line['closest_h'], prev_dir, args.lookahead * S)
                    if d is not None:
                        h = d if prev_dir is None else 0.75 * prev_dir + 0.25 * d
                        prev_dir = h / max(np.linalg.norm(h), 1e-6)
                    target, note = tgt, 'searching blue'
            elif phase == 'TRAVEL':
                # direction comes from continuity along the line (corners included), not from
                # where the red mark is: on a zig-zag the red mark is often "behind" the path
                if red is not None and np.linalg.norm(red - laser) < 60:
                    red_near_t = time.time()                 # red really seen close by, this frame
                if red is not None and np.linalg.norm(red - laser) < 20:
                    phase, t_done = 'ARRIVED', time.time()
                    print('reached END (red)', flush=True)
                    target = red
                elif on_line:
                    away = blue * S if (prev_dir is None and blue is not None) else None
                    tgt, d, at_end = path_ahead(mask, line['closest_h'], prev_dir, args.lookahead * S,
                                                away_from=away)
                    note = f'line dist {line["dist"]:.0f}px'
                    end_frames = end_frames + 1 if at_end else 0
                    if at_end and os.environ.get('FOLLOW_DUMP'):     # debug: raw frame at a line end
                        np.savez(f"{os.environ['FOLLOW_DUMP']}_{n}.npz", frame=frame, laser=laser,
                                 closest=line['closest_h'], prev_dir=np.array([np.nan, np.nan]) if prev_dir is None else prev_dir)
                    if not at_end:
                        target = tgt
                        if prev_dir is not None:
                            # look ~2x further along the line: slow down BEFORE a sharp turn
                            ch = line['closest_h']
                            far = _branches_at(mask, int(round(ch[0])), int(round(ch[1])), 24)[0]
                            turn = min([float(d @ prev_dir)] +
                                       [float(x[1] @ prev_dir) for x in far if float(x[1] @ prev_dir) > -0.6])
                            if turn < 0.7:
                                slow, note = 0.4, 'corner, line dist %.0fpx' % line['dist']
                        # heading = recent travel direction (~0.3 s), so "where we came from"
                        # does not flip at a corner
                        h = d if prev_dir is None else 0.75 * prev_dir + 0.25 * d
                        prev_dir = h / max(np.linalg.norm(h), 1e-6)
                    elif red is not None and np.linalg.norm(red - laser) < 80:
                        target, note = red, 'final approach to red'
                    elif time.time() - red_near_t < 0.7:
                        # line ends under the laser and the small red dot was seen right here
                        # a moment ago (now hidden by the laser glare): this is the end
                        phase, t_done = 'ARRIVED', time.time()
                        print('reached END (line end at the red mark)', flush=True)
                    elif end_frames >= 5:
                        raise RuntimeError('line ended (5 frames) but the red end mark is not here')
                    else:
                        note = 'line end? confirming'
                    offline = offline + 1 if line['dist'] > 30 else 0
                    if offline > 15:
                        raise RuntimeError(f'laser left the line ({line["dist"]:.0f}px) for 0.5 s')
                elif red is not None and np.linalg.norm(red - laser) < 80:
                    target, note = red, 'final approach to red'
            else:  # ARRIVED
                if red is not None:
                    target = red
                if time.time() - t_done > 2.0:
                    result = 'completed'
                    break

            # ---- act only on what was observed this frame ----
            if not lt.usable and phase != 'ARRIVED':
                target, note = None, 'laser not seen - holding'
            if target is None:
                lost += 1
                note = note or 'nothing usable in view - holding'
                if lost > 30 and phase != 'ARRIVED':     # after arrival the laser covers the red mark
                    raise RuntimeError('lost the line/marks for 1 s - stopped (no blind moves)')
                err = np.zeros(2)
                target_f = None
            else:
                lost = 0
                tgt = np.asarray(target, float)
                target_f = tgt if target_f is None else args.smooth * target_f + (1 - args.smooth) * tgt
                err = (target_f - laser) * slow
            now = time.time()
            period = 0.8 * period + 0.2 * min(max(now - getattr(main, '_t_last', now - period), 0.01), 0.2)
            main._t_last = now
            J = J_fn(q_cmd)
            dq = np.zeros(len(IDS))
            if target is not None and np.linalg.norm(err) > 1.5:
                dq = servo_step(J, err, args.gain, args.max_px, args.max_step)
            if target is None:
                target_f = None
                dq_f = np.zeros(len(IDS))                               # nothing seen: stop, no residual drift
            dq_f = args.smooth * dq_f + (1 - args.smooth) * dq          # low-pass the command
            q_new = np.clip(q_cmd + dq_f, qmin, qmax)
            speed = np.clip(np.abs(q_new - q_cmd) / period * 1.3, 20, 1500)   # arrive by next cycle
            lt.moved(J, q_new - q_cmd)
            q_cmd = q_new; q = q_cmd
            goal_with_speed(bus, q_cmd, speed)

            if now - last_load > 0.5:                                   # ~1 ms per read now
                sid = CAL_IDS[load_idx % len(CAL_IDS)]; load_idx += 1
                ld = T.loads(bus, [sid]); last_load = time.time()
                if ld[sid] > LOAD_LIMIT:
                    raise RuntimeError(f'load too high {ld}')
            n += 1
            if log:
                log.writerow([round(now - t0, 3), phase, note, *np.round(laser, 1),
                              *(np.round(blue, 1) if blue is not None else ['', '']),
                              *(np.round(red, 1) if red is not None else ['', '']),
                              '' if line is None else round(line['dist'], 1), *np.round(err, 1),
                              *q.astype(int).tolist()])
            if n % 15 == 0:
                print(f'{phase:10s} {note:28s} err=({err[0]:+4.0f},{err[1]:+4.0f}) '
                      f'blue={"Y" if blue is not None else "-"} red={"Y" if red is not None else "-"} '
                      f'q={q.astype(int).tolist()}', flush=True)

            vis = frame.copy()
            m = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            vis[m] = (0.5 * vis[m] + (0, 90, 0)).astype(np.uint8)
            if blue is not None:
                cv2.circle(vis, T.P(blue), 14, (255, 0, 0), 2)
            if red is not None:
                cv2.circle(vis, T.P(red), 14, (255, 0, 255), 2)
            if target is not None:
                cv2.circle(vis, T.P(target), 7, (0, 255, 0), 2)
            cv2.circle(vis, T.P(laser), 16, (0, 255, 255), 1)   # hollow: never hides the dot
            cv2.putText(vis, f'{phase}  {note}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if video is not None:
                video.put(vis)
            if args.snap and now - last_snap > 0.5:
                cv2.imwrite(args.snap + '.tmp.jpg', vis); os.replace(args.snap + '.tmp.jpg', args.snap)
                last_snap = now
            if gui:
                cv2.imshow('follow line: blue -> red', vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    result = 'stopped by user'
                    break
        else:
            result = 'timeout'
    except KeyboardInterrupt:
        result = 'interrupted'
    except RuntimeError as e:
        result = f'STOP: {e}'
    finally:
        print('result:', result, flush=True)
        try:
            T.goal(bus, read_q(bus))              # freeze where it is
            print('back to observe pose...', flush=True)
            move_slow(bus, observe)
            if rest is not None:
                print('down to rest pose...', flush=True)
                move_slow(bus, rest)
                bus.torque(False)
                print('at rest, torque OFF', flush=True)
            else:
                print('no rest pose saved - holding observe pose with torque ON', flush=True)
        except Exception as e:                    # noqa: BLE001
            print('return failed:', e, '- torque left as is', flush=True)
        finally:
            if 'video' in locals() and video is not None:
                video.close()
            bus.close()
            cam.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)          # kill -USR1 <pid>: print where it is stuck
    signal.signal(signal.SIGTERM, T._graceful)
    signal.signal(signal.SIGINT, T._graceful)
    main()
