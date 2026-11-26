"""Laser black-line tracking (eye-in-hand image-based visual servoing).

    python3 track.py                 # follow the line (LOOKAHEAD from config)
    python3 track.py --hold          # just keep the laser on the nearest line point
    python3 track.py --headless      # no GUI window (SSH)
    python3 track.py --record out.avi

Keys (GUI): q/ESC quit, space pause, f toggle follow/hold, l toggle laser, r reverse direction
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import config as C
from arm import Arm
from vision import Camera, LineTracker


class Controller:
    """dq = gain * W J^T (J W J^T + lambda^2 I)^-1 (-e)  +  null-space pull toward home."""

    def __init__(self, J, home):
        self.J = np.asarray(J, float)
        self.home = np.asarray(home, float)
        W = np.diag(C.JOINT_WEIGHTS)
        A = self.J @ W @ self.J.T + (C.DAMPING ** 2) * np.eye(2)
        self.Jinv = W @ self.J.T @ np.linalg.inv(A)          # 3x2
        self.N = np.eye(3) - np.linalg.pinv(self.J) @ self.J  # exact null-space projector
        self.qmin, self.qmax = np.array(C.JOINT_MIN, float), np.array(C.JOINT_MAX, float)

    def step(self, q, err):
        dq = np.zeros(3)
        if np.linalg.norm(err) > C.DEADBAND_PX:
            dq = C.GAIN * (self.Jinv @ (-err))   # target must move by -err in the image
        dq += C.NULLSPACE_GAIN * (self.N @ (self.home - q))
        dq = np.clip(dq, -C.MAX_STEP, C.MAX_STEP)
        return np.clip(q + dq, self.qmin, self.qmax)


def P(v):
    return tuple(int(round(float(x))) for x in v)


def draw(frame, laser_px, res, state):
    vis = frame.copy()
    if res is not None:
        m = cv2.resize(res["mask"], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis[m > 0] = (0.5 * vis[m > 0] + (0, 90, 0)).astype(np.uint8)
        c, t = res["closest"], res["target"]
        cv2.circle(vis, P(c), 5, (255, 255, 0), -1)
        cv2.circle(vis, P(t), 7, (0, 255, 0), 2)
        cv2.arrowedLine(vis, P(c), P(c + 40 * res["dir"]), (0, 255, 255), 2)
        cv2.line(vis, P(laser_px), P(t), (0, 0, 255), 1)
    cv2.drawMarker(vis, P(laser_px), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(vis, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--no-laser", action="store_true", help="track without switching the laser on")
    args = ap.parse_args()
    gui = not args.headless and bool(os.environ.get("DISPLAY"))

    with open(C.CALIB_FILE) as f:
        cal = json.load(f)
    laser_px = np.array(cal["laser_px"], float)
    ctrl = Controller(cal["J"], cal["home"])
    tracker = LineTracker(lookahead=0 if args.hold else C.LOOKAHEAD)

    arm, cam = Arm(), Camera()
    writer = None
    try:
        print("servos:", arm.ping())
        arm.set_speed()
        q = np.array(arm.read_positions(), float)
        if np.any(q < 0):
            raise RuntimeError("position read failed: %s" % q)
        arm.set_positions(q)                  # hold current pose before enabling torque
        arm.torque(True)
        laser = not args.no_laser
        arm.laser(laser)

        paused, lost, seq = False, 0, None
        t_prev, fps = time.time(), 0.0
        while True:
            seq, frame = cam.read(newer_than=seq)
            res = tracker.update(frame, laser_px)

            if res is None:
                lost += 1
                state = "LOST" if lost > C.LOST_FRAMES_LIMIT else "searching"
                arm.heartbeat()
            else:
                lost = 0
                err = res["target"] - laser_px
                if not paused:
                    q = ctrl.step(q, err)
                    arm.set_positions(q)
                else:
                    arm.heartbeat()
                state = "%s err=(%+.0f,%+.0f)" % ("PAUSED" if paused else
                                                  ("HOLD" if tracker.lookahead == 0 else "FOLLOW"),
                                                  err[0], err[1])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            state += "  q=%s  %.0ffps" % (q.astype(int).tolist(), fps)

            if gui or args.record:
                vis = draw(frame, laser_px, res, state)
                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"MJPG"), C.FPS,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if gui:
                    cv2.imshow("laser tracker", vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord(" "):
                        paused = not paused
                    if k == ord("f"):
                        tracker.lookahead = C.LOOKAHEAD if tracker.lookahead == 0 else 0
                    if k == ord("l"):
                        laser = not laser
                        arm.laser(laser)
                    if k == ord("r") and tracker.prev_dir is not None:
                        tracker.prev_dir = -tracker.prev_dir
            elif seq % 30 == 0:
                print(state)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cam.close()
        arm.close()                           # switches laser off; servos hold position
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
