"""Calibration: home pose (teach), laser pixel, and image Jacobian (px per servo tick).

Usage:
    python3 calibrate.py            # teach home by hand, then auto-calibrate
    python3 calibrate.py --no-teach # use current pose as home

Point the arm at a TEXTURED, flat surface (e.g. a printed page or patterned paper) at the
working distance. Plain white paper with a single line is not enough for the Jacobian step.
"""
import argparse
import json
import time

import numpy as np

import config as C
from arm import Arm
from vision import Camera, detect_laser_spot, image_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-teach", action="store_true")
    ap.add_argument("--delta", type=int, default=C.CALIB_DELTA)
    args = ap.parse_args()

    arm, cam = Arm(), Camera()
    try:
        print("servos:", arm.ping())
        arm.set_speed()
        if not args.no_teach:
            arm.torque(False)
            input("Torque OFF. Move the arm by hand so the camera looks at the work surface, then press Enter...")
        home = arm.read_positions()
        if any(p < 0 for p in home):
            raise RuntimeError("position read failed: %s" % home)
        arm.set_positions(home)
        arm.torque(True)
        print("home:", home)

        # 1) laser pixel (ON/OFF difference)
        arm.laser(False); time.sleep(0.3); off = cam.fresh()
        arm.laser(True);  time.sleep(0.3); on = cam.fresh()
        arm.laser(False)
        laser_px, strength = detect_laser_spot(on, off)
        print("laser pixel: (%.1f, %.1f)  strength=%.1f" % (laser_px[0], laser_px[1], strength))
        if strength < 20:
            print("  WARNING: weak laser response - check laser wiring/pin, or set laser_px manually")

        # 2) image Jacobian, central differences per joint
        J = np.zeros((2, 3))
        for j in range(3):
            shifts = []
            for sign in (-1, 1):
                q = list(home); q[j] += sign * args.delta
                arm.set_positions(q); time.sleep(C.CALIB_SETTLE_S)
                shifts.append(cam.fresh())
                arm.heartbeat()
            arm.set_positions(home); time.sleep(C.CALIB_SETTLE_S)
            d, resp = image_shift(shifts[0], shifts[1])
            J[:, j] = d / (2.0 * args.delta)
            print("joint %d: scene shift per +tick = (%+.3f, %+.3f) px  [confidence %.2f]%s"
                  % (j, J[0, j], J[1, j], resp, "  <-- LOW, add texture" if resp < 0.1 else ""))

        sv = np.linalg.svd(J, compute_uv=False)
        print("Jacobian singular values:", np.round(sv, 4))
        if sv[-1] < 0.05 * sv[0]:
            print("  WARNING: Jacobian nearly singular - joints barely move the view in one direction")

        with open(C.CALIB_FILE, "w") as f:
            json.dump({"home": home, "laser_px": laser_px, "J": J.tolist(),
                       "delta": args.delta, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        print("saved", C.CALIB_FILE)
    finally:
        cam.close()
        arm.close()


if __name__ == "__main__":
    main()
