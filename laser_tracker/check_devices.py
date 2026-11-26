"""Hardware check: camera, serial port, STS3215 servos, laser."""
import glob
import os
import time

import cv2

import config as C


def check_camera():
    print("[camera] devices:", sorted(glob.glob("/dev/video*")))
    cap = cv2.VideoCapture(C.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[camera] FAIL: cannot open index", C.CAMERA_INDEX)
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    t0, n, f = time.time(), 0, None
    while time.time() - t0 < 2.0:
        ok, fr = cap.read()
        if ok:
            n, f = n + 1, fr
    cap.release()
    if f is None:
        print("[camera] FAIL: no frames")
        return False
    cv2.imwrite("camera_check.jpg", f)
    print("[camera] OK %dx%d  %.1f fps  (saved camera_check.jpg)" % (f.shape[1], f.shape[0], n / 2.0))
    return True


def check_arm():
    port = C.SERIAL_PORT
    if not os.path.exists(port):
        print("[serial] FAIL: %s not found (ports: %s)" % (port, glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")))
        return False
    if not os.access(port, os.R_OK | os.W_OK):
        print("[serial] FAIL: no permission on %s -> run: sudo usermod -aG dialout $USER  (then log out/in)" % port)
        return False
    from arm import Arm
    try:
        arm = Arm()
    except Exception as e:
        print("[serial] FAIL:", e)
        return False
    try:
        print("[serial] OK", port)
        pong = arm.ping()
        print("[servo] ping:", pong, "OK" if all(pong.values()) else "<-- some servos not responding")
        print("[servo] positions:", arm.read_positions())
        print("[laser] blinking 3x ...")
        for _ in range(3):
            arm.laser(True); time.sleep(0.3); arm.laser(False); time.sleep(0.3)
        return all(pong.values())
    except Exception as e:
        print("[serial] FAIL: firmware not responding (%s). Upload firmware/laser_tracker_fw first." % e)
        return False
    finally:
        arm.close()


if __name__ == "__main__":
    ok = check_camera() & check_arm()
    print("\nALL OK" if ok else "\nSome checks failed.")
    raise SystemExit(0 if ok else 1)
