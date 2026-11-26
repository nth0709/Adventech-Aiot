"""Live view of the arm-mounted Arducam (q or ESC to quit).

Also saves the latest frame every 0.5 s to /tmp/arm_cam_latest.jpg so other
tools can look at the camera while this window holds the device.
"""
import os, time
import cv2

SNAP = os.environ.get('ARM_CAM_SNAP', '/tmp/arm_cam_latest.jpg')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit('camera /dev/video0 could not be opened')
cv2.namedWindow('Arm camera', cv2.WINDOW_NORMAL)
last_snap, frames, t0, fps = 0.0, 0, time.time(), 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        print('frame read failed'); time.sleep(0.1); continue
    frames += 1
    now = time.time()
    if now - t0 >= 1.0:
        fps, frames, t0 = frames / (now - t0), 0, now
    if now - last_snap >= 0.5:
        cv2.imwrite(SNAP + '.tmp.jpg', frame); os.replace(SNAP + '.tmp.jpg', SNAP)
        last_snap = now
    view = frame.copy()
    cv2.putText(view, f'{fps:.1f} fps  q: quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Arm camera', view)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break
cap.release()
cv2.destroyAllWindows()
