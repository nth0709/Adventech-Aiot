"""Live camera: original | synthetic arc light (arc_filter) side by side. No arm motion.

The arc is placed at the laser dot, found in the ORIGINAL frame every frame.
Keys:  0 off  1 weak  2 medium  3 strong   l: line detection on the arc view   s: snapshot   q/ESC: quit
Sliders change each effect live.

    python3 arc_live.py [--preset medium] [--scale 0.65] [--observe]
--observe: move the arm rest -> observe pose first (camera sees the sheet); on quit it goes
           back observe -> rest and turns torque off (same moves as follow5.py).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import config as C
import vision
import line_detect
from laser_spot import find_laser
from arc_filter import ArcSim, PRESETS

SLIDERS = [('intensity', 100), ('radius', 80), ('glow', 300), ('flicker', 100),
           ('spatter', 100), ('smoke', 100), ('darken', 100), ('streaks', 100)]
UNIT = {'radius': 1, 'glow': 1}                       # px sliders; the rest are 0..100 -> 0..1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='medium', choices=list(PRESETS))
    ap.add_argument('--scale', type=float, default=0.65, help='display scale of each view')
    ap.add_argument('--snap', default=None, help='save the side-by-side view here every 0.5 s')
    ap.add_argument('--observe', action='store_true', help='move the arm to the observe pose while viewing')
    args = ap.parse_args()

    bus = None
    if args.observe:
        import follow5 as F
        observe, rest = F.load_pose(F.OBSERVE_FILE), F.load_pose(F.REST_FILE)
        bus = F.T.ServoBus(ids=F.IDS).connect()
        bus.torque(True)
        print('moving to observe pose', observe.astype(int).tolist(), flush=True)
        F.move_slow(bus, observe)
        print('at observe pose', flush=True)
    try:
        view_loop(args)
    finally:
        if bus is not None:
            try:
                print('back to observe pose...', flush=True)
                F.T.goal(bus, F.read_q(bus)); F.move_slow(bus, observe)
                if rest is not None:
                    print('down to rest pose...', flush=True)
                    F.move_slow(bus, rest); bus.torque(False)
                    print('at rest, torque OFF', flush=True)
            except Exception as e:  # noqa: BLE001
                print('return failed:', e, '- torque left as is', flush=True)
            finally:
                bus.close()


def view_loop(args):
    cam = vision.Camera()
    sim = ArcSim(C.FRAME_W, C.FRAME_H, **PRESETS[args.preset])
    win = 'Arc light simulation  (left: camera / right: arc filter)'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def to_slider(k, v):
        return int(v if k in UNIT else round(v * 100))

    def set_sliders(p):
        for k, _ in SLIDERS:
            cv2.setTrackbarPos(k, win, to_slider(k, p[k]))

    for k, mx in SLIDERS:
        cv2.createTrackbar(k, win, to_slider(k, sim.p[k]), mx, lambda _v: None)

    try:
        fallback = np.array(json.load(open('laser_pixel.json'))['q'], float)
    except Exception:
        fallback = np.array([C.FRAME_W / 2, C.FRAME_H * 0.85])
    last_laser, last_seen = fallback, 0.0
    show_line, seq, n, t0, fps, last_snap = False, None, 0, time.time(), 0.0, 0.0
    try:
        while True:
            seq, f = cam.read(newer_than=seq)
            for k, _ in SLIDERS:
                v = cv2.getTrackbarPos(k, win)
                sim.p[k] = float(v) if k in UNIT else v / 100.0

            lp, _ = find_laser(f)                                # observed in the clean frame
            if lp is not None:
                last_laser, last_seen = lp, time.time()
            laser_ok = time.time() - last_seen < 0.5
            arc = sim.apply(f, last_laser)

            left = f.copy()
            cv2.drawMarker(left, tuple(int(v) for v in last_laser),
                           (0, 255, 0) if laser_ok else (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            right = arc.copy()
            if show_line:
                m = cv2.resize(line_detect.line_mask(arc), (f.shape[1], f.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
                right[m] = (0.4 * right[m] + [0, 150, 0]).astype(np.uint8)
            view = np.hstack([cv2.resize(left, None, fx=args.scale, fy=args.scale),
                              cv2.resize(right, None, fx=args.scale, fy=args.scale)])
            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps, n, t0 = n / (now - t0), 0, now
            txt = f'{fps:.0f} fps   laser {"ok" if laser_ok else "not seen (last pos)"}' \
                  f'   line overlay {"ON" if show_line else "off"}   [0-3 preset, l, s, q]'
            cv2.putText(view, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(view, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(win, view)
            if args.snap and now - last_snap > 0.5:
                cv2.imwrite(args.snap + '.tmp.jpg', view); os.replace(args.snap + '.tmp.jpg', args.snap)
                last_snap = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('0'), ord('1'), ord('2'), ord('3')):
                name = ['off', 'weak', 'medium', 'strong'][key - ord('0')]
                sim.set(**PRESETS[name]); set_sliders(sim.p)
            elif key == ord('l'):
                show_line = not show_line
            elif key == ord('s'):
                fn = time.strftime('arc_snap_%H%M%S.jpg')
                cv2.imwrite(fn, view); print('saved', fn, flush=True)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
