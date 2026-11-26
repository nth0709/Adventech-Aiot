"""Read-only comparison of USB serial DTR/RTS states; never writes servo registers."""
import termios
import time

import serial

from direct_port import find_port
from jog_direct import exchange


def check(s):
    good = 0
    for sid in (1, 2, 3):
        try:
            data = exchange(s, sid, 2, (56, 2))
            if len(data) != 2:
                raise RuntimeError(f'Unexpected payload: {data.hex()}')
            print(f'  ID={sid}: OK position={int.from_bytes(data, "little")}', flush=True)
            good += 1
        except (TimeoutError, RuntimeError) as exc:
            print(f'  ID={sid}: {exc}', flush=True)
    return good


def main():
    port = find_port()
    # Delayed open allows explicit initial control-line values.
    s = serial.Serial(port=None, baudrate=1000000, timeout=0.01,
                      write_timeout=1, exclusive=True)
    s.dtr = True
    s.rts = True
    s.port = port
    s.open()
    with s:
        attrs = termios.tcgetattr(s.fileno())
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(s.fileno(), termios.TCSANOW, attrs)
        time.sleep(2)
        results = []
        try:
            for dtr, rts in ((1, 1), (0, 1), (0, 0), (1, 0), (1, 1)):
                print(f'\nDTR={dtr} RTS={rts}', flush=True)
                s.dtr = bool(dtr)
                s.rts = bool(rts)
                time.sleep(0.5)
                first = check(s)
                time.sleep(0.2)
                second = check(s)
                results.append((dtr, rts, first + second))
        finally:
            s.dtr = True
            s.rts = True
        print('\nSUMMARY (valid replies / 6):', flush=True)
        for dtr, rts, count in results:
            print(f'DTR={dtr} RTS={rts}: {count}/6', flush=True)
        print('No servo motion, torque, ID, baud-rate or EEPROM commands were sent.')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'STOP: {exc}')
