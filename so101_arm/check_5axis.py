"""Read-only diagnosis for the existing CH343 / Feetech STS bus.

Run on MIC: python3 ~/so101_arm/check_5axis.py
Scans IDs 0..253, then reads responding servos five times on ONE port open.
Only READ instructions are sent; no torque, position, or EEPROM writes.
Register interpretation assumes STS-compatible servos, as in servo_bus.py.
"""
import argparse
from pathlib import Path
import time

import serial


def find_port():
    for tty in sorted(Path('/sys/class/tty').glob('tty*')):
        device = (tty / 'device').resolve()
        for parent in (device, *device.parents):
            try:
                vid = (parent / 'idVendor').read_text().strip()
                pid = (parent / 'idProduct').read_text().strip()
            except OSError:
                continue
            if (vid, pid) == ('1a86', '55d3'):
                node = Path('/dev') / tty.name
                if not node.exists():
                    raise RuntimeError(
                        f'CH343 is present in sysfs, but {node} is unavailable. '
                        'Run this script in the MIC desktop terminal with device access.')
                return str(node)
            break
    raise RuntimeError('No CH343 serial interface (USB 1a86:55d3) found.')


def read_registers(port, sid, address, count, timeout=0.12):
    body = bytes((sid, 4, 2, address, count))
    packet = b'\xff\xff' + body + bytes((~sum(body) & 255,))
    port.reset_input_buffer()
    port.write(packet)
    port.flush()
    buf = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buf.extend(port.read(port.in_waiting or 1))
        while len(buf) >= 4:
            start = buf.find(b'\xff\xff')
            if start < 0:
                buf[:] = buf[-1:]
                break
            del buf[:start]
            length = buf[3]
            if not 2 <= length <= 64:
                del buf[0]
                continue
            size = length + 4
            if len(buf) < size:
                break
            reply = bytes(buf[:size])
            del buf[:size]
            if reply == packet or reply[2] != sid or sum(reply[2:]) & 255 != 255:
                continue
            if len(reply[5:-1]) != count:
                raise RuntimeError(f'ID {sid}: unexpected response length; error={reply[4]:02x}')
            return reply[4], reply[5:-1]
    raise TimeoutError(f'ID {sid}: no valid reply')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='Explicit CH343 device path')
    parser.add_argument('--ids', nargs='+', type=int, help='Restrict scan to these IDs')
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    ids = sorted(set(args.ids)) if args.ids else list(range(254))
    if any(i < 0 or i > 253 for i in ids) or args.rounds < 1:
        parser.error('IDs must be 0..253; rounds must be positive')
    node = args.port or find_port()
    print(f'Port: {node}; READ ONLY; STS-compatible register map assumed', flush=True)
    print('One port open; scanning IDs (full scan takes about 35 seconds).', flush=True)
    found = []
    with serial.Serial(node, 115200, timeout=0.01, write_timeout=1, exclusive=True) as port:
        # Refresh line coding using the setup already used by direct_port.py.
        import termios
        attrs = termios.tcgetattr(port.fileno())
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(port.fileno(), termios.TCSANOW, attrs)
        time.sleep(0.1)
        port.baudrate = 1_000_000
        time.sleep(0.2)
        for sid in ids:
            try:
                error, data = read_registers(port, sid, 56, 8)
                found.append(sid)
                print(f'Found ID {sid}: raw={data.hex()} error=0x{error:02x}', flush=True)
            except TimeoutError:
                pass
            except RuntimeError as exc:
                print(str(exc), flush=True)
        if not found:
            raise RuntimeError('No valid status replies. Check servo power/model/baud/IDs. '
                               'The existing driver notes also describe USB reconnect recovery.')
        counts = {sid: 0 for sid in found}
        for cycle in range(1, args.rounds + 1):
            for sid in found:
                try:
                    error, data = read_registers(port, sid, 56, 8)
                    torque_error, torque = read_registers(port, sid, 40, 1)
                    position = int.from_bytes(data[:2], 'little')
                    if position & 0x8000:
                        position = -(position & 0x7fff)
                    print(f'Round {cycle} ID {sid}: position={position} '
                          f'voltage={data[6]/10:.1f}V temp={data[7]}C '
                          f'torque_raw={torque[0]} error=0x{error | torque_error:02x}', flush=True)
                    if not (error | torque_error):
                        counts[sid] += 1
                except (TimeoutError, RuntimeError) as exc:
                    print(f'Round {cycle}: {exc}', flush=True)
            time.sleep(0.3)
        print(f'Responding IDs: {found}; count={len(found)}')
        for sid, count in counts.items():
            print(f'ID {sid}: clean status+torque replies {count}/{args.rounds}')
        print('Servo IDs do not identify physical joint order without mapping.')
        return 0 if all(n == args.rounds for n in counts.values()) else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f'STOP: {exc}')
