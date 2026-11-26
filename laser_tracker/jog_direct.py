"""STS3215 USB jog: ID 1, bounded relative movement, then return."""
import argparse
import time

from direct_port import open_port


def exchange(s, sid, instruction, params):
    body = bytes([sid, len(params) + 2, instruction, *params])
    packet = b"\xff\xff" + body + bytes([(~sum(body)) & 255])
    s.reset_input_buffer()
    s.write(packet)
    s.flush()
    buf = bytearray()
    received = bytearray()
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        chunk = s.read(s.in_waiting or 1)
        received.extend(chunk)
        buf.extend(chunk)
        while len(buf) >= 4:
            start = buf.find(b"\xff\xff")
            if start < 0:
                buf[:] = buf[-1:]
                break
            if start:
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
            if reply[4]:
                raise RuntimeError(f"ID {sid}: servo error 0x{reply[4]:02x}")
            return reply[5:-1]
    raise TimeoutError(f"ID {sid}: no valid response (instruction={instruction}, "
                       f"params={list(params)}, RX={received.hex() or 'EMPTY'})")


def read(s, address, length):
    for attempt in range(3):
        try:
            data = exchange(s, 1, 2, (address, length))
            break
        except TimeoutError:
            if attempt == 2:
                raise
            time.sleep(0.1)
    if len(data) != length:
        raise RuntimeError("Unexpected response length")
    return data


def position(s):
    value = int.from_bytes(read(s, 56, 2), "little")
    if not 0 <= value <= 4095:
        raise RuntimeError("Position outside single-turn range; stopped")
    return value


def write_registers(s, address, values):
    # Sync-write targets only ID 1 and does not request a write status packet.
    params = bytes([address, len(values), 1, *values])
    body = bytes([254, len(params) + 2, 0x83]) + params
    s.write(b'\xff\xff' + body + bytes([(~sum(body)) & 255]))
    s.flush()
    time.sleep(0.03)


def goal(s, target):
    # ACC=20, goal position, goal time=0, speed=100 ticks/s.
    write_registers(s, 41, (20, target & 255, target >> 8, 0, 0, 100, 0))


def wait_at(s, target):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        actual = position(s)
        if abs(actual - target) <= 5:
            return actual
        time.sleep(0.1)
    raise RuntimeError(f"Target not reached: target={target}, actual={actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta', type=int, default=20,
                        help='Relative movement in ticks (-100 to 100; default 20)')
    args = parser.parse_args()
    if not 1 <= abs(args.delta) <= 100:
        parser.error('--delta must be nonzero and between -100 and 100')
    with open_port() as s:
        print('Port open; waiting 2 seconds for adapter startup...', flush=True)
        time.sleep(2)
        print('Reading ID 1 position (register 56)...', flush=True)
        original = position(s)
        print(f'Position={original}; reading torque (register 40)...', flush=True)
        if read(s, 40, 1) != b'\x01':
            raise RuntimeError("ID 1 torque is OFF; no movement commanded")
        target = original + args.delta
        if not (20 <= original <= 4075 and 20 <= target <= 4075):
            raise RuntimeError("Too close to position boundary; stopped")
        # Preserve the acceleration/speed/time settings changed by this jog.
        print('Reading speed/acceleration settings...', flush=True)
        old_acc = read(s, 41, 1)
        old_time_speed = read(s, 44, 4)
        print(f'ID 1: {original} -> {target} -> {original}', flush=True)
        try:
            goal(s, target)
            print('Moved:', wait_at(s, target), flush=True)
            time.sleep(1.5)
        finally:
            # Return may fail if power or serial communication is lost.
            goal(s, original)
            print('Returned:', wait_at(s, original), flush=True)
            write_registers(s, 41, old_acc)
            write_registers(s, 44, old_time_speed)
        print('Small movement and return confirmed')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'STOP: {exc}')
