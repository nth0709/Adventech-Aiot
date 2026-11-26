"""Keep the direct USB port open for repeated reads and bounded ID 1 jogs."""
import time

from direct_port import open_port
from jog_direct import exchange, goal, position, read, wait_at, write_registers


def show_positions(s):
    for sid in (1, 2, 3):
        data = exchange(s, sid, 2, (56, 2))
        if len(data) != 2:
            raise RuntimeError(f'ID {sid}: unexpected position response')
        value = int.from_bytes(data, 'little')
        if value & 0x8000:
            value = -(value & 0x7fff)
        print(f'ID {sid}: {value}', flush=True)


def jog(s, delta):
    original = position(s)
    target = original + delta
    if not (20 <= original <= 4075 and 20 <= target <= 4075):
        raise RuntimeError('Position boundary reached')
    if read(s, 40, 1) != b'\x01':
        raise RuntimeError('ID 1 torque is OFF; no movement commanded')
    old_acc = read(s, 41, 1)
    old_time_speed = read(s, 44, 4)
    print(f'ID 1: {original} -> {target} -> {original}', flush=True)
    try:
        goal(s, target)
        print('Moved:', wait_at(s, target), flush=True)
        time.sleep(1.5)
    finally:
        goal(s, original)
        print('Returned:', wait_at(s, original), flush=True)
        write_registers(s, 41, old_acc)
        write_registers(s, 44, old_time_speed)


def main():
    with open_port() as s:
        time.sleep(2)
        show_positions(s)
        print('Port stays open. r=read, + or -=ID 1 jog/return (100 ticks), q=quit.', flush=True)
        while True:
            command = input('servo> ').strip().lower()
            if command == 'q':
                break
            if command == 'r':
                show_positions(s)
            elif command in ('+', '-'):
                jog(s, 100 if command == '+' else -100)
            else:
                print('Use r, +, -, or q.')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'STOP: {exc}')
