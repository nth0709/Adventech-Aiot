"""Read-only repeated position checks for the direct STS USB connection."""
import time

from direct_port import open_port

from jog_direct import exchange


def main():
    with open_port() as s:
        print('Port open. Read-only check; no motion commands.', flush=True)
        time.sleep(2)
        counts = {sid: 0 for sid in (1, 2, 3)}
        for cycle in range(1, 11):
            for sid in counts:
                try:
                    data = exchange(s, sid, 2, (56, 2))
                    if len(data) != 2:
                        raise RuntimeError(f'Unexpected data: {data.hex()}')
                    raw = int.from_bytes(data, 'little')
                    pos = -(raw & 0x7fff) if raw & 0x8000 else raw
                    counts[sid] += 1
                    print(f'Round {cycle:02d} ID={sid}: OK position={pos}', flush=True)
                except (RuntimeError, TimeoutError) as exc:
                    print(f'Round {cycle:02d} ID={sid}: FAIL {exc}', flush=True)
                time.sleep(0.05)
            time.sleep(0.5)
        for sid, count in counts.items():
            print(f'ID={sid}: valid position replies {count}/10')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'STOP: {exc}')
