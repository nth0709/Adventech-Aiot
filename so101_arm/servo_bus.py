"""Feetech STS3215 bus control for the 3-servo arm (pyserial only).

Usage:
    from servo_bus import ServoBus
    with ServoBus(ids=(1, 2, 3)) as bus:
        print(bus.status())
        bus.move({1: 2100, 2: 2000, 3: 2050})

CLI:
    python3 servo_bus.py            # status of all servos
    python3 servo_bus.py torque off # torque off on all servos
"""
import glob
import os
import sys
import time

import serial

# USB-serial chip on the controller board (WCH CH343)
USB_VID, USB_PID = "1a86", "55d3"
BAUD = 1_000_000

# STS3215 registers
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
GOAL_SPEED = 46
PRESENT_POSITION = 56
PRESENT_LOAD = 60
PRESENT_VOLTAGE = 62
PRESENT_TEMPERATURE = 63
MOVING = 66

INST_PING, INST_READ, INST_WRITE, INST_SYNC_WRITE = 0x01, 0x02, 0x03, 0x83
BROADCAST = 0xFE


def find_port():
    """Find the controller board's serial port by USB VID:PID (name can change)."""
    for dev in sorted(glob.glob("/dev/ttyCH343USB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        path = os.path.realpath(f"/sys/class/tty/{os.path.basename(dev)}/device")
        for _ in range(4):  # walk up from the interface to the USB device
            try:
                vid = open(os.path.join(path, "idVendor")).read().strip()
                pid = open(os.path.join(path, "idProduct")).read().strip()
                if (vid, pid) == (USB_VID, USB_PID):
                    return dev
                break
            except OSError:
                path = os.path.dirname(path)
    raise RuntimeError("controller board not found (USB 1a86:55d3). Is the USB cable connected?")


class ServoBus:
    def __init__(self, ids=(1, 2, 3), port=None, limits=None):
        self.ids = tuple(ids)
        self.port = port
        # optional per-servo (min, max) position limits; default: full range
        self.limits = limits or {}
        self.s = None

    # ---------- connection ----------
    def connect(self, attempts=5):
        """Open the port and check every servo answers.

        With the current CH343 driver, about 1 in 3 port opens gives a dead
        session; closing and reopening fixes it, so retry the open.
        """
        port = self.port or find_port()
        for n in range(1, attempts + 1):
            self.s = serial.Serial(port, BAUD, timeout=0.03, write_timeout=0.5)
            time.sleep(0.02)
            try:
                missing = [i for i in self.ids if not self.ping(i)]
            except serial.SerialTimeoutException:
                missing = list(self.ids)   # CH343 wedged on write: close and reopen
            if not missing:
                self.port = port
                self.open_attempts = n
                return self
            self.s.close()
            self.s = None
            time.sleep(0.1)
        raise RuntimeError(f"servos {missing} did not respond on {port} after {attempts} opens "
                           "(check servo power and cable)")

    def close(self):
        if self.s:
            self.s.close()
            self.s = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---------- protocol ----------
    @staticmethod
    def _packet(sid, inst, params=b""):
        body = bytes([sid, len(params) + 2, inst]) + bytes(params)
        return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    def _read_reply(self, timeout=0.03):
        """Read exactly one status packet (header, then its declared length).

        Returns as soon as the packet is complete (~1 ms) instead of always waiting
        for the serial timeout as read(64) did (30 ms per call).
        """
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            k = buf.find(b"\xff\xff")
            if k > 0:
                del buf[:k]
            if len(buf) >= 4 and buf[:2] == b"\xff\xff":
                need = buf[3] + 4
                if len(buf) >= need:
                    return bytes(buf[:need])
                buf += self.s.read(need - len(buf))
            else:
                buf += self.s.read(max(1, 4 - len(buf)))
        return bytes(buf)

    def _txrx(self, sid, inst, params=b"", expect_reply=True):
        self.s.reset_input_buffer()
        # no flush(): pyserial's flush() is tcdrain() with NO timeout and hangs forever when the
        # CH343 session wedges; write() itself is bounded by write_timeout
        self.s.write(self._packet(sid, inst, params))
        if not expect_reply:
            return None
        r = self._read_reply()
        k = r.find(b"\xff\xff")
        if k < 0 or len(r) < k + 6:
            return None
        ln = r[k + 3]
        d = r[k + 2:k + 4 + ln]
        if len(d) < ln + 2 or d[0] != sid:
            return None
        if (~sum(d[:-1])) & 0xFF != d[-1]:
            return None  # checksum mismatch
        return d[2], d[3:-1]  # (error byte, params)

    def _retry(self, fn, what, tries=5):
        for _ in range(tries):
            r = fn()
            if r is not None:
                return r
            time.sleep(0.01)
        raise IOError(f"no reply: {what}")

    def ping(self, sid):
        return self._txrx(sid, INST_PING) is not None

    def read(self, sid, addr, n):
        def f():
            r = self._txrx(sid, INST_READ, bytes([addr, n]))
            return r[1] if r and len(r[1]) == n else None
        return self._retry(f, f"read id{sid} addr{addr}")

    def write(self, sid, addr, data):
        self._retry(lambda: self._txrx(sid, INST_WRITE, bytes([addr]) + bytes(data)),
                    f"write id{sid} addr{addr}")

    def sync_write(self, addr, values):
        """Write the same register on several servos in one packet (they start together)."""
        values = dict(values)
        n = len(next(iter(values.values())))
        params = bytes([addr, n])
        for sid, data in values.items():
            params += bytes([sid]) + bytes(data)
        self._txrx(BROADCAST, INST_SYNC_WRITE, params, expect_reply=False)

    # ---------- high level ----------
    @staticmethod
    def _u16(b):
        return b[0] | (b[1] << 8)

    def position(self, sid):
        return self._u16(self.read(sid, PRESENT_POSITION, 2))

    def positions(self):
        return {i: self.position(i) for i in self.ids}

    def status(self):
        out = {}
        for i in self.ids:
            b = self.read(i, PRESENT_POSITION, 8)  # pos, speed, load, volt, temp
            load = self._u16(b[4:6])
            out[i] = {
                "position": self._u16(b[0:2]),
                "load_pct": (load & 0x3FF) / 10 * (-1 if load & 0x400 else 1),
                "voltage": b[6] / 10,
                "temp_c": b[7],
                "torque": bool(self.read(i, TORQUE_ENABLE, 1)[0]),
            }
        return out

    def torque(self, on, ids=None):
        for i in ids or self.ids:
            if on:
                # set goal = current first so enabling torque never jumps
                p = self.position(i)
                self.write(i, GOAL_POSITION, [p & 0xFF, p >> 8])
            self.write(i, TORQUE_ENABLE, [1 if on else 0])

    def set_speed(self, speed=200, accel=20, ids=None):
        """speed in steps/s (4096 steps = 1 turn), accel 0-254 (0 = max)."""
        for i in ids or self.ids:
            self.write(i, ACCELERATION, [accel])
            self.write(i, GOAL_SPEED, [speed & 0xFF, speed >> 8])

    def _clamp(self, sid, pos):
        lo, hi = self.limits.get(sid, (0, 4095))
        return max(lo, min(hi, int(pos)))

    def move(self, targets, wait=True, timeout=5.0):
        """targets: {id: position}. Moves all servos together via sync write."""
        targets = {i: self._clamp(i, p) for i, p in targets.items()}
        self.sync_write(GOAL_POSITION, {i: [p & 0xFF, p >> 8] for i, p in targets.items()})
        if wait:
            t0 = time.time()
            time.sleep(0.05)
            while time.time() - t0 < timeout:
                if not any(self.read(i, MOVING, 1)[0] for i in targets):
                    break
                time.sleep(0.02)
        return self.positions()


if __name__ == "__main__":
    with ServoBus() as bus:
        if sys.argv[1:] == ["torque", "off"]:
            bus.torque(False)
        elif sys.argv[1:] == ["torque", "on"]:
            bus.torque(True)
        print(f"port: {bus.port}")
        for sid, st in bus.status().items():
            print(f"ID {sid}: {st}")
