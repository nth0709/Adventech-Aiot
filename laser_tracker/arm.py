"""Serial link to the Mega firmware (laser_tracker_fw.ino)."""
import time

import serial

import config as C


class Arm:
    def __init__(self, port=C.SERIAL_PORT, baud=C.SERIAL_BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.5, write_timeout=1.0)
        time.sleep(2.0)                 # Mega resets when the port opens
        self.ser.reset_input_buffer()

    def _cmd(self, s, expect=None):
        self.ser.write((s + "\n").encode())
        if expect is None:
            return None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ln = self.ser.readline().decode(errors="ignore").strip()
            if ln.startswith(expect):
                return ln
            if ln.startswith("ERR"):
                raise RuntimeError("firmware: %s (cmd=%r)" % (ln, s))
        raise TimeoutError("no %r reply to %r" % (expect, s))

    def ping(self):
        ln = self._cmd("PING", "PONG")
        return {int(k): v == "1" for k, v in (t.split(":") for t in ln.split()[1:])}

    def read_positions(self):
        return [int(v) for v in self._cmd("R", "POS").split()[1:]]

    def set_positions(self, q):
        self._cmd("P " + " ".join(str(int(round(v))) for v in q))

    def heartbeat(self):
        self._cmd("H")

    def torque(self, on):
        self._cmd("T %d" % (1 if on else 0), "OK")

    def laser(self, on):
        self._cmd("L %d" % (1 if on else 0), "OK")

    def set_speed(self, speed=C.SERVO_SPEED, acc=C.SERVO_ACC):
        self._cmd("V %d %d" % (speed, acc), "OK")

    def close(self):
        try:
            self.laser(False)
        except Exception:
            pass
        self.ser.close()
