"""Shared port setup for investigating USB close/reopen failures."""
import termios
import time
from pathlib import Path
from types import SimpleNamespace

import serial
from serial.tools import list_ports


ADAPTER_SERIAL = '5B90149782'


def available_ports(sys_tty=Path('/sys/class/tty'), dev_root=Path('/dev')):
    """Also enumerate vendor ttyCH343USB nodes omitted by pySerial on Linux."""
    ports = {p.device: p for p in list_ports.comports()}
    for tty in sys_tty.glob('ttyCH*'):
        node = dev_root / tty.name
        if not node.exists():
            continue
        device = (tty / 'device').resolve()
        for parent in (device, *device.parents):
            try:
                vid = int((parent / 'idVendor').read_text().strip(), 16)
                pid = int((parent / 'idProduct').read_text().strip(), 16)
                serial_number = (parent / 'serial').read_text().strip()
            except (OSError, ValueError):
                continue
            ports[str(node)] = SimpleNamespace(
                device=str(node), vid=vid, pid=pid, serial_number=serial_number,
                hwid=f'USB VID:PID={vid:04x}:{pid:04x} SER={serial_number}')
            break
    return list(ports.values())


def find_port(wait_s=10):
    """Find the previously identified adapter, never an arbitrary ACM port."""
    deadline = time.monotonic() + wait_s
    while True:
        ports = available_ports()
        matches = [p for p in ports if p.vid == 0x1a86 and p.pid == 0x55d3
                   and p.serial_number == ADAPTER_SERIAL]
        if len(matches) == 1:
            print(f'Adapter found: {matches[0].device}, serial={ADAPTER_SERIAL}', flush=True)
            return matches[0].device
        if len(matches) > 1:
            raise RuntimeError('Multiple matching ports; refusing to choose automatically')
        if time.monotonic() >= deadline:
            visible = '\n'.join(f'  {p.device}: {p.hwid}' for p in ports) or '  (none)'
            raise RuntimeError(
                f'Adapter {ADAPTER_SERIAL} has no visible serial port after {wait_s}s.\n'
                f'Visible ports:\n{visible}\n'
                'Run: ls -l /dev/ttyCH* /dev/ttyACM*; lsusb; '
                'sudo journalctl -k -n 60 --no-pager')
        time.sleep(0.25)


def open_port():
    port = find_port()
    # Open at a different nonzero rate so reopening must send a fresh line
    # coding configuration. No bus packets are sent at this temporary rate.
    s = serial.Serial(port, 115200, timeout=0.01,
                      write_timeout=1, exclusive=True)
    try:
        attrs = termios.tcgetattr(s.fileno())
        was_enabled = bool(attrs[2] & termios.HUPCL)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(s.fileno(), termios.TCSANOW, attrs)
        if termios.tcgetattr(s.fileno())[2] & termios.HUPCL:
            raise RuntimeError('Failed to clear HUPCL')
        time.sleep(0.1)
        s.baudrate = 1000000
        time.sleep(0.1)
        s.reset_input_buffer()
        print(f'Port setup: HUPCL {int(was_enabled)} -> 0, '
              f'DTR={int(s.dtr)}, RTS={int(s.rts)}, '
              'baud refreshed 115200 -> 1000000', flush=True)
        return s
    except BaseException:
        s.close()
        raise
