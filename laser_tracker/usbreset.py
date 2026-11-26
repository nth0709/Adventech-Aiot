"""Software re-plug of the Arduino Mega (USBDEVFS_RESET) and wait for its tty to come back.

    python3 usbreset.py [/dev/ttyACM0]
Needs write access to the USB device node (udev rule from `./init.sh setup`).
"""
import fcntl
import os
import sys
import time

USBDEVFS_RESET = 21780          # _IO('U', 20)


def usb_node(tty):
    dev = os.path.realpath("/sys/class/tty/%s/device/.." % os.path.basename(tty))
    bus = int(open(os.path.join(dev, "busnum")).read())
    num = int(open(os.path.join(dev, "devnum")).read())
    return "/dev/bus/usb/%03d/%03d" % (bus, num)


def reset(tty="/dev/ttyACM0", wait_s=10.0):
    node = usb_node(tty)
    fd = os.open(node, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
        os.close(fd)
    time.sleep(1.0)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if os.path.exists(tty):
            time.sleep(0.5)
            return node
        time.sleep(0.2)
    raise TimeoutError("%s did not come back after USB reset" % tty)


if __name__ == "__main__":
    tty = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    try:
        print("USB reset OK:", reset(tty))
    except PermissionError:
        sys.exit("USB reset: no permission on USB node -> run ./init.sh setup and re-plug once")
