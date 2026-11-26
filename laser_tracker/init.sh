#!/usr/bin/env bash
# Laser line tracker - single entry point
#
#   ./init.sh setup          # one-time (sudo): dialout group + stop ModemManager probing the Arduino
#   ./init.sh flash          # compile + upload Mega firmware
#   ./init.sh check          # camera / serial / servo / laser check
#   ./init.sh calib          # calibration -> calib.json
#   ./init.sh track [args]   # tracking (args passed to track.py: --hold --headless --record x.avi)
#   ./init.sh all            # flash -> check -> calib (if no calib.json) -> track
#   ./init.sh                # check -> calib (if no calib.json) -> track
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-/dev/ttyACM0}"
ARDUINO_DIR="${ARDUINO_DIR:-$HOME/Downloads/arduino-1.8.19}"
SKETCH="$DIR/firmware/laser_tracker_fw/laser_tracker_fw.ino"
BUILD="$DIR/firmware/build"
FQBN="arduino:avr:mega:cpu=atmega2560"

log() { printf '\033[1;36m[init]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[init] %s\033[0m\n' "$*" >&2; exit 1; }

# dialout added but this login session doesn't have it yet -> re-run under sg
need_dialout() {
  [[ "${1:-}" == "setup" ]] && return 0
  if ! id -nG | grep -qw dialout; then
    if getent group dialout | grep -qw "$USER"; then
      log "dialout not active in this session -> re-running via 'sg dialout'"
      exec sg dialout -c "$(printf '%q ' "$0" "$@")"
    else
      die "user not in dialout group. Run: ./init.sh setup"
    fi
  fi
}

wait_port() {
  for _ in $(seq 1 20); do [[ -e "$PORT" ]] && return 0; sleep 0.5; done
  die "$PORT not found. Is the Mega plugged in?"
}

# keep DTR high when a program closes the port (closing with DTR drop broke the next open on this Jetson)
prep_port() {
  wait_port
  stty -F "$PORT" 115200 raw -echo -hupcl || die "cannot configure $PORT"
}

do_setup() {
  log "adding $USER to dialout"
  sudo usermod -aG dialout "$USER"
  log "installing udev rule for Arduino Mega (ModemManager ignore, -hupcl at plug-in, USB reset access)"
  sudo rm -f /etc/udev/rules.d/99-arduino-no-modemmanager.rules
  sudo tee /etc/udev/rules.d/99-arduino-mega.rules >/dev/null <<'EOF'
# ModemManager must not probe the Mega
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ENV{ID_MM_DEVICE_IGNORE}="1"
# keep DTR high when a program closes the port, from the very first open after plug-in
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", RUN+="/bin/stty -F /dev/%k -hupcl"
# allow dialout users to USB-reset the board (software re-plug)
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="2341", MODE="0664", GROUP="dialout"
EOF
  sudo udevadm control --reload-rules && sudo udevadm trigger
  sudo systemctl stop ModemManager || true
  log "done. Re-plug the Mega USB. (log out/in to use the group without sg)"
}

do_flash() {
  [[ -x "$ARDUINO_DIR/arduino-builder" ]] || die "Arduino IDE not found at $ARDUINO_DIR (set ARDUINO_DIR)"
  mkdir -p "$BUILD"
  log "compiling firmware"
  ( cd "$ARDUINO_DIR" && ./arduino-builder -compile -hardware hardware -tools tools-builder \
      -tools hardware/tools/avr -built-in-libraries libraries -fqbn "$FQBN" \
      -build-path "$BUILD" "$SKETCH" ) | tail -2
  wait_port
  local avr="$ARDUINO_DIR/hardware/tools/avr"
  local out
  for try in 1 2 3; do
    prep_port
    log "uploading to $PORT (try $try)"
    if out=$("$avr/bin/avrdude" -C "$avr/etc/avrdude.conf" -p atmega2560 -c wiring -P "$PORT" \
               -b 115200 -D -U "flash:w:$BUILD/laser_tracker_fw.ino.hex:i" 2>&1); then
      grep -E "signature|verified" <<<"$out"
      log "upload OK"
      return 0
    fi
    grep -E "getsync|error" <<<"$out" | tail -1
    sleep 1
  done
  die "upload failed. Re-plug the Mega USB and run './init.sh flash' again."
}

py() { ( cd "$DIR" && python3 "$@" ); }

do_check() { prep_port; py check_devices.py; }
do_calib() { prep_port; py calibrate.py; }
do_track() { prep_port; py track.py "$@"; }
calib_if_missing() { [[ -f "$DIR/calib.json" ]] && log "using existing calib.json" || do_calib; }

need_dialout "$@"
cmd="${1:-run}"; shift || true
case "$cmd" in
  setup) do_setup ;;
  flash) do_flash ;;
  check) do_check ;;
  calib) do_calib ;;
  track) do_track "$@" ;;
  all)   do_flash; do_check; calib_if_missing; do_track "$@" ;;
  run)   do_check; calib_if_missing; do_track "$@" ;;
  *)     sed -n '2,10p' "$0"; exit 1 ;;
esac
