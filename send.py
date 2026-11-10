"""Send one JSON command to the Wave Rover over USB serial and print its replies.

Usage:
    uv run python send.py '{"T":405}'
    uv run python send.py '{"T":1,"L":0,"R":0}' --port /dev/cu.usbmodemXXXX --wait 2
    uv run python send.py '{"T":405}' --all      # also show raw debug output
    uv run python send.py --reset --all --wait 15  # reboot the board and watch it boot

The firmware interleaves IMU debug text with its JSON replies, sometimes on the
same line, so replies are extracted as JSON objects rather than filtered by line.
"""

import argparse
import glob
import json
import time

import serial

STATUS_TYPE = 1001  # periodic status/IMU feedback, hidden unless --all


def find_port() -> str:
    ports = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.wchusbserial*")
    if not ports:
        raise SystemExit("No rover serial port found. Is the USB cable plugged in (and a data cable)?")
    return ports[0]


def json_objects(text: str):
    """Yield every JSON object embedded anywhere in text."""
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                yield obj
            i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", help='JSON command, e.g. \'{"T":405}\'')
    p.add_argument("--reset", action="store_true", help="reboot the board over USB (before sending any command)")
    p.add_argument("--port", default=None, help="serial port (auto-detected if omitted)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--wait", type=float, default=5.0, help="seconds to listen for replies")
    p.add_argument("--all", action="store_true", help="print raw output, including debug text")
    args = p.parse_args()

    if not args.command and not args.reset:
        p.error("give a JSON command, --reset, or both")

    port = args.port or find_port()
    with serial.Serial(port, args.baud, timeout=0.2) as s:
        if args.reset:
            # Standard ESP32 auto-reset wiring: RTS drives EN, DTR drives GPIO0 (BOOT).
            s.dtr = False      # keep GPIO0 high -> normal boot, not flashing mode
            s.rts = True       # pull EN low -> hold in reset
            time.sleep(0.2)
            s.rts = False      # release -> reboot
            if args.command:
                time.sleep(3)  # let the firmware start before talking to it
        else:
            s.reset_input_buffer()  # drop stale output; after a reset we want the boot log
        if args.command:
            s.write((args.command.strip() + "\n").encode())
        end = time.time() + args.wait
        while time.time() < end:
            line = s.readline().decode(errors="replace").strip()
            if not line:
                continue
            if args.all:
                print(line)
                continue
            for obj in json_objects(line):
                if obj.get("T") != STATUS_TYPE:
                    print(json.dumps(obj))


if __name__ == "__main__":
    main()
