"""Wi-Fi control for the Wave Rover (Waveshare JSON protocol, as served by Pathfinder firmware).

The rover accepts commands as  GET http://<host>/js?json={...}
    {"T":1,"L":l,"R":r}  wheel speeds, roughly -0.5..0.5
    {"T":0}              emergency stop      {"T":999}  clear emergency stop
    {"T":130}            status (replies with a {"T":1001,...} feedback object)

Safety model:
  * Speeds are clamped to `max_speed` (default 0.25, gentle).
  * A background thread re-sends the current command several times a second.
    That keeps the ESP32's Wi-Fi awake (cold requests can take ~1 s) and lets
    us enforce a dead-man timer: if drive() isn't called again within
    `deadman_s`, the rover is told to stop. A crashed or stalled control loop
    therefore stops the rover instead of leaving it driving.
  * Leaving a `with Rover(...)` block (including via an exception or Ctrl-C)
    always sends stop.

CLI (wheels off the ground for anything that moves!):
    uv run python rover.py status
    uv run python rover.py stop
    uv run python rover.py wiggle             # each wheel briefly forward/back
    uv run python rover.py drive 0.15 0.15 1  # left, right, seconds
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass

import requests

DEFAULT_HOST = os.environ.get("ROVER_HOST", "192.168.68.88")


@dataclass
class Status:
    left: float
    right: float
    roll: float
    pitch: float
    accel_z: float
    voltage_raw: float
    raw: dict

    @property
    def upside_down(self) -> bool:
        # Accelerometer Z reads about +17000 sitting normally, about -16000 flipped.
        return self.accel_z < -8000


class Rover:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        max_speed: float = 0.25,
        rate_hz: float = 5.0,
        deadman_s: float = 0.6,
        timeout_s: float = 0.8,
        verbose: bool = False,
    ):
        self.base = f"http://{host}/js"
        self.max_speed = max_speed
        self.period = 1.0 / rate_hz
        self.deadman_s = deadman_s
        self.timeout_s = timeout_s
        self.verbose = verbose

        self._lock = threading.Lock()
        self._target = (0.0, 0.0)
        self._last_drive = 0.0
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self.consecutive_failures = 0
        self.last_latency_s: float | None = None

    # ---- low level -------------------------------------------------------

    def send(self, cmd: dict) -> str:
        """Send one JSON command and return the raw reply text."""
        t0 = time.monotonic()
        r = requests.get(self.base, params={"json": json.dumps(cmd, separators=(",", ":"))}, timeout=self.timeout_s)
        r.raise_for_status()
        self.last_latency_s = time.monotonic() - t0
        return r.text

    def status(self) -> Status:
        text = self.send({"T": 130})
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return Status(
            left=d.get("L", 0.0), right=d.get("R", 0.0),
            roll=d.get("r", 0.0), pitch=d.get("p", 0.0),
            accel_z=d.get("az", 0.0), voltage_raw=d.get("v", 0.0), raw=d,
        )

    # ---- driving ---------------------------------------------------------

    def _clamp(self, v: float) -> float:
        return max(-self.max_speed, min(self.max_speed, float(v)))

    def drive(self, left: float, right: float) -> None:
        """Set wheel speeds. Must be called again within deadman_s or the rover stops."""
        with self._lock:
            self._target = (self._clamp(left), self._clamp(right))
            self._last_drive = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._target = (0.0, 0.0)
        self._send_speeds(0.0, 0.0)

    def emergency_stop(self) -> None:
        self.stop()
        self.send({"T": 0})

    def clear_emergency_stop(self) -> None:
        self.send({"T": 999})

    def _send_speeds(self, left: float, right: float) -> None:
        try:
            self.send({"T": 1, "L": round(left, 3), "R": round(right, 3)})
            self.consecutive_failures = 0
        except requests.RequestException as e:
            self.consecutive_failures += 1
            if self.verbose or self.consecutive_failures in (1, 5, 20):
                print(f"[rover] send failed ({self.consecutive_failures}x): {e.__class__.__name__}")

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            with self._lock:
                if t0 - self._last_drive > self.deadman_s:
                    self._target = (0.0, 0.0)
                left, right = self._target
            self._send_speeds(left, right)
            self._stop_evt.wait(max(0.0, self.period - (time.monotonic() - t0)))

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> "Rover":
        if self._thread is None:
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, name="rover-keepalive", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        for _ in range(3):  # make sure the stop lands even on a flaky link
            self._send_speeds(0.0, 0.0)

    def __enter__(self) -> "Rover":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


# ---- CLI -----------------------------------------------------------------

def _hold(rover: Rover, left: float, right: float, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rover.drive(left, right)
        time.sleep(0.1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("stop")
    sub.add_parser("wiggle")
    d = sub.add_parser("drive")
    d.add_argument("left", type=float)
    d.add_argument("right", type=float)
    d.add_argument("seconds", type=float)
    args = p.parse_args()

    rover = Rover(args.host)

    if args.cmd == "status":
        lat = []
        for _ in range(3):
            s = rover.status()
            lat.append(rover.last_latency_s * 1000)
        print(f"wheels L={s.left} R={s.right}  roll={s.roll:.2f} pitch={s.pitch:.2f}  v={s.voltage_raw}"
              f"  upside_down={s.upside_down}")
        print("latency ms: " + ", ".join(f"{x:.0f}" for x in lat))
        return

    if args.cmd == "stop":
        rover.stop()
        print("stop sent")
        return

    with rover:
        try:
            if args.cmd == "wiggle":
                for label, l, r in [("left fwd", .15, 0), ("left back", -.15, 0),
                                    ("right fwd", 0, .15), ("right back", 0, -.15)]:
                    print(label)
                    _hold(rover, l, r, 0.8)
                    _hold(rover, 0, 0, 0.4)
            elif args.cmd == "drive":
                _hold(rover, args.left, args.right, args.seconds)
        except KeyboardInterrupt:
            pass
    print("stopped")


if __name__ == "__main__":
    main()
