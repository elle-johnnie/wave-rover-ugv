"""Drive the rover: manual with a game controller, or autonomous from the detector.

    uv run python drive.py <phone-ip>
    uv run python drive.py <phone-ip> --max-speed 0.2 --max-age 0.5

Controller (Xbox / PlayStation / Switch Pro, paired over Bluetooth in macOS settings):
    left stick        manual drive (up/down throttle, left/right steer)
    hold RB / R1      AUTONOMOUS: follow the detector. Release -> stop (dead-man switch)
    B / Circle        stop now and cancel autonomous
    Y / Triangle      start/stop recording frames to data/raw/<session>/ (2 per second)
    Touching the stick always overrides autonomous.

Keyboard (preview window focused; works with or without a controller):
    space  arm/disarm autonomous (latched -- no dead-man, so stay near x!)
    x      stop now and disarm      r  record on/off      q  quit

Safety, on top of rover.py's speed cap and 0.6 s dead-man stop:
    * starts stopped; autonomous is never on until you ask for it
    * autonomous stops if the newest camera frame is older than --max-age seconds
    * quitting, Ctrl-C, or a crash always sends stop
    * the firmware itself cuts the motors if the rover tips past ~34 degrees
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np

from camera import PhoneCamera, stream_url
from detect import Decision, Detector, decide, draw
from rover import DEFAULT_HOST, Rover

DEADZONE = 0.15
TURN_GAIN = 0.6  # how much steering adds/subtracts relative to max speed


@dataclass
class Inputs:
    throttle: float = 0.0  # -1..1, forward positive
    steer: float = 0.0  # -1..1, right positive
    auto_held: bool = False
    stop_pressed: bool = False
    record_pressed: bool = False
    connected: bool = False
    name: str = "no controller"


class Gamepad:
    """Standardized controller input via SDL's GameController mapping (same layout for Xbox/PS/Switch)."""

    def __init__(self) -> None:
        import pygame
        from pygame._sdl2 import controller

        self.pg, self.ctl = pygame, controller
        pygame.init()
        controller.init()
        self.pad = None
        self.name = "no controller"
        self._prev = {"stop": False, "record": False}
        self._last_scan = 0.0
        self._scan()

    def _scan(self) -> None:
        self._last_scan = time.monotonic()
        for i in range(self.pg.joystick.get_count()):
            if self.ctl.is_controller(i):
                self.pad = self.ctl.Controller(i)
                self.name = self.pad.name
                print(f"[pad] using {self.name}")
                return

    def _axis(self, a: int) -> float:
        v = self.pad.get_axis(a) / 32767.0
        return 0.0 if abs(v) < DEADZONE else max(-1.0, min(1.0, v))

    def _edge(self, key: str, now_down: bool) -> bool:
        pressed = now_down and not self._prev[key]
        self._prev[key] = now_down
        return pressed

    def poll(self) -> Inputs:
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.CONTROLLERDEVICEREMOVED and self.pad is not None:
                print(f"[pad] {self.name} disconnected")
                self.pad, self.name = None, "no controller"
        if self.pad is None or not self.pad.attached():
            self.pad, self.name = None, "no controller"
            if time.monotonic() - self._last_scan > 2.0:
                self._scan()
            return Inputs()
        b = self.pad.get_button
        return Inputs(
            throttle=-self._axis(pg.CONTROLLER_AXIS_LEFTY),
            steer=self._axis(pg.CONTROLLER_AXIS_LEFTX),
            auto_held=bool(b(pg.CONTROLLER_BUTTON_RIGHTSHOULDER)),
            stop_pressed=self._edge("stop", bool(b(pg.CONTROLLER_BUTTON_B))),
            record_pressed=self._edge("record", bool(b(pg.CONTROLLER_BUTTON_Y))),
            connected=True,
            name=self.name,
        )


def mix(throttle: float, steer: float, max_speed: float) -> tuple[float, float]:
    """Arcade drive: stick position -> left/right wheel speeds."""
    left = throttle + TURN_GAIN * steer
    right = throttle - TURN_GAIN * steer
    m = max(1.0, abs(left), abs(right))
    return left / m * max_speed, right / m * max_speed


def choose(inp: Inputs, auto_latched: bool, decision: Decision | None, frame_age: float,
           max_age: float, max_speed: float) -> tuple[str, tuple[float, float]]:
    """Return (mode label, wheel speeds). Pure, so the safety rules are testable."""
    if inp.throttle or inp.steer:
        return "MANUAL", mix(inp.throttle, inp.steer, max_speed)
    if inp.auto_held or auto_latched:
        if decision is None or frame_age > max_age:
            return "AUTO: camera stale -> STOP", (0.0, 0.0)
        return f"AUTO: {decision.action}", decision.speeds
    return "IDLE", (0.0, 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="phone IP (IP Webcam) or stream URL")
    p.add_argument("--host", default=DEFAULT_HOST, help="rover address")
    p.add_argument("--model", default="yolov8n-640")
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--near", type=float, default=0.55)
    p.add_argument("--max-speed", type=float, default=0.2, help="cap for manual driving (and rover clamp)")
    p.add_argument("--max-age", type=float, default=0.5, help="autonomous stops if frames are older than this (s)")
    p.add_argument("--no-pad", action="store_true", help="don't look for a game controller")
    args = p.parse_args()

    lanes = (0.3, 0.7)
    detector = Detector(args.model, args.confidence)
    pad = None if args.no_pad else Gamepad()
    rec_dir = Path("data/raw") / datetime.now().strftime("%Y%m%d-%H%M%S")
    recording, last_saved = False, 0.0
    auto_latched = False
    last_count, decision, overlay_src = -1, None, None
    last_mode = ""

    print(f"opening {stream_url(args.source)}; rover at {args.host}. Rover starts STOPPED.")
    with Rover(args.host, max_speed=args.max_speed) as rover, PhoneCamera(stream_url(args.source)) as cam:
        try:
            while True:
                inp = pad.poll() if pad else Inputs()

                frame, age = cam.read()
                if frame is not None and cam.frame_count != last_count:  # only run the model on new frames
                    last_count = cam.frame_count
                    det, labels, ms = detector(frame)
                    h, w = frame.shape[:2]
                    decision = decide(det.xyxy, w, h, near_frac=args.near, lane_split=lanes)
                    overlay_src = (frame, det, labels, ms)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    auto_latched = not auto_latched
                    print("autonomous ARMED (keyboard)" if auto_latched else "autonomous disarmed")
                if key == ord("x") or inp.stop_pressed:
                    auto_latched = False
                    rover.stop()
                    print("STOP")
                if key == ord("r") or inp.record_pressed:
                    recording = not recording
                    print(f"recording -> {rec_dir}" if recording else "recording stopped")

                mode, (left, right) = choose(inp, auto_latched, decision, age, args.max_age, args.max_speed)
                if inp.stop_pressed or key == ord("x"):
                    mode, left, right = "STOP", 0.0, 0.0
                rover.drive(left, right)
                if mode != last_mode:
                    print(f"{datetime.now():%H:%M:%S}  {mode}")
                    last_mode = mode

                if recording and frame is not None and time.monotonic() - last_saved >= 0.5:
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(rec_dir / f"{datetime.now():%H%M%S-%f}.jpg"), frame)
                    last_saved = time.monotonic()

                if overlay_src is not None:
                    f, det, labels, ms = overlay_src
                    info = f"model {ms:.0f} ms | frame age {age*1000:.0f} ms | {inp.name}"
                    out = draw(f, det, labels, decision, args.near, lanes, info)
                else:
                    out = np.zeros((480, 640, 3), np.uint8)
                hud = f"{mode}  L={left:+.2f} R={right:+.2f}" + ("  [REC]" if recording else "")
                color = (60, 60, 255) if mode.startswith(("STOP", "AUTO: camera")) else \
                        (0, 200, 255) if mode.startswith("AUTO") else (255, 255, 255)
                cv2.putText(out, hud, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, hud, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                cv2.imshow("rover drive", out)
        except KeyboardInterrupt:
            pass
        finally:
            rover.stop()
    cv2.destroyAllWindows()
    print("rover stopped")


if __name__ == "__main__":
    main()
