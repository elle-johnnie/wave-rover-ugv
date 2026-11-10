"""Drive the rover: manual with a game controller, or autonomous from the detector.

    uv run python drive.py <phone-ip>
    uv run python drive.py <phone-ip> --max-speed 0.2 --boost-speed 0.4 --max-age 0.5

Controller -- Nintendo Switch Pro Controller (paired over Bluetooth in macOS settings).
Buttons are mapped by their printed Nintendo labels:

                 X  (record)
          Y            A  (STOP)
                 B

    hold L       enable manual drive with the left stick at --max-speed (default 0.2)
    hold ZL      same, at --boost-speed (default 0.4) -- e.g. to get over a threshold
                 Without L or ZL held the stick does nothing; let go of them to stop.
    left stick   up/down throttle, left/right steer (only while L or ZL is held)
    hold R       AUTONOMOUS: follow the detector. Release -> stop (dead-man switch)
    A            stop now and cancel autonomous   (right face button)
    X            start/stop recording frames to data/raw/<session>/ (2 per second)  (top face button)
    Driving with L/ZL + stick overrides autonomous.

    Other controllers work too, by label: on Xbox, A is the bottom button and X the
    left one; on PlayStation, A = Cross and X = Square. Check before relying on stop.

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
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Map Nintendo controllers by printed label (A means the button marked "A"), not by
# Xbox position. Must be set before SDL starts, i.e. before pygame is imported.
os.environ["SDL_GAMECONTROLLER_USE_BUTTON_LABELS"] = "1"

import cv2
import numpy as np

from camera import PhoneCamera, stream_url
from detect import Decision, Detector, decide, draw
from rover import DEFAULT_HOST, Rover, Status

# Round deadzone, sized for this Switch Pro's left stick: after release it can rest anywhere
# up to about 0.41 from center, toward back-right (e.g. (+0.26, -0.31) for 2.5 s). Outside it, speed ramps
# up from zero, so there's no jump at the edge.
DEADZONE = 0.5
TURN_GAIN = 0.6  # how much steering adds/subtracts relative to max speed
@dataclass
class Inputs:
    throttle: float = 0.0  # -1..1, forward positive
    steer: float = 0.0  # -1..1, right positive
    auto_held: bool = False
    drive_held: bool = False  # L: stick drives at --max-speed
    boost_held: bool = False  # ZL: stick drives at --boost-speed
    stop_pressed: bool = False
    record_pressed: bool = False
    connected: bool = False
    name: str = "no controller"
    raw_x: float = 0.0  # stick before deadzone, for the --log diagnostics
    raw_y: float = 0.0


class Gamepad:
    """Controller input via SDL's GameController API, set up for the Switch Pro Controller:
    hold L (or ZL for boost) + left stick drives, hold R for autonomous, A stops, X records."""

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
                print(f"[pad] using {self.name}: hold L+stick=drive, ZL+stick=boost, hold R=autonomous, A=STOP, X=record")
                return

    def _stick(self) -> tuple[float, float]:
        """(throttle, steer) after the round deadzone; forward and right are positive."""
        pg = self.pg
        return apply_deadzone(self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTX) / 32767.0,
                              -self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTY) / 32767.0)

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
        throttle, steer = self._stick()
        return Inputs(
            throttle=throttle,
            steer=steer,
            auto_held=bool(b(pg.CONTROLLER_BUTTON_RIGHTSHOULDER)),  # Switch: R
            drive_held=bool(b(pg.CONTROLLER_BUTTON_LEFTSHOULDER)),  # Switch: L
            boost_held=self.pad.get_axis(pg.CONTROLLER_AXIS_TRIGGERLEFT) > 16000,  # Switch: ZL (digital)
            stop_pressed=self._edge("stop", bool(b(pg.CONTROLLER_BUTTON_A))),  # Switch: A (right)
            record_pressed=self._edge("record", bool(b(pg.CONTROLLER_BUTTON_X))),  # Switch: X (top)
            connected=True,
            name=self.name,
            raw_x=self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTX) / 32767.0,
            raw_y=-self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTY) / 32767.0,
        )


def apply_deadzone(x: float, y: float, dz: float = DEADZONE) -> tuple[float, float]:
    """Round deadzone on stick (x right, y forward) -> (throttle, steer), rescaled so the
    output starts at 0 at the deadzone edge and reaches 1 at full deflection."""
    m = math.hypot(x, y)
    if m < dz:
        return 0.0, 0.0
    k = min(1.0, (m - dz) / (1.0 - dz)) / m
    return y * k, x * k


def mix(throttle: float, steer: float, max_speed: float) -> tuple[float, float]:
    """Arcade drive: stick position -> left/right wheel speeds."""
    left = throttle + TURN_GAIN * steer
    right = throttle - TURN_GAIN * steer
    m = max(1.0, abs(left), abs(right))
    return left / m * max_speed, right / m * max_speed


def choose(inp: Inputs, auto_latched: bool, decision: Decision | None, frame_age: float,
           max_age: float, max_speed: float, boost_speed: float | None = None) -> tuple[str, tuple[float, float]]:
    """Return (mode label, wheel speeds). Pure, so the safety rules are testable.

    The stick only drives while L (normal) or ZL (boost) is held, so a stick that rests
    off-center can't move the rover, and letting go of L/ZL always stops manual driving."""
    if (inp.drive_held or inp.boost_held) and (inp.throttle or inp.steer):
        if inp.boost_held:
            return "MANUAL BOOST", mix(inp.throttle, inp.steer, boost_speed or max_speed)
        return "MANUAL", mix(inp.throttle, inp.steer, max_speed)
    if inp.auto_held or auto_latched:
        if decision is None or frame_age > max_age:
            return "AUTO: camera stale -> STOP", (0.0, 0.0)
        return f"AUTO: {decision.action}", decision.speeds
    return "IDLE", (0.0, 0.0)


BATT_LOW_V, BATT_CRIT_V = 10.8, 10.2  # 3S pack: ~12.6 V full
MAIN_POWER_OFF_V = 5.0  # below this the board is on USB power only; motors are dead
TILT_WARN_DEG = 25.0  # firmware cuts the motors at ~34 deg pitch
STALE_S = 1.5
WHITE, AMBER, RED, GREY = (255, 255, 255), (0, 200, 255), (60, 60, 255), (170, 170, 170)


def rover_lines(st: Status | None, age: float, latency_s: float | None) -> list[tuple[str, tuple]]:
    """Telemetry line plus any warnings, as (text, BGR color). Pure, so it's testable."""
    if st is None or age > STALE_S:
        return [("NO ROVER STATUS - off, or Wi-Fi link lost" if st is None
                 else f"NO ROVER STATUS for {age:.1f}s - Wi-Fi link lost?", RED)]
    pitch, roll = (round(a) + 0 for a in st.tilt_deg)  # +0 turns -0 into 0
    link = f"{latency_s * 1000:.0f}ms" if latency_s else "?"
    lines = [(f"battery {st.volts:.1f}V   pitch {pitch:+d}  roll {roll:+d} deg   link {link}", WHITE)]
    if st.volts < MAIN_POWER_OFF_V:
        lines.append(("MAIN POWER OFF - USB only, motors won't run", RED))
    elif st.volts < BATT_CRIT_V:
        lines.append((f"BATTERY CRITICAL {st.volts:.1f} V - stop and charge", RED))
    elif st.volts < BATT_LOW_V:
        lines.append((f"battery low {st.volts:.1f} V", AMBER))
    if st.upside_down:
        lines.append(("UPSIDE DOWN", RED))
    elif abs(pitch) >= TILT_WARN_DEG or abs(roll) >= TILT_WARN_DEG:
        lines.append((f"TILTING pitch {pitch:+.0f} roll {roll:+.0f} deg (motor cutoff ~34)", RED))
    return lines


def status_panel(width: int, mode: str, left: float, right: float, recording: bool,
                 rover: list[tuple[str, tuple]]) -> np.ndarray:
    """Dark panel shown under the video: big mode line, then rover status and warnings."""
    k = max(0.8, width / 640)  # scale text with the video width
    big, small = 1.1 * k, 0.7 * k
    row_big, row_small = int(44 * k), int(30 * k)
    panel = np.full((int(14 * k) + row_big + row_small * len(rover), width, 3), 30, np.uint8)
    color = RED if mode.startswith(("STOP", "AUTO: camera")) else AMBER if mode.startswith("AUTO") else WHITE
    margin = int(12 * k)

    def put(text: str, y: int, scale: float, c: tuple, thick: int) -> None:
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        scale *= min(1.0, (width - 2 * margin) / w)  # shrink long lines to fit
        cv2.putText(panel, text, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, thick, cv2.LINE_AA)

    y = row_big
    put(f"{mode}   L={left:+.2f} R={right:+.2f}" + ("   [REC]" if recording else ""),
        y, big, color, max(2, int(2.5 * k)))
    for text, c in rover:
        y += row_small
        put(text, y, small, c, max(1, int(1.6 * k)))
    return panel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="phone IP (IP Webcam) or stream URL")
    p.add_argument("--host", default=DEFAULT_HOST, help="rover address")
    p.add_argument("--model", default="yolov8n-640")
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--near", type=float, default=0.55)
    p.add_argument("--max-speed", type=float, default=0.2, help="manual driving speed while holding L")
    p.add_argument("--boost-speed", type=float, default=0.4,
                   help="manual driving speed while holding ZL (rover range is about 0.5)")
    p.add_argument("--max-age", type=float, default=0.5, help="autonomous stops if frames are older than this (s)")
    p.add_argument("--no-pad", action="store_true", help="don't look for a game controller")
    p.add_argument("--log", action="store_true",
                   help="write every loop and every command sent to data/logs/<time>.log (diagnostics)")
    args = p.parse_args()

    lanes = (0.3, 0.7)
    detector = Detector(args.model, args.confidence)
    pad = None if args.no_pad else Gamepad()
    rec_dir = Path("data/raw") / datetime.now().strftime("%Y%m%d-%H%M%S")
    recording, last_saved = False, 0.0
    auto_latched = False
    last_count, decision, overlay_src = -1, None, None
    last_mode = ""
    last_warnings: list[str] = []
    log = None
    if args.log:
        log_path = Path("data/logs") / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w", buffering=1)
        print(f"logging to {log_path}")
    t_start = time.monotonic()

    def log_send(left: float, right: float, ok: bool, thread: str) -> None:
        if log:
            log.write(f"{time.monotonic() - t_start:9.3f} SENT  L={left:+.3f} R={right:+.3f} "
                      f"{'ok' if ok else 'FAILED'} ({thread})\n")

    print(f"opening {stream_url(args.source)}; rover at {args.host}. Rover starts STOPPED.")
    with Rover(args.host, max_speed=max(args.max_speed, args.boost_speed), on_send=log_send) as rover, PhoneCamera(stream_url(args.source)) as cam:
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

                mode, (left, right) = choose(inp, auto_latched, decision, age, args.max_age,
                                       args.max_speed, args.boost_speed)
                if inp.stop_pressed or key == ord("x"):
                    mode, left, right = "STOP", 0.0, 0.0
                rover.drive(left, right)
                if log and (mode != last_mode or left or right or inp.raw_x or inp.raw_y):
                    log.write(f"{time.monotonic() - t_start:9.3f} LOOP  {mode:28s} stick x={inp.raw_x:+.3f} "
                              f"y={inp.raw_y:+.3f} {'L' if inp.drive_held else '-'}{'ZL' if inp.boost_held else '--'} "
                              f"-> L={left:+.3f} R={right:+.3f}\n")
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
                st, st_age = rover.telemetry()
                rlines = rover_lines(st, st_age, rover.last_latency_s)
                warnings = [t for t, c in rlines if c != WHITE]
                for w in warnings:
                    if w not in last_warnings and not w.startswith("NO ROVER STATUS for"):
                        print(f"{datetime.now():%H:%M:%S}  WARNING: {w}")
                last_warnings = warnings
                panel = status_panel(out.shape[1], mode, left, right, recording, rlines)
                cv2.imshow("rover drive", np.vstack([out, panel]))
        except KeyboardInterrupt:
            pass
        finally:
            rover.stop()
    if log:
        log.close()
    cv2.destroyAllWindows()
    print("rover stopped")


if __name__ == "__main__":
    main()
