"""Vision-only obstacle check: run a Roboflow model on the phone stream and show
the steering decision it *would* make. Does NOT move the rover.

    # put ROBOFLOW_API_KEY=... in .env (free key: app.roboflow.com -> Settings -> API Keys)
    uv run python detect.py <phone-ip>
    uv run python detect.py --image some.jpg   # one still image, writes some.annotated.jpg

How the decision works (camera mounted low, tilted slightly down):
  * Objects higher in the frame are farther away; an object whose bottom edge
    is below the NEAR line (default 55% down the frame) is close to the rover.
  * The lower part of the frame is split into LEFT / CENTER / RIGHT lanes.
    Each lane's "blocked" score is how far down its closest obstacle reaches.
  * CENTER clear -> FORWARD; otherwise turn toward the clearer side; if both
    sides are blocked too -> STOP.

Limitation: a pretrained COCO model only sees its 80 classes (people, chairs,
bottles, ...). Walls, table legs, boxes etc. are invisible to it. That's what
a custom floor-segmentation model trained on our own frames will fix.

Keys: q quit   s save annotated frame to data/annotated/
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv

from camera import PhoneCamera, stream_url

# Pick up ROBOFLOW_API_KEY from .env next to this file (a real env var still wins).
load_dotenv(Path(__file__).with_name(".env"))

# Wheel speeds each decision would map to once we connect the motors.
ACTIONS = {
    "FORWARD": (0.15, 0.15),
    "LEFT": (-0.12, 0.12),
    "RIGHT": (0.12, -0.12),
    "STOP": (0.0, 0.0),
}
COLORS = {"FORWARD": (80, 200, 80), "LEFT": (0, 200, 255), "RIGHT": (0, 200, 255), "STOP": (60, 60, 255)}


@dataclass
class Decision:
    action: str
    lanes: dict[str, float]  # 0 = clear ... 1 = obstacle reaches bottom of frame

    @property
    def speeds(self) -> tuple[float, float]:
        return ACTIONS[self.action]


def decide(
    boxes: np.ndarray, width: int, height: int,
    near_frac: float = 0.55, lane_split: tuple[float, float] = (0.3, 0.7), block_at: float = 0.25,
) -> Decision:
    """boxes: Nx4 array of x1,y1,x2,y2 pixels. Pure function so it's easy to test and reuse."""
    edges = {"LEFT": (0.0, lane_split[0]), "CENTER": lane_split, "RIGHT": (lane_split[1], 1.0)}
    lanes = {k: 0.0 for k in edges}
    for x1, _y1, x2, y2 in boxes:
        reach = (y2 / height - near_frac) / (1.0 - near_frac)  # 0 at NEAR line, 1 at bottom
        if reach <= 0:
            continue
        for lane, (a, b) in edges.items():
            if x2 > a * width and x1 < b * width:  # box overlaps this lane
                lanes[lane] = max(lanes[lane], min(1.0, reach))
    if lanes["CENTER"] < block_at:
        action = "FORWARD"
    elif min(lanes["LEFT"], lanes["RIGHT"]) >= block_at:
        action = "STOP"
    else:
        action = "LEFT" if lanes["LEFT"] <= lanes["RIGHT"] else "RIGHT"
    return Decision(action, lanes)


def draw(frame: np.ndarray, det: sv.Detections, labels: list[str], d: Decision,
         near_frac: float, lane_split: tuple[float, float], info: str) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    near_y = int(h * near_frac)
    overlay = out.copy()
    for lane, (a, b) in {"LEFT": (0, lane_split[0]), "CENTER": lane_split, "RIGHT": (lane_split[1], 1)}.items():
        x1, x2 = int(a * w), int(b * w)
        shade = int(255 * d.lanes[lane])
        cv2.rectangle(overlay, (x1, near_y), (x2, h), (0, 255 - shade, shade), -1)
        cv2.putText(out, f"{lane} {d.lanes[lane]:.2f}", (x1 + 6, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    cv2.line(out, (0, near_y), (w, near_y), (255, 255, 255), 1)
    out = sv.BoxAnnotator(thickness=2).annotate(out, det)
    out = sv.LabelAnnotator(text_scale=0.5).annotate(out, det, labels=labels)
    l, r = d.speeds
    cv2.putText(out, f"{d.action}  (L={l:+.2f} R={r:+.2f})", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, f"{d.action}  (L={l:+.2f} R={r:+.2f})", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[d.action], 2, cv2.LINE_AA)
    cv2.putText(out, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class Detector:
    def __init__(self, model_id: str, confidence: float):
        from inference import get_model  # slow import; keep it out of module import time

        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise SystemExit("ROBOFLOW_API_KEY not found. Add ROBOFLOW_API_KEY=... to .env "
                             "(app.roboflow.com -> Settings -> API Keys).")
        self.model = get_model(model_id=model_id, api_key=key)
        self.confidence = confidence

    def __call__(self, frame: np.ndarray) -> tuple[sv.Detections, list[str], float]:
        t0 = time.monotonic()
        result = self.model.infer(frame, confidence=self.confidence)[0]
        ms = (time.monotonic() - t0) * 1000
        det = sv.Detections.from_inference(result)
        names = det.data.get("class_name", [""] * len(det))
        labels = [f"{n} {c:.2f}" for n, c in zip(names, det.confidence)]
        return det, labels, ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", help="phone IP (IP Webcam) or stream URL")
    p.add_argument("--image", help="run on one image file instead of the stream")
    p.add_argument("--model", default="yolov8n-640", help="Roboflow model id (default: COCO yolov8n-640)")
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--near", type=float, default=0.55, help="NEAR line as a fraction of frame height")
    args = p.parse_args()
    if not args.source and not args.image:
        p.error("give a phone IP/stream URL or --image")

    lanes = (0.3, 0.7)
    detector = Detector(args.model, args.confidence)

    def process(frame: np.ndarray, extra: str = "") -> tuple[np.ndarray, Decision]:
        det, labels, ms = detector(frame)
        h, w = frame.shape[:2]
        d = decide(det.xyxy, w, h, near_frac=args.near, lane_split=lanes)
        return draw(frame, det, labels, d, args.near, lanes, f"model {ms:.0f} ms {extra}"), d

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"can't read {args.image}")
        out, d = process(frame)
        dest = Path(args.image).with_suffix(".annotated.jpg")
        cv2.imwrite(str(dest), out)
        print(f"{d.action}  lanes={ {k: round(v, 2) for k, v in d.lanes.items()} }  -> {dest}")
        return

    url = stream_url(args.source)
    print(f"opening {url}  (q quit, s save)")
    last_action, loop_t = None, time.monotonic()
    with PhoneCamera(url) as cam:
        while True:
            frame, age = cam.read()
            if frame is None:
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            now = time.monotonic()
            loop_fps, loop_t = 1.0 / max(1e-6, now - loop_t), now
            out, d = process(frame, f"| frame age {age*1000:.0f} ms | loop {loop_fps:4.1f}/s")
            if d.action != last_action:
                print(f"{datetime.now():%H:%M:%S}  {d.action:7s} lanes " +
                      " ".join(f"{k[0]}={v:.2f}" for k, v in d.lanes.items()))
                last_action = d.action
            cv2.imshow("rover vision (no motors)", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                path = Path("data/annotated") / f"{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), out)
                print(f"saved {path}")
            if key == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
