"""Read the Android phone's camera stream with minimal lag.

Phone setup (IP Webcam app, free on Play Store):
  * Video preferences -> Video resolution 640x480, Quality ~50
  * Start server; it shows the phone's address, e.g. http://192.168.68.70:8080
  * Stream URL is that address + /video

A background thread keeps grabbing frames and only the newest one is kept, so
the consumer (preview, Roboflow model, ...) never works on a stale, buffered
frame. Also works with DroidCam (http://<ip>:4747/video) or any RTSP/MJPEG URL.

Preview / latency test:
    uv run python camera.py 192.168.68.70          # bare IP -> IP Webcam URL
    uv run python camera.py http://192.168.68.70:8080/video

  Keys:  q quit   s save a frame   r start/stop recording frames (2 per second)
         p freeze/unfreeze the preview (for the latency check)
  Frames are saved to data/raw/ -- handy for building a Roboflow dataset later.
  Every 5 s a summary (fps, worst gap between frames) prints in the terminal.

  Latency check: point the phone at this preview window so it films the big
  "now" clock, then press p. The frozen image shows two clocks: the bottom one
  is when the frame arrived on the Mac, the one inside the phone's picture is
  when the phone captured it. Bottom minus inner = end-to-end latency.
  Repeat a few times; the "frame age" number at the top is NOT this latency,
  only how long the newest frame has waited on the Mac.
"""

from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def stream_url(arg: str) -> str:
    if "://" in arg:
        return arg
    host = arg if ":" in arg else f"{arg}:8080"
    return f"http://{host}/video"


class PhoneCamera:
    def __init__(self, url: str, reconnect_s: float = 2.0):
        self.url = url
        self.reconnect_s = reconnect_s
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps_window: list[float] = []
        self._max_gap = 0.0  # longest wait between frames since last stats() call

    def start(self) -> "PhoneCamera":
        self._thread = threading.Thread(target=self._loop, name="phone-camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"[camera] can't open {self.url}; retrying in {self.reconnect_s}s")
                self._stop.wait(self.reconnect_s)
                continue
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    print("[camera] stream dropped; reconnecting")
                    break
                now = time.monotonic()
                with self._lock:
                    if self._frame_time:
                        self._max_gap = max(self._max_gap, now - self._frame_time)
                    self._frame, self._frame_time = frame, now
                    self._count += 1
                    self._fps_window = [t for t in self._fps_window if now - t < 2.0] + [now]
            cap.release()
            self._stop.wait(self.reconnect_s)

    def read(self) -> tuple[np.ndarray | None, float]:
        """Return (newest frame or None, seconds since it arrived)."""
        with self._lock:
            if self._frame is None:
                return None, float("inf")
            return self._frame.copy(), time.monotonic() - self._frame_time

    @property
    def fps(self) -> float:
        with self._lock:
            w = self._fps_window
            return (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else 0.0

    def take_max_gap(self) -> float:
        """Longest gap between frames (seconds) since the previous call; resets it."""
        with self._lock:
            gap, self._max_gap = self._max_gap, 0.0
            return gap

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._count

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def __enter__(self) -> "PhoneCamera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


def _label(img: np.ndarray, text: str, y: int, scale: float = 0.7) -> None:
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="phone IP (assumes IP Webcam) or full stream URL")
    p.add_argument("--save-dir", default="data/raw")
    args = p.parse_args()

    url = stream_url(args.source)
    save_dir = Path(args.save_dir)
    recording, last_saved = False, 0.0
    frozen: np.ndarray | None = None
    stats_t0, stats_n0 = time.monotonic(), 0
    print(f"opening {url}  (q quit, s save, r record, p freeze)")

    with PhoneCamera(url) as cam:
        while True:
            now = time.monotonic()
            if now - stats_t0 >= 5.0:
                n = cam.frame_count
                print(f"[stats] {(n - stats_n0) / (now - stats_t0):4.1f} fps avg, "
                      f"worst gap between frames {cam.take_max_gap() * 1000:4.0f} ms")
                stats_t0, stats_n0 = now, n

            if frozen is not None:
                cv2.imshow("phone camera", frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("p"):
                    frozen = None
                elif key == ord("q"):
                    break
                continue

            frame, age = cam.read()
            if frame is None:
                frame = np.zeros((480, 640, 3), np.uint8)
                _label(frame, f"waiting for {url} ...", 240)
            else:
                h, w = frame.shape[:2]
                _label(frame, f"{w}x{h}  {cam.fps:4.1f} fps  frame age {age*1000:4.0f} ms", 30)
            _label(frame, "now " + datetime.now().strftime("%H:%M:%S.%f")[:-3], frame.shape[0] - 20, 1.1)
            if recording:
                cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)

            cv2.imshow("phone camera", frame)
            key = cv2.waitKey(1) & 0xFF

            raw, _ = cam.read()
            want_save = key == ord("s") or (recording and time.monotonic() - last_saved >= 0.5)
            if raw is not None and want_save:
                save_dir.mkdir(parents=True, exist_ok=True)
                path = save_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"
                cv2.imwrite(str(path), raw)  # saved without the overlay text
                last_saved = time.monotonic()
                if key == ord("s"):
                    print(f"saved {path}")
            if key == ord("p"):
                frozen = frame.copy()
                _label(frozen, "FROZEN - press p to resume", 60)
                print("frozen: bottom clock minus the clock inside the picture = latency")
            if key == ord("r"):
                recording = not recording
                print("recording" if recording else "stopped recording")
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
