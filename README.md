# Wave Rover obstacle avoidance (prototype)

Waveshare Wave Rover running HashiCorp's **Pathfinder v1.3.5** firmware (built on
Waveshare's stock JSON protocol), an Android phone as the camera, and a Mac running
the vision model and control loop.

```
[Android phone, IP Webcam] --MJPEG/Wi-Fi--> [Mac: camera.py + model + logic] --HTTP JSON--> [rover ESP32]
```

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --reinstall opencv-python==4.11.0.86   # see note in requirements.txt
echo 'ROBOFLOW_API_KEY=...' >> .env && chmod 600 .env   # git-ignored; detect.py loads it
```

Rover address defaults to `192.168.68.88`; override with `ROVER_HOST=... ` or `--host`.

## Scripts

| Script | What it does |
| --- | --- |
| `rover.py` | Wi-Fi drive control with speed clamp, keep-alive and dead-man stop |
| `camera.py` | Low-lag phone stream reader; preview, latency check, frame capture to `data/raw/` |
| `detect.py` | Roboflow model on the stream + LEFT/CENTER/RIGHT lane check; shows the decision, **doesn't drive** |
| `drive.py` | **Moves the rover.** Controller manual drive, hold RB/R1 for autonomous, Y records frames |
| `send.py` | Send any JSON command over **USB serial**; `--reset` reboots the board |

```bash
uv run python rover.py status
uv run python rover.py wiggle               # wheels off the ground!
uv run python camera.py <phone-ip>
uv run python detect.py <phone-ip>          # vision only, no motors
uv run python drive.py <phone-ip>           # controller: stick=drive, hold RB=auto, B=stop, Y=record
uv run python send.py '{"T":405}'           # Wi-Fi info over USB
```

## Rover notes

- HTTP: `GET http://<rover>/js?json={...}` — no API key needed (the Pathfinder
  `x-api-key` is only for its `/v1`, `/v2` movement-plan API).
- `{"T":1,"L":l,"R":r}` wheel speeds (about -0.5..0.5) · `{"T":0}` emergency stop ·
  `{"T":999}` clear it · `{"T":130}` status.
- Firmware stops motors itself if pitch exceeds ~34° (`IMU SAFETY` in serial log).
- Wi-Fi config lives in `wifiConfig.json` on the rover's flash. `T:404`/`T:407`
  didn't stick while the rover was retrying a missing network; rewriting the file
  with `T:206` (replace line 1) + `send.py --reset` worked.
- The rover ignores serial commands while stuck in its "connecting…" loop (dots in
  the log); wait for "WiFi reconnection failed" and retry.
- The built-in web page's buttons fire on both touch and mouse events, so phones
  send each command twice — harmless.
