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
| `drive.py` | **Moves the rover.** Switch Pro Controller: hold L (or ZL to boost) + stick drives, hold R for autonomous, A stops, X records |
| `send.py` | Send any JSON command over **USB serial**; `--reset` reboots the board |

```bash
uv run python rover.py status
uv run python rover.py wiggle               # wheels off the ground!
uv run python camera.py <phone-ip>
uv run python detect.py <phone-ip>          # vision only, no motors
uv run python drive.py <phone-ip>           # Switch Pro: L+stick=drive, ZL+stick=boost, hold R=auto, A=STOP, X=record
uv run python send.py '{"T":405}'           # Wi-Fi info over USB
```

### Controller (Nintendo Switch Pro Controller)

Pair it in macOS System Settings → Bluetooth (hold the sync button on top until the
lights run). Buttons are mapped by their **printed labels**:

```
            X  record
     Y            A  STOP
            B
```

| Input | Action |
| --- | --- |
| Hold **L** + left stick | Manual drive at `--max-speed` (default 0.2): up/down throttle, left/right steer |
| Hold **ZL** + left stick | Manual drive at `--boost-speed` (default 0.4), e.g. to get over a threshold |
| Left stick alone | Nothing. Let go of L/ZL to stop. Manual driving overrides autonomous |
| Hold **R** | Autonomous: follow the detector. Release → stop (dead-man switch) |
| **A** | Stop now and cancel autonomous |
| **X** | Record frames to `data/raw/<session>/` (2 per second) |

Why the L/ZL enable: this controller's left stick doesn't always return to center. After
release it has rested at about (+0.22, -0.30) for 30+ seconds, which read as "reverse".
Requiring L or ZL means a stick resting off-center can't move the rover, and A's stop
holds. The stick also has a round 0.5 deadzone: its resting points reached 0.41 from
center (always back-right), and while L/ZL is held those would otherwise creep the rover
backwards. Speed ramps up from zero at the deadzone edge, so you need to push past
halfway before it moves. Recalibrating the stick on a Switch (System Settings →
Controllers and Sensors) may help, and would allow a smaller deadzone.

The panel under the video shows the mode and wheel speeds in large text, then the rover's
status over Wi-Fi (polled about once a second): battery volts, pitch/roll from the
accelerometer, and link latency. Warnings appear there in amber/red and print in the
terminal: main power off (USB only), battery low (< 10.8 V) or critical (< 10.2 V),
tilting past 25°, upside down, and no status from the rover.

Keyboard fallback in the preview window: `space` arm/disarm autonomous (latched — no
dead-man), `x` stop, `r` record, `q` quit. Add `--log` to record every stick reading and
rover command to `data/logs/`.

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
