# Ornimetrics

Edge AI trap controller for birds/critters. Runs YOLO on a Pi-class device, draws live boxes, and drives a servo via the Adafruit PCA9685 HAT. Optional Firebase logging for images/events.

## ✨ Features
- **YOLO detection (CPU)** with forced on-screen bounding boxes every frame
- **Per-species behavior** from `trap_settings.json`  
  (own `open_duration`, `cooldown_duration`, `confidence_threshold`)
- **Servo control** via PCA9685 with jitter-safe timing + optional **slew/ramp**
- **Quiet logging** mode that prints only key events (`[TRIG]`, `[BLOCK]`, `[PHOTO]`, `[FPS]`)
- **Optional** Firebase snapshots + counters
- Hotkeys in non-quiet builds: `o` open, `c` close, `t` test trigger, `p` print state

---

## 🛠 Hardware (tested)
- Raspberry Pi (4/5 recommended)  
- **Adafruit PCA9685 Servo HAT** (I²C @ `0x40`)  
- Hobby servo (5–6 V)  
- **External 5–6 V supply** to HAT **V+** (shared ground with Pi)  
- Recommended: **1000–2200 µF** capacitor across V+ / GND at the HAT

> ⚠️ Don’t power servos from the Pi’s 5 V pin. Use a separate 5–6 V rail and tie **grounds** together.

---

## 📦 Software
- Python 3.9–3.11
- `ultralytics`, `opencv-python`, `adafruit-circuitpython-servokit`  
- (Optional) deps for your `firebase_logger.py` (e.g., `requests`)

Enable I²C and verify the HAT:
```bash
sudo raspi-config   # Interface Options → I2C → Enable
sudo apt install -y i2c-tools
sudo i2cdetect -y 1   # you should see 0x40
```

---

## 🚀 Quickstart

**Quiet build (recommended for deployment):**
```bash
export QT_QPA_PLATFORM=xcb
python3 "path_to_detect1_v3_cpu_boxlog_quiet.py"   --model "path_to_pytorch.pt"   --source 0 --conf 0.45 --imgsz 320   --cam_w 640 --cam_h 480 --fourcc MJPG   --max_fps 12 --every_n 1   --trap_settings "path_to_trapsettings.json"   --db "https://ornimetrics-default-rtdb.firebaseio.com"   --session_key "session_1"   --servo_channel 1 --servo_closed 5 --servo_open 60   --servo_min_us 600 --servo_max_us 2400   --servo_slew 240   --log_fps_interval 20 --preview --preview_scale 0.6 --quiet
```

**Verbose/interactive build (for debugging):**
```bash
export QT_QPA_PLATFORM=xcb
python3 "path_to_detect1_v3_cpu_boxlog.py"   --model "path_to_pytorch.pt"   --source 0 --conf 0.45 --imgsz 320   --cam_w 640 --cam_h 480 --fourcc MJPG   --max_fps 12 --every_n 1   --trap_settings "path_to_trapsettings.json"   --db "https://ornimetrics-default-rtdb.firebaseio.com"   --session_key "session_1"   --servo_channel 1 --servo_closed 5 --servo_open 60   --servo_min_us 600 --servo_max_us 2400   --servo_slew 240   --log_fps_interval 20 --preview --preview_scale 0.6
```

> Tip: If your desktop uses Wayland, the `QT_QPA_PLATFORM=xcb` export avoids Qt plugin errors.

---

## ⚙️ CLI Flags (most useful)

- `--model` Path to your YOLO `.pt` weights  
- `--source` Camera index or path (e.g., `0`)  
- `--conf` Detection confidence threshold (default `0.45`)  
- `--imgsz` Inference image size (default `320`)  
- `--cam_w / --cam_h / --fourcc` Camera capture setup  
- `--max_fps` Max detector FPS (preview still updates smoothly)  
- `--every_n` Run detection every Nth frame to save CPU  
- `--trap_settings` Path to JSON with per-species behavior  
- `--db` Firebase RTDB URL (omit to disable logging)  
- `--session_key` Firebase session bucket/key  
- **Servo tuning**  
  - `--servo_channel` PCA9685 channel (0–15)  
  - `--servo_closed / --servo_open` Angles in degrees  
  - `--servo_min_us / --servo_max_us` Pulse range (wider range = more travel; stay within safe limits for your servo)  
  - `--servo_slew` Slew rate in deg/s (`0` = instant, `240` = gentle ramp)  
- **UI / logs**  
  - `--preview --preview_scale` Show camera window  
  - `--quiet` Reduced logs: `[Start]`, `[TRIG]`, `[BLOCK]`, `[PHOTO]`, `[FPS]`, warnings/errors only  
  - `--log_fps_interval` Print average FPS every N seconds

---

## `trap_settings.json`

Per-species override for confidence/hold/cooldown. Keys are matched case-insensitive, with `_`/`-` treated the same.

```json
{
  "_default": {
    "open_duration": 1.5,
    "cooldown_duration": 8,
    "confidence_threshold": 0.45
  },
  "bird": {
    "open_duration": 1.2,
    "cooldown_duration": 10,
    "confidence_threshold": 0.50
  },
  "squirrel": {
    "open_duration": 0.0
  }
}
```

- `open_duration` seconds the trap stays open (0 disables that species)  
- `cooldown_duration` per-species cooldown after a trigger  
- `confidence_threshold` per-species conf override

There’s also a short **global busy** window (`hold + min_inter_trigger`) to protect the mechanics; it does **not** share the per-species cooldown.

---

## 🔧 Tuning under load (jitter / short travel)
1) **Power first**: external 5–6 V to HAT **V+**, Pi GND tied to HAT GND, add **1000–2200 µF** across V+/GND.  
2) **Travel**: increase `--servo_max_us` (e.g., 2400–2500) and/or decrease `--servo_min_us` (e.g., 500–600) until you get the desired movement, stopping short of mechanical end-stops (buzzing = too far).  
3) **Slew**: lower `--servo_slew` (e.g., 120) for heavier loads to reduce current spikes; raise it (360–480) if you need snappier action.  
4) **Angles**: if buzzing at the extremes, back off `--servo_open` a few degrees.  
5) **Mechanics**: reduce friction / leverage (shorter horn radius) or add a counter-spring for heavy doors.

---

## test (hardware check)
If the app is bugging, confirm the HAT + servo are fine:

```bash
python3 servo_smoketest_ch1.py --channel 1 --freq 50 --min-us 1000 --max-us 2000   --closed 5 --open 60 --cycles 3 --hold 1.0 --pause 0.8 --sweep
```

If the smoketest works but the app doesn’t, it’s a config/timing issue (see `trap_settings.json`, cooldowns, or logs).

---

## Troubleshooting
- **No GUI / Qt error** → set `export QT_QPA_PLATFORM=xcb`  
- **No HAT** → `sudo i2cdetect -y 1` should show `0x40`  
- **Servo jitters** → see *Tuning under load* (power + slew + travel)  
- **Trigger never repeats** → species still in cooldown or global busy; try non-quiet build and watch `[BLOCK] ...` lines  
- **Slow FPS** → lower `--imgsz` or increase `--every_n` (e.g., `2`)

---

## 📄 License
MIT (or your preferred license).

## 🤝 Contributing
PRs welcome! For bug reports, include:
- your full run command,
- a snippet of console logs (especially `[TRIG] / [BLOCK] / [FPS]`),
- `trap_settings.json`.
