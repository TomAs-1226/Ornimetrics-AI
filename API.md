# Ornimetrics API Reference

## Quick Start

```bash
# One-command setup
chmod +x setup.sh && ./setup.sh

# Run headless (detection only)
python3 run.py

# Run with web dashboard
python3 run.py --web

# Run with local GUI preview
python3 run.py --preview

# Disable depth camera
python3 run.py --no-3d

# Custom config file
python3 run.py --config my_config.json
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config_3d_detection.json` | Path to config JSON |
| `--web` | off | Enable web dashboard mode |
| `--preview` | off | Show local GUI preview window |
| `--no-3d` | off | Disable all depth processing |
| `--quiet` | off | Minimal console logging |
| `--model` | from config | Override YOLO model path |
| `--port` | 5000 | Override web server port |

## Configuration (`config_3d_detection.json`)

### Detection
```json
{
  "detection": {
    "model_path": "weights.pt",       // YOLO model (.pt or .hef for Hailo)
    "confidence_threshold": 0.45,     // Min detection confidence
    "input_size": 320                 // YOLO input resolution
  }
}
```

### Camera Setup
```json
{
  "camera": {
    "rgb_source": 0,                  // V4L2 device index for detection camera
    "width": 640,
    "height": 480,
    "fourcc": "MJPG"
  },
  "individual_camera": {
    "enabled": false,                 // Enable second camera for individual re-ID
    "source": 1,                      // V4L2 device index
    "width": 640,
    "height": 480
  }
}
```

**Single camera mode** (default): The main camera is used for both species
detection and individual recognition. The YOLO bounding box crop is extracted
from the main camera frame for re-ID.

**Dual camera mode**: Camera 0 runs YOLO for species detection. Camera 1
provides higher-quality crops for individual re-identification. Set
`individual_camera.enabled: true` and `individual_camera.source: 1`.

### Depth Source
```json
{
  "depth": {
    "source": "mono",                 // "cs20" | "mono" | "none"
    "enabled": true,
    "mode": "320x240",                // CS20 resolution (if using TOF)
    "mono_model": "auto",             // "auto" | "midas" | "depth_anything"
    "mono_input_size": 256            // Monocular model input resolution
  }
}
```

| Source | Description | Hardware Needed |
|--------|-------------|-----------------|
| `cs20` | Hardware TOF depth sensor | CS20 sensor |
| `mono` | Monocular depth estimation from RGB | Any RGB camera |
| `none` | No depth (appearance-only re-ID) | None |

### Individual Re-Identification
```json
{
  "reid": {
    "mode": "appearance",             // "appearance" | "point_cloud" | "tflite"
    "appearance_model": null,         // Optional TorchScript model path
    "appearance_input_size": 128      // Crop resize resolution
  }
}
```

| Mode | Description | Needs Depth? | Params |
|------|-------------|-------------|--------|
| `appearance` | RGB crop embeddings (color+texture) | No | ~1.5M or 0 (histogram fallback) |
| `point_cloud` | 3D point cloud via DGCNN | Yes | 35K (micro), 137K (light), 3.5M (heavy) |
| `tflite` | Depth+mask via TFLite model | Yes | ~100K |

### Point Cloud (when using `point_cloud` re-ID mode)
```json
{
  "point_cloud": {
    "backbone": "micro",              // "micro" | "light" | "heavy"
    "fps_points": 512,                // Points per cloud (512 for RPi, 1024 for PC)
    "voxel_size": 0.01,               // Voxel downsampling grid size
    "plane_removal": true             // RANSAC background plane removal
  }
}
```

| Backbone | Parameters | k-neighbors | Embedding dims | RPi speed |
|----------|-----------|-------------|----------------|-----------|
| `micro` | 35K | 10 | 64 | Fast (~15ms) |
| `light` | 137K | 20 | 256 | Medium (~45ms) |
| `heavy` | 3.5M | 40 | 512 | Slow (~200ms) |

### Tracking
```json
{
  "tracking": {
    "database_path": "birdid.sqlite", // SQLite database for individuals
    "match_threshold": 0.32,          // Cosine distance threshold
    "cooldown_seconds": 300,          // Seconds between dispenses per bird
    "max_per_day": 20                 // Max dispenses per bird per day
  }
}
```

### Servo/Trap
```json
{
  "servo": {
    "channel": 1,
    "closed_angle": 5.0,
    "open_angle": 60.0,
    "min_inter_trigger_sec": 1.5,
    "slew_deg_per_s": 240.0,
    "min_us": 1000,
    "max_us": 2000
  },
  "trap_settings": "trap_settings_full.json"
}
```

Per-species trap behavior is configured in `trap_settings_full.json`.

### Performance
```json
{
  "performance": {
    "max_fps": 12.0,                  // Max processing FPS
    "every_n_frames": 1               // Process every Nth frame
  }
}
```

### Firebase
```json
{
  "firebase": {
    "enabled": true,
    "database_url": "https://ornimetrics-default-rtdb.firebaseio.com",
    "session_key": "session_1"
  }
}
```

---

## Web API Endpoints

### `GET /`
Dashboard HTML page with live video feed, FPS counter, and detection stats.
When dual camera is active, shows both detection and individual camera feeds.

### `GET /video_feed`
MJPEG video stream of the detection camera with bounding boxes, labels,
individual names, and trigger indicators overlaid.

**Response**: `multipart/x-mixed-replace; boundary=frame`

### `GET /video_feed_individual`
MJPEG video stream from the individual recognition camera (if dual-camera
mode is enabled). Returns blank frames if no individual camera.

**Response**: `multipart/x-mixed-replace; boundary=frame`

### `GET /api/stats`
Real-time system statistics.

**Response**:
```json
{
  "fps": 11.2,
  "dets": 142,
  "trigs": 8,
  "individuals": 34,
  "has_indiv_cam": false
}
```

### `GET /api/status` (web_detection_server.py only)
Full system status including hardware info.

### `GET /api/individuals` (web_detection_server.py only)
List of all known individual birds in the database.

### `GET /api/detections/recent` (web_detection_server.py only)
Last 100 detections with timestamps, species, and individual info.

### `POST /api/control/detection` (web_detection_server.py only)
Toggle detection on/off.

### `POST /api/control/reset_db` (web_detection_server.py only)
Reset the individual bird database.

---

## Flat Photo Guard

The system guards against flat/printed photos of birds by measuring the
Laplacian variance of the detection crop. Real birds have natural depth
variation and texture gradients that flat photos lack. Detections flagged
as flat are marked with `[FLAT]` in the video overlay and are not enrolled
or matched in the individual database.

Threshold: Laplacian variance < 5.0 = flat (configurable in `_is_flat_crop`).

---

## Architecture

```
RGB Camera 0 ──► YOLO Detection (Hailo/PyTorch)
                     │
                     ├──► Bounding box crops
                     │         │
Individual Camera 1 ─┘    ┌────┴────┐
(optional)                │         │
                     Appearance   Point Cloud
                     Embedder     (DGCNN)
                          │         │
                          ▼         ▼
                     ┌─────────────────┐
                     │  Bird Database   │
                     │  (SQLite)       │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │ Match / Enroll   │
                     │ Cooldown Logic   │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │  Servo Trigger   │
                     │  Firebase Log    │
                     └─────────────────┘
```

## Recommended Configurations

### Raspberry Pi 5 + Hailo-8 (no depth sensor)
```json
{
  "detection": {"model_path": "best.hef", "input_size": 320},
  "depth": {"source": "none", "enabled": false},
  "reid": {"mode": "appearance"},
  "performance": {"max_fps": 12}
}
```

### Raspberry Pi 5 + Hailo-8 + CS20 TOF sensor
```json
{
  "detection": {"model_path": "best.hef", "input_size": 320},
  "depth": {"source": "cs20", "enabled": true},
  "reid": {"mode": "point_cloud"},
  "point_cloud": {"backbone": "micro", "fps_points": 512}
}
```

### Raspberry Pi 5 + dual RGB cameras (no depth sensor)
```json
{
  "detection": {"model_path": "best.hef", "input_size": 320},
  "camera": {"rgb_source": 0},
  "individual_camera": {"enabled": true, "source": 1},
  "depth": {"source": "mono", "enabled": true},
  "reid": {"mode": "appearance"}
}
```

### PC/Laptop (full quality)
```json
{
  "detection": {"model_path": "weights.pt", "input_size": 640},
  "depth": {"source": "mono", "enabled": true, "mono_model": "midas"},
  "reid": {"mode": "point_cloud"},
  "point_cloud": {"backbone": "light", "fps_points": 1024},
  "performance": {"max_fps": 30}
}
```

---

## Bluetooth Setup & Multi-Feeder Support

### How Bluetooth Pairing Works

Each feeder advertises itself via Bluetooth RFCOMM with a unique name that
includes the last 6 characters of its device ID:

```
Ornimetrics OS-a1b2c3
Ornimetrics OS-d4e5f6
```

This ensures multiple feeders in the same area are distinguishable in the
mobile app's Bluetooth scan. The device ID is auto-generated on first boot
(UUID v4) and stored in `ornimetrics_os_config.json`.

### Bluetooth Protocol (JSON over RFCOMM)

The mobile app communicates with the feeder using JSON messages over
Bluetooth Serial Port Profile (SPP). Service UUID: `00001101-0000-1000-8000-00805F9B34FB`.

#### Connection Flow

1. App scans for devices named `Ornimetrics OS-XXXXXX`
2. App connects via RFCOMM
3. Feeder sends welcome message with device info
4. App sends `pair` command to establish session
5. App sends `link_account`, `configure_wifi`, etc.

#### Commands

**`pair`** — Establish session
```json
// Request
{"type": "pair", "app_id": "com.ornimetrics.app", "device_model": "iPhone 15"}

// Response
{"type": "pair_success", "session_token": "abc123...", "device_id": "uuid..."}
```

**`link_account`** — Link user account to feeder
```json
// Request
{"type": "link_account", "session_token": "abc123...", "user_id": "uid_123",
 "account_email": "user@example.com", "account_token": "firebase_token",
 "feeder_name": "Backyard Feeder"}

// Response
{"type": "account_linked", "user_id": "uid_123", "device_id": "uuid..."}
```

**`configure_wifi`** — Send WiFi credentials
```json
// Request
{"type": "configure_wifi", "session_token": "abc123...",
 "ssid": "MyNetwork", "password": "mypassword"}

// Response
{"type": "wifi_configured", "ssid": "MyNetwork", "static_ip": "192.168.1.200"}
```

**`update_settings`** — Update feeder settings
```json
// Request
{"type": "update_settings", "session_token": "abc123...",
 "settings": {"feeder_name": "Garden Feeder",
              "features": {"individual_recognition": true}}}

// Response
{"type": "settings_updated", "success": true}
```

**`get_status`** — Get device status
```json
// Response
{"type": "status", "device_id": "uuid...", "device_name": "My Feeder",
 "version": "1.0.0", "account_linked": true, "wifi_configured": true,
 "static_ip": "192.168.1.200",
 "streaming": {"enabled": true,
   "mjpeg_url": "http://192.168.1.200:5000/video_feed",
   "rtsp_url": "rtsp://192.168.1.200:8554/ornimetrics/stream"}}
```

### Multi-Feeder Configuration

Each feeder is identified by its unique `device_id` (UUID). The Firebase
data structure isolates data per-user per-feeder:

```
/users/{user_id}/feeders/{device_id}/
  ├── detections/
  ├── individuals/
  ├── statistics/
  └── images/
```

To run multiple feeders:
1. Each Pi runs its own instance of Ornimetrics
2. Each generates a unique `device_id` on first boot
3. Bluetooth names include device ID suffix for disambiguation
4. All feeders linked to the same account appear in the mobile app
5. Data is isolated per-feeder in Firebase

### Ornimetrics OS Config (`ornimetrics_os_config.json`)

| Section | Key Fields | Description |
|---------|-----------|-------------|
| `system` | `device_id`, `device_name` | Unique feeder identity |
| `account` | `linked`, `user_id`, `account_email` | User account link |
| `network` | `static_ip`, `wifi_ssid` | Network configuration |
| `bluetooth` | `device_name`, `pin_code`, `paired_devices` | BT settings |
| `streaming` | `mjpeg_port`, `rtsp_port` | Stream endpoints |
| `firebase` | `database_url`, `path_template` | Cloud sync config |
| `features` | `detection`, `individual_recognition` | Feature toggles |

---

## Hailo-8 Model Recommendations

### Pre-compiled HEF models available in Hailo Model Zoo

| Use Case | Model | HEF Available | FPS on Hailo-8 |
|----------|-------|--------------|----------------|
| Object Detection | YOLOv8n/v11n | Yes | 80+ FPS |
| Person/Bird ReID | RepVGG-A0 (512-dim) | Yes | 100+ FPS |
| Depth Estimation | fast_depth (224x224) | Yes | 1000+ FPS |
| Depth Estimation | scdepthv3 (256x320) | Yes | 145 FPS |

### What to use for individual bird recognition

**Best option**: Fine-tune `repvgg_a0_person_reid_512.hef` on bird images
using the [Hailo ReID training guide](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_models/reid/docs/TRAINING_GUIDE.rst),
then recompile to HEF. This runs the full YOLO + ReID pipeline entirely
on the Hailo-8 with zero CPU overhead.

**Current implementation**: The `appearance` re-ID mode uses MobileNetV3-Small
or a color/texture histogram embedding (128-dim). This runs on the RPi CPU
and works without any depth camera.

### Why monocular depth is NOT recommended for individual re-ID

- Available Hailo depth models output **relative** depth (not metric)
- Monocular depth has ~0.45m MAE in outdoor wildlife settings
- Cannot distinguish flat photos from real birds (no anti-spoof)
- Bird bodies are 10-30cm — monocular error exceeds the measurement range

The `mono` depth mode is provided as a convenience for scene visualization
but should not be relied upon for accurate 3D point cloud generation.

---

## Dependencies

### Required
- `ultralytics` — YOLO object detection
- `torch`, `torchvision` — PyTorch runtime
- `numpy` — Numerical computation
- `opencv-python` — Image processing
- `sqlite-utils` — Individual bird database
- `flask`, `flask-cors` — Web dashboard

### Optional
- `tflite-runtime` — TFLite embedder (lightweight alternative to PyTorch)
- `hailo_platform` — Hailo-8 AI Hat acceleration
- `pytorch-metric-learning` — Training individual recognition models
- `open3d` — PLY/PCD file loading fallback (not needed at runtime)

### Removed (no longer required)
- ~~`open3d`~~ — Replaced with pure numpy point cloud preprocessing
- ~~`scipy`~~ — No longer used
- ~~`filterpy`~~ — No longer used
- ~~`pytz`~~ — No longer used
