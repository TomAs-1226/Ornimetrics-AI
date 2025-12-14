# BirdID

Depth-assisted individual bird re-identification for Raspberry Pi 5 using a
DFRobot CS20 depth-only ToF camera alongside an RGB+YOLO detector.

## Features
- Timestamp sync and calibration between RGB and CS20 depth.
- Depth cropping, quality gating, and background suppression.
- Deterministic baseline embedding plus optional TFLite embedding backend.
- SQLite prototype DB per species with cosine matching and open-set creation.
- Tracklet buffering for motion robustness.
- CLI demo for live (stdin JSON) and recorded (synthetic) modes.

## Setup
1. Install Python dependencies: `pip install numpy` (and `tflite-runtime` if
   using the TFLite embedder).
2. Install CS20 driver/SDK and ensure the depth device appears under
   `/dev/video*`.
3. Calibrate RGB-to-depth mapping:
   ```bash
   python -m birdid.calib.calibrate_rgb_to_depth --pairs pairs.json \
     --output birdid/calib/rgb_to_depth.json
   ```
4. Adjust thresholds in `birdid/config_default.json` for your feeder geometry.

### Outdoor sun mitigation
- Add a short hood/shroud to reduce direct IR from the sun.
- Reject frames where `mask.mean()` falls below the configured threshold (already
  enabled).
- Prefer 320x240 depth mode when motion is high to maintain frame rate.

## Running the CLI demo
Recorded mode (synthetic data):
```bash
python -m birdid.cli_demo --mode recorded
```

Live mode (expects JSON detections from YOLO on stdin):
```bash
python -m birdid.cli_demo --mode live
```
Each emitted line corresponds to a tracklet summary:
```
{time_range, species, track_id, individual_id, confidence, min_dist, frames_used}
```

## Profiling notes
- Depth capture runs in a background thread; processing avoids unnecessary
  copies.
- Use 320x240 mode for lower latency. Keep `max_queue` small in
  `CS20Camera` to bound buffering.

## Firebase
Integrate your existing Firebase upload by consuming the tracklet outputs from
`BirdIDEngine` and pushing them to your backend. The engine returns dictionaries
with stable IDs you can forward directly.
