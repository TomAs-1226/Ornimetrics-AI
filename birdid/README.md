# BirdID

Depth-assisted individual bird re-identification for Raspberry Pi 5 using a
DFRobot CS20 depth-only ToF camera alongside an RGB+YOLO detector.

## Features
- Timestamp sync and calibration between RGB and CS20 depth.
- Depth cropping, quality gating, and background suppression.
- Anti-spoof depth validation (planarity, thickness, centroid stability).
- Deterministic baseline embedding plus optional TFLite embedding backend.
- SQLite prototype DB per species with cosine matching and open-set creation
  plus cooldown/daily dispense limits.
- Tracklet buffering for motion robustness.
- CLI demo for live (stdin JSON) and recorded (synthetic) modes.

## Setup
1. Install Python dependencies: `pip install -r requirements.txt` (and
   `tflite-runtime` if using the TFLite embedder).
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
{
  time_range, species, track_id, individual_id, decision, deny_reason,
  confidence, min_dist, second_best_dist, margin,
  quality_score, frames_used, cooldown_remaining
}
```

## PC upload demo
For quick testing on a PC with file uploads instead of live cameras, use the
Streamlit app:

```bash
pip install -r requirements-pc.txt
streamlit run pc_demo/app.py
```

Upload an RGB image and either a depth PNG/NPY or a point cloud (PLY/PCD) to run
the same validation, matching, enrollment, cooldown, and decision logic. The UI
can seed a known bird, force enrollment, and shows debug overlays for bbox and
depth. Firebase uploads are enabled when `GOOGLE_APPLICATION_CREDENTIALS` is set
and the Google Cloud packages are installed.

### Enrollment/dispense behavior
- Depth must pass anti-spoof validation; otherwise we DENY with
  `invalid_depth_or_spoof`.
- Validation now also checks plausible physical size at ~40 cm and rejects
  stale depth frames that fall outside the configurable timestamp tolerance.
- If a valid tracklet does not confidently match an existing individual it is
  ENROLLed with a new `individual_id` but does not dispense (unless
  `allow_first_seen_dispense=true`).
- Dispense requires stronger gates: `min_dist < dispense.threshold`,
  `margin > dispense.margin_threshold`, minimum frames and depth quality, and
  passing cooldown/daily limits.
- Cooldown/daily limits are persisted per individual in SQLite. Adjust under
  `cooldown` in the config.
- YOLO species confidence gates dispensing via `yolo.min_confidence`.

### ROI fallback strategy
Depth ROIs from RGB bboxes are clamped to the CS20 dimensions, padded, and auto-expanded when too small. If the mapped ROI is still invalid, the engine falls back to the last good ROI or a foreground valid-depth blob to avoid empty crops. ROI metrics are included in decision outputs for debugging.

### Partial pointcloud handling
Tracklets keep per-frame quality and aggregate the best frames so partially visible birds still produce embeddings. Anti-spoof checks remain strict (planarity, thickness, centroid stability), while incomplete coverage only reduces quality. Dispense gating still enforces quality/frames thresholds.

### Weekly refresh policy
Each individual tracks `last_seen_ts` and `last_refresh_ts`. If a bird is re-matched after `matching.refresh_days` (default 7) with confidence above `matching.refresh_confidence_threshold` and quality above `matching.refresh_quality_min`, its prototype is refreshed with EMA to keep identities current without drift.

### Migration note
Existing SQLite databases gain additional columns (`last_seen_ts`,
`last_refresh_ts`, prototype `created_ts`) plus the `individual_stats` table the
next time the engine runs. Previously enrolled individuals receive the new
metadata on first use; no manual migration is required.

## Profiling notes
- Depth capture runs in a background thread; processing avoids unnecessary
  copies.
- Use 320x240 mode for lower latency. Keep `max_queue` small in
  `CS20Camera` to bound buffering.

## Firebase
Integrate your existing Firebase upload by consuming the tracklet outputs from
`BirdIDEngine` and pushing them to your backend. The engine returns dictionaries
with stable IDs you can forward directly.
