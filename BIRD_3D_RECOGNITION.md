# 3D Bird Individual Recognition System

## Overview

This enhanced bird detection system uses a **3D time-of-flight (ToF) camera** and **spatial AI models** to recognize individual birds, like a "3D Face ID" for birds. The system intelligently falls back to optics-only mode when the 3D camera is unavailable.

## Key Features

### 1. **Dual-Mode Operation**

#### Full 3D Mode (When CS20 Camera Available)
- YOLO species detection
- 3D point cloud capture from CS20 ToF camera
- Individual bird recognition using DGCNN spatial AI model
- Per-individual cooldown and daily limits
- Anti-spoof validation (detects photos/printouts)
- Individual bird logging to database and Firebase

#### Optics-Only Mode (Fallback)
- YOLO species detection
- Traditional servo triggering by species
- No individual tracking
- Automatic fallback when 3D camera not detected

### 2. **AI Hat Acceleration**

The system supports the **Raspberry Pi AI Hat 1+** with Hailo-8 chip (26 TOPS):
- Converts YOLO models to Hailo HEF format for hardware acceleration
- Graceful fallback to PyTorch CPU if Hailo not available
- Significantly improves FPS and reduces CPU load

### 3. **Individual Bird Tracking**

Each bird gets a unique identity based on their 3D point cloud "fingerprint":
- **Enrollment**: First visit registers the bird as a new individual
- **Recognition**: Subsequent visits match against database
- **Cooldown**: Per-bird cooldown period (default 5 minutes)
- **Daily Limits**: Max dispenses per bird per day (default 20)
- **Prototype Learning**: Bird embeddings improve over time with EMA updates

### 4. **Anti-Spoof Protection**

The system validates 3D structure to prevent spoofing:
- Planarity detection (rejects flat photos)
- Thickness measurement (ensures 3D volume)
- Point cloud quality scoring
- Geometry consistency checks

## Hardware Requirements

### Required
- Raspberry Pi (tested on Pi 5)
- USB camera for RGB (bird detection)
- Servo trap mechanism

### Optional (for Full 3D Mode)
- **DFRobot CS20 ToF Camera** (Arducam compatible)
  - Provides depth/point cloud data
  - 320x240 or 640x480 modes
  - Appears as `/dev/video*` on Linux

### Optional (for AI Hat Acceleration)
- **Raspberry Pi AI Hat 1+**
  - Hailo-8 accelerator (26 TOPS)
  - Requires HailoRT library
  - Requires converted HEF model file

## Installation

### 1. Install Dependencies

```bash
# Core dependencies (if not already installed)
pip install -r requirements.txt

# Point cloud dependencies
pip install -r requirements-pc.txt

# For Hailo-8 AI Hat (optional)
# Follow Hailo installation guide for your platform
# https://github.com/hailo-ai/hailo-rpi5-examples
```

### 2. Install CS20 Camera (Optional)

```bash
# Install DFRobot CS20 driver
git clone https://github.com/DFRobot-official/DFRobot_CX20_driver
cd DFRobot_CX20_driver
# Follow installation instructions

# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect
```

### 3. Convert Model for Hailo (Optional)

```bash
# If you have a Hailo AI Hat and want acceleration
# Convert your YOLO PyTorch model to HEF format
# This requires the Hailo Dataflow Compiler
# See: https://hailo.ai/developer-zone/documentation/
```

## Configuration

Edit `config_3d_detection.json` to customize:

### Point Cloud Settings
```json
{
  "point_cloud": {
    "backbone": "heavy",  // "light" or "heavy" DGCNN model
    "preprocessing": {
      "plane_removal": true,  // Remove background planes
      "normalization": "center_only",
      "voxel_size": 0.01,
      "fps_points": 2048
    }
  }
}
```

### Individual Tracking
```json
{
  "individual_tracking": {
    "match_threshold": 0.32,  // Lower = stricter matching
    "cooldown_seconds": 300,  // 5 minutes between dispenses
    "max_dispenses_per_day": 20
  }
}
```

### Camera Intrinsics
Calibrate your CS20 camera and update:
```json
{
  "point_cloud": {
    "intrinsics": {
      "fx": 525.0,
      "fy": 525.0,
      "cx": 319.5,
      "cy": 239.5
    }
  }
}
```

## Usage

### Basic Usage (Auto-detect hardware)

```bash
python detect_3d_individual.py
```

The system will automatically:
1. Try to use Hailo-8 if HEF model available
2. Try to initialize CS20 depth camera
3. Fall back gracefully if hardware unavailable

### Force Optics-Only Mode

```bash
python detect_3d_individual.py --no-3d
```

### Use Hailo AI Hat

```bash
python detect_3d_individual.py \
  --model best.hef \
  --use-3d
```

### Custom Configuration

```bash
python detect_3d_individual.py \
  --model weights.pt \
  --use-3d \
  --backbone heavy \
  --match-threshold 0.30 \
  --cooldown 600 \
  --max-per-day 15 \
  --depth-mode 640x480
```

### All Options

```bash
python detect_3d_individual.py --help
```

## How It Works

### 1. Detection Pipeline

```
RGB Frame → YOLO → Species Detection
     ↓
Depth Frame → Point Cloud → DGCNN → Embedding
     ↓
Embedding → Database Match → Individual ID
     ↓
Individual ID → Cooldown Check → Trigger/Block
```

### 2. Individual Recognition Flow

```
New Bird Visit:
1. YOLO detects species and bounding box
2. CS20 captures depth in that region
3. Convert depth to 3D point cloud
4. Validate 3D structure (anti-spoof)
5. DGCNN generates 512-dim embedding
6. Match against database (cosine similarity)
7. If match: Check cooldown/limits
8. If no match: Enroll as new individual
9. Trigger servo if allowed
10. Log to Firebase with individual ID
```

### 3. Graceful Degradation

```
Best Case: Hailo + CS20 + Individual Tracking
↓ (no Hailo)
Good: PyTorch CPU + CS20 + Individual Tracking
↓ (no CS20)
Fallback: PyTorch CPU + Species-only Detection
```

## Database Structure

Individual birds are stored in `birdid.sqlite`:

### Tables

- **individuals**: Bird records (species, last_seen, last_refresh)
- **prototypes**: Embeddings for each bird (vector, weight, timestamp)
- **individual_stats**: Cooldown and daily limits tracking

### Querying the Database

```python
from birdid.db import BirdIDDatabase

db = BirdIDDatabase("birdid.sqlite")

# Get all prototypes for a species
prototypes = db.get_prototypes("cardinal")

# Get individual stats
stats = db.get_stats(individual_id=1, now_ts=time.time())
print(f"Last dispense: {stats['last_dispense_ts']}")
print(f"Dispenses today: {stats['dispense_count_today']}")
```

## Logging and Monitoring

### Console Output

```
[Start] model=weights.pt cam=640x480 conf=0.45 imgsz=320
[Mode] Full 3D | Hailo-8 (26 TOPS)
[3D] CS20 depth camera initialized successfully
[3D] Individual tracker initialized (backbone: heavy)

[ENROLL] New bird registered: Bird_1
[TRIG] cardinal conf=0.89 | Bird_1 (known)
[BLOCK] cardinal | Bird_1 | cooldown (287.3s remaining)
[FPS] avg=11.84
```

### Firebase Logging

Events logged to Firebase include:
- Species detected
- Individual bird ID (if 3D mode)
- Whether bird is new enrollment
- Confidence scores
- Detection mode (Full 3D vs Optics-Only)
- Cropped bird images

## Troubleshooting

### CS20 Camera Not Detected

```
[3D] CS20 hardware not detected - falling back to optics-only mode
```

**Solutions:**
1. Check camera is connected: `ls /dev/video*`
2. Verify driver installed: Check DFRobot CS20 driver
3. Check permissions: `groups $USER` should include `video`
4. Try different video device: `--source 1` or `--source 2`

### Hailo Not Working

```
[WARN] Hailo libraries not available
```

**Solutions:**
1. Install HailoRT: Follow Hailo documentation
2. Use `.hef` model file, not `.pt`
3. Verify Hailo device: Check with Hailo utilities
4. System will auto-fallback to PyTorch

### Poor Individual Recognition

If birds aren't being recognized consistently:

1. **Adjust match threshold**: Lower = stricter
   ```bash
   --match-threshold 0.25
   ```

2. **Use heavy backbone**: Better accuracy
   ```bash
   --backbone heavy
   ```

3. **Check point cloud quality**: Ensure good lighting and stable depth readings

4. **Calibrate camera intrinsics**: Use proper calibration values

### Database Issues

Reset the database if needed:
```bash
rm birdid.sqlite
# System will create new database on next run
```

## Performance Tips

### For Maximum FPS (Hailo)
- Use HEF model with Hailo-8
- Use 320x240 depth mode
- Use light backbone for faster point cloud processing

### For Best Accuracy
- Use heavy backbone
- Use 640x480 depth mode
- Enable plane removal preprocessing
- Calibrate camera intrinsics properly

### For Resource-Constrained Systems
- Use optics-only mode (`--no-3d`)
- Reduce FPS (`--max_fps 8`)
- Use light backbone
- Process fewer frames (`--every_n 2`)

## Integration with Existing System

The new system is **fully compatible** with your existing setup:

✅ **Preserved Features:**
- Firebase logging
- Servo trap control
- Species-specific settings (`trap_settings.json`)
- All command-line arguments from original `detect1_v3_cpu_boxlog_quiet.py`

✅ **Enhanced Features:**
- Individual bird tracking (when 3D available)
- Per-individual cooldowns and limits
- Hailo acceleration support
- Automatic hardware detection

✅ **No Breaking Changes:**
- Can run in optics-only mode (same as before)
- All Firebase schemas unchanged
- Servo control logic unchanged

## Future Enhancements

Potential improvements for the system:

1. **Multi-camera support**: Track birds across multiple feeders
2. **Behavior analysis**: Track feeding patterns per individual
3. **Cloud sync**: Sync individual birds across multiple devices
4. **Advanced anti-spoof**: Motion analysis, temporal consistency
5. **Species-specific models**: Fine-tuned point cloud models per species
6. **Web dashboard**: Real-time individual bird tracking UI

## Credits

- **Point Cloud AI**: DGCNN (Dynamic Graph CNN) for point cloud feature learning
- **Bird Detection**: YOLOv8 (Ultralytics)
- **Depth Camera**: DFRobot CS20 ToF Camera
- **AI Acceleration**: Hailo-8 on Raspberry Pi AI Hat 1+
- **Original System**: Ornimetrics bird feeder project

## License

Same license as the parent Ornimetrics-AI project.

## Support

For issues or questions:
1. Check this documentation first
2. Review console output for error messages
3. Test in optics-only mode to isolate 3D issues
4. Check hardware connections and permissions
