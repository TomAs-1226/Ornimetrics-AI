# Quick Start: 3D Bird Individual Recognition

Get up and running with 3D individual bird recognition in 5 minutes!

## TL;DR

```bash
# Try it now (will auto-detect available hardware)
python detect_3d_individual.py

# Or force optics-only mode
python detect_3d_individual.py --no-3d
```

The system automatically detects:
- ✅ Hailo-8 AI Hat (if available)
- ✅ CS20 depth camera (if available)
- ✅ Falls back gracefully if hardware missing

## What You Get

### With 3D Camera (Full Mode)
```
[Mode] Full 3D | Hailo-8 (26 TOPS)
[3D] CS20 depth camera initialized
[3D] Individual tracker initialized (backbone: heavy)
```
- Individual bird recognition
- Per-bird cooldowns and daily limits
- Anti-spoof protection
- Individual bird logging

### Without 3D Camera (Optics-Only)
```
[Mode] Optics-Only | PyTorch CPU
[3D] 3D mode disabled - using optics-only mode
```
- Species detection only
- Traditional servo triggering
- Same as your original system

## Hardware Setup

### Minimum (Optics-Only Mode)
- ✅ Raspberry Pi
- ✅ USB camera
- ✅ Servo trap

### Recommended (Full 3D Mode)
- ✅ Raspberry Pi 5
- ✅ USB RGB camera
- ✅ DFRobot CS20 ToF camera (or compatible Arducam)
- ✅ Servo trap

### Optional (AI Hat Acceleration)
- ✅ Raspberry Pi AI Hat 1+ with Hailo-8
- ✅ HEF model file (converted from PyTorch)

## Installation

### 1. Update Your Python Environment

```bash
# Install point cloud dependencies (if not already)
pip install torch torchvision open3d scikit-learn

# Or use the requirements file
pip install -r requirements-pc.txt
```

### 2. Connect CS20 Camera (Optional)

```bash
# Check if camera is detected
ls /dev/video*

# Should see /dev/video0, /dev/video1, etc.
# CS20 usually shows as one of these

# Install CS20 driver if needed
# Follow: https://github.com/DFRobot-official/DFRobot_CX20_driver
```

### 3. Install Hailo Libraries (Optional)

```bash
# Only if you have Raspberry Pi AI Hat 1+
# Follow official guide:
# https://github.com/hailo-ai/hailo-rpi5-examples
```

## Quick Test

### Test 1: Check What Hardware Is Detected

```bash
python detect_3d_individual.py --quiet
```

Look for the `[Mode]` line:
- `Full 3D` = Everything working!
- `Optics-Only` = RGB camera only (normal fallback)

### Test 2: Run With Your Existing Model

```bash
python detect_3d_individual.py \
  --model /home/pi/Desktop/FinalPrototype/weights.pt \
  --source 0 \
  --use-3d
```

### Test 3: Run Without 3D (Optics-Only)

```bash
python detect_3d_individual.py \
  --model /home/pi/Desktop/FinalPrototype/weights.pt \
  --no-3d
```

## Common Options

### Basic Detection
```bash
# Auto-detect everything
./detect_3d_individual.py

# Custom model
./detect_3d_individual.py --model path/to/weights.pt

# Different camera
./detect_3d_individual.py --source 1

# Higher confidence threshold
./detect_3d_individual.py --conf 0.6
```

### 3D Configuration
```bash
# Use light backbone (faster, less accurate)
./detect_3d_individual.py --backbone light

# Stricter individual matching
./detect_3d_individual.py --match-threshold 0.25

# Longer cooldown (10 minutes)
./detect_3d_individual.py --cooldown 600

# Higher daily limit
./detect_3d_individual.py --max-per-day 30
```

### Performance
```bash
# Limit FPS
./detect_3d_individual.py --max_fps 10

# Process every other frame
./detect_3d_individual.py --every_n 2

# Quiet mode (less logging)
./detect_3d_individual.py --quiet
```

## Understanding the Output

### Startup Messages

```
[Start] model=weights.pt cam=640x480 conf=0.45 imgsz=320
```
Shows configuration being used.

```
[Mode] Full 3D | Hailo-8 (26 TOPS)
```
- **Full 3D** = 3D camera + individual tracking active
- **Optics-Only** = Standard detection only
- **Hailo-8** = Using AI Hat acceleration
- **PyTorch CPU** = Using CPU inference

### During Operation

```
[ENROLL] New bird registered: Bird_1
```
New individual bird enrolled in database.

```
[TRIG] cardinal conf=0.89 | Bird_1 (known)
```
Known bird triggered servo dispense.

```
[BLOCK] cardinal | Bird_2 | cooldown (287.3s remaining)
```
Bird blocked due to cooldown.

```
[FPS] avg=11.84
```
Average frames per second.

## Troubleshooting

### "CS20 hardware not detected"

**This is normal!** System automatically falls back to optics-only mode.

To use 3D mode:
1. Connect CS20 camera
2. Check `ls /dev/video*` shows camera
3. Install CS20 driver if needed
4. Run with `--use-3d` flag

### "Hailo libraries not available"

**This is normal!** System uses PyTorch CPU fallback.

To use Hailo:
1. Install Raspberry Pi AI Hat 1+
2. Install HailoRT libraries
3. Convert model to HEF format
4. Use: `--model your_model.hef`

### Poor Individual Recognition

Try these in order:
```bash
# 1. Use heavy backbone
./detect_3d_individual.py --backbone heavy

# 2. Stricter threshold
./detect_3d_individual.py --match-threshold 0.25

# 3. Check camera has good view of birds
# Make sure birds are 0.5-2 meters from camera

# 4. Reset database and re-enroll
rm birdid.sqlite
./detect_3d_individual.py
```

### Low FPS

```bash
# Use Hailo if available
./detect_3d_individual.py --model weights.hef

# Or reduce resolution
./detect_3d_individual.py --depth-mode 320x240

# Or use light backbone
./detect_3d_individual.py --backbone light

# Or process fewer frames
./detect_3d_individual.py --every_n 2
```

## Replacing Your Current Script

### Option 1: Run Side-by-Side (Recommended)

Keep both scripts and test the new one:
```bash
# Old script (still works)
python detect1_v3_cpu_boxlog_quiet.py --quiet

# New script with 3D
python detect_3d_individual.py --quiet
```

### Option 2: Switch Completely

Update your startup script:
```bash
# Replace this:
python detect1_v3_cpu_boxlog_quiet.py --model weights.pt --quiet

# With this:
python detect_3d_individual.py --model weights.pt --quiet
```

## What's Different from Old Script

### Preserved (Same as Before)
- ✅ Firebase logging
- ✅ Servo control
- ✅ Species detection
- ✅ All command-line arguments
- ✅ Preview window
- ✅ FPS limiting
- ✅ Trap settings JSON

### New Features
- ✨ Individual bird recognition (with 3D camera)
- ✨ Per-bird cooldowns and limits
- ✨ Hailo AI Hat support
- ✨ Automatic hardware detection
- ✨ Graceful fallback modes
- ✨ Anti-spoof validation

### No Breaking Changes
If you run without 3D camera, it works **exactly like before**.

## Next Steps

1. **Test it**: Run the script and watch the console output
2. **Check database**: Look at `birdid.sqlite` after a few birds visit
3. **Monitor Firebase**: Individual IDs are logged to Firebase
4. **Tune settings**: Adjust thresholds in `config_3d_detection.json`
5. **Read full docs**: See `BIRD_3D_RECOGNITION.md` for details

## Getting Help

1. Run with `--help` to see all options
2. Check console output for error messages
3. Try optics-only mode to isolate 3D issues: `--no-3d`
4. Read full documentation: `BIRD_3D_RECOGNITION.md`

## Example Systemd Service

To run automatically on boot:

```ini
[Unit]
Description=Bird Detection with 3D Individual Recognition
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Ornimetrics-AI
ExecStart=/usr/bin/python3 detect_3d_individual.py --quiet
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save as `/etc/systemd/system/birddetect.service` and:
```bash
sudo systemctl daemon-reload
sudo systemctl enable birddetect
sudo systemctl start birddetect
```

## That's It!

You're ready to go. The system will automatically use whatever hardware is available and fall back gracefully when needed.

**Happy bird watching!** 🐦
