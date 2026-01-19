# OOBE (Out Of Box Experience) System Guide

## Overview

The Ornimetrics OOBE system automatically configures your bird detection system on first boot. It's designed to make setup effortless and only runs once.

## What OOBE Does

### 1. **Dependency Management**
- ✅ Checks for all required Python packages
- ✅ Automatically installs missing dependencies
- ✅ Verifies Python version compatibility
- ✅ Handles optional packages gracefully

### 2. **Directory Structure**
Creates necessary directories:
- `data/` - Runtime data storage
- `data/species_prototypes/` - 3D point cloud models
- `logs/` - System logs
- `backups/` - Database backups
- `models/` - AI model storage

### 3. **Species Prototypes**
- ✅ Generates synthetic 3D point clouds for common bird species
- ✅ Creates 3 variations per species for diversity
- ✅ Supports 20+ common backyard feeder birds
- ✅ Seeds the individual bird identification system

### 4. **Hardware Detection**
- ✅ Detects Hailo-8 AI Hat
- ✅ Identifies USB cameras
- ✅ Checks for depth camera support
- ✅ Saves hardware capabilities for auto-configuration

### 5. **Feeder Configuration**
- ✅ Creates unique feeder ID
- ✅ Sets up future app connectivity framework
- ✅ Prepares multi-feeder network support
- ✅ Configures monitoring options

### 6. **Completion Marker**
- ✅ Saves `.oobe_completed` file
- ✅ Prevents running setup multiple times
- ✅ Stores system information and setup date

## When Does OOBE Run?

OOBE runs automatically when:
1. **First system startup** - No `.oobe_completed` marker exists
2. **After marker deletion** - User manually removed `.oobe_completed`
3. **Manual invocation** - User runs `python3 oobe_setup.py`

OOBE will **NOT** run if:
- ✅ Setup already completed (marker file exists)
- ✅ System is already configured

## Running OOBE

### Automatic (Recommended)

Just start the system normally:
```bash
./start_detection_system.sh
```

OOBE will automatically run if needed.

### Manual

Run OOBE directly:
```bash
python3 oobe_setup.py
```

### Force Re-run

To force OOBE to run again:
```bash
rm .oobe_completed
python3 oobe_setup.py
```

## Species with 3D Support

OOBE generates initial prototypes for these common feeder birds:

### Excellent Detection Quality
- **Northern Cardinal** - Distinctive crest, robust build
- **Blue Jay** - Large with prominent crest
- **Eastern Towhee** - Large body, long tail
- **Rose-breasted Grosbeak** - Robust with massive bill
- **Northern Flicker** - Large woodpecker
- **Tufted Titmouse** - Distinctive crest

### Very Good Detection Quality
- **White-breasted Nuthatch** - Compact, distinctive posture
- **Carolina Wren** - Plump body, upright tail
- **Downy Woodpecker** - Vertical posture, compact

### Good Detection Quality
- **American Goldfinch** - Small, compact
- **Carolina Chickadee** - Small, round body
- **Dark-eyed Junco** - Rounded head
- **Chipping Sparrow** - Small, slender
- **Song Sparrow** - Medium build
- **Indigo Bunting** - Compact with conical bill
- **House Sparrow** - Chunky build

## Feeder Configuration

OOBE creates `feeder_config.json` with this structure:

### Basic Configuration
```json
{
  "feeder": {
    "id": "unique-uuid",
    "name": "hostname",
    "location": {
      "description": "Your location",
      "latitude": null,
      "longitude": null
    }
  }
}
```

### App Connectivity (Future Feature)
```json
{
  "app_connectivity": {
    "enabled": false,
    "api_endpoint": "http://localhost:5000",
    "authentication": {
      "enabled": false
    }
  }
}
```

### Multi-Feeder Network (Future Feature)
```json
{
  "multi_feeder_network": {
    "enabled": false,
    "network_id": null,
    "connected_feeders": [],
    "share_individual_database": false
  }
}
```

This allows your mobile app to:
- 🔗 Connect to individual feeders
- 📊 View stats from multiple locations
- 🐦 Sync individual bird databases across feeders
- 🔔 Receive notifications from any feeder

## What Gets Created

### Files
```
.oobe_completed              # Completion marker
feeder_config.json          # Feeder configuration
species_3d_support.json     # Species with 3D support
```

### Directories
```
data/
├── species_prototypes/     # PLY point cloud models
│   ├── Cardinal_prototype_0.ply
│   ├── Blue_Jay_prototype_0.ply
│   └── ... (20+ species)
logs/                       # System logs
backups/                    # Database backups
models/                     # AI models
```

### Prototype Files
Each species gets 3 PLY files:
- `{Species}_prototype_0.ply` - Primary variant
- `{Species}_prototype_1.ply` - Variation 1
- `{Species}_prototype_2.ply` - Variation 2

Total: ~60-75 PLY files (20+ species × 3 variants)

## Customizing OOBE

### Add New Species

Edit `species_3d_support.json`:

```json
{
  "supported_species": {
    "Your_Species": {
      "enabled": true,
      "common_name": "Your Species",
      "scientific_name": "Genus species",
      "typical_size_cm": 15,
      "typical_weight_g": 20,
      "feature_notes": "Description of body shape",
      "individual_detection_quality": "good"
    }
  }
}
```

Then regenerate prototypes:
```bash
rm data/species_prototypes/*.ply
python3 src/generate_species_prototypes.py
```

### Skip Prototype Generation

Edit `oobe_setup.py` and comment out the prototype generation step, or just delete existing prototypes when prompted.

### Change Feeder Configuration

Edit `feeder_config.json` after OOBE completes:

```bash
nano feeder_config.json
```

Update:
- Feeder name and location
- API endpoints
- Monitoring settings
- Multi-feeder network options

## Troubleshooting

### OOBE Fails to Install Packages

**Manual installation:**
```bash
pip3 install numpy opencv-python torch torchvision ultralytics flask flask-cors
pip3 install open3d scipy scikit-learn  # Optional for 3D
```

### OOBE Runs Every Time

Check if marker file exists:
```bash
ls -la .oobe_completed
```

If missing, OOBE will run. If it keeps getting deleted, check:
- File permissions
- Any cleanup scripts
- Service configuration

### Want to Reset Everything

```bash
# Remove all OOBE artifacts
rm .oobe_completed
rm feeder_config.json
rm -rf data/species_prototypes/

# Re-run OOBE
python3 oobe_setup.py
```

### Prototypes Not Generating

Check dependencies:
```bash
python3 -c "import numpy; print('numpy OK')"
```

Manually generate:
```bash
python3 src/generate_species_prototypes.py \
  --config species_3d_support.json \
  --output data/species_prototypes
```

### Hardware Not Detected

OOBE hardware detection is informational only. The actual detection happens at runtime. If hardware isn't working:

1. **Check connections**: Verify cameras and AI Hat are connected
2. **Check drivers**: Ensure CS20 drivers installed
3. **Check permissions**: Add user to `video` group
4. **Check runtime logs**: Look for detection messages in startup

## OOBE Completion Marker

The `.oobe_completed` file contains:

```json
{
  "completed": true,
  "completed_date": "2024-01-17T12:00:00",
  "version": "1.0.0",
  "system_info": {
    "platform": "Linux",
    "is_raspberry_pi": true,
    "python_version": "3.9.2"
  },
  "detected_hardware": {
    "hailo": false,
    "usb_camera": true
  }
}
```

This file:
- ✅ Prevents re-running OOBE
- ✅ Stores setup information
- ✅ Can be used by monitoring tools
- ✅ Helps with troubleshooting

## Integration with Startup

The startup script (`start_detection_system.sh`) automatically:

1. Checks for `.oobe_completed`
2. Runs `oobe_setup.py` if missing
3. Continues to normal startup after OOBE

This ensures:
- ✅ Zero-configuration first boot
- ✅ Automatic dependency installation
- ✅ No manual setup required
- ✅ System works out of the box

## Future App Integration

The feeder configuration is designed for future mobile app connectivity:

### Phase 1 (Current)
- ✅ Feeder ID generation
- ✅ Configuration framework
- ✅ Local API endpoint

### Phase 2 (Planned)
- 🔜 Mobile app connection
- 🔜 Cloud sync support
- 🔜 Authentication system
- 🔜 Remote monitoring

### Phase 3 (Future)
- 🔜 Multi-feeder networks
- 🔜 Shared individual databases
- 🔜 Cross-feeder analytics
- 🔜 Bird migration tracking

## Security Considerations

### API Keys
When app connectivity is enabled, OOBE creates placeholders for:
- API keys
- Authentication tokens
- User credentials

**Never commit these to version control!**

Add to `.gitignore`:
```
feeder_config.json
.oobe_completed
data/
logs/
```

### Network Configuration
Default configuration:
- ✅ Local network only (`0.0.0.0:5000`)
- ✅ No authentication required
- ✅ Suitable for trusted home networks

For remote access, enable authentication in `feeder_config.json`.

## Performance Impact

OOBE has minimal performance impact:

- **Runtime**: 30-60 seconds (first time only)
- **Storage**: ~50MB for prototypes
- **CPU**: Negligible after completion
- **Memory**: No runtime overhead

The completion marker check is instant (<1ms).

## Logs

OOBE logs to console and includes:
- ✅ Dependency check results
- ✅ Installation progress
- ✅ Hardware detection
- ✅ Prototype generation stats
- ✅ Any errors or warnings

Save logs for troubleshooting:
```bash
python3 oobe_setup.py 2>&1 | tee oobe_setup.log
```

## Support

If OOBE fails:

1. **Check logs**: Review console output
2. **Check Python**: Ensure Python 3.7+
3. **Check internet**: Package installation needs network
4. **Check permissions**: Some operations need write access
5. **Manual setup**: Install dependencies manually if needed

For issues, check:
- `oobe_setup.py` - Main OOBE script
- `species_3d_support.json` - Species configuration
- `.oobe_completed` - Completion status

## Summary

The OOBE system provides:

- 🚀 **Zero-config setup** - Works out of the box
- 🔧 **Smart detection** - Only runs when needed
- 📦 **Auto-install** - Handles dependencies
- 🐦 **Species support** - Pre-generates 20+ bird models
- 🔌 **Future-ready** - App connectivity framework
- 🔐 **Idempotent** - Safe to run multiple times
- 📊 **Informative** - Clear progress and status

Just power on your Raspberry Pi and start detecting birds! 🐦
