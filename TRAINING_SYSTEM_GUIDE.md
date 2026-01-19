# Automated Self-Learning Training System Guide

## Overview

The Ornimetrics Training System is a **fully autonomous** self-learning pipeline that continuously improves 3D bird recognition models without user intervention.

### Key Features

- 🌙 **Night-time training** - Runs automatically during scheduled hours
- 🔄 **Continuous learning** - Learns from captured bird data
- ✅ **Quality control** - Validates data before training
- 🧪 **Self-testing** - Tests models before deployment
- 🚀 **Auto-deployment** - Deploys successful models automatically
- 🔙 **Version control** - Keeps previous model for rollback
- 📊 **Web dashboard** - Monitor training progress live
- 🤖 **Zero maintenance** - No user input required

---

## How It Works

### Daily Cycle

```
06:00 - 23:00  │  DAY MODE - Detection & Data Collection
               │  • Full 3D detection active
               │  • High-quality bird samples collected
               │  • Data validated and stored
               │
23:00 - 05:00  │  NIGHT MODE - Training & Testing
               │  • System enters hibernation mode
               │  • Detection continues at reduced rate (2 FPS)
               │  • TOF display disabled in web UI
               │  • CPU trains 3D model on collected data
               │  • New model tested against baseline
               │  • Successful model deployed automatically
               │
05:00          │  RETURN TO DAY MODE
               │  • New model active (if training succeeded)
               │  • Full detection resumes
               │  • Improved individual recognition
```

---

## System Components

### 1. Data Validator (`src/data_validator.py`)

**Quality Control Pipeline:**

✅ **Point Count Check**
- Minimum: 200 points
- Maximum: 10,000 points
- Optimal: 1000-3000 points

✅ **Planarity Filter**
- Rejects flat surfaces (photos, printouts)
- Uses PCA eigenvalue analysis
- Max planarity: 0.3

✅ **Thickness Validation**
- Ensures 3D volume (0.01m - 0.30m)
- Rejects paper-thin objects
- Typical birds: 0.05m - 0.15m

✅ **Size Validation**
- Min size: 0.03m (3cm)
- Max size: 0.35m (35cm)
- Filters out non-bird objects

✅ **YOLO Confidence**
- Minimum: 0.6 confidence
- Ensures species accuracy
- Prevents misclassification

✅ **Aspect Ratio Check**
- Bird-like proportions
- Length > width > height
- Rejects anomalous shapes

✅ **3D Structure Check**
- Not just noise or random points
- Proper clustering
- Coefficient of variation: 0.2-1.0

✅ **Anomaly Detection**
- Statistical outlier detection
- MAD (Median Absolute Deviation)
- Rejects if >10% outliers

**Result:** Only high-quality, bird-only data enters training

---

### 2. Training Dataset (`src/data_validator.py` - TrainingDataset)

**Data Management:**

```
data/training_samples/
├── Cardinal/
│   ├── 20240117_143052_892.npy  # Quality score: 0.892
│   ├── 20240117_151423_947.npy  # Quality score: 0.947
│   └── ...
├── Blue_Jay/
│   ├── 20240117_093311_865.npy
│   └── ...
└── index.json  # Metadata for all samples
```

**Index Structure:**
```json
{
  "Cardinal": [
    {
      "sample_id": "20240117_143052_892",
      "path": "Cardinal/20240117_143052_892.npy",
      "timestamp": 1705502052.5,
      "quality_score": 0.892,
      "yolo_confidence": 0.89,
      "num_points": 1523,
      "metadata": {
        "individual_id": 3,
        "validation": {...}
      }
    }
  ]
}
```

**Automatic Collection:**
- Samples added during detection
- Only validated data saved
- Filename includes quality score
- Organized by species

---

### 3. Model Trainer (`src/model_trainer.py`)

**Training Pipeline:**

**Configuration:**
```json
{
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.001,
  "early_stopping_patience": 10
}
```

**Features:**
- DGCNN Heavy architecture (512D embeddings)
- Data augmentation (rotation, jitter, scale)
- Train/validation split (80/20)
- Learning rate scheduling
- Early stopping
- Regular checkpointing

**During Training:**
- Runs on CPU (Hailo handles detection)
- Monitors progress every epoch
- Saves best model automatically
- Reports loss and accuracy
- Updates web dashboard live

---

### 4. Model Tester (`src/model_tester.py`)

**Self-Testing System:**

**Tests Performed:**

✅ **Baseline Accuracy Test**
- Minimum: 85% accuracy required
- Tests against holdout set
- Per-species accuracy checked

✅ **Improvement Test**
- Compares to previous model
- Must not degrade significantly (<5% drop)
- Ideally improves by >2%

✅ **Per-Species Test**
- No species below 60% accuracy
- Ensures no species "forgotten"
- Prevents catastrophic forgetting

✅ **Quality Test**
- Tests on high-quality samples only
- 20 samples per species minimum
- Ensures robust performance

**Test Results Saved:**
```json
{
  "passed": true,
  "overall_accuracy": 0.892,
  "per_species_accuracy": {
    "Cardinal": 0.95,
    "Blue_Jay": 0.88,
    ...
  },
  "test_duration_seconds": 45.2
}
```

---

### 5. Training Orchestrator (`src/training_orchestrator.py`)

**Complete Automation:**

**Schedule-Based Triggering:**
- Monitors time continuously
- Checks timezone configuration
- Triggers training at scheduled hours
- Prevents multiple runs per day

**5-Step Pipeline:**

**Step 1: Data Preparation (0-20%)**
- Load training samples
- Count per species
- Verify minimum samples met
- Create train/val split

**Step 2: Model Initialization (20-30%)**
- Initialize DGCNN Heavy
- Load architecture
- Set up optimizer
- Prepare data loaders

**Step 3: Training (30-80%)**
- Train for configured epochs
- Track loss and accuracy
- Update progress live
- Save checkpoints
- Early stopping if needed

**Step 4: Testing (80-90%)**
- Load best checkpoint
- Run comprehensive tests
- Compare to previous model
- Validate all criteria

**Step 5: Deployment (90-100%)**
- Backup current model
- Deploy new model
- Save deployment info
- Resume normal operation

**Failure Handling:**
- If tests fail, keep old model
- Log failure reasons
- Retry next night
- Alert via web dashboard

---

## Configuration

### Main Config (`training_config.json`)

```json
{
  "training": {
    "enabled": true,
    "schedule": {
      "start_time": "23:00",
      "end_time": "05:00",
      "timezone": "America/New_York"
    },
    "data_collection": {
      "min_samples_per_species": 50,
      "max_samples_per_species": 1000,
      "quality_threshold": 0.7
    },
    "model": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001
    },
    "validation": {
      "baseline_accuracy_threshold": 0.85,
      "improvement_threshold": 0.02
    },
    "deployment": {
      "auto_deploy_on_pass": true,
      "keep_previous_versions": 1,
      "rollback_on_accuracy_drop": true
    }
  },
  "night_mode": {
    "enabled": true,
    "disable_tof_display": true,
    "detection_fps": 2
  }
}
```

---

## Web Dashboard Integration

### Training Status Card

**Real-Time Information:**

📊 **Training Status**
- Idle / Preparing / Training / Testing / Completed / Failed
- Color-coded status badges
- Current operation displayed

📈 **Progress Bar**
- 0-100% completion
- Current step indicator
- Estimated time remaining
- Epoch counter (e.g., "Epoch 15/50")

💾 **Dataset Info**
- Total samples collected
- Samples per species
- Readiness for training
- Min samples required

🌙 **Night Mode Indicator**
- Shows when night mode active
- TOF display disabled indicator
- Training schedule display

**Controls:**
- **Start Training Now** - Manual trigger
- **Rollback Model** - Revert to previous version

**Updates:** Status refreshes every 5 seconds

---

## REST API Endpoints

### GET /api/training/status

**Response:**
```json
{
  "training_enabled": true,
  "is_training": false,
  "night_mode_active": false,
  "should_train_now": false,
  "training_status": {
    "status": "idle",
    "progress": 0.0,
    "message": "Waiting for scheduled time"
  },
  "dataset_ready": true,
  "species_counts": {
    "Cardinal": 127,
    "Blue_Jay": 89
  },
  "active_model_exists": true,
  "last_training_time": "2024-01-17T02:30:00"
}
```

### POST /api/training/start

Manually trigger training (even outside schedule).

**Response:**
```json
{
  "success": true,
  "message": "Training started"
}
```

### POST /api/training/rollback

Rollback to previous model version.

**Response:**
```json
{
  "success": true,
  "message": "Rolled back to previous model"
}
```

### GET /api/training/dataset

Get dataset statistics.

**Response:**
```json
{
  "enabled": true,
  "species_counts": {
    "Cardinal": 127,
    "Blue_Jay": 89
  },
  "total_samples": 216,
  "is_ready": true,
  "min_samples_required": 50
}
```

---

## Model Versioning

### File Structure

```
models/
├── active_model.pth           # Currently deployed model
├── previous_model.pth         # Backup (for rollback)
├── deployment_info.json       # Deployment metadata
├── checkpoints/
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   └── best_model.pth         # Best from training
└── test_results/
    ├── test_results_20240117_033022.json
    └── test_results_20240118_032544.json
```

### Deployment Info

```json
{
  "deployed_at": "2024-01-17T03:45:12",
  "model_path": "models/checkpoints/best_model.pth",
  "test_results": {
    "overall_accuracy": 0.892,
    "per_species_accuracy": {...}
  },
  "version": "20240117_034512"
}
```

### Rollback Process

1. Swap `active_model.pth` ↔ `previous_model.pth`
2. System uses rolled-back model immediately
3. Can roll forward again if needed
4. Only keeps 1 previous version (saves space)

---

## Usage Examples

### Enable Training

Edit `training_config.json`:
```json
{
  "training": {
    "enabled": true
  }
}
```

Restart system:
```bash
sudo systemctl restart ornimetrics-detection
```

### Change Schedule

```json
{
  "training": {
    "schedule": {
      "start_time": "22:00",  // Start 10 PM
      "end_time": "06:00",    // End 6 AM
      "timezone": "America/Los_Angeles"
    }
  }
}
```

### Manual Training

Via web dashboard:
1. Navigate to `http://raspberry-pi-ip:5000/`
2. Scroll to "Self-Learning System" card
3. Click "Start Training Now"
4. Confirm dialog
5. Monitor progress in real-time

Via API:
```bash
curl -X POST http://raspberry-pi-ip:5000/api/training/start
```

### Check Status

```bash
# Via API
curl http://raspberry-pi-ip:5000/api/training/status | jq

# Via logs
sudo journalctl -u ornimetrics-detection -f | grep TRAINING
```

### Rollback Model

If new model performs worse:

Via web dashboard:
1. Click "Rollback Model"
2. Confirm
3. Previous model activated immediately

Via API:
```bash
curl -X POST http://raspberry-pi-ip:5000/api/training/rollback
```

---

## Performance Considerations

### Resource Usage

**During Day (Detection Mode):**
- CPU: 60-80% (PyTorch) or 20-30% (Hailo)
- RAM: 2-3 GB
- Hailo: 100% (if available)

**During Night (Training Mode):**
- CPU: 90-100% (training)
- RAM: 3-4 GB
- Hailo: Idle or light detection duty
- Training duration: 2-6 hours (typical)

### Optimization Tips

**Faster Training:**
- Reduce epochs: 30 instead of 50
- Increase batch size: 32 (if RAM allows)
- Use light backbone: 256D instead of 512D

**Better Accuracy:**
- More epochs: 75-100
- More samples: 100+ per species
- Data augmentation: Enabled (default)

**Less Resource Usage:**
- Smaller batch size: 8
- Fewer epochs: 30
- Training every 2-3 days instead of daily

---

## Troubleshooting

### Training Never Starts

**Check:**
1. Training enabled: `training.enabled = true` in config
2. Enough data: 50+ samples per species minimum
3. Time window: Currently in scheduled hours?
4. Already trained today?

**Solution:**
```bash
# Check status
curl http://localhost:5000/api/training/status | jq

# Check dataset
curl http://localhost:5000/api/training/dataset | jq

# Manual trigger (bypass schedule)
curl -X POST http://localhost:5000/api/training/start
```

### Training Fails Tests

**Check Logs:**
```bash
sudo journalctl -u ornimetrics-detection -n 100 | grep -A 10 "TRAINING"
```

**Common Reasons:**
- Accuracy below 85% threshold
- Model degraded compared to previous
- Some species <60% accuracy
- Not enough test samples

**Solutions:**
- Collect more diverse data
- Adjust thresholds in config
- Check for bad data in dataset
- Review test results JSON files

### Model Worse After Training

**Immediate Rollback:**
```bash
curl -X POST http://localhost:5000/api/training/rollback
```

**Prevention:**
- Increase `improvement_threshold` to 0.05
- Enable `rollback_on_accuracy_drop`
- Raise `baseline_accuracy_threshold` to 0.90

### Out of Memory During Training

**Reduce Resources:**
```json
{
  "model": {
    "batch_size": 8,  // Down from 16
    "epochs": 30      // Down from 50
  }
}
```

### Training Takes Too Long

**Speed Up:**
```json
{
  "model": {
    "epochs": 30,
    "early_stopping_patience": 5
  }
}
```

Or disable training:
```json
{
  "training": {
    "enabled": false
  }
}
```

---

## Monitoring & Maintenance

### Check Training History

```bash
# View test results
ls -lh models/test_results/
cat models/test_results/test_results_*.json | jq

# View deployment info
cat models/deployment_info.json | jq
```

### Monitor Live Training

```bash
# Follow logs
sudo journalctl -u ornimetrics-detection -f

# Watch web dashboard
# Navigate to http://raspberry-pi-ip:5000/
```

### Dataset Maintenance

```bash
# Check dataset size
du -sh data/training_samples/

# Count samples per species
find data/training_samples/ -name "*.npy" | wc -l

# Clean old samples (if needed)
# Manual cleanup - remove oldest low-quality samples
```

### Backup Important Models

```bash
# Backup current models
cp models/active_model.pth backups/active_model_$(date +%Y%m%d).pth
cp models/previous_model.pth backups/previous_model_$(date +%Y%m%d).pth
```

---

## Advanced Configuration

### Custom Training Schedule

**Weekend Training:**
```json
{
  "schedule": {
    "mode": "custom",
    "allowed_days": [0, 6],  // Sunday, Saturday
    "start_time": "20:00",
    "end_time": "08:00"
  }
}
```

### Multi-Phase Training

**Curriculum Learning:**
```json
{
  "model": {
    "phase1_epochs": 20,
    "phase1_lr": 0.001,
    "phase2_epochs": 30,
    "phase2_lr": 0.0001
  }
}
```

### Species-Specific Training

**Focus on Specific Birds:**
```json
{
  "data_collection": {
    "priority_species": ["Cardinal", "Blue_Jay"],
    "min_samples_priority": 100,
    "min_samples_others": 30
  }
}
```

---

## Benefits

### Continuous Improvement

✅ Model accuracy increases over time
✅ Adapts to your specific birds
✅ Learns individual variations
✅ Handles seasonal changes (plumage)

### Zero Maintenance

✅ No manual retraining needed
✅ No model updates to install
✅ No expertise required
✅ Runs autonomously

### Quality Assurance

✅ Only validated data used
✅ Models tested before deployment
✅ Previous model kept for safety
✅ Automatic rollback if issues

### Efficiency

✅ Uses idle CPU time (night)
✅ Doesn't interfere with detection
✅ Optimizes resource usage
✅ Scales with your data

---

## Summary

The training system provides:

🤖 **Fully Automated** - No user intervention needed
🌙 **Night-Time Learning** - Trains while you sleep
✅ **Quality Controlled** - Only best data used
🧪 **Self-Tested** - Validates before deployment
🔄 **Continuous Improvement** - Gets better over time
📊 **Transparent** - Monitor via web dashboard
🔙 **Safe** - Rollback capability built-in
🚀 **Production Ready** - Used in real deployment

**Just enable it and let it learn!** 🐦🧠
