#!/usr/bin/env python3
"""
Training System Integration for Web Detection Server

Add these imports and modifications to web_detection_server.py to integrate
the automated training system.

This file shows the additions needed - merge with existing web_detection_server.py
"""

# ==============================================================================
# ADD TO IMPORTS SECTION (after existing imports)
# ==============================================================================

# Import training system
try:
    from src.training_orchestrator import TrainingOrchestrator
    TRAINING_AVAILABLE = True
except Exception as e:
    TRAINING_AVAILABLE = False
    print(f"[WARN] Training system not available: {e}")


# ==============================================================================
# ADD TO DetectionState class
# ==============================================================================

class DetectionState:
    def __init__(self):
        # ... existing fields ...

        # Training system
        self.training_orchestrator = None
        self.training_enabled = False


# ==============================================================================
# ADD TO initialize_hardware() function
# ==============================================================================

def initialize_hardware():
    """... existing hardware initialization ..."""

    # Initialize training orchestrator
    if TRAINING_AVAILABLE and cfg.get("training", {}).get("enabled", False):
        logger.info("Initializing training orchestrator...")
        try:
            state.training_orchestrator = TrainingOrchestrator("training_config.json")
            state.training_enabled = True

            # Start monitoring thread
            training_monitor_thread = threading.Thread(
                target=state.training_orchestrator.run_monitoring_loop,
                daemon=True
            )
            training_monitor_thread.start()

            logger.info("Training orchestrator initialized and monitoring started")
        except Exception as e:
            logger.error(f"Failed to initialize training orchestrator: {e}")
            state.training_enabled = False
    else:
        logger.info("Training system disabled")


# ==============================================================================
# ADD TO process_detection_frame() function
# ==============================================================================

def process_detection_frame(frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """... existing detection code ..."""

    # Check if in night mode
    if state.training_orchestrator and state.training_orchestrator.night_mode_active:
        # Night mode: Show reduced UI, collect data for training
        cv2.putText(frame, "NIGHT MODE - Training Data Collection", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    # Add training data sample (if available and valid)
    if state.training_orchestrator and individual_info:
        if individual_info.get("can_dispense") or individual_info.get("is_new"):
            # This is a good quality sample
            try:
                state.training_orchestrator.add_training_sample(
                    points=depth_frame_points,  # From backproject_depth
                    species=label,
                    yolo_confidence=conf,
                    timestamp=time.time(),
                    metadata={
                        "individual_id": individual_info.get("individual_id"),
                        "validation": individual_info.get("validation")
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to add training sample: {e}")


# ==============================================================================
# ADD NEW API ENDPOINTS
# ==============================================================================

@app.route('/api/training/status')
def api_training_status():
    """Get training system status."""
    if not state.training_enabled or not state.training_orchestrator:
        return jsonify({
            "enabled": False,
            "message": "Training system not available"
        })

    status = state.training_orchestrator.get_status()
    return jsonify(status)


@app.route('/api/training/start', methods=['POST'])
def api_training_start():
    """Manually trigger training."""
    if not state.training_enabled or not state.training_orchestrator:
        return jsonify({"success": False, "message": "Training system not available"}), 400

    if state.training_orchestrator.is_training:
        return jsonify({"success": False, "message": "Training already in progress"}), 400

    try:
        state.training_orchestrator.start_training_async()
        return jsonify({"success": True, "message": "Training started"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/training/rollback', methods=['POST'])
def api_training_rollback():
    """Rollback to previous model."""
    if not state.training_enabled or not state.training_orchestrator:
        return jsonify({"success": False, "message": "Training system not available"}), 400

    success = state.training_orchestrator.rollback_model()
    if success:
        return jsonify({"success": True, "message": "Rolled back to previous model"})
    else:
        return jsonify({"success": False, "message": "Rollback failed"}), 500


@app.route('/api/training/dataset')
def api_training_dataset():
    """Get training dataset information."""
    if not state.training_enabled or not state.training_orchestrator:
        return jsonify({"enabled": False})

    species_counts = state.training_orchestrator.dataset.get_species_count()
    min_samples = state.training_orchestrator.config["training"]["data_collection"].get("min_samples_per_species", 50)
    is_ready, info = state.training_orchestrator.dataset.is_ready_for_training(min_samples)

    return jsonify({
        "enabled": True,
        "species_counts": species_counts,
        "total_samples": sum(species_counts.values()),
        "is_ready": is_ready,
        "ready_info": info,
        "min_samples_required": min_samples
    })


# ==============================================================================
# UPDATE DASHBOARD HTML to include training status
# ==============================================================================

TRAINING_STATUS_HTML = """
<!-- Training System Status Card -->
<div class="card" id="training-card" style="display: none;">
    <h2 style="margin-bottom: 15px;">🧠 Self-Learning System</h2>

    <!-- Training Status -->
    <div id="training-status-container">
        <div class="status-badge status-inactive">Initializing...</div>
    </div>

    <!-- Training Progress (shown when training) -->
    <div id="training-progress" style="display: none; margin-top: 15px;">
        <div style="background: #243447; padding: 10px; border-radius: 5px;">
            <div style="margin-bottom: 5px; display: flex; justify-content: space-between;">
                <span id="training-progress-text">Training...</span>
                <span id="training-progress-pct">0%</span>
            </div>
            <div style="background: #1c2938; border-radius: 3px; height: 20px; overflow: hidden;">
                <div id="training-progress-bar" style="background: #5bc0de; height: 100%; width: 0%; transition: width 0.3s;"></div>
            </div>
            <div style="margin-top: 5px; font-size: 0.85em; color: #95a5a6;">
                <span id="training-epoch-info"></span>
            </div>
        </div>
    </div>

    <!-- Dataset Info -->
    <div style="margin-top: 15px;">
        <h3 style="font-size: 1em; margin-bottom: 10px;">Training Dataset</h3>
        <div class="stats-grid" style="grid-template-columns: repeat(2, 1fr);">
            <div class="stat-box">
                <div class="stat-value" id="training-samples">--</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="training-species">--</div>
                <div class="stat-label">Species</div>
            </div>
        </div>
    </div>

    <!-- Training Controls -->
    <div style="margin-top: 15px;">
        <h3 style="font-size: 1em; margin-bottom: 10px;">Controls</h3>
        <button class="control-btn" onclick="manualTriggerTraining()" id="train-btn">
            Start Training Now
        </button>
        <button class="control-btn" onclick="rollbackModel()" id="rollback-btn" style="background: #f39c12;">
            Rollback Model
        </button>
    </div>

    <!-- Night Mode Indicator -->
    <div id="night-mode-indicator" style="margin-top: 15px; display: none;">
        <div style="background: #243447; padding: 10px; border-radius: 5px; text-align: center;">
            <span style="font-size: 1.2em;">🌙</span>
            <span style="margin-left: 10px;">Night Mode Active</span>
        </div>
    </div>
</div>

<script>
// Training status update function
function updateTrainingStatus() {
    fetch('/api/training/status')
        .then(r => r.json())
        .then(data => {
            if (!data.enabled) {
                document.getElementById('training-card').style.display = 'none';
                return;
            }

            document.getElementById('training-card').style.display = 'block';

            // Update status badge
            let statusHtml = '';
            const status = data.training_status.status;

            if (status === 'idle') {
                statusHtml = '<div class="status-badge status-inactive">Idle - Waiting for schedule</div>';
            } else if (status === 'training') {
                statusHtml = '<div class="status-badge status-active">Training In Progress</div>';
            } else if (status === 'testing') {
                statusHtml = '<div class="status-badge status-active">Testing Model</div>';
            } else if (status === 'completed') {
                statusHtml = '<div class="status-badge status-active">Training Completed</div>';
            } else if (status === 'failed') {
                statusHtml = '<div class="status-badge status-inactive" style="background: #e74c3c;">Training Failed</div>';
            }

            document.getElementById('training-status-container').innerHTML = statusHtml;

            // Show/hide progress bar
            const showProgress = ['preparing', 'initializing', 'training', 'testing', 'deploying'].includes(status);
            document.getElementById('training-progress').style.display = showProgress ? 'block' : 'none';

            if (showProgress) {
                const progress = data.training_status.progress || 0;
                document.getElementById('training-progress-bar').style.width = progress + '%';
                document.getElementById('training-progress-pct').textContent = progress.toFixed(0) + '%';
                document.getElementById('training-progress-text').textContent = data.training_status.message || 'Processing...';

                if (data.training_status.current_epoch && data.training_status.total_epochs) {
                    document.getElementById('training-epoch-info').textContent =
                        `Epoch ${data.training_status.current_epoch}/${data.training_status.total_epochs}`;
                }
            }

            // Night mode indicator
            document.getElementById('night-mode-indicator').style.display =
                data.night_mode_active ? 'block' : 'none';

            // Disable training button if training active
            document.getElementById('train-btn').disabled = data.is_training;
        })
        .catch(err => console.error('Failed to fetch training status:', err));

    // Update dataset info
    fetch('/api/training/dataset')
        .then(r => r.json())
        .then(data => {
            if (data.enabled) {
                document.getElementById('training-samples').textContent = data.total_samples || 0;
                document.getElementById('training-species').textContent = Object.keys(data.species_counts || {}).length;
            }
        });
}

// Manual training trigger
function manualTriggerTraining() {
    if (!confirm('Manually start training now? This may take several hours.')) {
        return;
    }

    fetch('/api/training/start', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                alert('Training started successfully!');
            } else {
                alert('Failed to start training: ' + data.message);
            }
        });
}

// Rollback model
function rollbackModel() {
    if (!confirm('Rollback to previous model version? This will replace the current model.')) {
        return;
    }

    fetch('/api/training/rollback', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                alert('Successfully rolled back to previous model');
            } else {
                alert('Rollback failed: ' + data.message);
            }
        });
}

// Update training status every 5 seconds
setInterval(updateTrainingStatus, 5000);
updateTrainingStatus();  // Initial call
</script>
"""

# Add TRAINING_STATUS_HTML to the main dashboard HTML where appropriate
