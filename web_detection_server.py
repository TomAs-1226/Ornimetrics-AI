#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_detection_server.py

Web server that streams bird detection with 3D individual recognition.
Provides both a web UI and REST API for external app integration.

Features:
- Live MJPEG video stream with detections overlaid
- Real-time stats (FPS, mode, hardware status)
- Individual bird database viewer
- REST API endpoints for external apps
- Auto-detects all hardware (Hailo, CS20, etc.)
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

# Environment setup
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from flask import Flask, render_template_string, Response, jsonify, request

# Import detection components
try:
    from src.hailo_detector import HailoYOLODetector
    HAILO_AVAILABLE = True
except Exception:
    HAILO_AVAILABLE = False

try:
    from ultralytics import YOLO as PyTorchYOLO
    PYTORCH_AVAILABLE = True
except Exception:
    PYTORCH_AVAILABLE = False
    print("[ERROR] PyTorch/Ultralytics not available")
    sys.exit(2)

# Import 3D processing
try:
    from birdid.camera.cs20 import CS20Camera
    from src.depth_to_points import backproject_depth
    from src.reid_embedder import PointReID
    from src.pc_preprocess import PreprocessConfig
    from birdid.db import BirdIDDatabase
    from birdid.validate_3d import validate_point_cloud
    POINT_CLOUD_AVAILABLE = True
except Exception as e:
    POINT_CLOUD_AVAILABLE = False
    print(f"[WARN] Point cloud modules not available: {e}")

# Import servo and Firebase
try:
    from servoMain_fixed import ServoTrap
except Exception:
    try:
        from servoMain import ServoTrap
    except Exception as e:
        print(f"[ERROR] Servo module not available: {e}")
        sys.exit(2)

try:
    from firebase_logger import FirebaseLogger
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ornimetrics-bird-detection-2024'

# Global state
class DetectionState:
    def __init__(self):
        self.detector = None
        self.depth_camera = None
        self.individual_tracker = None
        self.trap = None
        self.flog = None
        self.birdid_db = None

        # Hardware status
        self.has_3d = False
        self.has_hailo = False
        self.has_individual_id = False
        self.backend_name = "Unknown"

        # Stream state
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Stats
        self.fps_history = deque(maxlen=120)
        self.detection_history = deque(maxlen=100)
        self.recent_individuals = {}  # individual_id -> last_seen_info
        self.stats = {
            'total_detections': 0,
            'total_triggers': 0,
            'total_enrollments': 0,
            'uptime_start': time.time(),
        }

        # Config
        self.config = {}

        # Control flags
        self.running = False
        self.detection_enabled = True

state = DetectionState()


def load_config(config_path: str = "config_3d_detection.json") -> Dict:
    """Load configuration from JSON file."""
    try:
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")

    # Default config (matches simplified config_3d_detection.json format)
    return {
        "detection": {
            "model_path": "weights.pt",
            "confidence_threshold": 0.45,
            "input_size": 320
        },
        "camera": {
            "rgb": {"source": 0, "width": 640, "height": 480},
            "rgb_source": 0, "width": 640, "height": 480,
            "depth": {"enabled": True, "mode": "320x240"}
        },
        "depth": {"enabled": True, "mode": "320x240"},
        "point_cloud": {
            "backbone": "light",
            "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
            "preprocessing": {
                "plane_removal": True,
                "normalization": "center_only",
                "voxel_size": 0.01,
                "fps_points": 1024,
                "depth_gate_k": 2.5
            }
        },
        "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
        "individual_tracking": {
            "enabled": True,
            "database_path": "birdid.sqlite",
            "match_threshold": 0.32,
            "cooldown_seconds": 300,
            "max_dispenses_per_day": 20
        },
        "tracking": {
            "database_path": "birdid.sqlite",
            "match_threshold": 0.32,
            "cooldown_seconds": 300,
            "max_per_day": 20
        },
        "web": {
            "host": "0.0.0.0",
            "port": 5000,
            "stream_fps": 15
        }
    }


def initialize_hardware():
    """Initialize all hardware components with auto-detection."""
    logger.info("Initializing hardware...")

    cfg = state.config

    # Initialize YOLO detector
    model_path = cfg["detection"]["model_path"]
    if HAILO_AVAILABLE and Path(model_path).suffix == ".hef":
        logger.info("Attempting Hailo-8 accelerator...")
        state.detector = HailoYOLODetector(
            model_path=model_path,
            fallback_pytorch_model=model_path.replace(".hef", ".pt"),
            confidence_threshold=cfg["detection"]["confidence_threshold"],
            input_size=cfg["detection"]["input_size"]
        )
        state.has_hailo = state.detector.is_hailo_active()
        state.backend_name = state.detector.get_backend_name()
    else:
        logger.info("Using PyTorch backend...")
        state.detector = HailoYOLODetector(
            model_path=None,
            fallback_pytorch_model=model_path,
            confidence_threshold=cfg["detection"]["confidence_threshold"],
            input_size=cfg["detection"]["input_size"]
        )
        state.backend_name = "PyTorch CPU"

    logger.info(f"Detection backend: {state.backend_name}")

    # Initialize 3D camera if enabled
    if cfg["camera"]["depth"]["enabled"] and POINT_CLOUD_AVAILABLE:
        logger.info("Attempting to initialize CS20 depth camera...")
        try:
            state.depth_camera = CS20Camera(
                mode=cfg["camera"]["depth"]["mode"],
                fallback_to_synthetic=False
            )
            state.depth_camera.start()
            time.sleep(0.5)

            if state.depth_camera.is_hardware_available():
                state.has_3d = True
                logger.info("CS20 depth camera initialized successfully")

                # Initialize individual tracker
                if cfg["individual_tracking"]["enabled"]:
                    logger.info("Initializing individual bird tracker...")
                    state.birdid_db = BirdIDDatabase(cfg["individual_tracking"]["database_path"])

                    preprocess_cfg = PreprocessConfig(
                        plane_removal=True,
                        normalization="center_only",
                        voxel_size=0.01,
                        fps_points=2048,
                        depth_gate_k=2.5
                    )

                    from detect_3d_individual import IndividualBirdTracker
                    state.individual_tracker = IndividualBirdTracker(
                        db=state.birdid_db,
                        embedder=PointReID(preprocess_config=preprocess_cfg, model_name=cfg["point_cloud"]["backbone"]),
                        intrinsics=cfg["point_cloud"]["intrinsics"],
                        match_threshold=cfg["individual_tracking"]["match_threshold"],
                        cooldown_seconds=cfg["individual_tracking"]["cooldown_seconds"],
                        max_per_day=cfg["individual_tracking"]["max_dispenses_per_day"]
                    )
                    state.has_individual_id = True
                    logger.info(f"Individual tracker initialized (backbone: {cfg['point_cloud']['backbone']})")
            else:
                logger.info("CS20 hardware not detected - using optics-only mode")
                state.depth_camera.stop()
                state.depth_camera = None
        except Exception as e:
            logger.warning(f"Failed to initialize depth camera: {e}")
            if state.depth_camera:
                state.depth_camera.stop()
            state.depth_camera = None
    else:
        logger.info("3D mode disabled - using optics-only mode")

    # Initialize servo trap
    logger.info("Initializing servo trap...")
    state.trap = ServoTrap(
        channel=1,
        closed_angle=5.0,
        open_angle=60.0,
        min_inter_trigger_sec=1.5,
        debug=False,
        slew_deg_per_s=240.0,
        min_us=1000,
        max_us=2000
    )

    # Initialize Firebase
    if FIREBASE_AVAILABLE:
        try:
            state.flog = FirebaseLogger("https://ornimetrics-default-rtdb.firebaseio.com", session_key="web_session")
            logger.info("Firebase enabled")
        except Exception as e:
            logger.warning(f"Firebase init failed: {e}")

    mode = "Full 3D" if state.has_3d and state.has_individual_id else "Optics-Only"
    logger.info(f"System ready: {mode} | {state.backend_name}")


def process_detection_frame(frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Process a frame and return annotated frame + detections."""
    if not state.detection_enabled:
        return frame, []

    t_start = time.time()

    # Run detection
    detections = state.detector.detect(frame)
    state.stats['total_detections'] += len(detections)

    # Get depth frame if available
    depth_frame = None
    if state.depth_camera:
        depth_frame = state.depth_camera.get_latest(timeout=0.05)

    # Process each detection
    processed_dets = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"]
        conf = det["confidence"]

        # Individual tracking
        individual_info = None
        if state.individual_tracker and depth_frame:
            individual_info = state.individual_tracker.process_detection(
                (x1, y1, x2, y2), label, depth_frame, time.time()
            )

            if individual_info:
                if individual_info.get("is_new"):
                    state.stats['total_enrollments'] += 1
                    logger.info(f"New bird enrolled: {individual_info['individual_name']}")

                # Track recent individuals
                ind_id = individual_info.get("individual_id")
                if ind_id:
                    state.recent_individuals[ind_id] = {
                        "name": individual_info.get("individual_name"),
                        "species": label,
                        "last_seen": time.time(),
                        "can_dispense": individual_info.get("can_dispense", False),
                        "reason": individual_info.get("reason", "")
                    }

        # Draw detection
        col = (0, 255, 0) if individual_info and individual_info.get("can_dispense") else (0, 128, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        text = f"{label} {conf:.2f}"
        if individual_info:
            text += f" | {individual_info.get('individual_name', 'Unknown')}"

        cv2.putText(frame, text, (x1, max(0, y1 - 6)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

        det_info = {
            "bbox": det["bbox"],
            "class_name": label,
            "confidence": conf,
            "timestamp": time.time(),
            "individual": individual_info
        }
        processed_dets.append(det_info)

        # Add to history
        state.detection_history.append({
            "time": datetime.now().isoformat(),
            "species": label,
            "confidence": conf,
            "individual_id": individual_info.get("individual_id") if individual_info else None,
            "individual_name": individual_info.get("individual_name") if individual_info else None
        })

    # Update FPS
    dt = time.time() - t_start
    if dt > 0:
        state.fps_history.append(1.0 / dt)

    # Draw stats overlay
    mode_text = "Mode: Full 3D" if state.has_3d else "Mode: Optics-Only"
    backend_text = f"Backend: {state.backend_name}"
    fps_text = f"FPS: {sum(state.fps_history) / len(state.fps_history):.1f}" if state.fps_history else "FPS: --"

    cv2.putText(frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, backend_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, fps_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, processed_dets


def detection_loop():
    """Main detection loop running in background thread."""
    logger.info("Starting detection loop...")

    # Open camera
    cfg = state.config
    cap = cv2.VideoCapture(cfg["camera"]["rgb"]["source"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["rgb"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["rgb"]["height"])
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while state.running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        state.trap.tick()

        # Process frame
        annotated_frame, detections = process_detection_frame(frame)

        # Update current frame for streaming
        with state.frame_lock:
            state.current_frame = annotated_frame.copy()

        # Small delay to control FPS
        time.sleep(1.0 / cfg["web"]["stream_fps"])

    cap.release()
    logger.info("Detection loop stopped")


def generate_frames():
    """Generator for MJPEG streaming."""
    while True:
        with state.frame_lock:
            if state.current_frame is None:
                # Send blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Initializing...", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            else:
                _, buffer = cv2.imencode('.jpg', state.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS max for stream


# HTML Templates
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bird Detection Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1419; color: #e4e6eb; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { color: #5bc0de; margin-bottom: 10px; }
        .subtitle { color: #95a5a6; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
        .card { background: #1c2938; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .video-container { position: relative; }
        .video-stream { width: 100%; border-radius: 8px; background: #000; }
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-box { background: #243447; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #5bc0de; }
        .stat-label { color: #95a5a6; margin-top: 5px; font-size: 0.9em; }
        .status-badge { display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 0.85em; margin: 5px; }
        .status-active { background: #27ae60; color: white; }
        .status-inactive { background: #e74c3c; color: white; }
        .status-fallback { background: #f39c12; color: white; }
        .individuals-list { max-height: 300px; overflow-y: auto; }
        .individual-item { background: #243447; padding: 10px; margin: 5px 0; border-radius: 5px; display: flex; justify-content: space-between; align-items: center; }
        .individual-name { font-weight: bold; color: #5bc0de; }
        .individual-status { font-size: 0.85em; color: #95a5a6; }
        .control-btn { background: #5bc0de; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; font-size: 1em; }
        .control-btn:hover { background: #4a9fc4; }
        .control-btn.danger { background: #e74c3c; }
        .control-btn.danger:hover { background: #c0392b; }
        .api-endpoint { background: #243447; padding: 10px; border-radius: 5px; margin: 10px 0; font-family: monospace; font-size: 0.9em; }
        .scrollable { max-height: 400px; overflow-y: auto; }
        @media (max-width: 968px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐦 Ornimetrics Bird Detection System</h1>
        <p class="subtitle">3D Individual Recognition with AI Acceleration</p>

        <div class="grid">
            <!-- Video Stream -->
            <div class="card video-container">
                <h2 style="margin-bottom: 15px;">Live Detection Stream</h2>
                <img src="/video_feed" class="video-stream" alt="Live Stream">
            </div>

            <!-- System Status -->
            <div class="card">
                <h2 style="margin-bottom: 15px;">System Status</h2>
                <div id="status">
                    <div class="status-badge status-active">Loading...</div>
                </div>

                <div class="stats-grid" style="margin-top: 20px;">
                    <div class="stat-box">
                        <div class="stat-value" id="fps">--</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="detections">--</div>
                        <div class="stat-label">Total Detections</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="triggers">--</div>
                        <div class="stat-label">Servo Triggers</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="enrollments">--</div>
                        <div class="stat-label">New Enrollments</div>
                    </div>
                </div>

                <h3 style="margin: 20px 0 10px 0;">Controls</h3>
                <button class="control-btn" onclick="toggleDetection()">Toggle Detection</button>
                <button class="control-btn danger" onclick="resetDatabase()">Reset Database</button>
            </div>
        </div>

        <!-- Individual Birds -->
        <div class="card" id="individuals-card" style="display: none;">
            <h2 style="margin-bottom: 15px;">Known Individual Birds</h2>
            <div class="individuals-list" id="individuals-list">
                <p style="color: #95a5a6;">No individuals tracked yet...</p>
            </div>
        </div>

        <!-- API Documentation -->
        <div class="card">
            <h2 style="margin-bottom: 15px;">API Endpoints (for your app)</h2>
            <div class="scrollable">
                <div class="api-endpoint">GET /api/status - System status and hardware info</div>
                <div class="api-endpoint">GET /api/stats - Detection statistics</div>
                <div class="api-endpoint">GET /api/individuals - List of known birds</div>
                <div class="api-endpoint">GET /api/detections/recent - Recent detection history</div>
                <div class="api-endpoint">POST /api/control/detection - Enable/disable detection</div>
                <div class="api-endpoint">GET /video_feed - MJPEG video stream</div>
            </div>
        </div>
    </div>

    <script>
        // Update stats every second
        setInterval(updateStats, 1000);

        function updateStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('detections').textContent = data.total_detections;
                    document.getElementById('triggers').textContent = data.total_triggers;
                    document.getElementById('enrollments').textContent = data.total_enrollments;
                });

            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    html += `<div class="status-badge ${data.mode === 'Full 3D' ? 'status-active' : 'status-fallback'}">${data.mode}</div>`;
                    html += `<div class="status-badge ${data.has_hailo ? 'status-active' : 'status-inactive'}">${data.backend}</div>`;
                    html += `<div class="status-badge ${data.has_3d ? 'status-active' : 'status-inactive'}">3D Camera: ${data.has_3d ? 'Active' : 'Offline'}</div>`;
                    document.getElementById('status').innerHTML = html;
                });

            fetch('/api/individuals')
                .then(r => r.json())
                .then(data => {
                    if (data.individuals && data.individuals.length > 0) {
                        document.getElementById('individuals-card').style.display = 'block';
                        let html = '';
                        data.individuals.forEach(ind => {
                            const status = ind.can_dispense ? 'Ready' : ind.reason;
                            const statusColor = ind.can_dispense ? '#27ae60' : '#e74c3c';
                            html += `
                                <div class="individual-item">
                                    <div>
                                        <div class="individual-name">${ind.name}</div>
                                        <div class="individual-status">${ind.species} • Last seen: ${new Date(ind.last_seen * 1000).toLocaleTimeString()}</div>
                                    </div>
                                    <div style="color: ${statusColor}; font-weight: bold;">${status}</div>
                                </div>
                            `;
                        });
                        document.getElementById('individuals-list').innerHTML = html;
                    }
                });
        }

        function toggleDetection() {
            fetch('/api/control/detection', { method: 'POST' })
                .then(r => r.json())
                .then(data => alert('Detection ' + (data.enabled ? 'enabled' : 'disabled')));
        }

        function resetDatabase() {
            if (confirm('This will delete all individual bird records. Continue?')) {
                fetch('/api/control/reset_db', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => alert(data.message));
            }
        }

        // Initial load
        updateStats();
    </script>
</body>
</html>
"""


# Flask Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """System status API."""
    return jsonify({
        "mode": "Full 3D" if state.has_3d and state.has_individual_id else "Optics-Only",
        "backend": state.backend_name,
        "has_hailo": state.has_hailo,
        "has_3d": state.has_3d,
        "has_individual_id": state.has_individual_id,
        "detection_enabled": state.detection_enabled,
        "uptime": time.time() - state.stats['uptime_start']
    })


@app.route('/api/stats')
def api_stats():
    """Detection statistics API."""
    fps = sum(state.fps_history) / len(state.fps_history) if state.fps_history else 0.0
    return jsonify({
        "fps": fps,
        "total_detections": state.stats['total_detections'],
        "total_triggers": state.stats['total_triggers'],
        "total_enrollments": state.stats['total_enrollments'],
        "uptime_seconds": time.time() - state.stats['uptime_start']
    })


@app.route('/api/individuals')
def api_individuals():
    """List of known individual birds API."""
    individuals = []
    for ind_id, info in state.recent_individuals.items():
        individuals.append({
            "id": ind_id,
            "name": info["name"],
            "species": info["species"],
            "last_seen": info["last_seen"],
            "can_dispense": info["can_dispense"],
            "reason": info["reason"]
        })
    individuals.sort(key=lambda x: x["last_seen"], reverse=True)
    return jsonify({"individuals": individuals})


@app.route('/api/detections/recent')
def api_recent_detections():
    """Recent detection history API."""
    return jsonify({"detections": list(state.detection_history)})


@app.route('/api/control/detection', methods=['POST'])
def api_control_detection():
    """Toggle detection on/off."""
    state.detection_enabled = not state.detection_enabled
    return jsonify({"enabled": state.detection_enabled})


@app.route('/api/control/reset_db', methods=['POST'])
def api_reset_database():
    """Reset individual bird database."""
    if state.birdid_db:
        try:
            # Close existing connection
            state.birdid_db.close()
            # Remove database file
            db_path = Path(state.config["individual_tracking"]["database_path"])
            if db_path.exists():
                db_path.unlink()
            # Reinitialize
            state.birdid_db = BirdIDDatabase(state.config["individual_tracking"]["database_path"])
            state.recent_individuals.clear()
            state.stats['total_enrollments'] = 0
            return jsonify({"success": True, "message": "Database reset successfully"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
    return jsonify({"success": False, "message": "3D mode not active"}), 400


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Web-based bird detection server")
    parser.add_argument("--config", default="config_3d_detection.json", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    # Load config
    state.config = load_config(args.config)

    # Override from args
    if args.host:
        state.config["web"]["host"] = args.host
    if args.port:
        state.config["web"]["port"] = args.port

    # Initialize hardware
    initialize_hardware()

    # Start detection loop in background thread
    state.running = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    # Start Flask server
    host = state.config["web"]["host"]
    port = state.config["web"]["port"]
    logger.info(f"Starting web server at http://{host}:{port}")
    logger.info(f"Dashboard: http://{host}:{port}/")
    logger.info(f"Video stream: http://{host}:{port}/video_feed")
    logger.info(f"API: http://{host}:{port}/api/status")

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        state.running = False
        if state.depth_camera:
            state.depth_camera.stop()
        if state.birdid_db:
            state.birdid_db.close()


if __name__ == "__main__":
    main()
