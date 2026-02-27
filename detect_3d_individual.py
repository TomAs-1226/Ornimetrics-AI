#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_3d_individual.py

Enhanced bird detection with 3D point cloud individual identification.

Features:
- Hailo-8 AI Hat accelerated YOLO inference (with PyTorch fallback)
- CS20 depth camera for 3D point cloud capture (with optics-only fallback)
- Individual bird recognition using spatial AI models (DGCNN)
- Firebase logging with individual bird tracking
- Servo trap control per-individual with cooldown management
- Graceful degradation to optics-only mode when 3D camera unavailable

Modes:
1. Full 3D Mode: YOLO + CS20 depth + individual recognition + servo
2. Optics-Only Mode: YOLO + servo (no individual tracking)
"""

import os
import sys
import time
import json
import argparse
from time import monotonic as now_mono
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

# Environment setup
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

# Import detection backends
try:
    from src.hailo_detector import HailoYOLODetector
    HAILO_AVAILABLE = True
except Exception as e:
    HAILO_AVAILABLE = False
    print(f"[WARN] Hailo detector not available: {e}")

try:
    from ultralytics import YOLO as PyTorchYOLO
    PYTORCH_AVAILABLE = True
except Exception as e:
    PYTORCH_AVAILABLE = False
    print(f"[ERROR] PyTorch/Ultralytics not available: {e}")
    sys.exit(2)

# Import 3D processing modules
try:
    from birdid.camera.cs20 import CS20Camera
    from src.depth_to_points import backproject_depth
    from src.reid_embedder import PointReID
    from src.pc_preprocess import PreprocessConfig
    from birdid.db import BirdIDDatabase
    from birdid.validate_3d import validate_point_cloud, quality_score
    POINT_CLOUD_AVAILABLE = True
except Exception as e:
    POINT_CLOUD_AVAILABLE = False
    print(f"[WARN] Point cloud modules not fully available: {e}")

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
    print("[WARN] Firebase logger not available")

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(p: str) -> str:
    """Resolve path relative to script directory."""
    q = Path(p)
    return str(q) if q.is_absolute() else str((SCRIPT_DIR / q).resolve())


def normalize_key(s: str) -> str:
    """Normalize string to lowercase with underscores."""
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


def load_trap_settings(
    path: str, hold_default: float, cooldown_default: float, conf_default: float
) -> Dict[str, dict]:
    """Load species-specific trap settings from JSON."""
    p = Path(path)
    if not p.is_file() and p.name == "trap_settings.json":
        cand = SCRIPT_DIR / "trap_settings.json"
        if cand.is_file():
            path = str(cand)
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("trap_settings.json is not a dict")
        return cfg
    except Exception as e:
        print(f"[WARN] Could not load trap_settings from {path}: {e}")
        return {
            "_default": {
                "open_duration": hold_default,
                "cooldown_duration": cooldown_default,
                "confidence_threshold": conf_default,
            }
        }


def get_action_for(
    label: str,
    cfg: Dict[str, dict],
    hold_default: float,
    cooldown_default: float,
    conf_default: float,
):
    """Get trap action settings for a given species label."""
    dflt = {
        "open_duration": hold_default,
        "cooldown_duration": cooldown_default,
        "confidence_threshold": conf_default,
    }
    raw = (label or "").strip()
    slug = normalize_key(raw)
    for k in (raw, slug, slug.replace("_", "-"), slug.replace("-", "_")):
        if k in cfg and isinstance(cfg[k], dict):
            return cfg[k]
    # Fallbacks
    if "bird" in cfg and "bird" in slug:
        return cfg["bird"]
    if "squirrel" in cfg and "squirrel" in slug:
        return cfg["squirrel"]
    if "critter" in cfg:
        return cfg["critter"]
    return cfg.get("_default", dflt) or dflt


class DetectionMode:
    """Tracks current detection mode and capabilities."""

    def __init__(self):
        self.has_3d_camera = False
        self.has_hailo = False
        self.has_individual_id = False
        self.backend_name = "Unknown"

    def __str__(self):
        mode = "Full 3D" if self.has_3d_camera and self.has_individual_id else "Optics-Only"
        return f"{mode} | {self.backend_name}"


class IndividualBirdTracker:
    """Tracks individual birds using 3D point clouds and DGCNN embeddings."""

    def __init__(
        self,
        db: "BirdIDDatabase",
        embedder: "PointReID",
        intrinsics: Dict[str, float],
        match_threshold: float = 0.32,
        cooldown_seconds: float = 300.0,
        max_per_day: int = 20,
    ):
        self.db = db
        self.embedder = embedder
        self.intrinsics = intrinsics
        self.match_threshold = match_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_per_day = max_per_day

    def process_detection(
        self, bbox: Tuple[int, int, int, int], species: str, depth_frame, now_ts: float
    ) -> Optional[Dict]:
        """Process a detection with point cloud analysis.

        Returns:
            Dict with individual_id, individual_name, is_new, can_dispense, reason
            None if processing fails
        """
        try:
            # Extract point cloud from depth
            if depth_frame is None:
                return None

            points, stats = backproject_depth(
                depth_frame.depth, self.intrinsics, bbox, valid_mask=depth_frame.valid
            )

            if points.shape[0] < 150:
                return {"reason": "insufficient_points", "can_dispense": False}

            # Validate 3D structure (anti-spoof)
            validation = validate_point_cloud(points)
            if not validation["is_valid"]:
                return {
                    "reason": f"validation_failed: {validation.get('reason', 'unknown')}",
                    "can_dispense": False,
                }

            # Generate embedding
            embed_result = self.embedder.embed(points)
            embedding = embed_result.embedding.cpu().numpy()

            # Match against database
            match_result = self.db.match(species, embedding, self.match_threshold)

            if match_result.is_match and match_result.individual_id is not None:
                # Known individual
                individual_id = match_result.individual_id
                self.db.mark_seen(individual_id, now_ts)

                # Check cooldown and daily limits
                stats = self.db.get_stats(individual_id, now_ts)
                time_since_last = now_ts - stats["last_dispense_ts"]
                can_dispense = True
                reason = "ok"

                if time_since_last < self.cooldown_seconds:
                    can_dispense = False
                    reason = f"cooldown ({self.cooldown_seconds - time_since_last:.1f}s remaining)"
                elif stats["dispense_count_today"] >= self.max_per_day:
                    can_dispense = False
                    reason = f"daily_limit_reached ({self.max_per_day})"

                # Update prototype with EMA
                if can_dispense:
                    self.db.update_prototype(
                        individual_id, embedding, ema=0.05, max_prototypes=20, now_ts=now_ts
                    )

                return {
                    "individual_id": individual_id,
                    "individual_name": f"Bird_{individual_id}",
                    "is_new": False,
                    "can_dispense": can_dispense,
                    "reason": reason,
                    "confidence": match_result.confidence,
                    "distance": match_result.distance,
                    "num_points": points.shape[0],
                    "validation": validation,
                }
            else:
                # New individual - register
                individual_id = self.db.add_individual(species, embedding, weight=1.0)
                self.db.mark_seen(individual_id, now_ts)

                return {
                    "individual_id": individual_id,
                    "individual_name": f"Bird_{individual_id}",
                    "is_new": True,
                    "can_dispense": False,  # Don't dispense on first enrollment
                    "reason": "new_enrollment",
                    "confidence": 0.0,
                    "distance": match_result.distance,
                    "num_points": points.shape[0],
                    "validation": validation,
                }

        except Exception as e:
            print(f"[ERROR] Individual tracking failed: {e}")
            return None


def main():
    ap = argparse.ArgumentParser(description="Enhanced bird detection with 3D individual ID")

    # Model and camera
    ap.add_argument(
        "--model",
        type=str,
        default="/home/pi/Desktop/FinalPrototype 2/FinalPrototype/GoodModel/weights(2).pt",
        help="Path to YOLO model (.pt or .hef)",
    )
    ap.add_argument("--source", type=str, default="0", help="RGB camera source")
    ap.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    ap.add_argument("--imgsz", type=int, default=320, help="YOLO input size")

    # 3D camera
    ap.add_argument("--use-3d", action="store_true", default=True, help="Enable 3D camera")
    ap.add_argument(
        "--no-3d", dest="use_3d", action="store_false", help="Disable 3D camera (optics-only)"
    )
    ap.add_argument(
        "--depth-mode", type=str, default="320x240", choices=["320x240", "640x480"]
    )

    # Individual tracking
    ap.add_argument(
        "--birdid-db", type=str, default="birdid.sqlite", help="Path to BirdID database"
    )
    ap.add_argument(
        "--match-threshold", type=float, default=0.32, help="Individual match threshold"
    )
    ap.add_argument(
        "--cooldown", type=float, default=300.0, help="Per-individual cooldown (seconds)"
    )
    ap.add_argument(
        "--max-per-day", type=int, default=20, help="Max dispenses per individual per day"
    )
    ap.add_argument(
        "--backbone",
        choices=["light", "heavy"],
        default="light",
        help="Point cloud backbone model",
    )

    # Trap settings
    ap.add_argument(
        "--trap_settings",
        type=str,
        default="/home/pi/Desktop/FinalPrototype/FinalPrototype/trap_settings.json",
    )
    ap.add_argument("--servo_channel", type=int, default=1)
    ap.add_argument("--servo_closed", type=float, default=5.0)
    ap.add_argument("--servo_open", type=float, default=60.0)
    ap.add_argument("--min_inter_trigger", type=float, default=1.5)
    ap.add_argument("--servo_min_us", type=int, default=1000)
    ap.add_argument("--servo_max_us", type=int, default=2000)
    ap.add_argument("--servo_slew", type=float, default=240.0)

    # Firebase
    ap.add_argument(
        "--db", type=str, default="https://ornimetrics-default-rtdb.firebaseio.com"
    )
    ap.add_argument("--session_key", type=str, default="session_1")

    # Display
    ap.add_argument("--preview", action="store_true", default=True)
    ap.add_argument("--preview_scale", type=float, default=0.6)
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--fourcc", type=str, default="MJPG")

    # Performance
    ap.add_argument("--max_fps", type=float, default=12.0)
    ap.add_argument("--every_n", type=int, default=1)
    ap.add_argument("--log_fps_interval", type=float, default=20.0)
    ap.add_argument("--box_ttl", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    args.model = resolve_path(args.model)
    args.trap_settings = resolve_path(args.trap_settings)

    def log(msg):
        if not args.quiet:
            print(msg)

    def log_key(msg):
        print(msg)

    # Detect available hardware and capabilities
    mode = DetectionMode()

    # Initialize YOLO detector (Hailo with PyTorch fallback)
    detector = None
    if HAILO_AVAILABLE and Path(args.model).suffix == ".hef":
        log("[YOLO] Attempting Hailo-8 accelerator...")
        detector = HailoYOLODetector(
            model_path=args.model,
            fallback_pytorch_model=args.model.replace(".hef", ".pt"),
            confidence_threshold=args.conf,
            input_size=args.imgsz,
        )
        mode.has_hailo = detector.is_hailo_active()
        mode.backend_name = detector.get_backend_name()
    else:
        log("[YOLO] Using PyTorch backend...")
        detector = HailoYOLODetector(
            model_path=None,
            fallback_pytorch_model=args.model,
            confidence_threshold=args.conf,
            input_size=args.imgsz,
        )
        mode.backend_name = "PyTorch CPU"

    # Initialize 3D camera if requested
    depth_camera = None
    individual_tracker = None

    if args.use_3d and POINT_CLOUD_AVAILABLE:
        log("[3D] Attempting to initialize CS20 depth camera...")
        try:
            depth_camera = CS20Camera(mode=args.depth_mode, fallback_to_synthetic=False)
            depth_camera.start()
            time.sleep(0.5)  # Allow camera to warm up

            if depth_camera.is_hardware_available():
                mode.has_3d_camera = True
                log("[3D] CS20 depth camera initialized successfully")

                # Initialize individual bird tracker
                log("[3D] Initializing individual bird tracker...")
                birdid_db = BirdIDDatabase(args.birdid_db)

                # Camera intrinsics (adjust for your camera calibration)
                intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}

                # Initialize point cloud embedder
                preprocess_cfg = PreprocessConfig(
                    plane_removal=True,
                    normalization="center_only",
                    voxel_size=0.01,
                    fps_points=2048,
                    depth_gate_k=2.5,
                )
                embedder = PointReID(preprocess_config=preprocess_cfg, model_name=args.backbone)

                individual_tracker = IndividualBirdTracker(
                    db=birdid_db,
                    embedder=embedder,
                    intrinsics=intrinsics,
                    match_threshold=args.match_threshold,
                    cooldown_seconds=args.cooldown,
                    max_per_day=args.max_per_day,
                )
                mode.has_individual_id = True
                log(f"[3D] Individual tracker initialized (backbone: {args.backbone})")
            else:
                log("[3D] CS20 hardware not detected - falling back to optics-only mode")
                depth_camera.stop()
                depth_camera = None
        except Exception as e:
            log(f"[3D] Failed to initialize depth camera: {e}")
            log("[3D] Falling back to optics-only mode")
            if depth_camera:
                depth_camera.stop()
            depth_camera = None
    else:
        log("[3D] 3D mode disabled - using optics-only mode")

    log_key(f"[Mode] {mode}")
    log_key(
        f"[Start] model={args.model} cam={args.cam_w}x{args.cam_h} conf={args.conf} imgsz={args.imgsz}"
    )

    # Initialize servo trap
    trap = ServoTrap(
        channel=args.servo_channel,
        closed_angle=args.servo_closed,
        open_angle=args.servo_open,
        min_inter_trigger_sec=args.min_inter_trigger,
        debug=(not args.quiet),
        slew_deg_per_s=args.servo_slew,
        min_us=args.servo_min_us,
        max_us=args.servo_max_us,
    )
    cfg = load_trap_settings(
        args.trap_settings, hold_default=1.5, cooldown_default=8.0, conf_default=args.conf
    )

    # Initialize Firebase
    flog = None
    if FIREBASE_AVAILABLE and args.db:
        try:
            flog = FirebaseLogger(args.db, session_key=args.session_key, run_id=None)
            log("[Firebase] Enabled")
        except Exception as e:
            log_key(f"[Firebase] Init failed: {e}")

    # Initialize RGB camera
    cap = cv2.VideoCapture(
        int(args.source) if str(args.source).isdigit() else args.source, cv2.CAP_V4L2
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Display setup
    show = bool(args.preview)
    scale = float(args.preview_scale)
    if show:
        try:
            tmp = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imshow("detect", tmp)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        except Exception as e:
            log("[Display] GUI not available; continuing headless")
            show = False

    # Performance tracking
    max_fps = max(1e-3, float(args.max_fps))
    last_proc = 0.0
    last_fps_log = now_mono()
    log_interval = max(5.0, float(args.log_fps_interval))
    fps_hist = deque(maxlen=120)
    frame_idx = 0
    last_block_print = 0.0

    # Detection cache
    last_dets: List = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            trap.tick()

            t_now = now_mono()
            shown = frame.copy()

            # Draw cached boxes
            if last_dets:
                for det_info in list(last_dets):
                    if (time.time() - det_info["ts"]) > float(args.box_ttl):
                        continue
                    x1, y1, x2, y2 = det_info["bbox"]
                    label = det_info["label"]
                    conf = det_info["conf"]
                    moved = det_info["moved"]
                    individual = det_info.get("individual")

                    col = (0, 255, 0) if moved else (0, 128, 255)
                    cv2.rectangle(shown, (x1, y1), (x2, y2), col, 2)

                    text = f"{label} {conf:.2f}"
                    if individual:
                        text += f" | {individual.get('individual_name', 'Unknown')}"
                    if moved:
                        text += " [TRIG]"

                    cv2.putText(
                        shown,
                        text,
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        col,
                        2,
                        cv2.LINE_AA,
                    )

            # FPS cap
            if t_now - last_proc < 1.0 / max_fps:
                if show:
                    disp = (
                        shown
                        if scale == 1.0
                        else cv2.resize(
                            shown, (int(shown.shape[1] * scale), int(shown.shape[0] * scale))
                        )
                    )
                    cv2.imshow("detect", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                continue

            last_proc = t_now
            t0 = now_mono()

            frame_idx += 1
            if frame_idx % max(1, int(args.every_n)) != 0:
                continue

            # Get depth frame if available
            depth_frame = None
            if depth_camera is not None:
                depth_frame = depth_camera.get_latest(timeout=0.05)

            # Run YOLO detection
            detections = detector.detect(frame)

            new_cache = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = det["class_name"]
                conf = det["confidence"]

                # Process individual identification if 3D available
                individual_info = None
                if individual_tracker and depth_frame:
                    individual_info = individual_tracker.process_detection(
                        (x1, y1, x2, y2), label, depth_frame, time.time()
                    )

                # Get trap action
                action = get_action_for(
                    label, cfg, hold_default=1.5, cooldown_default=8.0, conf_default=args.conf
                )
                open_dur = float(action.get("open_duration", 1.5))
                cooldown = float(action.get("cooldown_duration", 8.0))
                min_conf = float(action.get("confidence_threshold", args.conf))

                moved = False
                if open_dur > 0 and conf >= min_conf:
                    # Check if we should trigger based on individual tracking
                    can_trigger = True
                    block_reason = ""

                    if individual_info:
                        if not individual_info.get("can_dispense", False):
                            can_trigger = False
                            block_reason = individual_info.get("reason", "unknown")
                            if individual_info.get("is_new", False):
                                log_key(
                                    f"[ENROLL] New bird registered: {individual_info['individual_name']}"
                                )

                    # Check servo cooldown
                    if can_trigger and hasattr(trap, "can_trigger"):
                        ok, reason, sp_left, gb_left = trap.can_trigger(label=label, force=False)
                        if not ok:
                            can_trigger = False
                            block_reason = reason

                    if can_trigger:
                        moved = trap.trigger(
                            open_duration=open_dur,
                            cooldown_duration=cooldown,
                            label=label,
                            confidence=float(conf),
                            min_conf=min_conf,
                            force=False,
                        )
                        if moved:
                            dispense_msg = f"[TRIG] {label} conf={conf:.2f}"
                            if individual_info:
                                dispense_msg += (
                                    f" | {individual_info['individual_name']} (known)"
                                )
                                # Record dispense in database
                                individual_tracker.db.record_dispense(
                                    individual_info["individual_id"], time.time()
                                )
                            log_key(dispense_msg)
                    else:
                        # Log block reason
                        if (now_mono() - last_block_print) >= 1.5:
                            block_msg = f"[BLOCK] {label}"
                            if individual_info:
                                block_msg += f" | {individual_info.get('individual_name', 'Unknown')}"
                            block_msg += f" | {block_reason}"
                            log_key(block_msg)
                            last_block_print = now_mono()

                # Draw detection
                col = (0, 255, 0) if moved else (0, 128, 255)
                cv2.rectangle(shown, (x1, y1), (x2, y2), col, 2)

                text = f"{label} {conf:.2f}"
                if individual_info:
                    text += f" | {individual_info.get('individual_name', 'Unknown')}"
                if moved:
                    text += " [TRIG]"

                cv2.putText(
                    shown,
                    text,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    col,
                    2,
                    cv2.LINE_AA,
                )

                new_cache.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "conf": conf,
                        "moved": moved,
                        "ts": time.time(),
                        "individual": individual_info,
                    }
                )

                # Firebase logging on trigger
                if moved and flog:
                    try:
                        crop = frame[
                            max(0, y1) : min(frame.shape[0], y2),
                            max(0, x1) : min(frame.shape[1], x2),
                        ].copy()
                        if crop.size == 0:
                            crop = frame
                        ok2, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok2:
                            flog.push_photo_snapshot(image_bytes=buf.tobytes(), species=label)

                            event_info = {
                                "conf": float(conf),
                                "cooldown": cooldown,
                                "open_dur": open_dur,
                                "mode": str(mode),
                            }
                            if individual_info:
                                event_info["individual_id"] = individual_info.get("individual_id")
                                event_info["individual_name"] = individual_info.get(
                                    "individual_name"
                                )
                                event_info["is_new"] = individual_info.get("is_new", False)

                            flog.log_event(event_type="servo_trigger", species=label, info=event_info)
                            flog.increment_species(label, 1)
                            log_key(f"[PHOTO] {label} bbox=({x1},{y1},{x2},{y2})")
                    except Exception as e:
                        log_key(f"[Firebase] Log failed: {e}")

            if new_cache:
                last_dets = new_cache

            # FPS logging
            dt = now_mono() - t0
            if dt > 0:
                fps_hist.append(1.0 / dt)
            if now_mono() - last_fps_log >= log_interval:
                if len(fps_hist):
                    avg_fps = sum(fps_hist) / len(fps_hist)
                    log_key(f"[FPS] avg={avg_fps:.2f}")
                else:
                    log_key("[FPS] no frames yet")
                last_fps_log = now_mono()

            # Preview
            if show:
                disp = (
                    shown
                    if scale == 1.0
                    else cv2.resize(
                        shown, (int(shown.shape[1] * scale), int(shown.shape[0] * scale))
                    )
                )
                cv2.imshow("detect", disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        log_key("[INFO] Interrupted by user")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if depth_camera:
            try:
                depth_camera.stop()
            except Exception:
                pass
        log_key("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
