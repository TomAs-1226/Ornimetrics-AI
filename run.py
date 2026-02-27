#!/usr/bin/env python3
"""Ornimetrics — single entry point for bird detection.

Usage:
    python run.py                     # auto-detect everything, run headless
    python run.py --web               # run with web dashboard on port 5000
    python run.py --preview           # run with local GUI preview
    python run.py --config my.json    # custom config file
    python run.py --no-3d             # disable depth camera (optics-only)
    python run.py --quiet             # minimal logging
"""

import os
import sys
import time
import json
import argparse
import threading
from time import monotonic as _mono
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

DEFAULTS = {
    "detection": {"model_path": "weights.pt", "confidence_threshold": 0.45, "input_size": 320},
    "camera": {"rgb_source": 0, "width": 640, "height": 480, "fourcc": "MJPG"},
    "depth": {"enabled": True, "mode": "320x240"},
    "point_cloud": {"backbone": "light", "fps_points": 1024, "voxel_size": 0.01, "plane_removal": True},
    "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
    "tracking": {"database_path": "birdid.sqlite", "match_threshold": 0.32, "cooldown_seconds": 300, "max_per_day": 20},
    "servo": {"channel": 1, "closed_angle": 5.0, "open_angle": 60.0, "min_inter_trigger_sec": 1.5, "slew_deg_per_s": 240.0, "min_us": 1000, "max_us": 2000},
    "trap_settings": "trap_settings_full.json",
    "firebase": {"enabled": True, "database_url": "https://ornimetrics-default-rtdb.firebaseio.com", "session_key": "session_1"},
    "performance": {"max_fps": 12.0, "every_n_frames": 1},
    "display": {"preview": False, "scale": 0.6, "quiet": False},
    "web": {"enabled": False, "host": "0.0.0.0", "port": 5000, "stream_fps": 15},
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULTS)
    if path and Path(path).is_file():
        with open(path) as f:
            cfg = _deep_merge(cfg, json.load(f))
    return cfg


def _get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _resolve(p: str) -> str:
    q = Path(p)
    return str(q) if q.is_absolute() else str((SCRIPT_DIR / q).resolve())


# ---------------------------------------------------------------------------
# Trap-settings helpers  (unchanged logic)
# ---------------------------------------------------------------------------

def _normalize_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


def _load_trap_settings(path: str, conf_default: float) -> dict:
    p = Path(path)
    if not p.is_file():
        cand = SCRIPT_DIR / "trap_settings_full.json"
        if cand.is_file():
            p = cand
    try:
        with open(p) as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            return cfg
    except Exception as e:
        print(f"[WARN] trap_settings: {e}")
    return {"_default": {"open_duration": 1.5, "cooldown_duration": 8.0, "confidence_threshold": conf_default}}


def _action_for(label: str, cfg: dict, conf_default: float) -> dict:
    dflt = {"open_duration": 1.5, "cooldown_duration": 8.0, "confidence_threshold": conf_default}
    slug = _normalize_key(label)
    for k in (label.strip(), slug, slug.replace("_", "-"), slug.replace("-", "_")):
        if k in cfg and isinstance(cfg[k], dict):
            return cfg[k]
    if "bird" in cfg and "bird" in slug:
        return cfg["bird"]
    if "squirrel" in cfg and "squirrel" in slug:
        return cfg["squirrel"]
    return cfg.get("_defaults_for_unknown_label", cfg.get("_default", dflt)) or dflt


# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------

def _init_detector(cfg: dict):
    """Initialize YOLO detector (Hailo or PyTorch)."""
    model_path = _resolve(_get(cfg, "detection", "model_path", default="weights.pt"))
    conf = _get(cfg, "detection", "confidence_threshold", default=0.45)
    imgsz = _get(cfg, "detection", "input_size", default=320)

    try:
        from src.hailo_detector import HailoYOLODetector
        if Path(model_path).suffix == ".hef":
            det = HailoYOLODetector(model_path=model_path, fallback_pytorch_model=model_path.replace(".hef", ".pt"), confidence_threshold=conf, input_size=imgsz)
        else:
            det = HailoYOLODetector(model_path=None, fallback_pytorch_model=model_path, confidence_threshold=conf, input_size=imgsz)
        return det
    except Exception:
        pass

    # Fallback: direct ultralytics
    from ultralytics import YOLO
    _yolo = YOLO(model_path)

    class _Wrapper:
        def __init__(self, model, names, conf_thr, imgsz):
            self._model = model
            self._names = names
            self._conf = conf_thr
            self._imgsz = imgsz
        def detect(self, image):
            import torch
            with torch.inference_mode():
                res = self._model.predict(source=image, imgsz=self._imgsz, conf=self._conf, iou=0.5, verbose=False)
            dets = []
            if res:
                r0 = res[0]
                if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes):
                    xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
                    confs = r0.boxes.conf.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clss):
                        name = self._names[cid] if 0 <= cid < len(self._names) else f"class_{cid}"
                        dets.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "class_name": name, "class_id": int(cid), "confidence": float(c)})
            return dets
        def is_hailo_active(self): return False
        def get_backend_name(self): return "PyTorch CPU"

    names = list(_yolo.names.values()) if isinstance(_yolo.names, dict) else list(_yolo.names) if hasattr(_yolo, "names") else []
    return _Wrapper(_yolo, names, conf, imgsz)


def _init_3d(cfg: dict):
    """Initialize depth camera + individual tracker. Returns (camera, tracker) or (None, None)."""
    if not _get(cfg, "depth", "enabled", default=True):
        return None, None
    try:
        from birdid.camera.cs20 import CS20Camera
        from src.depth_to_points import backproject_depth
        from src.reid_embedder import PointReID
        from src.pc_preprocess import PreprocessConfig
        from birdid.db import BirdIDDatabase
        from birdid.validate_3d import validate_point_cloud
    except Exception as e:
        print(f"[WARN] 3D modules unavailable: {e}")
        return None, None

    mode = _get(cfg, "depth", "mode", default="320x240")
    cam = CS20Camera(mode=mode, fallback_to_synthetic=False)
    cam.start()
    time.sleep(0.3)
    if not cam.is_hardware_available():
        print("[INFO] CS20 depth camera not detected — optics-only mode")
        cam.stop()
        return None, None

    print("[INFO] CS20 depth camera ready")
    pc_cfg = PreprocessConfig(
        plane_removal=_get(cfg, "point_cloud", "plane_removal", default=True),
        normalization="center_only",
        voxel_size=_get(cfg, "point_cloud", "voxel_size", default=0.01),
        fps_points=_get(cfg, "point_cloud", "fps_points", default=1024),
        depth_gate_k=2.5,
    )
    backbone = _get(cfg, "point_cloud", "backbone", default="light")
    embedder = PointReID(preprocess_config=pc_cfg, model_name=backbone)
    db = BirdIDDatabase(_get(cfg, "tracking", "database_path", default="birdid.sqlite"))

    from detect_3d_individual import IndividualBirdTracker
    tracker = IndividualBirdTracker(
        db=db,
        embedder=embedder,
        intrinsics=_get(cfg, "intrinsics", default={"fx": 525, "fy": 525, "cx": 319.5, "cy": 239.5}),
        match_threshold=_get(cfg, "tracking", "match_threshold", default=0.32),
        cooldown_seconds=_get(cfg, "tracking", "cooldown_seconds", default=300),
        max_per_day=_get(cfg, "tracking", "max_per_day", default=20),
    )
    return cam, tracker


def _init_servo(cfg: dict):
    sv = _get(cfg, "servo", default={})
    try:
        from servoMain_fixed import ServoTrap
    except Exception:
        from servoMain import ServoTrap
    return ServoTrap(
        channel=sv.get("channel", 1),
        closed_angle=sv.get("closed_angle", 5.0),
        open_angle=sv.get("open_angle", 60.0),
        min_inter_trigger_sec=sv.get("min_inter_trigger_sec", 1.5),
        debug=not _get(cfg, "display", "quiet", default=False),
        slew_deg_per_s=sv.get("slew_deg_per_s", 240.0),
        min_us=sv.get("min_us", 1000),
        max_us=sv.get("max_us", 2000),
    )


def _init_firebase(cfg: dict):
    if not _get(cfg, "firebase", "enabled", default=False):
        return None
    try:
        from firebase_logger import FirebaseLogger
        url = _get(cfg, "firebase", "database_url", default="")
        sk = _get(cfg, "firebase", "session_key", default="session_1")
        return FirebaseLogger(url, session_key=sk, run_id=None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Detection loop  (shared between headless, preview and web modes)
# ---------------------------------------------------------------------------

def _process_frame(frame, detector, depth_camera, tracker, trap, trap_cfg, flog, conf_default, quiet):
    """Run detection on one frame. Returns (annotated_frame, detections_list)."""
    detections = detector.detect(frame)
    depth_frame = depth_camera.get_latest(timeout=0.05) if depth_camera else None
    shown = frame.copy()
    results = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class_name"]
        conf = det["confidence"]

        individual = None
        if tracker and depth_frame:
            individual = tracker.process_detection((x1, y1, x2, y2), label, depth_frame, time.time())

        action = _action_for(label, trap_cfg, conf_default)
        open_dur = float(action.get("open_duration", 1.5))
        cooldown = float(action.get("cooldown_duration", 8.0))
        min_conf = float(action.get("confidence_threshold", conf_default))

        moved = False
        if open_dur > 0 and conf >= min_conf:
            can_trigger = True
            if individual and not individual.get("can_dispense", False):
                can_trigger = False
                if individual.get("is_new"):
                    print(f"[ENROLL] {individual.get('individual_name', '?')}")
            if can_trigger and hasattr(trap, "can_trigger"):
                ok, reason, _, _ = trap.can_trigger(label=label, force=False)
                if not ok:
                    can_trigger = False
            if can_trigger:
                moved = trap.trigger(open_duration=open_dur, cooldown_duration=cooldown, label=label, confidence=float(conf), min_conf=min_conf, force=False)
                if moved:
                    print(f"[TRIG] {label} conf={conf:.2f}")
                    if individual and tracker:
                        tracker.db.record_dispense(individual["individual_id"], time.time())
                    if flog:
                        try:
                            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)].copy()
                            if crop.size == 0:
                                crop = frame
                            ok2, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok2:
                                flog.push_photo_snapshot(image_bytes=buf.tobytes(), species=label)
                                flog.log_event(event_type="servo_trigger", species=label, info={"conf": float(conf)})
                                flog.increment_species(label, 1)
                        except Exception:
                            pass

        col = (0, 255, 0) if moved else (0, 128, 255)
        cv2.rectangle(shown, (x1, y1), (x2, y2), col, 2)
        text = f"{label} {conf:.2f}"
        if individual:
            text += f" | {individual.get('individual_name', '?')}"
        if moved:
            text += " [TRIG]"
        cv2.putText(shown, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        results.append({"bbox": (x1, y1, x2, y2), "label": label, "conf": conf, "moved": moved, "individual": individual})
    return shown, results


# ---------------------------------------------------------------------------
# Web mode
# ---------------------------------------------------------------------------

def _run_web(cfg, detector, depth_camera, tracker, trap, trap_cfg, flog):
    from flask import Flask, Response, jsonify, render_template_string

    app = Flask(__name__)
    conf_default = _get(cfg, "detection", "confidence_threshold", default=0.45)
    quiet = _get(cfg, "display", "quiet", default=False)
    stream_fps = _get(cfg, "web", "stream_fps", default=15)

    _state = {"frame": None, "lock": threading.Lock(), "running": True, "stats": {"dets": 0, "trigs": 0}, "fps": deque(maxlen=60)}

    def _loop():
        cap = cv2.VideoCapture(_get(cfg, "camera", "rgb_source", default=0), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _get(cfg, "camera", "width", default=640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _get(cfg, "camera", "height", default=480))
        fourcc = _get(cfg, "camera", "fourcc", default="MJPG")
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while _state["running"]:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            trap.tick()
            t0 = _mono()
            shown, results = _process_frame(frame, detector, depth_camera, tracker, trap, trap_cfg, flog, conf_default, quiet)
            dt = _mono() - t0
            if dt > 0:
                _state["fps"].append(1.0 / dt)
            _state["stats"]["dets"] += len(results)
            _state["stats"]["trigs"] += sum(1 for r in results if r["moved"])
            with _state["lock"]:
                _state["frame"] = shown
            time.sleep(1.0 / stream_fps)
        cap.release()

    threading.Thread(target=_loop, daemon=True).start()

    DASHBOARD = """<!DOCTYPE html><html><head><title>Ornimetrics</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:sans-serif;background:#111;color:#eee}
    .c{max-width:900px;margin:0 auto;padding:20px}h1{color:#5bc0de;margin-bottom:10px}
    img{width:100%;border-radius:8px;background:#000}
    .s{display:flex;gap:20px;margin:20px 0}.sb{flex:1;background:#1c2938;padding:15px;border-radius:8px;text-align:center}
    .sv{font-size:2em;font-weight:bold;color:#5bc0de}.sl{color:#888;font-size:.9em}</style></head>
    <body><div class="c"><h1>Ornimetrics</h1><p style="color:#888;margin-bottom:15px">Bird Detection System</p>
    <img src="/video_feed" alt="stream">
    <div class="s"><div class="sb"><div class="sv" id="fps">--</div><div class="sl">FPS</div></div>
    <div class="sb"><div class="sv" id="dets">0</div><div class="sl">Detections</div></div>
    <div class="sb"><div class="sv" id="trigs">0</div><div class="sl">Triggers</div></div></div>
    </div><script>setInterval(()=>fetch('/api/stats').then(r=>r.json()).then(d=>{
    document.getElementById('fps').textContent=d.fps.toFixed(1);
    document.getElementById('dets').textContent=d.dets;
    document.getElementById('trigs').textContent=d.trigs;}),1000)</script></body></html>"""

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD)

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                with _state["lock"]:
                    f = _state["frame"]
                if f is None:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buf = cv2.imencode(".jpg", blank)
                else:
                    _, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                time.sleep(0.033)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/stats")
    def stats():
        fps = sum(_state["fps"]) / max(len(_state["fps"]), 1)
        return jsonify({"fps": fps, "dets": _state["stats"]["dets"], "trigs": _state["stats"]["trigs"]})

    host = _get(cfg, "web", "host", default="0.0.0.0")
    port = _get(cfg, "web", "port", default=5000)
    print(f"[WEB] http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ornimetrics bird detection system")
    ap.add_argument("--config", default="config_3d_detection.json", help="Path to config JSON")
    ap.add_argument("--web", action="store_true", help="Start web dashboard mode")
    ap.add_argument("--preview", action="store_true", help="Show local GUI preview")
    ap.add_argument("--no-3d", action="store_true", help="Disable depth camera")
    ap.add_argument("--quiet", action="store_true", help="Minimal logging")
    ap.add_argument("--model", type=str, default=None, help="Override model path")
    ap.add_argument("--port", type=int, default=None, help="Override web port")
    args = ap.parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    if args.no_3d:
        cfg["depth"]["enabled"] = False
    if args.quiet:
        cfg["display"]["quiet"] = True
    if args.preview:
        cfg["display"]["preview"] = True
    if args.web:
        cfg["web"]["enabled"] = True
    if args.model:
        cfg["detection"]["model_path"] = args.model
    if args.port:
        cfg["web"]["port"] = args.port

    quiet = _get(cfg, "display", "quiet", default=False)

    def log(msg):
        if not quiet:
            print(msg)

    # Initialize hardware
    log("[INIT] Loading YOLO model...")
    detector = _init_detector(cfg)
    log(f"[INIT] Backend: {detector.get_backend_name()}")

    log("[INIT] Initializing 3D camera...")
    depth_camera, tracker = _init_3d(cfg)
    mode = "Full 3D" if depth_camera and tracker else "Optics-Only"
    log(f"[INIT] Mode: {mode}")

    log("[INIT] Initializing servo...")
    trap = _init_servo(cfg)

    trap_settings_path = _resolve(_get(cfg, "trap_settings", default="trap_settings_full.json"))
    conf_default = _get(cfg, "detection", "confidence_threshold", default=0.45)
    trap_cfg = _load_trap_settings(trap_settings_path, conf_default)

    flog = _init_firebase(cfg)
    if flog:
        log("[INIT] Firebase enabled")

    print(f"[READY] {mode} | {detector.get_backend_name()}")

    # Web mode
    if _get(cfg, "web", "enabled", default=False):
        try:
            _run_web(cfg, detector, depth_camera, tracker, trap, trap_cfg, flog)
        except KeyboardInterrupt:
            pass
        finally:
            if depth_camera:
                depth_camera.stop()
        return

    # Headless / preview mode
    cap = cv2.VideoCapture(_get(cfg, "camera", "rgb_source", default=0), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _get(cfg, "camera", "width", default=640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _get(cfg, "camera", "height", default=480))
    fourcc = _get(cfg, "camera", "fourcc", default="MJPG")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    show = _get(cfg, "display", "preview", default=False)
    scale = _get(cfg, "display", "scale", default=0.6)
    if show:
        try:
            cv2.imshow("detect", np.zeros((50, 50, 3), dtype=np.uint8))
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        except Exception:
            log("[WARN] GUI not available, headless mode")
            show = False

    max_fps = max(1e-3, _get(cfg, "performance", "max_fps", default=12.0))
    every_n = max(1, _get(cfg, "performance", "every_n_frames", default=1))
    last_proc = 0.0
    fps_hist = deque(maxlen=120)
    last_fps_log = _mono()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            trap.tick()
            t_now = _mono()

            if t_now - last_proc < 1.0 / max_fps:
                if show:
                    disp = frame if scale == 1.0 else cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
                    cv2.imshow("detect", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                continue

            last_proc = t_now
            frame_idx += 1
            if frame_idx % every_n != 0:
                continue

            t0 = _mono()
            shown, results = _process_frame(frame, detector, depth_camera, tracker, trap, trap_cfg, flog, conf_default, quiet)
            dt = _mono() - t0
            if dt > 0:
                fps_hist.append(1.0 / dt)

            if _mono() - last_fps_log >= 20.0:
                if fps_hist:
                    print(f"[FPS] avg={sum(fps_hist) / len(fps_hist):.1f}")
                last_fps_log = _mono()

            if show:
                disp = shown if scale == 1.0 else cv2.resize(shown, (int(shown.shape[1] * scale), int(shown.shape[0] * scale)))
                cv2.imshow("detect", disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        print("[INFO] Stopped")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if depth_camera:
            depth_camera.stop()


if __name__ == "__main__":
    main()
