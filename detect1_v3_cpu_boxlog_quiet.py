#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect1_v3_cpu_boxlog_quiet.py
Same as detect1_v3_cpu_boxlog.py but with a --quiet flag for minimal logs.

What prints in --quiet mode:
  - Startup one-liner (model/cam).
  - [TRIG] species label, conf, open_dur, cooldown.
  - [BLOCK] species_cd or global_busy (throttled).
  - [FPS] avg every N seconds.
  - [PHOTO] when a snapshot is sent.
  - Errors/warnings.
"""

import os, sys, time, json, argparse
from time import monotonic as now_mono
from pathlib import Path
from collections import deque
from typing import Tuple, List, Dict

# Keep Qt stable even over SSH
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# Light CPU threading for better interactivity
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

# Servo + Firebase
try:
    from servoMain_fixed import ServoTrap
except Exception:
    from servoMain import ServoTrap  # fallback if user kept old filename

FirebaseLogger = None
try:
    from firebase_logger import FirebaseLogger as _FirebaseLogger
    FirebaseLogger = _FirebaseLogger
except Exception:
    FirebaseLogger = None

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics not installed:", e, file=sys.stderr)
    sys.exit(2)

SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_path(p: str) -> str:
    q = Path(p)
    return str(q) if q.is_absolute() else str((SCRIPT_DIR / q).resolve())

def normalize_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")

def load_trap_settings(path: str, hold_default: float, cooldown_default: float, conf_default: float) -> Dict[str, dict]:
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
        print(f"WARNING: could not load trap_settings from {path}: {e}")
        return {"_default": {"open_duration": hold_default, "cooldown_duration": cooldown_default, "confidence_threshold": conf_default}}

def get_action_for(label: str, cfg: Dict[str, dict], hold_default: float, cooldown_default: float, conf_default: float):
    dflt = {"open_duration": hold_default, "cooldown_duration": cooldown_default, "confidence_threshold": conf_default}
    raw = (label or "").strip(); slug = normalize_key(raw)
    for k in (raw, slug, slug.replace("_","-"), slug.replace("-","_")):
        if k in cfg and isinstance(cfg[k], dict):
            return cfg[k]
    # fallbacks
    if "bird" in cfg and "bird" in slug: return cfg["bird"]
    if "squirrel" in cfg and "squirrel" in slug: return cfg["squirrel"]
    if "critter" in cfg: return cfg["critter"]
    return cfg.get("_default", dflt) or dflt

def main():
    ap = argparse.ArgumentParser()
    # Same flags as v3 + quiet
    ap.add_argument("--model", type=str, default="/home/pi/Desktop/FinalPrototype 2/FinalPrototype/GoodModel/weights(2).pt")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--trap_settings", type=str, default="/home/pi/Desktop/FinalPrototype/FinalPrototype/trap_settings.json")
    ap.add_argument("--db", type=str, default="https://ornimetrics-default-rtdb.firebaseio.com")
    ap.add_argument("--session_key", type=str, default="session_1")
    ap.add_argument("--preview", action="store_true", default=True)
    ap.add_argument("--preview_scale", type=float, default=0.6)
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--fourcc", type=str, default="MJPG")
    ap.add_argument("--max_fps", type=float, default=12.0)
    ap.add_argument("--every_n", type=int, default=1)
    ap.add_argument("--log_fps_interval", type=float, default=20.0)
    ap.add_argument("--box_ttl", type=float, default=2.0, help="Seconds to keep last boxes on screen")
    ap.add_argument("--quiet", action="store_true", help="Reduce logs to essential events only")
    # Servo
    ap.add_argument("--servo_channel", type=int, default=1)
    ap.add_argument("--servo_closed", type=float, default=5.0)
    ap.add_argument("--servo_open", type=float, default=60.0)
    ap.add_argument("--min_inter_trigger", type=float, default=1.5)
    ap.add_argument("--servo_min_us", type=int, default=1000)
    ap.add_argument("--servo_max_us", type=int, default=2000)
    ap.add_argument("--servo_slew", type=float, default=240.0, help="Slew rate in deg/s; 0=instant")
    args = ap.parse_args()

    args.model = resolve_path(args.model)
    args.trap_settings = resolve_path(args.trap_settings)

    def log(msg):
        if not args.quiet:
            print(msg)

    def log_key(msg):
        # Always print essential info even in quiet mode
        print(msg)

    # Startup single line in quiet mode
    log_key(f"[Start] model={args.model} cam={args.cam_w}x{args.cam_h}@{args.fourcc} conf={args.conf} imgsz={args.imgsz}")

    # Servo + settings
    trap = ServoTrap(channel=args.servo_channel, closed_angle=args.servo_closed,
                     open_angle=args.servo_open, min_inter_trigger_sec=args.min_inter_trigger,
                     debug=(not args.quiet), slew_deg_per_s=args.servo_slew,
                     min_us=args.servo_min_us, max_us=args.servo_max_us)
    cfg = load_trap_settings(args.trap_settings, hold_default=1.5, cooldown_default=8.0, conf_default=args.conf)

    # Firebase
    flog = None
    if FirebaseLogger is not None and args.db:
        try:
            flog = FirebaseLogger(args.db, session_key=args.session_key, run_id=None)
            log("[Firebase] Enabled")
        except Exception as e:
            log_key(f"[Firebase] init failed: {e}")

    # YOLO
    model = YOLO(args.model)
    names = []
    if hasattr(model, "names"):
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)

    # Camera
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Display
    show = bool(args.preview)
    scale = float(args.preview_scale)
    if show:
        try:
            tmp = np.zeros((50,50,3), dtype=np.uint8)
            cv2.imshow("detect", tmp); cv2.waitKey(1); cv2.destroyAllWindows()
        except Exception as e:
            log("[Display] GUI not available; continuing headless")
            show = False

    # perf + logs
    max_fps = max(1e-3, float(args.max_fps))
    last_proc = 0.0
    last_fps_log = now_mono()
    log_interval = max(5.0, float(args.log_fps_interval))
    fps_hist = deque(maxlen=120)
    frame_idx = 0
    last_block_print = 0.0  # throttle block messages in quiet mode

    # cache for forced boxes
    last_dets: List[Tuple[int,int,int,int,float,str,bool,float]] = []  # (x1,y1,x2,y2,conf,label,moved,timestamp)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01); continue

            trap.tick()

            t_now = now_mono()
            shown = frame.copy()

            # Draw cached boxes every frame
            if last_dets:
                for (x1,y1,x2,y2,sc,label,moved,ts) in list(last_dets):
                    if (time.time() - ts) > float(args.box_ttl):
                        continue
                    col = (0,255,0) if moved else (0,128,255)
                    cv2.rectangle(shown, (x1,y1), (x2,y2), col, 2)
                    cv2.putText(shown, f"{label} {sc:.2f}" + (" [TRIG]" if moved else ""),
                                (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            # obey FPS cap but keep preview smooth
            if t_now - last_proc < 1.0 / max_fps:
                if show:
                    disp = shown if scale == 1.0 else cv2.resize(shown, (int(shown.shape[1]*scale), int(shown.shape[0]*scale)))
                    cv2.imshow("detect", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                continue
            last_proc = t_now
            t0 = now_mono()

            frame_idx += 1
            if frame_idx % max(1, int(args.every_n)) == 0:
                # YOLO forward
                with __import__("torch").inference_mode():
                    res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=0.5, verbose=False)
                dets = []
                if res:
                    r0 = res[0]
                    if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
                        xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
                        confs = r0.boxes.conf.cpu().numpy()
                        clss  = r0.boxes.cls.cpu().numpy().astype(int)
                        for (x1,y1,x2,y2), sc, cid in zip(xyxy, confs, clss):
                            if sc < args.conf: continue
                            label = names[cid] if (names and 0 <= cid < len(names)) else f"class_{cid}"
                            dets.append((int(x1), int(y1), int(x2), int(y2), float(sc), label))

                new_cache = []
                for (x1,y1,x2,y2,sc,label) in dets:
                    action = get_action_for(label, cfg, hold_default=1.5, cooldown_default=8.0, conf_default=args.conf)
                    open_dur = float(action.get("open_duration", 1.5))
                    cooldown = float(action.get("cooldown_duration", 8.0))
                    min_conf = float(action.get("confidence_threshold", args.conf))

                    moved = False
                    if open_dur > 0 and sc >= min_conf:
                        can_ok = True
                        reason = ""; sp_left = gb_left = 0.0
                        if hasattr(trap, "can_trigger"):
                            ok, reason, sp_left, gb_left = trap.can_trigger(label=label, force=False)
                            can_ok = ok

                        if can_ok:
                            moved = trap.trigger(open_duration=open_dur, cooldown_duration=cooldown,
                                                 label=label, confidence=float(sc), min_conf=min_conf, force=False)
                            if moved:
                                log_key(f"[TRIG] {label} conf={sc:.2f} open={open_dur}s cd={cooldown}s")
                        else:
                            # throttle block prints to avoid spam
                            if (now_mono() - last_block_print) >= 1.5:
                                if reason == "species_cooldown":
                                    log_key(f"[BLOCK] {label} species_cd {sp_left:.2f}s")
                                elif reason == "global_busy":
                                    log_key(f"[BLOCK] {label} busy {gb_left:.2f}s")
                                else:
                                    log_key(f"[BLOCK] {label} {reason}")
                                last_block_print = now_mono()

                    # draw now
                    col = (0,255,0) if moved else (0,128,255)
                    cv2.rectangle(shown, (x1,y1), (x2,y2), col, 2)
                    cv2.putText(shown, f"{label} {sc:.2f}" + (" [TRIG]" if moved else ""),
                                (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

                    new_cache.append((x1,y1,x2,y2,float(sc),label,bool(moved),time.time()))

                    # Firebase on move
                    if moved and (flog is not None):
                        try:
                            crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
                            if crop.size == 0: crop = frame
                            ok2, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok2:
                                flog.push_photo_snapshot(image_bytes=buf.tobytes(), species=label)
                                log_key(f"[PHOTO] {label} bbox=({x1},{y1},{x2},{y2})")
                                flog.log_event(event_type="servo_trigger", species=label,
                                               info={"conf": float(sc), "cooldown": cooldown, "open_dur": open_dur})
                                flog.increment_species(label, 1)
                        except Exception as e:
                            log_key(f"[Firebase] log_event failed: {e}")

                if new_cache:
                    last_dets = new_cache

            # FPS logging
            dt = now_mono() - t0
            if dt > 0:
                fps_hist.append(1.0/dt)
            if now_mono() - last_fps_log >= log_interval:
                if len(fps_hist):
                    avg_fps = sum(fps_hist)/len(fps_hist)
                    log_key(f"[FPS] avg={avg_fps:.2f}")
                else:
                    log_key("[FPS] no frames yet")
                last_fps_log = now_mono()

            # preview
            if show:
                disp = shown if scale == 1.0 else cv2.resize(shown, (int(shown.shape[1]*scale), int(shown.shape[0]*scale)))
                cv2.imshow("detect", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

if __name__ == "__main__":
    main()
