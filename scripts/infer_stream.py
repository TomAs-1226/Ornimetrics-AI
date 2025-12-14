#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import uuid
import cv2
import numpy as np

from src.yolo_detect import YOLODetector, load_rgb_image
from src.depth_to_points import backproject_depth
from src.reid_embedder import PointReID
from src.gallery import Gallery, cosine_distance
from src.tracker import Tracker
from src.naming import generate_name
from src.db import IdentityDB
from src.output_schema import format_detection
from src.pc_preprocess import PreprocessConfig


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + point-cloud re-id stream")
    parser.add_argument("--camera", default=None, help="Camera index or 'demo'")
    parser.add_argument("--rgb", help="RGB image path for offline", default=None)
    parser.add_argument("--depth", help="Depth image path for offline", default=None)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("fx", "fy", "cx", "cy"), default=[525.0, 525.0, 319.5, 239.5])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--margin-guard", type=float, default=0.05)
    parser.add_argument("--debug_identity", action="store_true", help="Enable verbose identity debugging")
    parser.add_argument("--plane-removal", action="store_true", default=True)
    parser.add_argument("--no-plane-removal", dest="plane_removal", action="store_false")
    parser.add_argument("--normalization", choices=["center_only", "center_and_scale", "center_and_scale_with_scale_feature"], default="center_only")
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--fps-points", type=int, default=2048)
    parser.add_argument("--depth-gate-k", type=float, default=2.5)
    parser.add_argument("--smooth-window", type=int, default=5)
    return parser.parse_args()


def load_depth(path):
    if path is None:
        return None
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32) / 1000.0
    return depth


def demo_depth(rgb):
    h, w, _ = rgb.shape
    depth = np.ones((h, w), dtype=np.float32) * 2.0
    return depth


def ensure_debug_dir():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path("runs") / "debug_identity" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def detection_debug_block(det, preprocess_stats, embedding, gallery_scores, decision, tracker_stats):
    sorted_scores = [{"id": i, "dist": float(d)} for i, d in gallery_scores]
    norm = float(np.linalg.norm(embedding)) if embedding is not None else 0.0
    block = {
        "class": det.get("class_name"),
        "bbox": det.get("bbox"),
        "preprocessing": preprocess_stats,
        "embedding_norm": norm,
        "top5": sorted_scores,
        "best_match_id": decision.get("best_id"),
        "best_dist": decision.get("best_dist"),
        "second_best_dist": decision.get("second_best"),
        "margin": decision.get("margin"),
        "threshold": decision.get("threshold"),
        "new_identity": decision.get("new_identity"),
        "reason": decision.get("reason"),
        "tracker": tracker_stats,
    }
    return block


def process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, threshold, debug_dir=None, smooth_window=5, debug=False):
    detections = det_model.detect(rgb)
    processed = []
    frame_debug = []
    for det in detections:
        bbox = det["bbox"]
        points, bp_stats = backproject_depth(depth, intrinsics, bbox)
        embed_result = embedder.embed(points)
        emb = embed_result.embedding.cpu().numpy()
        det["embedding"] = emb
        best_id, best_dist, scored = gallery.match(det["class_name"], emb)
        second_best = scored[1][1] if len(scored) > 1 else 1.0
        margin = second_best - best_dist
        create_new = best_id is None or gallery.needs_new_identity(det["class_name"], best_dist)
        reason = "empty_gallery" if best_id is None else ("above_threshold" if create_new else "matched")
        threshold_used = gallery.get_threshold(det["class_name"])
        det_info = {
            "best_id": best_id,
            "best_dist": float(best_dist),
            "second_best": float(second_best),
            "margin": float(margin),
            "threshold": float(threshold_used),
            "new_identity": bool(create_new),
            "reason": reason,
            "top5": scored,
        }
        if create_new:
            indiv_id = str(uuid.uuid4())
            name = generate_name()
            gallery.update(det["class_name"], indiv_id, emb, allow=True, name=name)
            db.upsert_individual(indiv_id, name, det["class_name"], emb)
            det["individual_id"] = indiv_id
            det["individual_name"] = name
            det["min_dist"] = 1.0
            det["second_best_dist"] = second_best
            det["margin"] = margin
        else:
            gallery.update(det["class_name"], best_id, emb, allow=gallery.should_update_identity(det["class_name"], best_dist))
            info = db.get_identity(best_id)
            det["individual_id"] = best_id
            det["individual_name"] = info[0] if info else None
            det["min_dist"] = float(best_dist)
            det["second_best_dist"] = float(second_best)
            det["margin"] = float(margin)
        det["bp_stats"] = bp_stats.__dict__
        det["preprocess_stats"] = embed_result.preprocess_stats
        det["decision"] = det_info
        processed.append(det)
    tracked = tracker.update(processed)
    for det in tracked:
        tracker_stats = det.get("tracker_debug", {})
        use_emb = det.get("embedding")
        if det.get("track_id") is not None and tracker_stats:
            # Replace with smoothed embedding for downstream logging/updating
            for t in tracker.tracks:
                if t.track_id == det["track_id"]:
                    smoothed = t.smoothed_embedding
                    if smoothed is not None:
                        use_emb = smoothed
                        det["frames_used"] = len(t.embedding_history)
                    break
        decision = det.get("decision", {})
        block = detection_debug_block(det, {**det.get("bp_stats", {}), **det.get("preprocess_stats", {})}, use_emb, decision.get("top5", []), decision, tracker_stats)
        frame_debug.append(block)
    outputs = [format_detection(d) for d in tracked]
    if debug and debug_dir is not None:
        frame_id = len(list(debug_dir.glob("frame_*.json")))
        with open(debug_dir / f"frame_{frame_id:04d}.json", "w", encoding="utf-8") as f:
            json.dump(frame_debug, f, indent=2, default=lambda o: o if isinstance(o, (int, float, str)) else str(o))
        print(json.dumps(frame_debug, indent=2))
    return outputs


def main():
    args = parse_args()
    intrinsics = {"fx": args.intrinsics[0], "fy": args.intrinsics[1], "cx": args.intrinsics[2], "cy": args.intrinsics[3]}
    det_model = YOLODetector(args.model)
    preprocess_cfg = PreprocessConfig(
        plane_removal=args.plane_removal,
        normalization=args.normalization,
        voxel_size=args.voxel,
        fps_points=args.fps_points,
        depth_gate_k=args.depth_gate_k,
    )
    embedder = PointReID(preprocess_config=preprocess_cfg)
    gallery = Gallery(default_threshold=args.threshold, margin_guard=args.margin_guard)
    tracker = Tracker(alpha=0.5, smooth_window=args.smooth_window)
    db = IdentityDB()
    debug_dir = ensure_debug_dir() if args.debug_identity else None

    if args.rgb:
        rgb = load_rgb_image(args.rgb)
        depth = load_depth(args.depth) if args.depth else demo_depth(rgb)
        outputs = process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, args.threshold, debug_dir, args.smooth_window, debug=args.debug_identity)
        print(json.dumps(outputs, indent=2))
        return

    if args.camera is None:
        raise SystemExit("Specify --camera or --rgb/--depth")

    if args.camera == "demo":
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = demo_depth(rgb)
        outputs = process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, args.threshold, debug_dir, args.smooth_window, debug=args.debug_identity)
        print(json.dumps(outputs, indent=2))
        return

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = demo_depth(rgb)
            outputs = process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, args.threshold, debug_dir, args.smooth_window, debug=args.debug_identity)
            print(json.dumps(outputs))
    finally:
        cap.release()


if __name__ == "__main__":
    main()
