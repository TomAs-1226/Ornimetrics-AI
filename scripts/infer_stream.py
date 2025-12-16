#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid
import cv2
import numpy as np

from src.yolo_detect import YOLODetector, load_rgb_image
from src.depth_to_points import backproject_depth, BackprojectStats
from src.reid_embedder import PointReID
from src.gallery import Gallery, cosine_distance
from src.tracker import Tracker
from src.naming import generate_name
from src.db import IdentityDB
from src.output_schema import format_detection
from src.pc_preprocess import PreprocessConfig
from src.bird_point_extractor import AntiSpoofConfig, BirdPointCloudExtractor, BirdExtractorConfig
from src.multi_instance_depth import MultiInstanceConfig, MultiInstanceDepthProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + point-cloud re-id stream")
    parser.add_argument("--camera", default=None, help="Camera index or 'demo'")
    parser.add_argument("--rgb", help="RGB image path for offline", default=None)
    parser.add_argument("--depth", help="Depth image path for offline", default=None)
    parser.add_argument("--model", default=None, help="YOLO model path (defaults to bundled best.pt when present)")
    parser.add_argument("--backbone", choices=["light", "heavy", "xheavy"], default="heavy", help="Point-cloud re-id backbone")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("fx", "fy", "cx", "cy"), default=[525.0, 525.0, 319.5, 239.5])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--margin-guard", type=float, default=0.02)
    parser.add_argument("--geometry-guard", type=float, default=0.02)
    parser.add_argument("--debug_identity", action="store_true", help="Enable verbose identity debugging")
    parser.add_argument("--plane-removal", action="store_true", default=True)
    parser.add_argument("--no-plane-removal", dest="plane_removal", action="store_false")
    parser.add_argument("--normalization", choices=["center_only", "center_and_scale", "center_and_scale_with_scale_feature"], default="center_only")
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--fps-points", type=int, default=2048)
    parser.add_argument("--depth-gate-k", type=float, default=2.5)
    parser.add_argument("--use-bird-extractor", action="store_true", help="Enable depth bird-only extractor")
    parser.add_argument("--bird-cluster-radius", type=float, default=0.02)
    parser.add_argument("--bird-min-points", type=int, default=64)
    parser.add_argument("--enable-anti-spoof", action="store_true", help="Reject planar/background point clouds")
    parser.add_argument("--anti-spoof-threshold", type=float, default=0.6, help="Threshold for anti-spoof score")
    parser.add_argument("--anti-spoof-model", type=Path, default=None, help="Optional TinyPointNet weights for anti-spoofing")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--enable-multi-instance", action="store_true", help="Enable depth clustering assignment per detection")
    parser.add_argument("--multi-voxel", type=float, default=0.01)
    parser.add_argument("--multi-eps", type=float, default=0.03)
    parser.add_argument("--multi-min-points", type=int, default=40)
    parser.add_argument("--multi-min-iou", type=float, default=0.05)
    parser.add_argument("--multi-max-cost", type=float, default=2.5)
    parser.add_argument("--pi-profile", action="store_true", help="Pi-friendly settings for clustering and preprocessing")
    parser.add_argument("--debug-multi-instance", action="store_true", help="Log clustering assignments and gating decisions")
    return parser.parse_args()


def _points_from_depth(depth, intrinsics, bbox):
    """Accept either a depth map or a precomputed Nx3 point cloud."""
    if depth is None:
        empty_stats = BackprojectStats(0, 0, (0, 0, 0, 0), 0.0, 0.0)
        return np.zeros((0, 3), dtype=np.float32), empty_stats
    if depth.ndim == 2 and depth.shape[1] == 3:
        pc = depth.astype(np.float32)
    elif depth.ndim == 2:
        return backproject_depth(depth, intrinsics, bbox)
    elif depth.ndim == 3 and depth.shape[1] == 3:
        pc = depth.reshape(-1, 3).astype(np.float32)
    else:
        raise ValueError("Depth input must be 2D depth map or Nx3 point cloud")
    stats = BackprojectStats(
        num_depth_pixels=pc.shape[0],
        num_points_raw=int(pc.shape[0]),
        bbox=tuple(int(x) for x in bbox) if bbox is not None else (0, 0, 0, 0),
        depth_min=float(pc[:, 2].min()) if pc.size else 0.0,
        depth_max=float(pc[:, 2].max()) if pc.size else 0.0,
    )
    return pc, stats


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


def save_debug_overlay(rgb, detections, debug_dir, frame_id=None):
    if rgb is None or debug_dir is None:
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(x) for x in bbox]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = det.get("class_name", "")
        action = det.get("action") or det.get("decision_state")
        cid = det.get("instance_cluster_id")
        cid_txt = f"c{cid}" if cid is not None else None
        label_parts = [p for p in [cls, cid_txt, action] if p]
        label = " | ".join(label_parts) if label_parts else cls
        if label:
            cv2.putText(bgr, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    frame_name = f"frame_{frame_id:04d}.jpg" if frame_id is not None else "frame.jpg"
    out_path = debug_dir / frame_name
    cv2.imwrite(str(out_path), bgr)
    return out_path


def detection_debug_block(det, preprocess_stats, embedding, gallery_scores, decision, tracker_stats):
    sorted_scores = [
        {"id": s.get("id"), "dist": float(s.get("dist", 0.0)), "geometry": float(s.get("geometry", float("inf")))}
        for s in gallery_scores
    ]
    norm = float(np.linalg.norm(embedding)) if embedding is not None else 0.0
    block = {
        "class": det.get("class_name"),
        "bbox": det.get("bbox"),
        "birdness": det.get("birdness"),
        "preprocessing": preprocess_stats,
        "embedding_norm": norm,
        "top5": sorted_scores,
        "best_match_id": decision.get("best_id"),
        "best_dist": decision.get("best_dist"),
        "second_best_dist": decision.get("second_best"),
        "margin": decision.get("margin"),
        "threshold": decision.get("threshold"),
        "geometry_dist": decision.get("geometry_dist"),
        "new_identity": decision.get("new_identity"),
        "reason": decision.get("reason"),
        "decision_state": det.get("decision_state"),
        "decision_reason": det.get("decision_reason"),
        "action": det.get("action"),
        "tracker": tracker_stats,
    }
    return block


def process_frame(
    rgb,
    depth,
    det_model,
    embedder,
    gallery,
    tracker,
    db,
    intrinsics,
    threshold,
    debug_dir=None,
    smooth_window=5,
    debug=False,
    return_debug=False,
    bird_extractor: Optional[BirdPointCloudExtractor] = None,
    multi_instance: Optional[MultiInstanceDepthProcessor] = None,
    debug_multi_instance: bool = False,
):
    detections = det_model.detect(rgb)
    processed = []
    frame_debug = []
    assignments = {}
    multi_debug = {}
    if multi_instance is not None:
        assignments, multi_debug = multi_instance.assign_clusters(depth, intrinsics, detections, debug=debug_multi_instance)
    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        bp_stats = None
        assign_info = assignments.get(idx, {}) if assignments else {}
        pre_points = None
        pre_stats = None
        if assign_info:
            if assign_info.get("cluster_id") is not None:
                det["instance_cluster_id"] = assign_info.get("cluster_id")
            if "birdness" in assign_info:
                det["birdness"] = {"score": assign_info.get("birdness"), "stats": assign_info.get("stats", {})}
                pre_stats = assign_info.get("stats", {})
            if assign_info.get("deny_reason"):
                det["deny_reason"] = assign_info.get("deny_reason")
                det["decision_state"] = "rejected"
                det["decision_reason"] = assign_info.get("deny_reason")
                det["action"] = "rejected"
                det["preprocess_stats"] = assign_info.get("stats", {})
                det["embedding"] = np.zeros((embedder.model.emb_dims,), dtype=np.float32)
                processed.append(det)
                continue
            pre_points = assign_info.get("points")
        if bird_extractor is not None:
            extraction = bird_extractor.extract(
                rgb,
                depth,
                bbox,
                intrinsics,
                mask=det.get("mask"),
                precluster_points=pre_points,
                precluster_stats=pre_stats,
            )
            det["birdness"] = {"score": extraction.quality, "stats": extraction.stats}
            bp_stats = BackprojectStats(
                num_depth_pixels=int(extraction.stats.get("num_depth_pixels", 0)),
                num_points_raw=int(extraction.stats.get("num_points_raw", 0)),
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                depth_min=float(extraction.stats.get("depth_min", 0.0)),
                depth_max=float(extraction.stats.get("depth_max", 0.0)),
            )
            points = extraction.points
            if extraction.deny_reason:
                det["deny_reason"] = extraction.deny_reason
                det["quality"] = extraction.quality
                det["preprocess_stats"] = extraction.stats
                det["decision"] = {"reason": extraction.deny_reason, "threshold": threshold, "new_identity": False}
                det["decision_state"] = "rejected"
                det["decision_reason"] = extraction.deny_reason
                det["action"] = "rejected"
                det["embedding"] = np.zeros((embedder.model.emb_dims,), dtype=np.float32)
                processed.append(det)
                continue
        else:
            if pre_points is not None:
                points = pre_points.astype(np.float32)
                bp_stats = BackprojectStats(
                    num_depth_pixels=int(pre_points.shape[0]),
                    num_points_raw=int(pre_points.shape[0]),
                    bbox=tuple(int(x) for x in bbox),
                    depth_min=float(pre_points[:, 2].min()) if pre_points.size else 0.0,
                    depth_max=float(pre_points[:, 2].max()) if pre_points.size else 0.0,
                )
            else:
                points, bp_stats = _points_from_depth(depth, intrinsics, bbox)
        embed_result = embedder.embed(points)
        emb = embed_result.embedding.cpu().numpy()
        det["embedding"] = emb
        best_id, best_dist, scored = gallery.match(det["class_name"], emb, candidate_points=embed_result.points)
        second_best = scored[1]["dist"] if len(scored) > 1 else 1.0
        best_geom = scored[0].get("geometry", float("inf")) if scored else float("inf")
        margin = second_best - best_dist
        geometry_block = best_geom < float("inf") and best_geom > gallery.geometry_guard
        create_new = best_id is None or gallery.needs_new_identity(det["class_name"], best_dist, geometry=best_geom)
        if geometry_block:
            create_new = False
        reason = "empty_gallery" if best_id is None else ("geometry_guard" if geometry_block else ("above_threshold" if create_new else "matched"))
        threshold_used = gallery.get_threshold(det["class_name"])
        det_info = {
            "best_id": best_id,
            "best_dist": float(best_dist),
            "second_best": float(second_best),
            "margin": float(margin),
            "threshold": float(threshold_used),
            "geometry_dist": float(best_geom),
            "new_identity": bool(create_new),
            "reason": reason,
            "top5": scored,
        }
        action = "rejected" if geometry_block else ("dispense" if not create_new and reason == "matched" else "accepted")
        det["decision_state"] = "rejected" if det.get("deny_reason") or geometry_block else ("accepted" if action == "accepted" else "dispensing")
        det["decision_reason"] = reason
        det["action"] = action
        if geometry_block:
            det["deny_reason"] = reason
            det["individual_id"] = None
            det["individual_name"] = None
            det["min_dist"] = float(best_dist)
            det["second_best_dist"] = float(second_best)
            det["margin"] = float(margin)
            processed.append(det)
            continue
        if create_new:
            indiv_id = str(uuid.uuid4())
            name = generate_name()
            gallery.update(det["class_name"], indiv_id, emb, allow=True, name=name, geometry_points=embed_result.points)
            db.upsert_individual(indiv_id, name, det["class_name"], emb)
            det["individual_id"] = indiv_id
            det["individual_name"] = name
            det["min_dist"] = 1.0
            det["second_best_dist"] = second_best
            det["margin"] = margin
        else:
            gallery.update(
                det["class_name"],
                best_id,
                emb,
                allow=gallery.should_update_identity(det["class_name"], best_dist),
                geometry_points=embed_result.points,
            )
            info = db.get_identity(best_id)
            det["individual_id"] = best_id
            det["individual_name"] = info[0] if info else None
            det["min_dist"] = float(best_dist)
            det["second_best_dist"] = float(second_best)
            det["margin"] = float(margin)
        det["bp_stats"] = bp_stats.__dict__ if bp_stats else {}
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
        block = detection_debug_block(
            det,
            {**det.get("bp_stats", {}), **det.get("preprocess_stats", {})},
            use_emb,
            decision.get("top5", []),
            decision,
            tracker_stats,
        )
        frame_debug.append(block)
    outputs = [format_detection(d) for d in tracked]
    if debug and debug_dir is not None:
        frame_id = len(list(debug_dir.glob("frame_*.json")))
        overlay_path = save_debug_overlay(rgb, tracked, debug_dir, frame_id=frame_id)
        with open(debug_dir / f"frame_{frame_id:04d}.json", "w", encoding="utf-8") as f:
            json.dump({"detections": frame_debug, "multi_instance": multi_debug}, f, indent=2, default=lambda o: o if isinstance(o, (int, float, str)) else str(o))
        if overlay_path is not None:
            print(f"Saved debug overlay: {overlay_path}")
        print(json.dumps(frame_debug, indent=2))
    if return_debug:
        return outputs, frame_debug
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
    embedder = PointReID(preprocess_config=preprocess_cfg, model_name=args.backbone)
    gallery = Gallery(
        default_threshold=args.threshold,
        margin_guard=args.margin_guard,
        geometry_guard=args.geometry_guard,
    )
    tracker = Tracker(alpha=0.5, smooth_window=args.smooth_window)
    db = IdentityDB()
    debug_dir = ensure_debug_dir() if args.debug_identity else None
    bird_extractor = None
    multi_instance = None
    mi_cfg = MultiInstanceConfig(
        enabled=args.enable_multi_instance,
        voxel_size=args.multi_voxel,
        eps=args.multi_eps,
        min_points=args.multi_min_points,
        min_iou=args.multi_min_iou,
        max_cost=args.multi_max_cost,
        pi_profile=args.pi_profile,
    )
    if args.enable_multi_instance:
        multi_instance = MultiInstanceDepthProcessor(mi_cfg)
    if args.use_bird_extractor:
        bird_extractor = BirdPointCloudExtractor(
            BirdExtractorConfig(
                depth_gate_k=args.depth_gate_k,
                cluster_radius=args.bird_cluster_radius,
                min_cluster_points=args.bird_min_points,
                debug_dir=debug_dir,
                anti_spoof=AntiSpoofConfig(
                    enabled=args.enable_anti_spoof,
                    threshold=args.anti_spoof_threshold,
                    model_path=args.anti_spoof_model,
                ),
            )
        )

    if args.rgb:
        rgb = load_rgb_image(args.rgb)
        depth = load_depth(args.depth) if args.depth else demo_depth(rgb)
        outputs, frame_debug = process_frame(
            rgb,
            depth,
            det_model,
            embedder,
            gallery,
            tracker,
            db,
            intrinsics,
            args.threshold,
            debug_dir,
            args.smooth_window,
            debug=args.debug_identity,
            bird_extractor=bird_extractor,
            multi_instance=multi_instance,
            debug_multi_instance=args.debug_multi_instance,
            return_debug=True,
        )
        print(json.dumps(outputs, indent=2))
        return

    if args.camera is None:
        raise SystemExit("Specify --camera or --rgb/--depth")

    if args.camera == "demo":
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = demo_depth(rgb)
        outputs, frame_debug = process_frame(
            rgb,
            depth,
            det_model,
            embedder,
            gallery,
            tracker,
            db,
            intrinsics,
            args.threshold,
            debug_dir,
            args.smooth_window,
            debug=args.debug_identity,
            bird_extractor=bird_extractor,
            multi_instance=multi_instance,
            debug_multi_instance=args.debug_multi_instance,
            return_debug=True,
        )
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
            outputs = process_frame(
                rgb,
                depth,
                det_model,
                embedder,
                gallery,
                tracker,
                db,
                intrinsics,
                args.threshold,
                debug_dir,
                args.smooth_window,
                debug=args.debug_identity,
                bird_extractor=bird_extractor,
                multi_instance=multi_instance,
                debug_multi_instance=args.debug_multi_instance,
            )
            print(json.dumps(outputs))
    finally:
        cap.release()


if __name__ == "__main__":
    main()
