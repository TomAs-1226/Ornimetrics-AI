#!/usr/bin/env python
import argparse
import json
import uuid
import cv2
import numpy as np

from src.yolo_detect import YOLODetector, load_rgb_image
from src.depth_to_points import backproject_depth
from src.reid_embedder import PointReID
from src.gallery import Gallery
from src.tracker import Tracker
from src.naming import generate_name
from src.db import IdentityDB
from src.output_schema import format_detection


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + point-cloud re-id stream")
    parser.add_argument("--camera", default=None, help="Camera index or 'demo'")
    parser.add_argument("--rgb", help="RGB image path for offline", default=None)
    parser.add_argument("--depth", help="Depth image path for offline", default=None)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("fx", "fy", "cx", "cy"), default=[525.0, 525.0, 319.5, 239.5])
    parser.add_argument("--threshold", type=float, default=0.3)
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


def process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, threshold):
    detections = det_model.detect(rgb)
    processed = []
    for det in detections:
        bbox = det["bbox"]
        pc = backproject_depth(depth, intrinsics, bbox)
        emb = embedder.embed(pc).cpu().numpy()
        det["embedding"] = emb
        indiv_id, dist = gallery.match(det["class_name"], emb)
        if indiv_id is None or gallery.needs_new_identity(det["class_name"], dist):
            indiv_id = str(uuid.uuid4())
            name = generate_name()
            gallery.update(det["class_name"], indiv_id, emb)
            db.upsert_individual(indiv_id, name, det["class_name"], emb)
            det["distance"] = 1.0
            det["margin"] = 0.0
            det["individual_name"] = name
        else:
            gallery.update(det["class_name"], indiv_id, emb)
            det["distance"] = dist
            det["margin"] = threshold - dist
            info = db.get_identity(indiv_id)
            det["individual_name"] = info[0] if info else None
        det["individual_id"] = indiv_id
        processed.append(det)
    tracked = tracker.update(processed)
    outputs = [format_detection(d) for d in tracked]
    return outputs


def main():
    args = parse_args()
    intrinsics = {"fx": args.intrinsics[0], "fy": args.intrinsics[1], "cx": args.intrinsics[2], "cy": args.intrinsics[3]}
    det_model = YOLODetector(args.model)
    embedder = PointReID()
    gallery = Gallery(default_threshold=args.threshold)
    tracker = Tracker(alpha=0.5)
    db = IdentityDB()

    if args.rgb:
        rgb = load_rgb_image(args.rgb)
        depth = load_depth(args.depth) if args.depth else demo_depth(rgb)
        outputs = process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, args.threshold)
        print(json.dumps(outputs, indent=2))
        return

    if args.camera is None:
        raise SystemExit("Specify --camera or --rgb/--depth")

    if args.camera == "demo":
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = demo_depth(rgb)
        outputs = process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, args.threshold)
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
            outputs = process_frame(rgb, depth, det_model, embedder, gallery, tracker, db, intrinsics, args.threshold)
            print(json.dumps(outputs))
    finally:
        cap.release()


if __name__ == "__main__":
    main()

