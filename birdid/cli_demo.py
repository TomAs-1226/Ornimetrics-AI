"""CLI demo for BirdID.

The live mode expects YOLO detections streamed in via STDIN as JSON lines or can
be easily adapted to your existing pipeline. For CI and development we simulate
YOLO detections and depth frames so the full path runs without hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterable, List

import numpy as np

from .camera.cs20 import CS20Camera, DepthFrame
from .engine import BirdIDConfig, BirdIDEngine


def _simulate_detections(num: int = 12) -> Iterable[dict]:
    now = time.time()
    for i in range(num):
        yield {
            "timestamp": now + i * 0.1,
            "bbox_xyxy": [40, 40, 120, 140],
            "conf": 0.9,
            "species": "sparrow",
            "track_id": 1,
        }


def _simulate_depth_frames(num: int = 20) -> List[DepthFrame]:
    frames = []
    now = time.time()
    for i in range(num):
        depth = np.full((240, 320), 0.38 + 0.002 * np.sin(i), dtype=np.float32)
        valid = np.ones_like(depth, dtype=bool)
        frames.append(DepthFrame(timestamp=now + i * 0.08, depth=depth, valid=valid))
    return frames


def run_live(engine: BirdIDEngine) -> None:
    cam = CS20Camera(mode=engine.config.depth_mode)
    cam.start()
    try:
        buffer = []
        for line in sys.stdin:
            try:
                det = json.loads(line)
            except json.JSONDecodeError:
                continue
            depth_frame = cam.get_latest(timeout=0.05)
            if depth_frame is None:
                continue
            res = engine.process_detection(det, depth_frame)
            if res:
                print(json.dumps(res, indent=2))
            for out in engine.flush_expired():
                print(json.dumps(out, indent=2))
    finally:
        cam.stop()


def run_recorded(engine: BirdIDEngine) -> None:
    detections = list(_simulate_detections())
    depth_frames = _simulate_depth_frames()
    for det in detections:
        # pick closest depth frame
        closest = min(depth_frames, key=lambda f: abs(f.timestamp - det["timestamp"]))
        res = engine.process_detection(det, closest)
        if res:
            print(json.dumps(res, indent=2))
    for out in engine.flush_expired(time.time() + 2):
        print(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdID CLI demo")
    parser.add_argument("--config", default="birdid/config_default.json")
    parser.add_argument("--mode", choices=["live", "recorded"], default="recorded")
    args = parser.parse_args()

    cfg = BirdIDConfig.from_file(args.config) if args.config else BirdIDConfig()
    engine = BirdIDEngine(config=cfg)
    if args.mode == "live":
        run_live(engine)
    else:
        run_recorded(engine)


if __name__ == "__main__":
    main()
