"""Utilities for timestamp matching between RGB detections and depth frames."""
from __future__ import annotations

from typing import Iterable, List, Optional


def closest_by_timestamp(target: float, candidates: Iterable[float], tolerance: float) -> Optional[float]:
    """Return the candidate timestamp closest to ``target`` within ``tolerance`` seconds."""
    best = None
    best_delta = tolerance
    for ts in candidates:
        delta = abs(ts - target)
        if delta <= best_delta:
            best_delta = delta
            best = ts
    return best


def match_depth_frames(detection_timestamp: float, depth_frames: List, tolerance_s: float):
    """Return the depth frame whose timestamp is closest to the detection.

    ``depth_frames`` is mutated: stale frames older than ``tolerance_s`` past the
    detection timestamp are discarded to keep latency bounded.
    """
    if not depth_frames:
        return None
    best_idx = None
    best_delta = tolerance_s
    for idx, frame in enumerate(depth_frames):
        delta = abs(frame.timestamp - detection_timestamp)
        if delta <= best_delta:
            best_delta = delta
            best_idx = idx
    # Drop frames older than detection - tolerance for freshness
    depth_frames[:] = [f for f in depth_frames if f.timestamp >= detection_timestamp - tolerance_s]
    if best_idx is None:
        return None
    return depth_frames[best_idx]
