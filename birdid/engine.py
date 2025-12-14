"""BirdID orchestration engine."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import embedding_baseline
from .db import BirdIDDatabase
from .segment import DepthSegmenter


@dataclass
class BirdIDConfig:
    depth_mode: str = "320x240"
    timestamp_tolerance_ms: float = 120
    min_valid_ratio: float = 0.4
    max_depth_variance: float = 0.06
    min_bbox_area: int = 900
    max_bbox_area: int = 40000
    buffer_seconds: float = 1.2
    min_frames: int = 3
    max_frames: int = 12
    aggregate_stride: int = 4
    cosine_threshold: float = 0.42
    ema: float = 0.05
    max_prototypes: int = 20

    @classmethod
    def from_file(cls, path: str) -> "BirdIDConfig":
        data = json.loads(Path(path).read_text())
        def _get(*keys, default=None):
            cur = data
            for k in keys:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return default
            return cur
        return cls(
            depth_mode=_get("depth", "mode", default="320x240"),
            timestamp_tolerance_ms=float(_get("depth", "timestamp_tolerance_ms", default=120)),
            min_valid_ratio=float(_get("depth", "min_valid_ratio", default=0.4)),
            max_depth_variance=float(_get("depth", "max_depth_variance", default=0.06)),
            min_bbox_area=int(_get("depth", "min_bbox_area", default=900)),
            max_bbox_area=int(_get("depth", "max_bbox_area", default=40000)),
            buffer_seconds=float(_get("tracklets", "buffer_seconds", default=1.2)),
            min_frames=int(_get("tracklets", "min_frames", default=3)),
            max_frames=int(_get("tracklets", "max_frames", default=12)),
            aggregate_stride=int(_get("tracklets", "aggregate_stride", default=4)),
            cosine_threshold=float(_get("matching", "cosine_threshold", default=0.42)),
            ema=float(_get("matching", "ema", default=0.05)),
            max_prototypes=int(_get("matching", "max_prototypes", default=20)),
        )


@dataclass
class TrackletBuffer:
    species: str
    track_id: int
    detections: List[Dict] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    last_timestamp: float = 0.0

    def add(self, detection: Dict, embedding: np.ndarray):
        self.detections.append(detection)
        self.embeddings.append(embedding)
        self.last_timestamp = detection["timestamp"]
        if len(self.detections) > 64:
            self.detections.pop(0)
            self.embeddings.pop(0)

    def aggregate(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros(1, dtype=np.float32)
        mat = np.stack(self.embeddings, axis=0)
        mean = mat.mean(axis=0)
        norm = np.linalg.norm(mean) + 1e-8
        return mean / norm


class Calibration:
    def __init__(self, path: str = "birdid/calib/rgb_to_depth.json"):
        self.path = Path(path)
        self.matrix = np.eye(3, dtype=np.float32)
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.matrix = np.asarray(data.get("matrix", np.eye(3)), dtype=np.float32)

    def map_bbox(self, bbox_xyxy: List[float], depth_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox_xyxy
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1],
        ], dtype=np.float32).T
        proj = self.matrix @ corners
        proj /= proj[2:3, :]
        xs = np.clip(proj[0], 0, depth_shape[1] - 1)
        ys = np.clip(proj[1], 0, depth_shape[0] - 1)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class BirdIDEngine:
    def __init__(self, config: Optional[BirdIDConfig] = None, embedder: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None, db_path: str = "birdid.sqlite"):
        self.config = config or BirdIDConfig()
        self.embedder = embedder or embedding_baseline.compute_baseline_embedding
        self.db = BirdIDDatabase(db_path)
        self.segmenter = DepthSegmenter(min_valid_ratio=self.config.min_valid_ratio)
        self.calib = Calibration()
        self.tracklets: Dict[int, TrackletBuffer] = {}

    def _crop_depth(self, depth_frame, bbox_xyxy: List[float]):
        x1, y1, x2, y2 = self.calib.map_bbox(bbox_xyxy, depth_frame.depth.shape)
        depth_crop = depth_frame.depth[y1 : y2 + 1, x1 : x2 + 1]
        valid_crop = depth_frame.valid[y1 : y2 + 1, x1 : x2 + 1]
        return depth_crop, valid_crop

    def _frame_quality_ok(self, depth: np.ndarray, mask: np.ndarray) -> bool:
        valid_ratio = mask.mean() if mask.size else 0.0
        if valid_ratio < self.config.min_valid_ratio:
            return False
        variance = np.nanvar(depth[mask]) if np.any(mask) else np.inf
        if variance > self.config.max_depth_variance:
            return False
        return True

    def process_detection(self, detection: Dict, depth_frame) -> Optional[Dict]:
        bbox = detection.get("bbox_xyxy")
        if bbox is None:
            return None
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < self.config.min_bbox_area or area > self.config.max_bbox_area:
            return None
        depth_crop, valid_crop = self._crop_depth(depth_frame, bbox)
        depth_clean, mask = self.segmenter.segment(depth_crop, valid_crop)
        if not self._frame_quality_ok(depth_clean, mask):
            return None
        embedding = self.embedder(depth_clean, mask)
        track_id = int(detection["track_id"])
        buf = self.tracklets.setdefault(track_id, TrackletBuffer(species=detection["species"], track_id=track_id))
        buf.add(detection, embedding)
        if len(buf.embeddings) >= self.config.max_frames:
            buf.detections.pop(0)
            buf.embeddings.pop(0)
        if len(buf.embeddings) >= self.config.min_frames and len(buf.embeddings) % self.config.aggregate_stride == 0:
            return self._finalize(track_id)
        return None

    def _finalize(self, track_id: int) -> Optional[Dict]:
        buf = self.tracklets.get(track_id)
        if not buf:
            return None
        agg = buf.aggregate()
        species = buf.species
        match = self.db.match(species, agg, self.config.cosine_threshold)
        self.db.update_prototype(match.individual_id, agg, self.config.ema, self.config.max_prototypes)
        start_ts = buf.detections[0]["timestamp"]
        end_ts = buf.detections[-1]["timestamp"]
        result = {
            "time_range": (start_ts, end_ts),
            "species": species,
            "track_id": track_id,
            "individual_id": match.individual_id,
            "confidence": match.confidence,
            "min_dist": match.distance,
            "frames_used": len(buf.embeddings),
        }
        return result

    def flush_expired(self, now: Optional[float] = None) -> List[Dict]:
        now = now or time.time()
        outputs = []
        expired = [tid for tid, buf in self.tracklets.items() if now - buf.last_timestamp > self.config.buffer_seconds]
        for tid in expired:
            res = self._finalize(tid)
            if res:
                outputs.append(res)
            self.tracklets.pop(tid, None)
        return outputs

    def close(self):
        self.db.close()
