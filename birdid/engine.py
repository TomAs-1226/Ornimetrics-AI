"""BirdID orchestration engine."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import embedding_baseline
from .db import BirdIDDatabase
from .decision import decide
from .segment import DepthSegmenter
from .utils.roi import clamp_roi, pad_roi, roi_area, safe_roi_from_center
from .validate_3d import validate_tracklet


LOGGER = logging.getLogger(__name__)


@dataclass
class BirdIDConfig:
    depth_mode: str = "320x240"
    timestamp_tolerance_ms: float = 120
    min_valid_ratio: float = 0.4
    max_depth_variance: float = 0.06
    min_bbox_area: int = 900
    max_bbox_area: int = 40000
    min_species_confidence: float = 0.5
    buffer_seconds: float = 1.2
    min_frames: int = 3
    max_frames: int = 12
    aggregate_stride: int = 4
    cosine_threshold: float = 0.42
    ema: float = 0.05
    max_prototypes: int = 20
    allow_first_seen_dispense: bool = False
    dispense_threshold: float = 0.32
    margin_threshold: float = 0.05
    min_frames_for_dispense: int = 4
    min_quality_score_for_dispense: float = 0.5
    cooldown_seconds: float = 300.0
    max_dispenses_per_day: int = 20
    validation_min_points: int = 150
    validation_planar_ratio: float = 0.02
    validation_min_thickness: float = 0.01
    validation_max_thickness: float = 0.25
    validation_max_centroid_jitter: float = 0.04
    validation_min_valid_ratio: float = 0.25
    validation_min_size_m: float = 0.04
    validation_max_size_m: float = 0.28
    validation_scattering_min: float = 0.002
    validation_planar_thickness_min: float = 0.006
    pointcloud_units: str = "auto"
    depth_roi_min_area: int = 400
    depth_roi_min_dim: int = 10
    depth_roi_pad_px: int = 8
    depth_roi_fallback: str = "last_or_valid"
    best_k_frames: int = 5
    refresh_days: int = 7
    refresh_confidence_threshold: float = 0.65
    refresh_quality_min: float = 0.4

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
            min_species_confidence=float(_get("yolo", "min_confidence", default=0.5)),
            buffer_seconds=float(_get("tracklets", "buffer_seconds", default=1.2)),
            min_frames=int(_get("tracklets", "min_frames", default=3)),
            max_frames=int(_get("tracklets", "max_frames", default=12)),
            aggregate_stride=int(_get("tracklets", "aggregate_stride", default=4)),
            cosine_threshold=float(_get("matching", "cosine_threshold", default=0.42)),
            ema=float(_get("matching", "ema", default=0.05)),
            max_prototypes=int(_get("matching", "max_prototypes", default=20)),
            allow_first_seen_dispense=bool(_get("dispense", "allow_first_seen_dispense", default=False)),
            dispense_threshold=float(_get("dispense", "threshold", default=0.32)),
            margin_threshold=float(_get("dispense", "margin_threshold", default=0.05)),
            min_frames_for_dispense=int(_get("dispense", "min_frames", default=4)),
            min_quality_score_for_dispense=float(_get("dispense", "min_quality_score", default=0.5)),
            cooldown_seconds=float(_get("cooldown", "seconds", default=300.0)),
            max_dispenses_per_day=int(_get("cooldown", "max_per_day", default=20)),
            validation_min_points=int(_get("validation", "min_points", default=150)),
            validation_planar_ratio=float(_get("validation", "planar_ratio", default=0.02)),
            validation_min_thickness=float(_get("validation", "min_thickness", default=0.01)),
            validation_max_thickness=float(_get("validation", "max_thickness", default=0.25)),
            validation_max_centroid_jitter=float(_get("validation", "max_centroid_jitter", default=0.04)),
            validation_min_valid_ratio=float(_get("validation", "min_valid_ratio", default=0.25)),
            validation_min_size_m=float(_get("validation", "min_size_m", default=0.04)),
            validation_max_size_m=float(_get("validation", "max_size_m", default=0.28)),
            validation_scattering_min=float(_get("validation", "scattering_min", default=0.002)),
            validation_planar_thickness_min=float(_get("validation", "planar_thickness_min", default=0.006)),
            pointcloud_units=str(_get("validation", "pointcloud_units", default="auto")),
            depth_roi_min_area=int(_get("depth", "roi_min_area", default=400)),
            depth_roi_min_dim=int(_get("depth", "roi_min_dim", default=10)),
            depth_roi_pad_px=int(_get("depth", "roi_pad_px", default=8)),
            depth_roi_fallback=str(_get("depth", "roi_fallback", default="last_or_valid")),
            best_k_frames=int(_get("tracklets", "best_k_frames", default=5)),
            refresh_days=int(_get("matching", "refresh_days", default=7)),
            refresh_confidence_threshold=float(_get("matching", "refresh_confidence_threshold", default=0.65)),
            refresh_quality_min=float(_get("matching", "refresh_quality_min", default=0.4)),
        )


@dataclass
class TrackletBuffer:
    species: str
    track_id: int
    detections: List[Dict] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    depth_masks: List[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    last_timestamp: float = 0.0

    def add(self, detection: Dict, embedding: np.ndarray, depth: np.ndarray, mask: np.ndarray, quality: float):
        self.detections.append(detection)
        self.embeddings.append(embedding)
        self.depth_masks.append((depth, mask))
        self.qualities.append(float(quality))
        self.last_timestamp = detection["timestamp"]
        if len(self.detections) > 64:
            self.detections.pop(0)
            self.embeddings.pop(0)
            self.depth_masks.pop(0)
            self.qualities.pop(0)

    def aggregate(self, top_k: int) -> np.ndarray:
        if not self.embeddings:
            return np.zeros(1, dtype=np.float32)
        if top_k <= 0 or top_k >= len(self.embeddings):
            selected = list(range(len(self.embeddings)))
        else:
            selected = np.argsort(self.qualities)[::-1][:top_k]
        mat = np.stack([self.embeddings[i] for i in selected], axis=0)
        mean = mat.mean(axis=0)
        norm = np.linalg.norm(mean) + 1e-8
        return mean / norm

    def top_depth_masks(self, top_k: int) -> List[tuple[np.ndarray, np.ndarray]]:
        if top_k <= 0 or top_k >= len(self.depth_masks):
            return list(self.depth_masks)
        selected = np.argsort(self.qualities)[::-1][:top_k]
        return [self.depth_masks[i] for i in selected]


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
        self._last_valid_roi: Dict[int, tuple[int, int, int, int]] = {}

    def _fallback_roi(self, depth_frame, track_id: int, clamped_roi):
        dh, dw = depth_frame.depth.shape[:2]
        fallback = None
        if self.config.depth_roi_fallback in {"last_or_valid", "last"}:
            last = self._last_valid_roi.get(track_id)
            if last:
                fallback = clamp_roi(last, dw, dh)
        if fallback is None and self.config.depth_roi_fallback in {"last_or_valid", "valid_mask", "blob"}:
            ys, xs = np.nonzero(depth_frame.valid)
            if len(xs) > 0:
                fallback = clamp_roi((xs.min(), ys.min(), xs.max(), ys.max()), dw, dh)
        if fallback is None:
            fallback = safe_roi_from_center(dw / 2.0, dh / 2.0, dw * 0.4, dh * 0.4, dw, dh)
        return fallback

    def _crop_depth(self, depth_frame, bbox_xyxy: List[float], track_id: int):
        dh, dw = depth_frame.depth.shape[:2]
        raw_roi = self.calib.map_bbox(bbox_xyxy, depth_frame.depth.shape)
        clamped = clamp_roi(raw_roi, dw, dh)
        padded = clamp_roi(pad_roi(clamped, self.config.depth_roi_pad_px), dw, dh)
        roi = padded
        width = roi[2] - roi[0]
        height = roi[3] - roi[1]
        if roi_area(roi) < self.config.depth_roi_min_area or width < self.config.depth_roi_min_dim or height < self.config.depth_roi_min_dim:
            roi = clamp_roi(pad_roi(clamped, self.config.depth_roi_pad_px * 2), dw, dh)
        width = roi[2] - roi[0]
        height = roi[3] - roi[1]
        if roi_area(roi) < self.config.depth_roi_min_area or width < self.config.depth_roi_min_dim or height < self.config.depth_roi_min_dim:
            roi = self._fallback_roi(depth_frame, track_id, clamped)
        x1, y1, x2, y2 = map(int, roi)
        depth_crop = depth_frame.depth[y1 : y2 + 1, x1 : x2 + 1]
        valid_crop = depth_frame.valid[y1 : y2 + 1, x1 : x2 + 1]
        metrics = {
            "raw_depth_roi": raw_roi,
            "clamped_depth_roi": clamped,
            "used_depth_roi": roi,
            "roi_area": roi_area(roi),
            "roi_valid_ratio": float(valid_crop.mean()) if valid_crop.size else 0.0,
        }
        if valid_crop.any():
            self._last_valid_roi[track_id] = (x1, y1, x2, y2)
        return depth_crop, valid_crop, metrics

    def _frame_quality_score(self, depth: np.ndarray, mask: np.ndarray) -> float:
        if depth.size == 0 or mask.size == 0:
            return 0.0
        valid_ratio = float(mask.mean())
        if valid_ratio <= 0:
            return 0.0
        variance = float(np.nanvar(depth[mask])) if np.any(mask) else float("inf")
        variance_score = 1.0 if variance == 0 else min(1.0, self.config.max_depth_variance / (variance + 1e-6))
        return float(max(0.0, min(1.0, 0.6 * valid_ratio + 0.4 * variance_score)))

    def process_detection(self, detection: Dict, depth_frame) -> Optional[Dict]:
        if detection.get("conf", 1.0) < self.config.min_species_confidence:
            return None
        bbox = detection.get("bbox_xyxy")
        if bbox is None:
            return None
        if abs(depth_frame.timestamp - detection.get("timestamp", 0.0)) > self.config.timestamp_tolerance_ms / 1000.0:
            return None
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < self.config.min_bbox_area or area > self.config.max_bbox_area:
            return None
        depth_crop, valid_crop, roi_metrics = self._crop_depth(depth_frame, bbox, int(detection.get("track_id", -1)))
        depth_clean, mask = self.segmenter.segment(depth_crop, valid_crop)
        if mask.sum() == 0 and valid_crop.any():
            depth_clean = depth_crop.astype(np.float32, copy=False)
            mask = valid_crop.astype(bool, copy=False)
        quality = self._frame_quality_score(depth_clean, mask)
        if mask.sum() == 0:
            return None
        embedding = self.embedder(depth_clean, mask)
        track_id = int(detection["track_id"])
        species = detection["species"]
        buf = self.tracklets.get(track_id)
        if buf and buf.species != species:
            self._finalize(track_id)
            buf = None
        if buf is None:
            buf = TrackletBuffer(species=species, track_id=track_id)
            self.tracklets[track_id] = buf
        buf.add(detection, embedding, depth_clean, mask, quality)
        detection["roi_metrics"] = roi_metrics
        if len(buf.embeddings) >= self.config.max_frames:
            buf.detections.pop(0)
            buf.embeddings.pop(0)
            buf.depth_masks.pop(0)
            buf.qualities.pop(0)
        if len(buf.embeddings) >= self.config.min_frames and len(buf.embeddings) % self.config.aggregate_stride == 0:
            return self._finalize(track_id)
        return None

    def _finalize(self, track_id: int) -> Optional[Dict]:
        buf = self.tracklets.get(track_id)
        if not buf:
            return None
        agg = buf.aggregate(self.config.best_k_frames)
        species = buf.species
        selected_masks = buf.top_depth_masks(self.config.best_k_frames)
        validation = validate_tracklet(selected_masks, self.config)
        match = self.db.match(species, agg, self.config.cosine_threshold)
        now_ts = time.time()
        if match.individual_id is not None:
            self.db.mark_seen(match.individual_id, now_ts)
        refresh_ok = (
            validation.is_valid
            and match.individual_id is not None
            and match.is_match
            and match.confidence >= self.config.refresh_confidence_threshold
            and validation.quality_score >= self.config.refresh_quality_min
            and self.db.is_refresh_stale(match.individual_id, now_ts, self.config.refresh_days)
        )
        if refresh_ok:
            self.db.update_prototype(match.individual_id, agg, self.config.ema, self.config.max_prototypes, now_ts=now_ts)
        decision = decide(
            species=species,
            track_id=track_id,
            embedding=agg,
            match=match,
            validation=validation,
            frames_used=len(buf.embeddings),
            config=self.config,
            db=self.db,
            now_ts=now_ts,
        )
        start_ts = buf.detections[0]["timestamp"]
        end_ts = buf.detections[-1]["timestamp"]
        result = {
            "time_range": (start_ts, end_ts),
            "species": species,
            "track_id": track_id,
            "individual_id": decision.individual_id,
            "confidence": match.confidence,
            "min_dist": match.distance,
            "second_best_dist": match.second_best,
            "margin": decision.margin,
            "frames_used": len(buf.embeddings),
            "quality_score": validation.quality_score,
            "decision": decision.decision,
            "deny_reason": decision.deny_reason,
            "cooldown_remaining": decision.cooldown_remaining,
            "roi_debug": buf.detections[-1].get("roi_metrics", {}),
        }
        LOGGER.info("birdid_decision", extra={"decision": result})
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
