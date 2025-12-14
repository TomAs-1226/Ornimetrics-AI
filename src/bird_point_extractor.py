"""Depth-driven bird-only point-cloud extraction with safety gating.

All logic is optional and lives behind configuration flags so existing
RGB+YOLO behaviour remains unchanged unless explicitly enabled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.depth_to_points import backproject_depth
from src.pc_preprocess import _get_o3d


DEFAULT_DENY = "low_confidence_cloud"


@dataclass
class BirdExtractorConfig:
    depth_gate_k: float = 2.5
    plane_distance: float = 0.01
    cluster_radius: float = 0.02
    min_cluster_points: int = 64
    planarity_ratio: float = 0.02
    max_aspect_ratio: float = 6.0
    max_depth_jump: float = 0.05
    quality_floor: float = 0.2
    allow_planar_override: bool = False
    use_mask: bool = False
    debug_dir: Optional[Path] = None


@dataclass
class BirdCloudResult:
    points: np.ndarray
    quality: float
    stats: Dict[str, float] = field(default_factory=dict)
    deny_reason: Optional[str] = None


class BirdPointCloudExtractor:
    def __init__(self, config: Optional[BirdExtractorConfig] = None):
        self.cfg = config or BirdExtractorConfig()
        if self.cfg.debug_dir:
            self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)

    def _depth_gate(self, points: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        stats: Dict[str, float] = {}
        if points.size == 0:
            return points, stats
        depths = points[:, 2]
        med = float(np.median(depths))
        mad = float(np.median(np.abs(depths - med)) + 1e-6)
        low, high = med - self.cfg.depth_gate_k * mad, med + self.cfg.depth_gate_k * mad
        mask = (depths >= low) & (depths <= high)
        gated = points[mask]
        stats.update({"depth_median": med, "depth_mad": mad, "depth_low": low, "depth_high": high, "after_gate": float(len(gated))})
        return gated, stats

    def _remove_plane(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        if points.shape[0] < 3:
            return points, 0.0
        o3d = _get_o3d()
        if o3d is None:
            return points, 0.0
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        plane_model, inliers = cloud.segment_plane(distance_threshold=self.cfg.plane_distance, ransac_n=3, num_iterations=32)
        if not inliers:
            return points, 0.0
        outlier_pc = cloud.select_by_index(inliers, invert=True)
        ratio = 1.0 - len(outlier_pc.points) / max(len(points), 1)
        return np.asarray(outlier_pc.points), float(ratio)

    def _cluster(self, points: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if points.shape[0] == 0:
            return points, {"num_clusters": 0}
        tree = cKDTree(points)
        visited = np.zeros(points.shape[0], dtype=bool)
        clusters: List[np.ndarray] = []
        for i in range(points.shape[0]):
            if visited[i]:
                continue
            inds = tree.query_ball_point(points[i], self.cfg.cluster_radius)
            stack = list(inds)
            comp: List[int] = []
            while stack:
                idx = stack.pop()
                if visited[idx]:
                    continue
                visited[idx] = True
                comp.append(idx)
                neighbours = tree.query_ball_point(points[idx], self.cfg.cluster_radius)
                stack.extend([n for n in neighbours if not visited[n]])
            clusters.append(np.asarray(comp, dtype=int))
        clusters = sorted(clusters, key=lambda c: (-len(c), float(points[c, 2].min())))
        best = clusters[0]
        return points[best], {"num_clusters": len(clusters), "largest_cluster": float(len(best))}

    def _shape_stats(self, points: np.ndarray) -> Dict[str, float]:
        if points.size == 0:
            return {"extent_x": 0.0, "extent_y": 0.0, "extent_z": 0.0, "compactness": 0.0, "aspect": 0.0, "planarity": 1.0}
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)
        planarity = float(eigvals[0] / (eigvals[2] + 1e-6)) if eigvals[2] > 0 else 1.0
        extent = points.max(axis=0) - points.min(axis=0)
        aspect = float(np.max(extent[:2]) / (np.min(extent[:2]) + 1e-6))
        compactness = float(np.mean(np.linalg.norm(centered, axis=1)))
        return {
            "extent_x": float(extent[0]),
            "extent_y": float(extent[1]),
            "extent_z": float(extent[2]),
            "compactness": compactness,
            "aspect": aspect,
            "planarity": planarity,
        }

    def _quality(self, points: np.ndarray, stats: Dict[str, float]) -> Tuple[float, Optional[str]]:
        if points.shape[0] < self.cfg.min_cluster_points:
            return 0.0, "depth_missing"
        if stats.get("planarity", 1.0) < self.cfg.planarity_ratio and not self.cfg.allow_planar_override:
            return 0.0, "planar_surface"
        if stats.get("aspect", 0.0) > self.cfg.max_aspect_ratio:
            return 0.0, "inconsistent_depth_object"
        if stats.get("extent_z", 0.0) > self.cfg.max_depth_jump:
            return 0.0, "no_valid_cluster"
        return max(self.cfg.quality_floor, 1.0 - stats.get("compactness", 0.0)), None

    def extract(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        bbox: Tuple[float, float, float, float],
        intrinsics: Dict[str, float],
        mask: Optional[np.ndarray] = None,
    ) -> BirdCloudResult:
        stats: Dict[str, float] = {}
        if depth is None or intrinsics is None:
            return BirdCloudResult(points=np.zeros((0, 3), dtype=np.float32), quality=0.0, stats=stats, deny_reason="depth_missing")
        x1, y1, x2, y2 = map(int, bbox)
        depth_crop = depth[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
        if depth_crop.size == 0:
            return BirdCloudResult(points=np.zeros((0, 3), dtype=np.float32), quality=0.0, stats=stats, deny_reason="depth_missing")
        if self.cfg.use_mask and mask is not None:
            if mask.shape[:2] != depth.shape:
                raise ValueError("Mask shape must match depth map")
            depth_crop = np.where(mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)], depth_crop, 0)
        depth_crop = depth_crop.astype(np.float32)
        depth_crop[~np.isfinite(depth_crop)] = 0
        points, bp_stats = backproject_depth(depth_crop, intrinsics, None)
        stats.update(bp_stats.__dict__)
        points, gate_stats = self._depth_gate(points)
        stats.update(gate_stats)
        points, plane_ratio = self._remove_plane(points)
        stats["plane_removed_ratio"] = float(plane_ratio)
        clustered, cluster_stats = self._cluster(points)
        stats.update(cluster_stats)
        shape = self._shape_stats(clustered)
        stats.update(shape)
        quality, deny = self._quality(clustered, stats)
        result = BirdCloudResult(points=clustered.astype(np.float32), quality=quality, stats=stats, deny_reason=deny)
        if self.cfg.debug_dir is not None:
            idx = len(list(self.cfg.debug_dir.glob("frame_*.json")))
            with open(self.cfg.debug_dir / f"frame_{idx:04d}.json", "w", encoding="utf-8") as f:
                json.dump({"bbox": bbox, "quality": quality, "deny_reason": deny, "stats": stats}, f, indent=2)
        return result


__all__ = ["BirdPointCloudExtractor", "BirdCloudResult", "BirdExtractorConfig"]
