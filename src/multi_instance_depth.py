"""Multi-instance depth clustering and assignment for bird isolation.

This module clusters depth points globally, scores clusters for birdness,
and assigns them one-to-one to YOLO detections using Hungarian matching.
All behaviour is behind configuration flags so the legacy single-bird path
remains unchanged when the feature is disabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.depth_to_points import BackprojectStats


def _get_o3d():  # pragma: no cover - optional dependency
    try:
        import open3d as o3d

        return o3d
    except Exception:
        return None


@dataclass
class MultiInstanceConfig:
    enabled: bool = False
    voxel_size: float = 0.01
    eps: float = 0.03
    min_points: int = 40
    max_points: int = 50000
    zone_margin: float = 0.05
    assign_w_iou: float = 0.7
    assign_w_depth: float = 0.2
    assign_w_birdness: float = 0.1
    min_iou: float = 0.05
    max_cost: float = 2.5
    split_iou_threshold: float = 0.05
    split_birdness: float = 0.4
    use_radius_clustering: bool = False
    radius: float = 0.04
    pi_profile: bool = False

    def for_pi(self) -> "MultiInstanceConfig":
        if not self.pi_profile:
            return self
        tuned = MultiInstanceConfig(**self.__dict__)
        tuned.voxel_size = max(self.voxel_size, 0.02)
        tuned.use_radius_clustering = True
        tuned.eps = 0.04
        tuned.radius = 0.05
        tuned.max_points = min(self.max_points, 20000)
        tuned.assign_w_depth = 0.1
        tuned.assign_w_birdness = 0.2
        return tuned


@dataclass
class ClusterInfo:
    id: int
    points: np.ndarray
    pixel_xy: np.ndarray
    bbox_2d: Tuple[int, int, int, int]
    centroid: np.ndarray
    depth_median: float
    birdness: float
    stats: Dict[str, float] = field(default_factory=dict)


def _compute_birdness(points: np.ndarray) -> Tuple[float, Dict[str, float]]:
    stats: Dict[str, float] = {}
    if points.shape[0] == 0:
        return 0.0, stats
    centroid = points.mean(axis=0)
    diffs = points - centroid
    cov = np.cov(diffs.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.clip(np.sort(eigvals), 1e-8, None)
    l1, l2, l3 = eigvals
    planarity = float((l2 - l1) / (l3 + 1e-6))
    linearity = float((l3 - l2) / (l3 + 1e-6))
    surface_variation = float(l1 / (l1 + l2 + l3 + 1e-6))
    extents = points.max(axis=0) - points.min(axis=0)
    compactness = float(points.shape[0] / (np.prod(extents + 1e-3)))
    depth_mad = float(np.median(np.abs(points[:, 2] - np.median(points[:, 2]))))
    stats.update(
        {
            "planarity": planarity,
            "linearity": linearity,
            "surface_variation": surface_variation,
            "compactness": compactness,
            "depth_mad": depth_mad,
            "extent_x": float(extents[0]),
            "extent_y": float(extents[1]),
            "extent_z": float(extents[2]),
            "num_points": int(points.shape[0]),
        }
    )
    size_penalty = np.clip(abs(extents.mean() - 0.08) * 4.0, 0.0, 1.0)
    planarity_penalty = np.clip(planarity * 2.0, 0.0, 1.0)
    compact_reward = np.clip(compactness / 2500.0, 0.0, 1.0)
    depth_reward = np.clip(depth_mad * 25.0, 0.0, 1.0)
    linearity_reward = np.clip(linearity, 0.0, 1.0)
    score = 0.35 * compact_reward + 0.25 * depth_reward + 0.2 * linearity_reward + 0.2 * (1 - planarity_penalty)
    score *= 1.0 - 0.3 * size_penalty
    return float(np.clip(score, 0.0, 1.0)), stats


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter_area / (area_a + area_b - inter_area + 1e-6))


class MultiInstanceDepthProcessor:
    def __init__(self, config: Optional[MultiInstanceConfig] = None):
        self.cfg = (config or MultiInstanceConfig()).for_pi()
        self.o3d = _get_o3d()

    @staticmethod
    def _zone_from_bboxes(detections: List[Dict]) -> Tuple[int, int, int, int]:
        xs, ys, xe, ye = [], [], [], []
        for det in detections:
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
            xs.append(x1)
            ys.append(y1)
            xe.append(x2)
            ye.append(y2)
        if not xs:
            return (0, 0, 0, 0)
        return (int(min(xs)), int(min(ys)), int(max(xe)), int(max(ye)))

    def _depth_to_points(self, depth: np.ndarray, intrinsics: Dict[str, float], zone: Tuple[int, int, int, int]):
        h, w = depth.shape
        x1, y1, x2, y2 = zone
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), w), min(int(y2), h)
        cropped = depth[y1:y2, x1:x2]
        ys, xs = np.nonzero(cropped > 0)
        if xs.size == 0:
            stats = BackprojectStats(num_depth_pixels=cropped.size, num_points_raw=0, bbox=(x1, y1, x2, y2), depth_min=0.0, depth_max=0.0)
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32), stats
        if xs.size > self.cfg.max_points:
            idx = np.random.choice(xs.size, self.cfg.max_points, replace=False)
            xs, ys = xs[idx], ys[idx]
        zs = cropped[ys, xs].astype(np.float32)
        xs_full = xs + x1
        ys_full = ys + y1
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        x3d = (xs_full - cx) * zs / fx
        y3d = (ys_full - cy) * zs / fy
        points = np.stack([x3d, y3d, zs], axis=-1).astype(np.float32)
        stats = BackprojectStats(
            num_depth_pixels=cropped.size,
            num_points_raw=int(points.shape[0]),
            bbox=(x1, y1, x2, y2),
            depth_min=float(zs.min()) if zs.size else 0.0,
            depth_max=float(zs.max()) if zs.size else 0.0,
        )
        pixels = np.stack([xs_full, ys_full], axis=-1)
        return points, pixels, stats

    def _cluster_points(self, points: np.ndarray, pixels: np.ndarray) -> List[ClusterInfo]:
        if points.shape[0] < self.cfg.min_points:
            return []
        labels = None
        if not self.cfg.use_radius_clustering and self.o3d is not None:
            pc = self.o3d.geometry.PointCloud()
            pc.points = self.o3d.utility.Vector3dVector(points)
            if self.cfg.voxel_size > 0:
                pc = pc.voxel_down_sample(self.cfg.voxel_size)
                points_ds = np.asarray(pc.points)
                # map pixels crudely by nearest neighbor to downsized points
                if points_ds.shape[0] > 0:
                    tree = cKDTree(points)
                    _, nn_idx = tree.query(points_ds, k=1)
                    pixels = pixels[nn_idx]
                    points = points_ds
            labels = np.array(pc.cluster_dbscan(eps=self.cfg.eps, min_points=self.cfg.min_points, print_progress=False))
        if labels is None or labels.size == 0 or labels.max() < 0:
            tree = cKDTree(points)
            labels = np.full(points.shape[0], -1, dtype=int)
            cluster_id = 0
            visited = np.zeros(points.shape[0], dtype=bool)
            for idx in range(points.shape[0]):
                if visited[idx]:
                    continue
                neighbours = tree.query_ball_point(points[idx], r=self.cfg.radius if self.cfg.use_radius_clustering else self.cfg.eps)
                if len(neighbours) < self.cfg.min_points:
                    visited[idx] = True
                    continue
                queue = list(neighbours)
                visited[queue] = True
                while queue:
                    current = queue.pop()
                    labels[current] = cluster_id
                    nbrs = tree.query_ball_point(points[current], r=self.cfg.radius if self.cfg.use_radius_clustering else self.cfg.eps)
                    for n in nbrs:
                        if not visited[n]:
                            visited[n] = True
                            queue.append(n)
                cluster_id += 1
        clusters: List[ClusterInfo] = []
        for cid in np.unique(labels):
            if cid < 0:
                continue
            mask = labels == cid
            pts = points[mask]
            px = pixels[mask]
            if pts.shape[0] < self.cfg.min_points:
                continue
            birdness, stats = _compute_birdness(pts)
            bbox_2d = (int(px[:, 0].min()), int(px[:, 1].min()), int(px[:, 0].max()), int(px[:, 1].max()))
            clusters.append(
                ClusterInfo(
                    id=len(clusters),
                    points=pts,
                    pixel_xy=px,
                    bbox_2d=bbox_2d,
                    centroid=pts.mean(axis=0),
                    depth_median=float(np.median(pts[:, 2])),
                    birdness=birdness,
                    stats=stats,
                )
            )
        return clusters

    def assign_clusters(
        self, depth: np.ndarray, intrinsics: Dict[str, float], detections: List[Dict], debug: bool = False
    ) -> Tuple[Dict[int, Dict], Dict]:
        if not self.cfg.enabled or depth is None or not len(detections):
            return {}, {}
        zone = self._zone_from_bboxes(detections)
        # expand zone
        x1, y1, x2, y2 = zone
        w, h = depth.shape[1], depth.shape[0]
        dx, dy = (x2 - x1) * self.cfg.zone_margin, (y2 - y1) * self.cfg.zone_margin
        zone = (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))
        points, pixels, stats = self._depth_to_points(depth, intrinsics, zone)
        clusters = self._cluster_points(points, pixels)
        if not clusters:
            return {i: {"deny_reason": "no_valid_cluster"} for i in range(len(detections))}, {"clusters": []}
        det_depths = []
        for det in detections:
            bx1, by1, bx2, by2 = det.get("bbox", (0, 0, 0, 0))
            crop = depth[int(by1) : int(by2), int(bx1) : int(bx2)]
            valid = crop[crop > 0]
            det_depths.append(float(np.median(valid)) if valid.size else float("inf"))
        cost = np.full((len(detections), len(clusters)), fill_value=5.0, dtype=float)
        ious = np.zeros_like(cost)
        for i, det in enumerate(detections):
            for j, cluster in enumerate(clusters):
                iou = _iou(det.get("bbox", (0, 0, 0, 0)), cluster.bbox_2d)
                ious[i, j] = iou
                if iou < self.cfg.min_iou:
                    continue
                depth_diff = abs(det_depths[i] - cluster.depth_median)
                bird_penalty = 1.0 - cluster.birdness
                cost[i, j] = (
                    self.cfg.assign_w_iou * (1 - iou)
                    + self.cfg.assign_w_depth * depth_diff
                    + self.cfg.assign_w_birdness * bird_penalty
                )
        row_ind, col_ind = linear_sum_assignment(cost)
        assignments: Dict[int, Dict] = {}
        for r, c in zip(row_ind, col_ind):
            chosen = clusters[c]
            det_cost = cost[r, c]
            if det_cost >= self.cfg.max_cost or ious[r, c] < self.cfg.min_iou:
                assignments[r] = {"deny_reason": "inconsistent_depth_object"}
                continue
            candidates = [cl for cl in clusters if _iou(detections[r].get("bbox", (0, 0, 0, 0)), cl.bbox_2d) >= self.cfg.split_iou_threshold]
            if len(candidates) > 1:
                candidates_sorted = sorted(candidates, key=lambda cl: cl.birdness, reverse=True)
                if len(candidates_sorted) > 1 and candidates_sorted[1].birdness >= self.cfg.split_birdness:
                    assignments[r] = {"deny_reason": "mixed_instances"}
                    continue
                chosen = candidates_sorted[0]
            assignments[r] = {
                "cluster_id": chosen.id,
                "points": chosen.points,
                "birdness": chosen.birdness,
                "cluster_bbox_2d": chosen.bbox_2d,
                "stats": {**chosen.stats, "depth_median": chosen.depth_median, "num_clusters": len(clusters)},
            }
        # handle unassigned detections
        for idx in range(len(detections)):
            if idx not in assignments:
                assignments[idx] = {"deny_reason": "no_valid_cluster"}
        debug_info = {
            "zone": zone,
            "num_clusters": len(clusters),
            "cluster_sizes": [c.points.shape[0] for c in clusters],
        }
        return assignments, debug_info


__all__ = ["MultiInstanceDepthProcessor", "MultiInstanceConfig"]
