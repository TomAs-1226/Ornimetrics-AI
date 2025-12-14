import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np

# Force headless Open3D to avoid libGL issues in minimal environments
os.environ.setdefault("OPEN3D_CPU_DISABLE_GL", "1")
import open3d as o3d

DEFAULT_POINTS = 2048


@dataclass
class PreprocessConfig:
    plane_removal: bool = True
    plane_distance: float = 0.01
    voxel_size: float = 0.01
    fps_points: int = DEFAULT_POINTS
    depth_gate_k: float = 2.5
    normalization: str = "center_only"  # options: center_only, center_and_scale, center_and_scale_with_scale_feature
    append_scale: bool = False
    random_seed: int = 0


@dataclass
class PreprocessResult:
    points: np.ndarray
    stats: Dict[str, float]
    scale_feature: Optional[float] = None


rng_cache: Dict[int, np.random.Generator] = {}


def _get_rng(seed: int) -> np.random.Generator:
    if seed not in rng_cache:
        rng_cache[seed] = np.random.default_rng(seed)
    return rng_cache[seed]


def _to_o3d(pc: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    return cloud


def _remove_plane(pc: np.ndarray, distance: float) -> Tuple[np.ndarray, float]:
    if pc.shape[0] < 10:
        return pc, 0.0
    cloud = _to_o3d(pc)
    plane_model, inliers = cloud.segment_plane(distance_threshold=distance, ransac_n=3, num_iterations=50)
    if len(inliers) == 0:
        return pc, 0.0
    inlier_pc = cloud.select_by_index(inliers)
    outlier_pc = cloud.select_by_index(inliers, invert=True)
    removed_ratio = 1.0 - (len(outlier_pc.points) / max(len(pc), 1))
    return np.asarray(outlier_pc.points), removed_ratio


def _depth_gate(pc: np.ndarray, k: float) -> Tuple[np.ndarray, float, float]:
    if pc.shape[0] == 0:
        return pc, 0.0, 0.0
    depths = pc[:, 2]
    med = np.median(depths)
    mad = np.median(np.abs(depths - med)) + 1e-6
    low, high = med - k * mad, med + k * mad
    mask = (depths >= low) & (depths <= high)
    kept = pc[mask]
    removed_pct = 1.0 - float(len(kept)) / max(len(pc), 1)
    depth_range = high - low
    return kept, removed_pct, depth_range


def _voxel_down(pc: np.ndarray, voxel_size: float) -> np.ndarray:
    if pc.shape[0] == 0:
        return pc
    cloud = _to_o3d(pc)
    down = cloud.voxel_down_sample(voxel_size)
    return np.asarray(down.points)


def _farthest_point_sample(pc: np.ndarray, n: int, seed: int) -> np.ndarray:
    if pc.shape[0] == 0:
        return pc
    n = min(n, pc.shape[0])
    rng = _get_rng(seed)
    centroids = np.zeros((n,), dtype=np.int64)
    distance = np.ones((pc.shape[0],)) * 1e10
    farthest = 0
    for i in range(n):
        centroids[i] = farthest
        centroid = pc[farthest, :]
        dist = np.sum((pc - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(rng.choice(np.flatnonzero(distance == distance.max())))
    return pc[centroids]


def _normalize(pc: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    if pc.shape[0] == 0:
        return pc, 1.0
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    if mode == "center_only":
        return pc, 1.0
    scale = np.linalg.norm(pc, axis=1).max()
    scale = scale if scale > 0 else 1.0
    pc_scaled = pc / scale
    return pc_scaled, scale


def preprocess_with_stats(pc: np.ndarray, config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    cfg = config or PreprocessConfig()
    stats = {
        "num_points_raw": float(pc.shape[0]),
        "num_points_after_crop": float(pc.shape[0]),
    }
    working = pc.astype(np.float32)

    if cfg.plane_removal:
        working, removed_ratio = _remove_plane(working, cfg.plane_distance)
        stats["num_points_after_plane_removal"] = float(working.shape[0])
        stats["plane_removed_ratio"] = removed_ratio
    else:
        stats["num_points_after_plane_removal"] = float(working.shape[0])
        stats["plane_removed_ratio"] = 0.0

    working, removed_pct, depth_range = _depth_gate(working, cfg.depth_gate_k)
    stats["num_points_after_depth_gate"] = float(working.shape[0])
    stats["depth_gate_removed_pct"] = removed_pct
    stats["depth_gate_range"] = depth_range

    working = _voxel_down(working, cfg.voxel_size)
    stats["num_points_after_voxel"] = float(working.shape[0])

    if working.shape[0] >= cfg.fps_points:
        working = _farthest_point_sample(working, cfg.fps_points, cfg.random_seed)
    elif working.shape[0] > 0:
        rng = _get_rng(cfg.random_seed)
        idx = rng.choice(working.shape[0], cfg.fps_points, replace=True)
        working = working[idx]
    stats["num_points_final"] = float(working.shape[0])
    pc_norm, scale = _normalize(working, cfg.normalization)
    stats["scale"] = scale

    scale_feature = scale if cfg.normalization == "center_and_scale_with_scale_feature" else None
    result_points = pc_norm
    if cfg.append_scale and scale_feature is not None:
        scale_column = np.full((result_points.shape[0], 1), scale_feature, dtype=result_points.dtype)
        result_points = np.concatenate([result_points, scale_column], axis=1)
    return PreprocessResult(points=result_points.astype(np.float32), stats=stats, scale_feature=scale_feature)


def preprocess(pc: np.ndarray, num_points: int = DEFAULT_POINTS) -> np.ndarray:
    cfg = PreprocessConfig(fps_points=num_points)
    return preprocess_with_stats(pc, cfg).points


__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_with_stats",
    "preprocess",
    "DEFAULT_POINTS",
]
