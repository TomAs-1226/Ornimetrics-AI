"""Point cloud preprocessing pipeline — pure numpy, no Open3D dependency.

Designed for low-latency inference on Raspberry Pi with Hailo AI module.
All operations use vectorized numpy for speed on ARM CPUs.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np

DEFAULT_POINTS = 1024


@dataclass
class PreprocessConfig:
    plane_removal: bool = True
    plane_distance: float = 0.01
    voxel_size: float = 0.01
    fps_points: int = DEFAULT_POINTS
    depth_gate_k: float = 2.5
    normalization: str = "center_only"
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


def _remove_plane(pc: np.ndarray, distance: float) -> Tuple[np.ndarray, float]:
    """RANSAC plane removal using pure numpy."""
    n = pc.shape[0]
    if n < 10:
        return pc, 0.0
    rng = np.random.default_rng(42)
    best_inlier_mask = None
    best_count = 0
    iters = 50
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pc[idx[0]], pc[idx[1]], pc[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p0)
        dists = np.abs(pc @ normal + d)
        inlier_mask = dists < distance
        count = int(inlier_mask.sum())
        if count > best_count:
            best_count = count
            best_inlier_mask = inlier_mask
    if best_inlier_mask is None or best_count == 0:
        return pc, 0.0
    outliers = pc[~best_inlier_mask]
    removed_ratio = 1.0 - (len(outliers) / max(n, 1))
    return outliers, removed_ratio


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
    """Voxel downsampling using integer grid hashing — pure numpy."""
    if pc.shape[0] == 0 or voxel_size <= 0:
        return pc
    keys = np.floor(pc / voxel_size).astype(np.int64)
    # Use structured array for unique voxel lookup
    _, unique_idx = np.unique(
        keys[:, 0] * 1000003 + keys[:, 1] * 1000033 + keys[:, 2],
        return_index=True,
    )
    return pc[np.sort(unique_idx)]


def _farthest_point_sample(pc: np.ndarray, n: int, seed: int) -> np.ndarray:
    if pc.shape[0] == 0:
        return pc
    n = min(n, pc.shape[0])
    rng = _get_rng(seed)
    centroids = np.zeros((n,), dtype=np.int64)
    distance = np.full(pc.shape[0], 1e10)
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
