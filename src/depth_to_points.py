from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class BackprojectStats:
    num_depth_pixels: int
    num_points_raw: int
    bbox: Tuple[int, int, int, int]
    depth_min: float
    depth_max: float


def backproject_depth(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_points: int = 50000,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, BackprojectStats]:
    """Convert a depth map to a 3D point cloud with cropping stats."""
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32), BackprojectStats(0, 0, (0, 0, 0, 0), 0.0, 0.0)
    h, w = depth.shape
    x1, y1, x2, y2 = bbox if bbox is not None else (0, 0, w, h)
    x1, y1 = int(max(x1, 0)), int(max(y1, 0))
    x2, y2 = int(min(x2, w)), int(min(y2, h))
    cropped = depth[y1:y2, x1:x2]
    ys, xs = np.nonzero(cropped > 0)
    if xs.size == 0:
        stats = BackprojectStats(num_depth_pixels=cropped.size, num_points_raw=0, bbox=(x1, y1, x2, y2), depth_min=0.0, depth_max=0.0)
        return np.zeros((0, 3), dtype=np.float32), stats
    if xs.size > max_points:
        idx = np.random.choice(xs.size, max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    zs = cropped[ys, xs].astype(np.float32)
    xs = xs + x1
    ys = ys + y1
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    points = np.stack([x3d, y3d, zs], axis=-1)
    stats = BackprojectStats(
        num_depth_pixels=cropped.size,
        num_points_raw=int(points.shape[0]),
        bbox=(x1, y1, x2, y2),
        depth_min=float(zs.min()) if zs.size else 0.0,
        depth_max=float(zs.max()) if zs.size else 0.0,
    )
    return points, stats


__all__ = ["backproject_depth", "BackprojectStats"]
