from typing import Dict, Tuple
import numpy as np


def backproject_depth(depth: np.ndarray, intrinsics: Dict[str, float], bbox: Tuple[float, float, float, float] = None,
                      max_points: int = 50000) -> np.ndarray:
    """Convert a depth map to a 3D point cloud.

    Args:
        depth: HxW depth map in meters.
        intrinsics: dict with fx, fy, cx, cy.
        bbox: optional (x1, y1, x2, y2) crop. If omitted, uses full frame.
        max_points: optional cap to avoid huge allocations.
    Returns:
        Nx3 array of 3D points in camera frame.
    """
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32)
    h, w = depth.shape
    x1, y1, x2, y2 = bbox if bbox is not None else (0, 0, w, h)
    x1, y1 = int(max(x1, 0)), int(max(y1, 0))
    x2, y2 = int(min(x2, w)), int(min(y2, h))
    cropped = depth[y1:y2, x1:x2]
    ys, xs = np.nonzero(cropped > 0)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
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
    return points


__all__ = ["backproject_depth"]

