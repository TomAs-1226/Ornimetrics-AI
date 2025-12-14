from typing import Tuple
import numpy as np
import open3d as o3d


DEFAULT_POINTS = 2048


def _to_o3d(pc: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc)
    return cloud


def denoise_and_sample(pc: np.ndarray, num_points: int = DEFAULT_POINTS) -> np.ndarray:
    if pc.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cloud = _to_o3d(pc)
    if len(pc) > 10:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = np.asarray(cloud.points)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(points) >= num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
    else:
        idx = np.random.choice(len(points), num_points, replace=True)
    return points[idx].astype(np.float32)


def normalize(pc: np.ndarray) -> Tuple[np.ndarray, float]:
    if pc.size == 0:
        return pc, 1.0
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    scale = np.linalg.norm(pc, axis=1).max()
    scale = scale if scale > 0 else 1.0
    pc = pc / scale
    return pc, scale


def preprocess(pc: np.ndarray, num_points: int = DEFAULT_POINTS) -> np.ndarray:
    pc = denoise_and_sample(pc, num_points=num_points)
    pc, _ = normalize(pc)
    return pc


__all__ = ["preprocess", "denoise_and_sample", "normalize", "DEFAULT_POINTS"]

