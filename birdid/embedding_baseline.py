"""Deterministic baseline embedding for bird depth crops."""
from __future__ import annotations

import math
import numpy as np


def _moments(mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    x_mean = xs.mean()
    y_mean = ys.mean()
    cov_xx = ((xs - x_mean) ** 2).mean()
    cov_yy = ((ys - y_mean) ** 2).mean()
    cov_xy = ((xs - x_mean) * (ys - y_mean)).mean()
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy ** 2
    ratio = (trace + math.sqrt(max(trace ** 2 - 4 * det, 0))) / (trace + 1e-5)
    return ratio, cov_xx, cov_yy


def compute_baseline_embedding(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if depth.shape != mask.shape:
        raise ValueError("depth and mask must align")
    if depth.size == 0:
        return np.zeros(16, dtype=np.float32)
    masked = np.where(mask, depth, np.nan)
    valid_depths = masked[~np.isnan(masked)]
    if valid_depths.size == 0:
        return np.zeros(16, dtype=np.float32)
    stats = [
        np.nanmean(valid_depths),
        np.nanstd(valid_depths),
        np.nanmin(valid_depths),
        np.nanmax(valid_depths),
        np.nanmedian(valid_depths),
    ]
    stats.append(float(np.nanmean(np.abs(valid_depths - stats[0]))))
    ratio, cov_xx, cov_yy = _moments(mask)
    stats.extend([ratio, cov_xx, cov_yy])
    bbox_area = mask.sum()
    stats.append(bbox_area / mask.size)
    depth_variance = float(np.nanvar(valid_depths))
    stats.append(depth_variance)
    depth_range = float(np.nanmax(valid_depths) - np.nanmin(valid_depths))
    stats.append(depth_range)
    z_proj = float(np.nanmean(valid_depths ** 2))
    stats.append(z_proj)
    center_depth = depth[mask].mean() if bbox_area else 0.0
    stats.append(float(center_depth))
    vector = np.array(stats, dtype=np.float32)
    norm = np.linalg.norm(vector) + 1e-8
    return vector / norm
