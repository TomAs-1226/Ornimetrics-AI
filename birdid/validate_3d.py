"""Depth validation and anti-spoof checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: float
    reason: str | None = None


def _compute_planarity(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 1.0
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.real(eigvals))
    if eigvals[-1] <= 0:
        return 1.0
    return float(eigvals[0] / (eigvals[-1] + 1e-8))


def validate_tracklet(depth_masks: List[Tuple[np.ndarray, np.ndarray]], config) -> ValidationResult:
    points_list: List[np.ndarray] = []
    centroids: List[np.ndarray] = []
    total_pixels = 0
    valid_pixels = 0
    for depth, mask in depth_masks:
        if depth.size == 0:
            continue
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            total_pixels += mask.size
            continue
        vals = depth[ys, xs]
        valid_pixels += len(vals)
        total_pixels += mask.size
        x_norm = (xs - xs.mean()) / (mask.shape[1] + 1e-6)
        y_norm = (ys - ys.mean()) / (mask.shape[0] + 1e-6)
        pts = np.stack([x_norm, y_norm, vals], axis=1)
        points_list.append(pts)
        centroids.append(np.array([x_norm.mean(), y_norm.mean(), float(vals.mean())]))
    if not points_list:
        return ValidationResult(False, 0.0, reason="no_points")
    points = np.concatenate(points_list, axis=0)
    valid_ratio = valid_pixels / max(total_pixels, 1)
    thickness = float(points[:, 2].max() - points[:, 2].min())
    depth_mean = float(points[:, 2].mean()) if points.size else 0.0
    centroid_jitter = 0.0
    if len(centroids) > 1:
        centroid_jitter = float(np.linalg.norm(np.std(np.stack(centroids, axis=0), axis=0)))

    # Approximate physical size using normalized extent scaled by mean depth.
    mask_stack = np.stack([m for _, m in depth_masks], axis=0)
    size_ok = True
    if mask_stack.any():
        _, ys, xs = np.nonzero(mask_stack)
        h, w = depth_masks[0][1].shape
        width_norm = (xs.max() - xs.min() + 1) / max(w, 1)
        height_norm = (ys.max() - ys.min() + 1) / max(h, 1)
        approx_size_m = depth_mean * max(width_norm, height_norm)
        if approx_size_m < config.validation_min_size_m or approx_size_m > config.validation_max_size_m:
            size_ok = False

    if valid_ratio < config.validation_min_valid_ratio:
        return ValidationResult(False, 0.0, reason="insufficient_depth")
    if points.shape[0] < config.validation_min_points:
        return ValidationResult(False, 0.0, reason="too_few_points")
    if not size_ok:
        return ValidationResult(False, 0.2, reason="size_out_of_bounds")
    if thickness < config.validation_min_thickness or thickness > config.validation_max_thickness:
        return ValidationResult(False, 0.2, reason="thickness_out_of_bounds")
    planar_ratio = _compute_planarity(points)
    if planar_ratio < config.validation_planar_ratio:
        return ValidationResult(False, 0.1, reason="planar_surface")
    if centroid_jitter > config.validation_max_centroid_jitter:
        return ValidationResult(False, 0.2, reason="centroid_instability")

    # Rough quality score combining multiple cues.
    planar_score = max(0.0, min(1.0, (planar_ratio - config.validation_planar_ratio) * 50))
    thickness_score = max(0.0, min(1.0, (thickness - config.validation_min_thickness) / (config.validation_max_thickness - config.validation_min_thickness + 1e-6)))
    density_score = max(0.0, min(1.0, valid_ratio))
    stability_score = max(0.0, min(1.0, 1.0 - centroid_jitter / max(config.validation_max_centroid_jitter, 1e-6)))
    size_score = 1.0 if size_ok else 0.0
    quality = float(0.2 * (planar_score + thickness_score + density_score + stability_score + size_score))
    return ValidationResult(True, quality, reason=None)
