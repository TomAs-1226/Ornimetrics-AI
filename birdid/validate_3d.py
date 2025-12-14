"""Depth validation and anti-spoof checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: float
    reason: str | None = None
    diagnostics: Dict[str, float] = field(default_factory=dict)


def _pca_metrics(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        return np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.real(eigvals[order])
    eigvecs = np.real(eigvecs[:, order])
    return eigvals, eigvecs


def _planarity_from_eigs(eigvals: np.ndarray) -> tuple[float, float, float]:
    if eigvals.size != 3 or eigvals[0] <= 0:
        return 0.0, 0.0, 0.0
    lam1, lam2, lam3 = eigvals
    scattering = float(lam3 / (lam1 + 1e-8))
    planarity_metric = float((lam2 - lam3) / (lam1 + 1e-8))
    linearity = float((lam1 - lam2) / (lam1 + 1e-8))
    return scattering, planarity_metric, linearity


def _thickness_from_points(points: np.ndarray, eigvecs: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    aligned = points - points.mean(axis=0, keepdims=True)
    if eigvecs.size:
        aligned = aligned @ eigvecs
    axis = aligned[:, 2] if aligned.shape[1] >= 3 else aligned[:, -1]
    return float(np.percentile(axis, 95) - np.percentile(axis, 5))


def _normalize_point_units(points: np.ndarray, units: str = "auto") -> np.ndarray:
    if units == "m":
        return points.astype(np.float32, copy=False)
    pts = points.astype(np.float32, copy=False)
    if units == "mm":
        return pts / 1000.0
    median_mag = float(np.nanmedian(np.linalg.norm(pts, axis=1))) if pts.size else 0.0
    if median_mag > 10.0:
        return pts / 1000.0
    return pts


def _validate_point_cloud(points: np.ndarray, config, trusted: bool = False) -> ValidationResult:
    diagnostics: Dict[str, float] = {}
    if points.size == 0:
        return ValidationResult(False, 0.0, reason="no_points", diagnostics=diagnostics)
    if points.shape[0] > 10000:
        idx = np.random.choice(points.shape[0], 10000, replace=False)
        pts_sample = points[idx]
    else:
        pts_sample = points
    eigvals, eigvecs = _pca_metrics(pts_sample)
    scattering, planarity_metric, linearity = _planarity_from_eigs(eigvals)
    thickness = _thickness_from_points(pts_sample, eigvecs)
    diagnostics.update(
        {
            "scattering": scattering,
            "planarity_metric": planarity_metric,
            "linearity": linearity,
            "thickness": thickness,
            "points": float(points.shape[0]),
        }
    )
    spoof_like = (
        scattering < getattr(config, "validation_scattering_min", 0.002)
        and thickness < getattr(config, "validation_planar_thickness_min", 0.006)
    )
    if spoof_like and not trusted:
        return ValidationResult(False, 0.1, reason="planar_surface", diagnostics=diagnostics)

    quality = 1.0
    if thickness < config.validation_min_thickness:
        quality *= 0.6
    if points.shape[0] < config.validation_min_points:
        quality *= 0.7
    quality = max(0.0, min(1.0, quality))
    return ValidationResult(True, quality, diagnostics=diagnostics)


def validate_tracklet(
    depth_masks: List[Tuple[np.ndarray, np.ndarray]],
    config,
    point_cloud: np.ndarray | None = None,
    trusted_ply_input: bool = False,
    point_units: str = "auto",
) -> ValidationResult:
    if point_cloud is not None:
        pts_m = _normalize_point_units(point_cloud, units=point_units)
        return _validate_point_cloud(pts_m, config, trusted=trusted_ply_input)

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

    mask_stack = np.stack([m for _, m in depth_masks], axis=0)
    size_ok = True
    approx_size_m = 0.0
    if mask_stack.any():
        _, ys, xs = np.nonzero(mask_stack)
        h, w = depth_masks[0][1].shape
        width_norm = (xs.max() - xs.min() + 1) / max(w, 1)
        height_norm = (ys.max() - ys.min() + 1) / max(h, 1)
        approx_size_m = depth_mean * max(width_norm, height_norm)
        if approx_size_m < config.validation_min_size_m or approx_size_m > config.validation_max_size_m:
            size_ok = False

    eigvals, eigvecs = _pca_metrics(points)
    scattering, planarity_metric, linearity = _planarity_from_eigs(eigvals)
    planar_thickness = _thickness_from_points(points, eigvecs)

    diagnostics = {
        "valid_ratio": valid_ratio,
        "thickness": thickness,
        "approx_size_m": approx_size_m,
        "centroid_jitter": centroid_jitter,
        "scattering": scattering,
        "planarity_metric": planarity_metric,
        "linearity": linearity,
        "planar_thickness": planar_thickness,
    }

    if valid_ratio <= 0:
        return ValidationResult(False, 0.0, reason="no_points", diagnostics=diagnostics)
    if thickness < config.validation_min_thickness or thickness > config.validation_max_thickness:
        return ValidationResult(False, 0.2, reason="thickness_out_of_bounds", diagnostics=diagnostics)
    spoof_like = (
        scattering < getattr(config, "validation_scattering_min", 0.002)
        and planar_thickness < getattr(config, "validation_planar_thickness_min", 0.006)
    )
    if spoof_like:
        return ValidationResult(False, 0.1, reason="planar_surface", diagnostics=diagnostics)
    if centroid_jitter > config.validation_max_centroid_jitter:
        return ValidationResult(False, 0.2, reason="centroid_instability", diagnostics=diagnostics)

    completeness_penalty = 0.0
    if valid_ratio < config.validation_min_valid_ratio:
        completeness_penalty += 0.3
    if points.shape[0] < config.validation_min_points:
        completeness_penalty += 0.4
    size_score = 1.0
    if not size_ok:
        completeness_penalty += 0.2
        size_score = 0.4

    planar_score = max(0.0, min(1.0, (scattering - getattr(config, "validation_scattering_min", 0.002)) * 50))
    thickness_score = max(
        0.0,
        min(
            1.0,
            (thickness - config.validation_min_thickness)
            / (config.validation_max_thickness - config.validation_min_thickness + 1e-6),
        ),
    )
    density_score = max(0.0, min(1.0, valid_ratio))
    stability_score = max(0.0, min(1.0, 1.0 - centroid_jitter / max(config.validation_max_centroid_jitter, 1e-6)))
    quality = float(0.2 * (planar_score + thickness_score + density_score + stability_score + size_score))
    quality = max(0.0, quality - completeness_penalty)
    return ValidationResult(True, quality, reason=None if completeness_penalty == 0 else "partial_low_quality", diagnostics=diagnostics)
