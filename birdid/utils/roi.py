"""ROI utility helpers for robust depth cropping."""
from __future__ import annotations

from typing import Tuple


ROI = Tuple[float, float, float, float]


def clamp_roi(roi: ROI, width: int, height: int) -> ROI:
    x1, y1, x2, y2 = roi
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return (x1, y1, x2, y2)


def pad_roi(roi: ROI, pad_px: float) -> ROI:
    x1, y1, x2, y2 = roi
    return (x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px)


def roi_area(roi: ROI) -> float:
    x1, y1, x2, y2 = roi
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def safe_roi_from_center(cx: float, cy: float, w: float, h: float, width: int, height: int) -> ROI:
    half_w = w / 2.0
    half_h = h / 2.0
    return clamp_roi((cx - half_w, cy - half_h, cx + half_w, cy + half_h), width, height)
