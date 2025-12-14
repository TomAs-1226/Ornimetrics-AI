"""Depth segmentation utilities for isolating birds inside YOLO crops."""
from __future__ import annotations

import numpy as np


def _largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    best_mask = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    stack = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack.append((y, x))
            visited[y, x] = True
            component = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) > best_mask.sum():
                best_mask[:] = False
                for cy, cx in component:
                    best_mask[cy, cx] = True
    return best_mask


class DepthSegmenter:
    def __init__(self, alpha: float = 0.02, min_valid_ratio: float = 0.35):
        self.alpha = alpha
        self.min_valid_ratio = min_valid_ratio
        self._baseline: np.ndarray | None = None

    def update_baseline(self, depth: np.ndarray, valid: np.ndarray) -> None:
        if self._baseline is None:
            self._baseline = depth.copy()
            return
        self._baseline = (1 - self.alpha) * self._baseline + self.alpha * depth
        self._baseline = np.where(valid, self._baseline, self._baseline)

    def segment(self, depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if depth.size == 0:
            return depth, np.zeros_like(depth, dtype=bool)
        depth = depth.astype(np.float32, copy=False)
        median = np.nanmedian(depth[valid]) if np.any(valid) else np.nan
        if np.isnan(median):
            return depth, np.zeros_like(depth, dtype=bool)
        if self._baseline is not None:
            delta = np.abs(depth - self._baseline)
        else:
            delta = np.abs(depth - median)
        mask = (delta > 0.015) & valid
        if mask.mean() < self.min_valid_ratio:
            return depth, np.zeros_like(mask)
        mask = _largest_component(mask)
        return depth * mask, mask
