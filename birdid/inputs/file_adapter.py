"""Adapters for file-based inputs used in the PC demo.

This module converts uploaded RGB/depth/point-cloud files into the same
structures consumed by the BirdID pipeline.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from birdid.camera.cs20 import DepthFrame


@dataclass
class FileInputs:
    detection: dict
    depth_frame: DepthFrame
    rgb: np.ndarray
    bbox_xyxy: List[float]
    point_cloud: Optional[np.ndarray]


def _read_bytes(src) -> bytes:
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    if hasattr(src, "read"):
        return src.read()
    raise ValueError("Unsupported file-like object")


def load_rgb_image(src) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - pillow missing on Pi
        raise RuntimeError("Pillow is required for the PC demo") from exc
    data = _read_bytes(src)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _normalize_depth(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    return depth, valid


def _depth_from_png(src) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for depth PNGs") from exc
    img = Image.open(io.BytesIO(_read_bytes(src)))
    arr = np.array(img)
    if arr.dtype == np.uint16:
        depth = arr.astype(np.float32) / 1000.0
    else:
        depth = arr.astype(np.float32)
    return _normalize_depth(depth)


def _depth_from_npy(src) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(io.BytesIO(_read_bytes(src)))
    return _normalize_depth(data)


def _depth_from_point_cloud(src, grid: Tuple[int, int] = (240, 320)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("open3d is required for point-cloud inputs") from exc
    if isinstance(src, (str, Path)):
        pc = o3d.io.read_point_cloud(str(src))
    else:
        # UploadedFile / BytesIO: persist to temp so Open3D can read it.
        import tempfile

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        try:
            tmp.write(_read_bytes(src))
            tmp.flush()
            pc = o3d.io.read_point_cloud(tmp.name)
        finally:
            path = tmp.name
            tmp.close()
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
    pts = np.asarray(pc.points)
    if pts.size == 0:
        depth = np.zeros(grid, dtype=np.float32)
        return depth, np.zeros_like(depth, dtype=bool), pts
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    x_norm = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-6)
    y_norm = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-6)
    w, h = grid[1], grid[0]
    xi = np.clip((x_norm * (w - 1)).astype(int), 0, w - 1)
    yi = np.clip((y_norm * (h - 1)).astype(int), 0, h - 1)
    depth = np.full((h, w), np.nan, dtype=np.float32)
    for x, y, z in zip(xi, yi, zs):
        if np.isnan(depth[y, x]) or z < depth[y, x]:
            depth[y, x] = z
    valid = np.isfinite(depth)
    depth = np.where(valid, depth, 0.0)
    return depth, valid, pts


def load_depth(src) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    suffix = ""
    if isinstance(src, (str, Path)):
        suffix = Path(src).suffix.lower()
    elif hasattr(src, "name"):
        suffix = Path(getattr(src, "name", "")).suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg"}:
        depth, valid = _depth_from_png(src)
        return depth, valid, None
    if suffix == ".npy":
        depth, valid = _depth_from_npy(src)
        return depth, valid, None
    if suffix in {".ply", ".pcd"}:
        depth, valid, pts = _depth_from_point_cloud(src)
        return depth, valid, pts
    raise ValueError("Unsupported depth/point-cloud format")


def build_inputs(
    rgb_file,
    depth_file,
    species: str,
    bbox_xyxy: Optional[List[float]] = None,
    track_id: int = 1,
    conf: float = 0.9,
) -> FileInputs:
    rgb = load_rgb_image(rgb_file)
    depth, valid, pts = load_depth(depth_file)
    h, w = rgb.shape[:2]
    bbox = bbox_xyxy or [0, 0, w - 1, h - 1]
    ts = time.time()
    detection = {
        "timestamp": ts,
        "bbox_xyxy": bbox,
        "conf": conf,
        "species": species,
        "track_id": track_id,
    }
    depth_frame = DepthFrame(timestamp=ts, depth=depth, valid=valid)
    return FileInputs(detection=detection, depth_frame=depth_frame, rgb=rgb, bbox_xyxy=bbox, point_cloud=pts)

