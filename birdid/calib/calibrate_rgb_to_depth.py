"""Estimate a homography/extrinsic mapping from RGB pixels to depth pixels.

Usage:
    python -m birdid.calib.calibrate_rgb_to_depth --pairs pairs.json \
        --output birdid/calib/rgb_to_depth.json

``pairs.json`` format:
{
  "image_size_rgb": [1280, 720],
  "image_size_depth": [320, 240],
  "pairs": [
    {"rgb": [x, y], "depth": [u, v]},
    ... at least 4 pairs ...
  ]
}

The calibration is intentionally simple (planar homography). For outdoor feeder
setups the cameras are roughly fronto-parallel; if your mounting deviates
significantly, collect more pairs and switch ``--solver extrinsic`` to estimate a
full pinhole + extrinsics using a least-squares fit.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    method: str
    matrix: np.ndarray  # 3x3 homography
    image_size_rgb: Tuple[int, int]
    image_size_depth: Tuple[int, int]

    def to_json(self) -> Dict:
        return {
            "method": self.method,
            "matrix": self.matrix.tolist(),
            "image_size_rgb": list(self.image_size_rgb),
            "image_size_depth": list(self.image_size_depth),
        }


def _build_dlt_matrix(pairs: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> np.ndarray:
    A_rows: List[List[float]] = []
    for (x, y), (u, v) in pairs:
        A_rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A_rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    return np.asarray(A_rows, dtype=float)


def compute_homography(rgb_points: np.ndarray, depth_points: np.ndarray) -> np.ndarray:
    if rgb_points.shape != depth_points.shape or rgb_points.shape[1] != 2:
        raise ValueError("points must be Nx2 and have matching shapes")
    if len(rgb_points) < 4:
        raise ValueError("need at least 4 point pairs for homography")
    pairs = list(zip(rgb_points.tolist(), depth_points.tolist()))
    A = _build_dlt_matrix(pairs)
    # Solve Ah=0 using SVD
    _, _, vh = np.linalg.svd(A)
    h = vh[-1, :]
    H = h.reshape(3, 3)
    return H / H[-1, -1]


def save_calibration(path: Path, result: CalibrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_json(), f, indent=2)


def load_pairs(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    data = json.loads(path.read_text())
    rgb_pts = np.asarray([p["rgb"] for p in data["pairs"]], dtype=float)
    depth_pts = np.asarray([p["depth"] for p in data["pairs"]], dtype=float)
    rgb_size = tuple(data.get("image_size_rgb", [0, 0]))
    depth_size = tuple(data.get("image_size_depth", [0, 0]))
    return rgb_pts, depth_pts, rgb_size, depth_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate RGB -> depth mapping")
    parser.add_argument("--pairs", required=True, help="JSON file containing point pairs")
    parser.add_argument("--output", default="birdid/calib/rgb_to_depth.json")
    args = parser.parse_args()

    rgb_pts, depth_pts, rgb_size, depth_size = load_pairs(Path(args.pairs))
    H = compute_homography(rgb_pts, depth_pts)
    result = CalibrationResult(method="homography", matrix=H, image_size_rgb=rgb_size, image_size_depth=depth_size)
    save_calibration(Path(args.output), result)
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
