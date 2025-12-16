#!/usr/bin/env python
"""Offline sanity checks for point-cloud re-id embeddings."""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial import cKDTree

from src.gallery import Gallery, cosine_distance
from src.reid_embedder import PointReID
from src.pc_preprocess import PreprocessConfig


def load_cloud(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)
    try:
        import open3d as o3d  # type: ignore  # pragma: no cover

        pc = o3d.io.read_point_cloud(str(path))
        return np.asarray(pc.points, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Unable to load {path}: {exc}")


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = ta.query(b, k=1)
    db, _ = tb.query(a, k=1)
    return float((da.mean() + db.mean()) / 2.0)


def run_checks(paths: List[Path], debug: bool = False) -> dict:
    cfg = PreprocessConfig(fps_points=256, normalization="center_only")
    embedder = PointReID(preprocess_config=cfg, model_name="light", emb_dims=128)
    embeddings = []
    for p in paths:
        pc = load_cloud(p)
        res = embedder.embed(pc)
        embeddings.append(res.embedding.cpu().numpy())
    embeds = np.stack(embeddings)
    pairwise = np.dot(embeds, embeds.T)
    variance = float(np.var(pairwise))
    same_self = float(1 - cosine_distance(embeds[0], embeds[0]))
    chamfer_far = chamfer(load_cloud(paths[0]), load_cloud(paths[-1]))
    gallery = Gallery(default_threshold=0.2)
    gallery.update("bird", "id_a", embeds[0])
    best_id, dist, _ = gallery.match("bird", embeds[-1])
    needs_new = gallery.needs_new_identity("bird", dist)
    if debug:
        print(json.dumps({"variance": variance, "self": same_self, "chamfer_far": chamfer_far, "new_identity": needs_new}, indent=2))
    checks = {
        "embedding_variance_ok": variance > 1e-4,
        "self_similarity_ok": same_self > 0.99,
        "distance_separates": chamfer_far == float("inf") or dist > 0.0,
        "gallery_new_identity": needs_new,
    }
    return checks


def main():
    parser = argparse.ArgumentParser(description="Run offline identity sanity checks")
    parser.add_argument("data", help="Folder of .ply or .npy point clouds")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    root = Path(args.data)
    paths = sorted(root.glob("*.ply")) + sorted(root.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No point clouds found in {root}")
    checks = run_checks(paths, debug=args.debug)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f"Failed checks: {failed}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
