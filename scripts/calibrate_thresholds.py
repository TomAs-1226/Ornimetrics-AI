#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d

from src.reid_embedder import PointReID
from src.gallery import cosine_distance


def load_dataset(root: Path):
    samples = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for indiv_dir in class_dir.iterdir():
            if not indiv_dir.is_dir():
                continue
            for ply in (indiv_dir / "samples").glob("*.ply"):
                pc = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
                samples.append((class_dir.name, indiv_dir.name, pc))
    return samples


def compute_thresholds(samples, embedder, percentile=95):
    by_class = {}
    for cls, indiv, pc in samples:
        by_class.setdefault(cls, []).append((indiv, pc))
    thresholds = {}
    for cls, pcs in by_class.items():
        embeds = [(indiv, embedder.embed(pc).embedding.cpu().numpy()) for indiv, pc in pcs]
        dists_same = []
        dists_diff = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                dist = cosine_distance(embeds[i][1], embeds[j][1])
                if embeds[i][0] == embeds[j][0]:
                    dists_same.append(dist)
                else:
                    dists_diff.append(dist)
        if not dists_same or not dists_diff:
            thresholds[cls] = 0.3
            continue
        pos = np.percentile(dists_same, percentile)
        neg = np.percentile(dists_diff, 100 - percentile)
        thresholds[cls] = float((pos + neg) / 2)
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-species thresholds from labeled validation set")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default=Path("runs/calibrated_thresholds.json"), type=Path)
    parser.add_argument("--percentile", type=int, default=95)
    args = parser.parse_args()
    embedder = PointReID()
    samples = load_dataset(args.data)
    thresholds = compute_thresholds(samples, embedder, args.percentile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
