#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
from pathlib import Path

os.environ.setdefault("OPEN3D_CPU_DISABLE_GL", "1")
import open3d as o3d

from src.reid_embedder import PointReID
from src.gallery import cosine_distance, Gallery


def load_clouds(folder: Path):
    clouds = []
    for ply in folder.glob("*.ply"):
        pc = o3d.io.read_point_cloud(str(ply))
        clouds.append(np.asarray(pc.points))
    return clouds


def embed_clouds(clouds, embedder):
    embs = []
    for pc in clouds:
        res = embedder.embed(pc)
        embs.append(res.embedding.cpu().numpy())
    return embs


def geometry_distance(pc1, pc2):
    if len(pc1) == 0 or len(pc2) == 0:
        return float("inf")
    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2))
    threshold = 0.05
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.inlier_rmse


def embedding_variance_test(embs):
    pairwise = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            pairwise.append(cosine_distance(embs[i], embs[j]))
    pairwise = np.array(pairwise)
    return pairwise.mean(), pairwise.std()


def self_consistency_test(embedder, pc):
    e1 = embedder.embed(pc).embedding.cpu().numpy()
    e2 = embedder.embed(pc).embedding.cpu().numpy()
    return cosine_distance(e1, e2)


def different_clouds_test(embedder, pc1, pc2, min_threshold):
    e1 = embedder.embed(pc1).embedding.cpu().numpy()
    e2 = embedder.embed(pc2).embedding.cpu().numpy()
    dist = cosine_distance(e1, e2)
    return dist, dist > min_threshold


def gallery_threshold_test():
    gallery = Gallery(default_threshold=0.3)
    emb_far = np.ones(4)
    emb_close = np.zeros(4)
    gallery.update("bird", "id1", emb_close)
    best, dist, _ = gallery.match("bird", emb_far)
    return gallery.needs_new_identity("bird", dist)


def run_checks(folder: Path, min_geometry_diff: float, min_embed_dist: float):
    embedder = PointReID()
    clouds = load_clouds(folder)
    if len(clouds) < 2:
        raise SystemExit("Need at least two .ply files for sanity checks")
    embs = embed_clouds(clouds, embedder)
    mean_d, std_d = embedding_variance_test(embs[: min(50, len(embs))])
    self_dist = self_consistency_test(embedder, clouds[0])
    geom = geometry_distance(clouds[0], clouds[1])
    embed_dist, embed_pass = different_clouds_test(embedder, clouds[0], clouds[1], min_embed_dist)
    gallery_pass = gallery_threshold_test()
    report = {
        "embedding_non_collapse": {
            "mean_distance": mean_d,
            "std_distance": std_d,
            "pass": std_d >= 0.01 and mean_d >= 0.02,
        },
        "self_consistency": {
            "distance": self_dist,
            "pass": self_dist < 0.01,
        },
        "different_clouds": {
            "geometry_rmse": geom,
            "embedding_distance": embed_dist,
            "pass": geom > min_geometry_diff and embed_pass,
        },
        "gallery_threshold_logic": {
            "pass": gallery_pass,
        },
    }
    print(json.dumps(report, indent=2))
    if not all(item["pass"] for item in report.values()):
        raise SystemExit("Sanity checks FAILED")


def main():
    parser = argparse.ArgumentParser(description="Offline sanity checks for identity embeddings")
    parser.add_argument("--ply-folder", required=True, type=Path)
    parser.add_argument("--min-geometry-diff", type=float, default=0.01)
    parser.add_argument("--min-embed-dist", type=float, default=0.2)
    args = parser.parse_args()
    run_checks(args.ply_folder, args.min_geometry_diff, args.min_embed_dist)


if __name__ == "__main__":
    main()
