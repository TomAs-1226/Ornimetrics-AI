"""Offline sanity checks for multi-instance depth clustering and assignment."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.multi_instance_depth import MultiInstanceConfig, MultiInstanceDepthProcessor


def save_cloud(path: Path, points: np.ndarray):  # pragma: no cover - util
    if points.size == 0:
        return
    try:
        import open3d as o3d

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(path), pc)
    except Exception:
        np.save(path.with_suffix(".npy"), points)


def main():
    parser = argparse.ArgumentParser(description="Sanity check for multi-instance clustering")
    parser.add_argument("--depth", required=True, help="Depth image as .npy or 16-bit PNG")
    parser.add_argument("--detections", required=True, help="JSON list of detections with bbox fields")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("fx", "fy", "cx", "cy"), default=[525.0, 525.0, 319.5, 239.5])
    parser.add_argument("--out", type=Path, default=Path("runs/multi_instance"))
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--min-points", type=int, default=40)
    parser.add_argument("--pi-profile", action="store_true")
    args = parser.parse_args()

    intrinsics = {"fx": args.intrinsics[0], "fy": args.intrinsics[1], "cx": args.intrinsics[2], "cy": args.intrinsics[3]}
    depth_path = Path(args.depth)
    if depth_path.suffix == ".npy":
        depth = np.load(depth_path)
    else:
        import cv2

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32) / 1000.0
    detections = json.loads(Path(args.detections).read_text())

    cfg = MultiInstanceConfig(
        enabled=True,
        voxel_size=args.voxel,
        eps=args.eps,
        min_points=args.min_points,
        pi_profile=args.pi_profile,
    )
    processor = MultiInstanceDepthProcessor(cfg)
    assignments, debug = processor.assign_clusters(depth, intrinsics, detections, debug=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"assignments": assignments, "debug": debug}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o if isinstance(o, (int, float, str)) else str(o)))
    for det_idx, info in assignments.items():
        if info.get("deny_reason"):
            continue
        out_path = args.out / f"det_{det_idx}_cluster_{info.get('cluster_id', 'x')}.ply"
        save_cloud(out_path, np.asarray(info.get("points")))

    expected = len(detections)
    produced = sum(1 for a in assignments.values() if not a.get("deny_reason"))
    if produced >= expected:
        print("PASS: produced isolated point clouds for all detections")
    else:
        print(f"FAIL: only produced {produced}/{expected} valid clouds")


if __name__ == "__main__":
    main()
