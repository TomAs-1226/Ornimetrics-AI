# Point-Cloud Identity PC Demo

This Streamlit app wraps the YOLO + point-cloud re-id pipeline so you can test
identity creation/matching locally with uploaded RGB and depth/point-cloud
files. YOLO is only used for detection/class labels; identity is driven by the
point-cloud embedding + gallery thresholds.

## Setup

```bash
pip install -r requirements-pc.txt
```

## Running

```bash
streamlit run pc_demo/app.py
```

Upload an RGB image plus one of:
- Depth image: 16-bit PNG (millimeters) or `.npy` float32 depth map.
- Point cloud: `.ply` or `.pcd` (requires `open3d`).

Controls mirror the pipeline features:
- YOLO detection (optional) with a toggle to override the class label using your
  species input.
- Preprocessing toggles: plane removal, depth gating, voxel/FPS sampling,
  normalization modes (including scale-feature option).
- Re-id backbone choice: `heavy` (wider DGCNN for harder separation) or `light`
  (smaller DGCNN for low-power), plus configurable embedding dimensionality.
- Thresholds and margin guard for open-set identity creation plus an embedding
  smoothing window for tracker stability.
- `Enable identity debug logging` to emit the per-detection debug blocks added
  in the main pipeline (preprocessing counts, depth gating range, top-5 gallery
  distances, threshold/margin decisions, tracker costs) and write them to
  `runs/debug_identity/<timestamp>/`.

Depth handling notes:
- Depth PNGs are decoded with `cv2.IMREAD_UNCHANGED`; uint16 values are assumed
  to be millimeters and converted to meters automatically.
- Point-cloud uploads bypass back-projection and feed directly into the
  preprocessing/embedding path, preserving identity-bearing geometry.

## Example debug block (`--debug_identity` enabled)

```
[
  {
    "class": "sparrow",
    "bbox": [12, 8, 140, 160],
    "preprocessing": {
      "num_points_raw": 8123,
      "num_points_after_crop": 8123,
      "num_points_after_plane_removal": 7900,
      "plane_removed_ratio": 0.027,
      "num_points_after_depth_gate": 7600,
      "depth_gate_removed_pct": 0.038,
      "depth_gate_range": 0.42,
      "num_points_after_voxel": 2100,
      "num_points_final": 2048,
      "scale": 0.37
    },
    "embedding_norm": 1.0,
    "top5": [{"id": "sparrow_0001", "dist": 0.18}, {"id": "sparrow_0003", "dist": 0.44}],
    "best_match_id": "sparrow_0001",
    "best_dist": 0.18,
    "second_best_dist": 0.44,
    "margin": 0.26,
    "threshold": 0.30,
    "new_identity": false,
    "reason": "matched",
    "tracker": {"motion_cost": 0.12, "appearance_cost": 0.18, "combined_cost": 0.15}
  }
]
```
