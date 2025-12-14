# BirdID PC Demo

This Streamlit app lets you exercise the BirdID pipeline on a PC using uploaded
RGB and depth/point-cloud files.

## Setup

```bash
pip install -r requirements-pc.txt
```

## Running

```bash
streamlit run pc_demo/app.py
```

Upload an RGB image plus one of:
- Depth image: 16-bit PNG (millimeters) or `.npy` float32 depth map
- Point cloud: `.ply` or `.pcd` (requires `open3d`)

Controls allow setting species, optional bounding box, seeding a known bird, or
forcing enrollment. If `best.pt` exists in the repo root and `ultralytics` is
installed, the demo can auto-detect species/bbox from the uploaded RGB image; a
manual fallback is always available. The output panel shows the decision
(DISPENSE/ENROLL/DENY), distances, quality score, and cooldown state. Debug
panes render the RGB with bbox, depth crop stats (dtype/min/max/valid percent),
point count, and PCA-derived diagnostics used for planar spoof rejection. A
`Trusted PLY input` toggle lets you skip planar spoof denial for uploads when
testing known-good point clouds on PC (kept strict by default).

Depth handling notes:
- Depth PNGs are decoded with `cv2.IMREAD_UNCHANGED`; uint16 values are assumed
  to be millimeters and converted to meters automatically.
- If segmentation returns zero points, the pipeline falls back to the raw valid
  depth crop to avoid erroneous `no_points` denials.

## Firebase (optional)

If `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account and
`google-cloud-firestore`/`google-cloud-storage` are installed, the demo will log
events and upload artifacts using the same paths as the production pipeline.
Without credentials it runs in local-only mode.
