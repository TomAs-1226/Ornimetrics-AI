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
forcing enrollment. The output panel shows the decision (DISPENSE/ENROLL/DENY),
distances, quality score, and cooldown state. Debug panes render the RGB with
bbox, depth crop stats, and point count.

## Firebase (optional)

If `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account and
`google-cloud-firestore`/`google-cloud-storage` are installed, the demo will log
events and upload artifacts using the same paths as the production pipeline.
Without credentials it runs in local-only mode.
