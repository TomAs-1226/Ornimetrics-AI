# Point-Cloud Individual Re-Identification

End-to-end pipeline that detects objects with RGB YOLO only for class labels, converts depth crops into point clouds, encodes them with a metric-learning model, and assigns per-class individual identities using open-set gallery matching.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset format

```
data/<class_name>/<individual_id>/samples/*.ply
```

Each `.ply` contains the cropped object point cloud. Training samples batches from one species at a time with multiple individuals and samples per batch.

## Training

```bash
python scripts/train_reid.py --data data/ --epochs 10 --batch-size 2 --k-individuals 8 --m-samples 4 --loss supcon
```

Outputs checkpoints to `checkpoints/reid.pt` and metrics to `runs/train_reid/<timestamp>_metrics.json` (Recall@1/5, mAP logged per epoch).

## Calibrating thresholds

```bash
python scripts/calibrate_thresholds.py --data data/ --output runs/calibrated_thresholds.json
```

Produces per-species cosine-distance thresholds for open-set identity creation.

## Sanity checks (offline)

Run discriminativeness checks on a folder of point clouds:

```bash
python scripts/sanity_identity_checks.py --ply-folder data/demo_class/demo_id/samples/
```

Ensures embeddings are non-collapsed, self-consistent, geometry-aware, and that gallery thresholds create new identities for far embeddings.

## Inference

```bash
# Live camera demo
python scripts/infer_stream.py --camera demo --debug_identity

# Offline RGB + depth
python scripts/infer_stream.py --rgb path/to/rgb.jpg --depth path/to/depth.png --debug_identity
```

Key options:
- `--debug_identity`: emit per-detection debug JSON and save to `runs/debug_identity/<timestamp>/frame_XXXX.json`.
- `--normalization`: `center_only` (default) avoids identity-erasing scaling; `center_and_scale`/`center_and_scale_with_scale_feature` optional.
- `--plane-removal`/`--no-plane-removal`, `--depth-gate-k`, `--voxel`, `--fps-points` control preprocessing.

Outputs JSON per detection with fields `species`, `track_id`, `individual_id`, `individual_name`, `min_dist`, `second_best_dist`, `margin`, `quality_score`, `frames_used`, `deny_reason`.

## Dataset builder

```bash
python scripts/build_dataset_from_pairs.py --source raw_pairs/ --output data/
```

## Notes

- YOLO is **only** used for detection and class labels. Identity comes from point-cloud embeddings and gallery thresholds.
- Matching uses cosine distance with per-class thresholds and margin guards to decide new individuals.
- Tracker blends motion + appearance and smooths embeddings over multiple frames to reduce ID flicker.
- Geometry baseline (ICP RMSE) is used in sanity checks to verify point clouds carry identity signal.
