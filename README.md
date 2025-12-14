# Point-Cloud Individual Re-Identification

This repository provides an end-to-end pipeline that detects objects with RGB YOLO, crops depth to build object point clouds, and re-identifies individuals within each detected class using a metric-learning point-cloud encoder.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

```
python scripts/train_reid.py --data data/ --epochs 2 --batch-size 4
```

Dataset layout:

```
data/<class_name>/<individual_id>/samples/*.ply
```

Each `.ply` is a point cloud of the cropped object. The trainer builds batches with multiple individuals per class and learns embeddings with triplet loss.

### Inference

```
# Live camera demo using YOLO for detection
python scripts/infer_stream.py --camera demo

# Offline frames
python scripts/infer_stream.py --rgb path/to/rgb.jpg --depth path/to/depth.png
```

The inference script detects objects with YOLO, extracts per-detection point clouds using the depth map and camera intrinsics, embeds them, matches against a per-class gallery, tracks with a Kalman/Hungarian tracker, and emits JSON per detection.

### Building a dataset from paired captures

```
python scripts/build_dataset_from_pairs.py --source raw_pairs/ --output data/
```

The script expects folders containing RGB, depth, and PLY captures and organizes them into the training directory structure.

## Components

- `src/yolo_detect.py` – YOLO detection wrapper (RGB only for class labels).
- `src/depth_to_points.py` – Intrinsics-aware back-projection and bbox cropping to build point clouds.
- `src/pc_preprocess.py` – Denoise, sample to 2048 points, normalize to zero-mean/unit-radius.
- `src/models/dgcnn.py` – Point-cloud encoder producing fixed-length embeddings.
- `src/reid_embedder.py` – Model wrapper for inference.
- `src/metric_train.py` – Metric-learning training loop (triplet or supervised contrastive).
- `src/gallery.py` – Per-class embedding gallery with thresholding and centroid updates.
- `src/tracker.py` – Kalman filter + Hungarian assignment to stabilize IDs.
- `src/naming.py` – Two-part random name generator.
- `src/db.py` – SQLite persistence for individuals and embeddings.
- `src/output_schema.py` – JSON-friendly output structure for detections.
- `scripts/` – CLIs for training, inference, and dataset building.

## Notes

- YOLO is used **only** for detection and class labels. Identity is decided from point-cloud embeddings.
- Matching uses cosine distance; new individuals are created when distance exceeds a per-class threshold.
- Point clouds are always sampled/normalized to a consistent shape to keep the encoder robust.
- The tracker blends motion and embedding distance to avoid ID flicker.

