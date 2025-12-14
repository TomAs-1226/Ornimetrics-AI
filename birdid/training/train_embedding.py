"""Simple contrastive training loop for BirdID embeddings (optional, PC-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_pairs(manifest: Path):
    data = json.loads(manifest.read_text())
    depths = []
    masks = []
    labels = []
    for idx, item in enumerate(data):
        depths.append(np.load(manifest.parent / item["depth"]))
        masks.append(np.load(manifest.parent / item["mask"]))
        labels.append(f"{item.get('species','unk')}:{item.get('track_id', idx)}")
    return depths, masks, labels


def main():  # pragma: no cover - training helper
    parser = argparse.ArgumentParser(description="Train a BirdID embedding model with contrastive pairs")
    parser.add_argument("manifest", type=Path, help="Path to manifest.json produced by collect_dataset.py")
    parser.add_argument("output", type=Path, help="Output directory for checkpoints")
    args = parser.parse_args()

    print("This script is a placeholder for a PC-only training workflow.")
    print("Load manifest", args.manifest)
    depths, masks, labels = load_pairs(args.manifest)
    print(f"Loaded {len(depths)} crops for weakly supervised training")
    print("Use your preferred DL stack (PyTorch/TF) here to train a MobileNetV3-Small embedding")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "TODO.txt").write_text("Run training with PyTorch/TF and export to TFLite via export_tflite.py")


if __name__ == "__main__":
    main()
