"""Stub to export a trained embedding model to int8 TFLite."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():  # pragma: no cover - helper script
    parser = argparse.ArgumentParser(description="Export BirdID embedding model to TFLite (int8)")
    parser.add_argument("saved_model_dir", type=Path, help="Path to TF SavedModel or PyTorch ONNX export")
    parser.add_argument("output_tflite", type=Path, help="Destination .tflite path")
    args = parser.parse_args()

    args.output_tflite.parent.mkdir(parents=True, exist_ok=True)
    args.output_tflite.write_text(
        "Replace this stub with TF Lite Converter logic to quantize to int8 for Pi deployment."
    )
    print("Wrote placeholder TFLite file to", args.output_tflite)


if __name__ == "__main__":
    main()
