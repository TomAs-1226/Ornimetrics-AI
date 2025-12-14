#!/usr/bin/env python
"""Export utilities for Raspberry Pi deployment."""

import argparse
from pathlib import Path

import torch

from src.reid_embedder import PointReID
from src.pc_preprocess import PreprocessConfig


def export(backbone: str, emb_dims: int, output_dir: Path, quantize: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig(fps_points=1024, voxel_size=0.015, depth_gate_k=2.0, plane_removal=False)
    model = PointReID(preprocess_config=cfg, model_name=backbone, emb_dims=emb_dims).model
    model.eval()
    dummy = torch.randn(1, cfg.fps_points, 3)
    traced = torch.jit.trace(model, dummy)
    ts_path = output_dir / "reid_pi.pt"
    traced.save(ts_path)
    onnx_path = output_dir / "reid_pi.onnx"
    torch.onnx.export(model, dummy, onnx_path, input_names=["points"], output_names=["emb"], opset_version=12)
    if quantize:
        qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        q_ts = output_dir / "reid_pi_int8.pt"
        torch.jit.trace(qmodel, dummy).save(q_ts)
    print(f"Saved TorchScript to {ts_path} and ONNX to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export point re-id model for Raspberry Pi")
    parser.add_argument("--backbone", choices=["light", "heavy", "xheavy"], default="light")
    parser.add_argument("--emb-dims", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("exports/pi"))
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    export(args.backbone, args.emb_dims, args.output, quantize=args.quantize)


if __name__ == "__main__":
    main()
