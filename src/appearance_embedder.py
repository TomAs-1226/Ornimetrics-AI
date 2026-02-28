"""Appearance-based bird re-identification using RGB crop embeddings.

This module provides individual bird recognition WITHOUT any depth camera.
It uses the bird detection bounding box crop from the RGB camera and generates
a compact embedding for matching against a gallery of known individuals.

Two backends:
  1. TorchScript MobileNetV3-Small feature extractor (default, ~1.5M params)
  2. Histogram-based fallback when PyTorch is unavailable (works everywhere)

Usage:
    embedder = AppearanceEmbedder()
    embedding = embedder.embed(rgb_crop)  # np.ndarray (H, W, 3) BGR
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class AppearanceEmbedder:
    """Generate re-ID embeddings from RGB bird crops."""

    def __init__(self, model_path: Optional[str] = None, input_size: int = 128, emb_dims: int = 128):
        self.input_size = input_size
        self.emb_dims = emb_dims
        self._torch_model = None
        self._backend = "histogram"

        # Try to load a TorchScript or PyTorch model
        if model_path and Path(model_path).exists():
            try:
                import torch
                self._torch_model = torch.jit.load(model_path, map_location="cpu")
                self._torch_model.eval()
                self._backend = "torchscript"
            except Exception:
                pass

        # If no custom model, try MobileNetV3 features from torchvision
        if self._torch_model is None:
            try:
                import torch
                import torchvision.models as models
                # Use MobileNetV3-Small — only 1.5M params, runs on RPi CPU
                base = models.mobilenet_v3_small(weights=None)
                # Remove classifier, keep feature extractor
                base.classifier = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(576, emb_dims, bias=False),
                )
                base.eval()
                self._torch_model = base
                self._backend = "mobilenet_v3_small"
            except Exception:
                pass

    @property
    def backend(self) -> str:
        return self._backend

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalize a BGR crop for embedding."""
        resized = cv2.resize(crop, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # ImageNet normalization
        img = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img

    def _histogram_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Fallback: color + texture histogram embedding (no PyTorch needed)."""
        resized = cv2.resize(crop, (64, 64))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Color histogram (H: 16 bins, S: 8 bins, V: 8 bins = 32 dims)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).ravel()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()

        # Texture: grayscale gradient magnitudes (32 bins = 32 dims)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        t_hist = cv2.calcHist([mag.astype(np.float32)], [0], None, [32], [0, mag.max() + 1e-6]).ravel()

        # Spatial: mean color per 4x4 grid (4*4*3 = 48 dims)
        grid = cv2.resize(resized, (4, 4)).astype(np.float32).ravel() / 255.0

        # Shape: aspect ratio + area ratio (8 dims padded)
        h, w = crop.shape[:2]
        aspect = w / max(h, 1)
        area_ratio = (h * w) / (640 * 480)
        shape_feats = np.array([aspect, area_ratio, h / 480, w / 640, 0, 0, 0, 0], dtype=np.float32)

        # Concatenate: 16 + 8 + 8 + 32 + 48 + 8 = 120 dims, pad to 128
        vec = np.concatenate([
            h_hist / (h_hist.sum() + 1e-8),
            s_hist / (s_hist.sum() + 1e-8),
            v_hist / (v_hist.sum() + 1e-8),
            t_hist / (t_hist.sum() + 1e-8),
            grid,
            shape_feats,
        ]).astype(np.float32)

        # Pad or truncate to emb_dims
        if len(vec) < self.emb_dims:
            vec = np.pad(vec, (0, self.emb_dims - len(vec)))
        else:
            vec = vec[:self.emb_dims]

        # L2 normalize
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Generate embedding from a BGR bird crop.

        Args:
            crop: BGR image crop of the bird (from detection bbox)

        Returns:
            L2-normalized embedding vector (emb_dims,)
        """
        if crop is None or crop.size == 0:
            return np.zeros(self.emb_dims, dtype=np.float32)

        if self._torch_model is not None:
            try:
                import torch
                img = self._preprocess(crop)
                # (H, W, C) -> (1, C, H, W)
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                with torch.inference_mode():
                    emb = self._torch_model(tensor).squeeze(0)
                vec = emb.cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(vec) + 1e-8
                return vec / norm
            except Exception:
                pass

        return self._histogram_embedding(crop)


__all__ = ["AppearanceEmbedder"]
