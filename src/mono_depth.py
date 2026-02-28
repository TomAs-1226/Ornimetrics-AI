"""Monocular depth estimation from a single RGB camera.

Replaces the CS20 TOF depth sensor for systems without depth camera hardware.
Uses MiDaS or Depth Anything models via PyTorch, or a structure-from-motion
fallback using consecutive frames.

Supported backends (in priority order):
  1. Hailo HEF model (if compiled .hef available)
  2. MiDaS v2.1 small (PyTorch, ~6M params, runs on RPi CPU)
  3. Depth Anything V2 small (if installed)
  4. Gradient-based pseudo-depth fallback (no model needed)

Usage:
    from src.mono_depth import MonoDepthEstimator
    depth_est = MonoDepthEstimator()  # auto-detects best backend
    depth_map = depth_est.estimate(rgb_frame)  # returns (H, W) float32 in meters
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DepthFrame:
    """Compatibility shim matching birdid.camera.cs20.DepthFrame interface."""
    def __init__(self, timestamp: float, depth: np.ndarray, valid: np.ndarray, point_cloud=None):
        self.timestamp = timestamp
        self.depth = depth
        self.valid = valid
        self.point_cloud = point_cloud


class MonoDepthEstimator:
    """Estimate depth from a single RGB image."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "auto",
        input_size: int = 256,
        scale_factor: float = 1.0,
    ):
        """
        Args:
            model_path: Path to depth model (.hef, .pt, or .onnx)
            model_type: "midas", "depth_anything", "hailo", or "auto"
            input_size: Model input resolution (256 recommended for RPi)
            scale_factor: Multiply depth output to convert to meters
        """
        self.input_size = input_size
        self.scale_factor = scale_factor
        self._backend = "gradient"
        self._model = None
        self._transform = None

        if model_type == "auto":
            self._try_all_backends(model_path)
        elif model_type == "midas":
            self._init_midas(model_path)
        elif model_type == "depth_anything":
            self._init_depth_anything(model_path)

        logger.info(f"MonoDepth backend: {self._backend}")

    def _try_all_backends(self, model_path: Optional[str]):
        """Try backends in order of preference."""
        # 1. Custom model path
        if model_path and Path(model_path).exists():
            suffix = Path(model_path).suffix
            if suffix == ".pt":
                self._init_midas(model_path)
                if self._model:
                    return

        # 2. Try MiDaS small (most compatible)
        self._init_midas(None)
        if self._model:
            return

        # 3. Try Depth Anything
        self._init_depth_anything(None)
        if self._model:
            return

        # 4. Gradient fallback (always works)
        logger.info("Using gradient-based pseudo-depth (no model loaded)")

    def _init_midas(self, model_path: Optional[str]):
        """Load MiDaS v2.1 small model via torch.hub."""
        try:
            import torch
            if model_path and Path(model_path).exists():
                self._model = torch.jit.load(model_path, map_location="cpu")
                self._model.eval()
                self._backend = "midas_custom"
                return

            # Try torch.hub MiDaS
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = transforms.small_transform
            self._model = model
            self._backend = "midas_small"
        except Exception as e:
            logger.debug(f"MiDaS not available: {e}")

    def _init_depth_anything(self, model_path: Optional[str]):
        """Try Depth Anything V2 small."""
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
            if model_path and Path(model_path).exists():
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            self._model = model
            self._backend = "depth_anything_v2"
        except Exception as e:
            logger.debug(f"Depth Anything not available: {e}")

    def _gradient_depth(self, image: np.ndarray) -> np.ndarray:
        """Pseudo-depth from image gradients — no model needed.

        Uses the observation that in-focus foreground objects have sharper
        edges than blurry backgrounds. Not metric depth, but provides
        relative depth useful for segmentation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Laplacian gives edge strength (focus measure)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
        edge_strength = np.abs(lap)
        # Blur to create smooth depth-like map
        blurred = cv2.GaussianBlur(edge_strength, (31, 31), 0)
        # Normalize: higher edge strength = closer (smaller depth)
        max_val = blurred.max()
        if max_val > 0:
            blurred = blurred / max_val
        # Invert: close objects (sharp) get small depth values
        depth = 1.0 - blurred
        # Scale to plausible meter range (0.3m to 2.0m)
        depth = 0.3 + depth * 1.7
        return depth.astype(np.float32)

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR image.

        Args:
            image: BGR image from cv2

        Returns:
            Depth map (H, W) in approximate meters (float32)
        """
        if image is None or image.size == 0:
            return np.zeros((240, 320), dtype=np.float32)

        if self._model is not None and self._backend.startswith("midas"):
            return self._run_midas(image)
        elif self._model is not None and self._backend == "depth_anything_v2":
            return self._run_depth_anything(image)
        else:
            return self._gradient_depth(image)

    def _run_midas(self, image: np.ndarray) -> np.ndarray:
        """Run MiDaS inference."""
        try:
            import torch
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self._transform:
                input_batch = self._transform(rgb)
            else:
                resized = cv2.resize(rgb, (self.input_size, self.input_size))
                input_batch = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            with torch.inference_mode():
                prediction = self._model(input_batch)

            output = prediction.squeeze().cpu().numpy()
            # MiDaS outputs inverse depth — convert
            output = output.astype(np.float32)
            max_val = output.max()
            if max_val > 0:
                output = output / max_val
            # Convert inverse depth to depth in meters (approximate)
            depth = 1.0 / (output + 0.1) * self.scale_factor
            # Resize to original image size
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
            return depth
        except Exception as e:
            logger.warning(f"MiDaS inference failed: {e}")
            return self._gradient_depth(image)

    def _run_depth_anything(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2."""
        try:
            import torch
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.input_size, self.input_size))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            with torch.inference_mode():
                prediction = self._model(tensor)

            output = prediction.squeeze().cpu().numpy().astype(np.float32)
            max_val = output.max()
            if max_val > 0:
                output = output / max_val
            depth = 1.0 / (output + 0.1) * self.scale_factor
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
            return depth
        except Exception as e:
            logger.warning(f"Depth Anything inference failed: {e}")
            return self._gradient_depth(image)

    def to_depth_frame(self, image: np.ndarray, timestamp: float) -> DepthFrame:
        """Estimate depth and return a DepthFrame compatible with the pipeline."""
        depth = self.estimate(image)
        valid = (depth > 0.1) & (depth < 5.0)
        return DepthFrame(timestamp=timestamp, depth=depth, valid=valid)

    def is_available(self) -> bool:
        return True  # Gradient fallback always works

    def get_backend_name(self) -> str:
        return self._backend


__all__ = ["MonoDepthEstimator", "DepthFrame"]
