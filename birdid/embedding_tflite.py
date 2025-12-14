"""TFLite embedding backend.

Designed for lightweight depth+mask inputs sized to 96x96 or 128x128. When the
TFLite runtime is unavailable, this module falls back to a deterministic hash so
that tests and the CLI still work without the model present.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Interpreter = None


class TFLiteEmbedder:
    def __init__(self, model_path: str, input_size: int = 96):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self._interp: Optional[Interpreter] = None
        if Interpreter is not None and self.model_path.exists():
            self._interp = Interpreter(model_path=str(self.model_path))
            self._interp.allocate_tensors()

    def _fallback(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        payload = depth.tobytes() + mask.astype(np.uint8).tobytes()
        digest = hashlib.sha256(payload).digest()
        arr = np.frombuffer(digest, dtype=np.uint8)[:32].astype(np.float32)
        vec = np.repeat(arr, 4)[:128]
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def _preprocess(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided

        h, w = depth.shape
        size = self.input_size
        # Simple resize using striding/nearest neighbor without pulling in cv2
        y_idx = (np.linspace(0, h - 1, size)).astype(int)
        x_idx = (np.linspace(0, w - 1, size)).astype(int)
        depth_resized = depth[y_idx][:, x_idx]
        mask_resized = mask[y_idx][:, x_idx]
        stacked = np.stack([depth_resized, mask_resized.astype(np.float32)], axis=-1)
        return stacked.astype(np.float32)

    def embed(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._interp is None:
            return self._fallback(depth, mask)
        input_data = self._preprocess(depth, mask)[None, ...]
        input_details = self._interp.get_input_details()[0]
        self._interp.set_tensor(input_details["index"], input_data.astype(input_details["dtype"]))
        self._interp.invoke()
        output_details = self._interp.get_output_details()[0]
        vec = self._interp.get_tensor(output_details["index"]).astype(np.float32).ravel()
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
