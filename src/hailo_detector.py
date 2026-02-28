"""Hailo-8 AI Hat+ accelerated YOLO detector for Raspberry Pi 5.

Supports both the AI Hat (Hailo-8L, 13 TOPS) and AI Hat+ (Hailo-8, 26 TOPS).
Gracefully falls back to CPU-based PyTorch inference if Hailo hardware is
not available.

Hardware Requirements:
- Raspberry Pi 5 with AI Hat+ (Hailo-8) attached via M.2 PCIe
- HailoRT >= 4.17 installed  (sudo apt install hailo-all)
- Compiled HEF model file

Setup (on Raspberry Pi OS Bookworm):
    sudo apt update && sudo apt install hailo-all
    sudo reboot
    hailortcli fw-control identify   # verify device detected

Usage:
    detector = HailoYOLODetector(
        model_path="models/model.hef",
        fallback_pytorch_model="models/model.pt",
    )
    detections = detector.detect(frame)
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — Hailo device detection
# ---------------------------------------------------------------------------

def hailo_device_available() -> bool:
    """Check if a Hailo device is physically present on this system."""
    # Check for /dev/hailo0 (PCIe device node created by hailort driver)
    if os.path.exists("/dev/hailo0"):
        return True
    # Fallback: check lspci for Hailo vendor ID (1e60)
    try:
        import subprocess
        out = subprocess.check_output(["lspci", "-d", "1e60:"], stderr=subprocess.DEVNULL, timeout=5)
        return len(out.strip()) > 0
    except Exception:
        pass
    return False


def hailo_runtime_available() -> Tuple[bool, str]:
    """Check if HailoRT Python bindings are importable.
    Returns (available, version_or_error).
    """
    try:
        from hailo_platform import __version__ as hv
        return True, hv
    except ImportError:
        pass
    try:
        import hailo_platform  # noqa: F811
        return True, "unknown"
    except ImportError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Model file resolution
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root

_MODEL_SEARCH_PATHS = [
    "models/model.hef",
    "models/best.hef",
    "models/yolov8n.hef",
    "model.hef",
    "best.hef",
]

_PT_SEARCH_PATHS = [
    "models/model.pt",
    "models/best.pt",
    "models/weights.pt",
    "model.pt",
    "best.pt",
    "weights.pt",
]


def find_model(model_path: Optional[str] = None, prefer_hef: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """Locate best available model files.

    Returns (hef_path_or_None, pt_path_or_None).
    If model_path is given, it is tried first. Then the standard search paths
    under the project root are checked.
    """
    hef_path = None
    pt_path = None

    # Explicit path
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            p = _SCRIPT_DIR / p
        if p.is_file():
            if p.suffix == ".hef":
                hef_path = str(p)
            elif p.suffix == ".pt":
                pt_path = str(p)
        # Also look for sibling file with the other extension
        if hef_path and not pt_path:
            sibling = Path(hef_path).with_suffix(".pt")
            if sibling.is_file():
                pt_path = str(sibling)
        if pt_path and not hef_path:
            sibling = Path(pt_path).with_suffix(".hef")
            if sibling.is_file():
                hef_path = str(sibling)

    # Search standard paths
    if not hef_path:
        for rel in _MODEL_SEARCH_PATHS:
            p = _SCRIPT_DIR / rel
            if p.is_file():
                hef_path = str(p)
                break

    if not pt_path:
        for rel in _PT_SEARCH_PATHS:
            p = _SCRIPT_DIR / rel
            if p.is_file():
                pt_path = str(p)
                break

    return hef_path, pt_path


def get_model_info(hef_path: str) -> Dict:
    """Read metadata from a HEF file (model name, input shape, etc.)."""
    info = {"path": hef_path, "name": Path(hef_path).stem, "size_mb": 0}
    try:
        info["size_mb"] = round(os.path.getsize(hef_path) / (1024 * 1024), 1)
    except OSError:
        pass
    try:
        from hailo_platform import HEF
        hef = HEF(hef_path)
        info["input_shapes"] = {
            name: hef.get_input_vstream_infos()[i].shape
            for i, name in enumerate(n.name for n in hef.get_input_vstream_infos())
        }
        info["output_shapes"] = {
            name: hef.get_output_vstream_infos()[i].shape
            for i, name in enumerate(n.name for n in hef.get_output_vstream_infos())
        }
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class HailoYOLODetector:
    """YOLO detector with Hailo-8/8L acceleration and PyTorch fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_pytorch_model: Optional[str] = None,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        input_size: int = 320,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._hailo_active = False
        self._vdevice = None
        self._network_group = None
        self._input_vstream_infos = None
        self._output_vstream_infos = None
        self._input_vstreams_params = None
        self._output_vstreams_params = None
        self._hef = None
        self._pytorch_model = None
        self._class_names = []
        self._hailo_chip = "unknown"
        self._model_path_used = None

        # Resolve model files
        hef_path, pt_path = find_model(model_path)
        if fallback_pytorch_model:
            p = Path(fallback_pytorch_model)
            if not p.is_absolute():
                p = _SCRIPT_DIR / p
            if p.is_file():
                pt_path = str(p)

        # Try Hailo first
        if hef_path and hailo_device_available():
            rt_ok, rt_info = hailo_runtime_available()
            if rt_ok:
                self._hailo_active = self._init_hailo(hef_path)
                if self._hailo_active:
                    self._model_path_used = hef_path
            else:
                logger.warning(f"Hailo device found but runtime not installed: {rt_info}")
                logger.warning("Install with: sudo apt install hailo-all")
        elif hef_path and not hailo_device_available():
            logger.info("HEF model found but no Hailo device detected (/dev/hailo0 missing)")
            logger.info("If AI Hat+ is attached, try: sudo apt install hailo-all && sudo reboot")

        # Fallback to PyTorch
        if not self._hailo_active and pt_path:
            self._init_pytorch(pt_path)
            if self._pytorch_model is not None:
                self._model_path_used = pt_path

        if not self._hailo_active and self._pytorch_model is None:
            logger.warning("No valid model loaded — detector will return empty results")
            if not hef_path and not pt_path:
                logger.warning("Place model.hef in models/ folder (or model.pt for CPU fallback)")

    def _init_hailo(self, model_path: str) -> bool:
        """Initialize Hailo-8/8L accelerator using the HailoRT VDevice API."""
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoStreamInterface,
                InferVStreams,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
            )
        except ImportError as e:
            logger.warning(f"hailo_platform import failed: {e}")
            return False

        try:
            logger.info(f"Loading HEF: {model_path}")
            hef = HEF(model_path)
            self._hef = hef

            # VDevice() is a context manager — we keep it alive for the
            # lifetime of this detector instance.
            self._vdevice = VDevice()

            # Identify chip type for logging
            try:
                dev_ids = self._vdevice.get_physical_devices()
                if dev_ids:
                    arch = str(dev_ids[0].get_architecture())
                    if "HAILO8L" in arch.upper() or "8L" in arch:
                        self._hailo_chip = "Hailo-8L (13 TOPS)"
                    else:
                        self._hailo_chip = "Hailo-8 (26 TOPS)"
            except Exception:
                self._hailo_chip = "Hailo (unknown variant)"

            # Configure the HEF on the device
            configure_params = ConfigureParams.create_from_hef(
                hef=hef, interface=HailoStreamInterface.PCIe
            )
            network_groups = self._vdevice.configure(hef, configure_params)
            if not network_groups:
                logger.warning("Failed to configure Hailo network group")
                return False

            self._network_group = network_groups[0]

            # Get stream info for I/O
            self._input_vstream_infos = hef.get_input_vstream_infos()
            self._output_vstream_infos = hef.get_output_vstream_infos()

            # Build vstream params
            self._input_vstreams_params = InputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )
            self._output_vstreams_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )

            # Extract class names from output shape (num_classes = output_dim - 5)
            for info in self._output_vstream_infos:
                shape = info.shape
                if len(shape) >= 2:
                    n_classes = shape[-1] - 5 if shape[-1] > 5 else shape[-1]
                    self._class_names = [f"class_{i}" for i in range(n_classes)]
                    break

            # Determine input size from HEF metadata
            for info in self._input_vstream_infos:
                shape = info.shape  # (H, W, C) or (C, H, W)
                if len(shape) >= 3:
                    h, w = shape[0], shape[1]
                    if shape[2] <= 4:  # (H, W, C) layout
                        self.input_size = max(h, w)
                    else:  # (C, H, W) layout
                        self.input_size = max(shape[1], shape[2])
                    break

            logger.info(f"Hailo initialized: {self._hailo_chip} | input={self.input_size}x{self.input_size}")
            return True

        except Exception as e:
            logger.warning(f"Hailo init failed: {e}")
            # Clean up on failure
            self._vdevice = None
            self._network_group = None
            return False

    def _init_pytorch(self, model_path: str) -> None:
        """Initialize PyTorch YOLO model as fallback."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading PyTorch YOLO model: {model_path}")
            self._pytorch_model = YOLO(model_path)

            if hasattr(self._pytorch_model, "names"):
                if isinstance(self._pytorch_model.names, dict):
                    self._class_names = list(self._pytorch_model.names.values())
                else:
                    self._class_names = list(self._pytorch_model.names)

            logger.info(f"PyTorch YOLO loaded: {len(self._class_names)} classes")

        except ImportError:
            logger.warning("ultralytics not installed — cannot load .pt model")
            logger.warning("Install with: pip install ultralytics")
            self._pytorch_model = None
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            self._pytorch_model = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_hailo(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Hailo inference."""
        resized = cv2.resize(image, (self.input_size, self.input_size))
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess_hailo(
        self, outputs: Dict[str, np.ndarray], orig_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Postprocess Hailo YOLO outputs to detection list."""
        detections = []
        if not outputs:
            return detections

        # Hailo output dict keys are output layer names.
        # For YOLO, there's typically one key with shape (batch, N, 5+C)
        # or multiple heads that need to be concatenated.
        all_outputs = list(outputs.values())
        if not all_outputs:
            return detections

        # Concatenate all output heads along detection dimension
        output = all_outputs[0]
        if output.ndim == 3:
            # (batch, N, 5+C) — standard
            pass
        elif output.ndim == 2:
            output = np.expand_dims(output, 0)
        elif output.ndim == 4:
            # Multi-head: flatten spatial dims
            b, h, w, c = output.shape
            output = output.reshape(b, h * w, c)

        orig_h, orig_w = orig_shape
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for batch_idx in range(output.shape[0]):
            for det in output[batch_idx]:
                if len(det) < 5:
                    continue

                # YOLOv8 raw output: x_center, y_center, w, h, class_scores...
                # (no separate objectness in v8, scores are directly class confs)
                x_center, y_center, width, height = det[:4]
                class_scores = det[4:]

                if len(class_scores) == 0:
                    continue

                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])

                # YOLOv5-style output has objectness at index 4
                if len(det) >= 6 and det[4] < 1.0:
                    objectness = det[4]
                    class_scores = det[5:]
                    if len(class_scores) == 0:
                        continue
                    class_id = int(np.argmax(class_scores))
                    confidence = float(objectness * class_scores[class_id])

                if confidence < self.confidence_threshold:
                    continue

                x1 = int((x_center - width / 2) * scale_x)
                y1 = int((y_center - height / 2) * scale_y)
                x2 = int((x_center + width / 2) * scale_x)
                y2 = int((y_center + height / 2) * scale_y)

                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                class_name = (
                    self._class_names[class_id]
                    if class_id < len(self._class_names)
                    else f"class_{class_id}"
                )

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                })

        return self._apply_nms(detections)

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return detections

        by_class: Dict[str, List[Dict]] = {}
        for det in detections:
            by_class.setdefault(det["class_name"], []).append(det)

        filtered = []
        for class_dets in by_class.values():
            class_dets = sorted(class_dets, key=lambda x: x["confidence"], reverse=True)
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                class_dets = [
                    d for d in class_dets
                    if self._iou(best["bbox"], d["bbox"]) < self.iou_threshold
                ]
            filtered.extend(keep)
        return filtered

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        inter = (x2_i - x1_i) * (y2_i - y1_i)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_hailo(self, image: np.ndarray) -> List[Dict]:
        """Run inference on the Hailo accelerator."""
        if self._network_group is None:
            return []

        try:
            from hailo_platform import InferVStreams

            input_data = self._preprocess_hailo(image)

            # Build input dict: {layer_name: data}
            input_dict = {
                info.name: input_data
                for info in self._input_vstream_infos
            }

            with InferVStreams(
                self._network_group,
                self._input_vstreams_params,
                self._output_vstreams_params,
            ) as pipeline:
                output_dict = pipeline.infer(input_dict)

            return self._postprocess_hailo(output_dict, image.shape[:2])

        except Exception as e:
            logger.error(f"Hailo inference failed: {e}")
            return []

    def _infer_pytorch(self, image: np.ndarray) -> List[Dict]:
        """Run inference using PyTorch YOLO."""
        if self._pytorch_model is None:
            return []
        try:
            import torch
            with torch.inference_mode():
                results = self._pytorch_model.predict(
                    source=image, imgsz=self.input_size,
                    conf=self.confidence_threshold, iou=self.iou_threshold,
                    verbose=False,
                )
            detections = []
            if results:
                r0 = results[0]
                if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
                    confs = r0.boxes.conf.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
                        name = self._class_names[cid] if cid < len(self._class_names) else f"class_{cid}"
                        detections.append({
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "class_name": name, "class_id": int(cid),
                            "confidence": float(conf),
                        })
            return detections
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on a BGR image.

        Returns list of dicts with keys: bbox, class_name, class_id, confidence.
        """
        if self._hailo_active:
            return self._infer_hailo(image)
        elif self._pytorch_model is not None:
            return self._infer_pytorch(image)
        return []

    def reload_model(self, model_path: str) -> bool:
        """Hot-reload a new model file (for auto-update support).

        Call this when a new model file is downloaded. It will re-initialize
        the appropriate backend without restarting the process.
        """
        p = Path(model_path)
        if not p.is_file():
            logger.warning(f"Model file not found: {model_path}")
            return False

        logger.info(f"Reloading model: {model_path}")

        if p.suffix == ".hef" and hailo_device_available():
            # Release old Hailo resources
            self._network_group = None
            self._vdevice = None
            self._hailo_active = False

            if self._init_hailo(model_path):
                self._model_path_used = model_path
                self._pytorch_model = None
                return True
            logger.warning("HEF reload failed, keeping previous model")
            return False

        if p.suffix == ".pt":
            self._init_pytorch(model_path)
            if self._pytorch_model is not None:
                self._hailo_active = False
                self._model_path_used = model_path
                return True
            return False

        logger.warning(f"Unsupported model format: {p.suffix}")
        return False

    def is_hailo_active(self) -> bool:
        return self._hailo_active

    def get_backend_name(self) -> str:
        if self._hailo_active:
            return self._hailo_chip
        elif self._pytorch_model is not None:
            return "PyTorch CPU"
        return "No backend loaded"

    def get_model_path(self) -> Optional[str]:
        return self._model_path_used
