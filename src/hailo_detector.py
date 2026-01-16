"""Hailo-8 AI Hat accelerated YOLO detector for Raspberry Pi AI Hat 1+.

This module provides a drop-in replacement for YOLODetector that uses the
Hailo-8 accelerator (26 TOPS) for faster inference. It gracefully falls back
to CPU-based PyTorch inference if the Hailo hardware is not available.

Hardware Requirements:
- Raspberry Pi AI Hat 1+ with Hailo-8 chip
- HailoRT library installed
- Converted HEF model file (from YOLO ONNX/PyTorch model)

Usage:
    # Try Hailo first, fallback to PyTorch
    detector = HailoYOLODetector(model_path="best.hef", fallback_pytorch_model="best.pt")

    # Check which backend is active
    if detector.is_hailo_active():
        print("Running on Hailo-8 accelerator")
    else:
        print("Running on CPU/PyTorch")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HailoYOLODetector:
    """YOLO detector with Hailo-8 acceleration and PyTorch fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_pytorch_model: Optional[str] = None,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        input_size: int = 320,
    ):
        """Initialize the detector with Hailo acceleration or PyTorch fallback.

        Args:
            model_path: Path to Hailo HEF model file (e.g., "best.hef")
            fallback_pytorch_model: Path to PyTorch model file (e.g., "best.pt")
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            input_size: Input image size (default 320)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._hailo_active = False
        self._hailo_network = None
        self._hailo_configured = False
        self._pytorch_model = None
        self._class_names = []

        # Try to load Hailo first
        if model_path and Path(model_path).suffix == ".hef":
            self._hailo_active = self._init_hailo(model_path)

        # Fallback to PyTorch if Hailo fails or no HEF provided
        if not self._hailo_active:
            if fallback_pytorch_model:
                self._init_pytorch(fallback_pytorch_model)
            elif model_path and Path(model_path).suffix == ".pt":
                self._init_pytorch(model_path)
            else:
                # Try to find a default model
                for default in ["best.pt", "weights.pt", "yolov8n.pt"]:
                    if Path(default).exists():
                        self._init_pytorch(default)
                        break

        if not self._hailo_active and self._pytorch_model is None:
            logger.warning("No valid model loaded - detector will return empty results")

    def _init_hailo(self, model_path: str) -> bool:
        """Initialize Hailo-8 accelerator."""
        try:
            # Import Hailo runtime libraries
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

            logger.info(f"Loading Hailo HEF model: {model_path}")

            # Load HEF file
            hef = HEF(model_path)

            # Get available Hailo devices
            devices = VDevice.create()
            if not devices:
                logger.warning("No Hailo devices found")
                return False

            # Configure network
            network_groups = devices[0].configure(hef)
            if not network_groups:
                logger.warning("Failed to configure Hailo network")
                return False

            network_group = network_groups[0]
            network_group_params = network_group.create_params()

            # Create input/output stream parameters
            input_vstreams_params = InputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            output_vstreams_params = OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32
            )

            # Store configuration
            self._hailo_network = {
                "device": devices[0],
                "network_group": network_group,
                "input_vstreams_params": input_vstreams_params,
                "output_vstreams_params": output_vstreams_params,
            }
            self._hailo_configured = True

            logger.info("Hailo-8 accelerator initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"Hailo libraries not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize Hailo: {e}")
            return False

    def _init_pytorch(self, model_path: str) -> None:
        """Initialize PyTorch YOLO model as fallback."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading PyTorch YOLO model: {model_path}")
            self._pytorch_model = YOLO(model_path)

            # Extract class names
            if hasattr(self._pytorch_model, "names"):
                if isinstance(self._pytorch_model.names, dict):
                    self._class_names = list(self._pytorch_model.names.values())
                else:
                    self._class_names = list(self._pytorch_model.names)

            logger.info(f"PyTorch YOLO model loaded with {len(self._class_names)} classes")

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            self._pytorch_model = None

    def _preprocess_hailo(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Hailo inference."""
        # Resize to input size
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(normalized, axis=0)

    def _postprocess_hailo(
        self, outputs: List[np.ndarray], orig_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Postprocess Hailo outputs to detection format."""
        detections = []

        # YOLO output format: [batch, num_detections, 5 + num_classes]
        # Where each detection is: [x_center, y_center, width, height, objectness, class_probs...]
        if not outputs or len(outputs) == 0:
            return detections

        output = outputs[0]  # Get first output tensor
        if output.ndim == 2:
            output = np.expand_dims(output, 0)

        orig_h, orig_w = orig_shape
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for batch_idx in range(output.shape[0]):
            for detection in output[batch_idx]:
                # Extract detection components
                x_center, y_center, width, height = detection[:4]
                objectness = detection[4]
                class_probs = detection[5:]

                # Get best class
                class_id = int(np.argmax(class_probs))
                class_conf = float(class_probs[class_id])
                confidence = objectness * class_conf

                if confidence < self.confidence_threshold:
                    continue

                # Convert to xyxy format (scaled to original image size)
                x1 = int((x_center - width / 2) * scale_x)
                y1 = int((y_center - height / 2) * scale_y)
                x2 = int((x_center + width / 2) * scale_x)
                y2 = int((y_center + height / 2) * scale_y)

                # Clip to image bounds
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                class_name = (
                    self._class_names[class_id]
                    if class_id < len(self._class_names)
                    else f"class_{class_id}"
                )

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                    }
                )

        # Apply NMS
        return self._apply_nms(detections)

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not detections:
            return detections

        # Group by class
        by_class: Dict[str, List[Dict]] = {}
        for det in detections:
            class_name = det["class_name"]
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(det)

        # Apply NMS per class
        filtered = []
        for class_name, class_dets in by_class.items():
            if not class_dets:
                continue

            # Sort by confidence
            class_dets = sorted(class_dets, key=lambda x: x["confidence"], reverse=True)

            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)

                # Remove overlapping boxes
                class_dets = [
                    det
                    for det in class_dets
                    if self._iou(best["bbox"], det["bbox"]) < self.iou_threshold
                ]

            filtered.extend(keep)

        return filtered

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        inter_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def _infer_hailo(self, image: np.ndarray) -> List[Dict]:
        """Run inference using Hailo accelerator."""
        if not self._hailo_configured or self._hailo_network is None:
            return []

        try:
            from hailo_platform import InferVStreams

            # Preprocess
            input_data = self._preprocess_hailo(image)

            # Inference
            with InferVStreams(
                self._hailo_network["network_group"],
                self._hailo_network["input_vstreams_params"],
                self._hailo_network["output_vstreams_params"],
            ) as infer_pipeline:
                # Get input/output names
                input_dict = {}
                for idx, input_vstream in enumerate(infer_pipeline.input):
                    input_dict[input_vstream.name] = input_data

                # Run inference
                output_dict = infer_pipeline.infer(input_dict)
                outputs = list(output_dict.values())

            # Postprocess
            return self._postprocess_hailo(outputs, image.shape[:2])

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
                    source=image,
                    imgsz=self.input_size,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )

            detections = []
            if results:
                r0 = results[0]
                if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
                    confs = r0.boxes.conf.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, clss):
                        class_name = (
                            self._class_names[class_id]
                            if class_id < len(self._class_names)
                            else f"class_{class_id}"
                        )
                        detections.append(
                            {
                                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                "class_name": class_name,
                                "class_id": int(class_id),
                                "confidence": float(conf),
                            }
                        )

            return detections

        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            return []

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on an image.

        Args:
            image: Input image (BGR format from cv2)

        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - class_name: str
                - class_id: int
                - confidence: float
        """
        if self._hailo_active:
            return self._infer_hailo(image)
        elif self._pytorch_model is not None:
            return self._infer_pytorch(image)
        else:
            return []

    def is_hailo_active(self) -> bool:
        """Check if Hailo accelerator is active."""
        return self._hailo_active

    def get_backend_name(self) -> str:
        """Get the name of the active backend."""
        if self._hailo_active:
            return "Hailo-8 (26 TOPS)"
        elif self._pytorch_model is not None:
            return "PyTorch CPU"
        else:
            return "No backend loaded"
