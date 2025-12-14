import os
from typing import List, Dict, Any
import cv2
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - allow import without ultralytics installed
    YOLO = None


WINDOWS_DEFAULT_MODEL = (
    r"C:\\Users\\User\\Downloads\\Ornimetrics-AI-codex-implement-individual-identification-pipeline"
    r"\\Ornimetrics-AI-codex-implement-individual-identification-pipeline\\best.pt"
)

WINDOWS_ROBUST_MODEL = (
    r"C:\\Users\\User\\Downloads\\Ornimetrics-AI-codex-implement-robust-point-cloud-identity-system"
    r"\\Ornimetrics-AI-codex-implement-robust-point-cloud-identity-system\\best.pt"
)

WINDOWS_MODEL_CANDIDATES = [WINDOWS_ROBUST_MODEL, WINDOWS_DEFAULT_MODEL]


def default_yolo_model_path(fallback: str = "yolov8n.pt") -> str:
    """Return a hardcoded Windows model path when present, otherwise fallback.

    The search prioritizes the latest requested repository layout while keeping
    compatibility with the prior default path and finally a portable fallback.
    """

    for candidate in WINDOWS_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return fallback


class YOLODetector:
    """Wrapper around ultralytics YOLO for RGB-only detection.

    The detector returns bboxes in xyxy format with class names and confidences.
    Identity is handled separately by the point-cloud re-id stack.
    """

    def __init__(self, model_path: str = None, device: str = None):
        if YOLO is None:
            raise ImportError("ultralytics not available; install per requirements.txt")
        model_to_load = model_path or default_yolo_model_path()
        self.model = YOLO(model_to_load)
        if device:
            self.model.to(device)

    def detect(self, image) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []
        results = self.model(image, verbose=False)[0]
        detections: List[Dict[str, Any]] = []
        names = results.names
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_idx = int(box.cls)
            detections.append(
                {
                    "bbox": xyxy,
                    "class_id": cls_idx,
                    "class_name": names.get(cls_idx, str(cls_idx)),
                    "score": float(box.conf),
                }
            )
        return detections


def load_rgb_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

