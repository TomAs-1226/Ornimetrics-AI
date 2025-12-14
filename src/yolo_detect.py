from typing import List, Dict, Any
import cv2
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - allow import without ultralytics installed
    YOLO = None


class YOLODetector:
    """Wrapper around ultralytics YOLO for RGB-only detection.

    The detector returns bboxes in xyxy format with class names and confidences.
    Identity is handled separately by the point-cloud re-id stack.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = None):
        if YOLO is None:
            raise ImportError("ultralytics not available; install per requirements.txt")
        self.model = YOLO(model_path)
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

