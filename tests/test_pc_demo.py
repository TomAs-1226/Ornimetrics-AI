import pytest

pytest.importorskip("numpy")

from pc_demo.app import HybridDetector


class DummyDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image):  # pragma: no cover - trivial
        return self._detections


def test_hybrid_detector_no_fallback_returns_empty_when_disabled():
    detector = HybridDetector(base=DummyDetector([]), fallback_bbox=None, class_name="sparrow", override_class=False, allow_fallback=False)
    out = detector.detect(image=[[0]])
    assert out == []


def test_hybrid_detector_preserves_base_class_when_not_overridden():
    base_det = DummyDetector([{"bbox": [0, 0, 1, 1], "class_name": "eagle", "score": 0.9}])
    detector = HybridDetector(base=base_det, fallback_bbox=None, class_name="sparrow", override_class=False, allow_fallback=False)
    out = detector.detect(image=[[0]])
    assert out[0]["class_name"] == "eagle"
