import os

from src import yolo_detect


def test_default_yolo_model_prefers_candidates(monkeypatch):
    called = []

    def fake_exists(path):
        called.append(path)
        return path == yolo_detect.WINDOWS_MODEL_CANDIDATES[0]

    monkeypatch.setattr(os.path, "exists", fake_exists)
    assert yolo_detect.default_yolo_model_path("fallback.pt") == yolo_detect.WINDOWS_MODEL_CANDIDATES[0]
    assert called[0] == yolo_detect.WINDOWS_MODEL_CANDIDATES[0]
