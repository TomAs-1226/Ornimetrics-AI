import numpy as np

from src.bird_point_extractor import BirdPointCloudExtractor, BirdExtractorConfig


def _intrinsics():
    return {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}


def test_extractor_rejects_empty_depth():
    depth = np.zeros((10, 10), dtype=np.float32)
    extractor = BirdPointCloudExtractor(BirdExtractorConfig(min_cluster_points=10))
    result = extractor.extract(np.zeros((10, 10, 3)), depth, (0, 0, 5, 5), _intrinsics())
    assert result.deny_reason == "depth_missing"
    assert result.quality == 0.0


def test_extractor_rejects_planar_surface():
    depth = np.ones((20, 20), dtype=np.float32)
    extractor = BirdPointCloudExtractor(BirdExtractorConfig(min_cluster_points=10, planarity_ratio=0.5))
    result = extractor.extract(np.zeros((20, 20, 3)), depth, (0, 0, 20, 20), _intrinsics())
    assert result.deny_reason in {"planar_surface", "depth_missing", "no_valid_cluster"}
