import numpy as np

from src.bird_point_extractor import AntiSpoofConfig, BirdPointCloudExtractor, BirdExtractorConfig


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


def test_extractor_flags_spoof_background():
    # flat plane with tiny jitter should be rejected when anti-spoof is enabled
    base = np.ones((30, 30), dtype=np.float32)
    noise = (np.random.rand(30, 30) - 0.5) * 0.001
    depth = base + noise
    cfg = BirdExtractorConfig(
        min_cluster_points=50,
        planarity_ratio=0.01,
        anti_spoof=AntiSpoofConfig(enabled=True, threshold=0.5),
    )
    extractor = BirdPointCloudExtractor(cfg)
    result = extractor.extract(np.zeros((30, 30, 3)), depth, (0, 0, 30, 30), _intrinsics())
    assert result.deny_reason in {"spoof_background", "planar_surface", "depth_missing"}
