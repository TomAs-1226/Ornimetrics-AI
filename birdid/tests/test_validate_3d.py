import numpy as np

from birdid.engine import BirdIDConfig
from birdid.validate_3d import validate_tracklet


def test_planar_surface_rejected():
    cfg = BirdIDConfig()
    depth = np.zeros((30, 30), dtype=np.float32)
    x = np.linspace(0, 0.05, depth.shape[1], dtype=np.float32)
    depth += x
    depth = depth + 0.35
    mask = np.ones_like(depth, dtype=bool)
    result = validate_tracklet([(depth, mask)], cfg)
    assert not result.is_valid
    assert result.reason in {"planar_surface", "thickness_out_of_bounds", "size_out_of_bounds"}


def test_size_out_of_bounds_rejected():
    cfg = BirdIDConfig()
    cfg.validation_planar_ratio = 0.0
    cfg.validation_min_thickness = 0.0
    rng = np.random.RandomState(0)
    depth = 0.4 + 0.05 * rng.rand(40, 40).astype(np.float32)
    mask = np.ones_like(depth, dtype=bool)
    result = validate_tracklet([(depth, mask)], cfg)
    assert result.is_valid
    assert result.quality_score < 1.0
