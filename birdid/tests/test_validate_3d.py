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
    assert result.reason in {"planar_surface", "thickness_out_of_bounds"}
