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


def test_planar_point_cloud_rejected():
    cfg = BirdIDConfig()
    # Large planar sheet with tiny thickness
    xs, ys = np.meshgrid(np.linspace(0, 0.1, 80), np.linspace(0, 0.1, 80))
    zs = np.full_like(xs, 0.4) + 0.0005 * np.random.randn(*xs.shape)
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    result = validate_tracklet([], cfg, point_cloud=pts)
    assert not result.is_valid
    assert result.reason == "planar_surface"


def test_non_planar_thin_cloud_passes():
    cfg = BirdIDConfig()
    rng = np.random.RandomState(0)
    pts = rng.rand(2000, 3).astype(np.float32)
    pts[:, 2] = 0.35 + 0.02 * rng.rand(2000)
    result = validate_tracklet([], cfg, point_cloud=pts)
    assert result.is_valid
    assert result.quality_score > 0


def test_mm_point_cloud_auto_converts():
    cfg = BirdIDConfig()
    pts_mm = np.array(
        [
            [20.0, 0.0, 400.0],
            [0.0, 15.0, 407.0],
            [5.0, 5.0, 410.0],
        ],
        dtype=np.float32,
    )
    result = validate_tracklet([], cfg, point_cloud=pts_mm, point_units="auto")
    assert result.is_valid
    assert result.diagnostics.get("thickness", 0) > 0.006 / 2  # meters after conversion
