import numpy as np
from src.depth_to_points import backproject_depth


def test_backproject_depth_shape():
    depth = np.ones((4, 4), dtype=np.float32)
    intr = {"fx": 1.0, "fy": 1.0, "cx": 1.5, "cy": 1.5}
    bbox = (1, 1, 3, 3)
    pts = backproject_depth(depth, intr, bbox)
    assert pts.shape[1] == 3
    assert pts.shape[0] > 0
