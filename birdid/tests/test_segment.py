import numpy as np
from birdid.segment import DepthSegmenter


def test_segment_extracts_component():
    depth = np.ones((5, 5), dtype=np.float32) * 0.4
    depth[1:4, 1:4] = 0.36
    valid = np.ones_like(depth, dtype=bool)
    seg = DepthSegmenter(alpha=0.0, min_valid_ratio=0.05)
    clean, mask = seg.segment(depth, valid)
    assert mask.sum() >= 4
    assert np.all(clean[mask] < 0.39)
