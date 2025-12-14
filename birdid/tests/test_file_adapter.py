import numpy as np
import pytest

from birdid.inputs.file_adapter import build_inputs


def test_build_inputs_with_npy_depth(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)

    depth = np.full((10, 12), 0.4, dtype=np.float32)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, depth)

    inputs = build_inputs(rgb_path, depth_path, species="sparrow", bbox_xyxy=[1, 2, 5, 6])
    assert inputs.detection["species"] == "sparrow"
    assert inputs.depth_frame.depth.shape == depth.shape
    assert inputs.bbox_xyxy == [1, 2, 5, 6]
    assert inputs.point_cloud is None
