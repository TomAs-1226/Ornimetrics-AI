import io

import numpy as np
import pytest

from birdid.inputs.file_adapter import build_inputs, load_rgb_image


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


def test_depth_png_uint16_mm_to_m(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)

    depth_mm = np.full((8, 8), 400, dtype=np.uint16)
    depth_path = tmp_path / "depth.png"
    Image.fromarray(depth_mm).save(depth_path)

    inputs = build_inputs(rgb_path, depth_path, species="sparrow")
    depth = inputs.depth_frame.depth
    valid = inputs.depth_frame.valid
    assert depth.dtype == np.float32
    assert np.isclose(depth.max(), 0.4)
    assert valid.sum() == depth.size


def test_bbox_none_defaults_full_frame(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)

    depth = np.ones((4, 6), dtype=np.float32)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, depth)

    inputs = build_inputs(rgb_path, depth_path, species="finch", bbox_xyxy=None)
    assert inputs.bbox_xyxy == [0, 0, 5, 3]


def test_rgb_loader_rewinds_stream(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 128
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")

    # Simulate Streamlit UploadedFile being read multiple times
    # Single buffer reused twice (seeking should be handled internally)
    shared = io.BytesIO(buf.getvalue())
    img1 = load_rgb_image(shared)
    img2 = load_rgb_image(shared)

    assert img1.shape == img2.shape == (4, 4, 3)
    assert img1[0, 0, 1] == 128
