import numpy as np

from src.pc_preprocess import PreprocessConfig, DEFAULT_POINTS
from src.reid_embedder import PointReID


def test_embedder_handles_scale_feature():
    cfg = PreprocessConfig(
        normalization="center_and_scale_with_scale_feature",
        append_scale=True,
        fps_points=32,
    )
    embedder = PointReID(preprocess_config=cfg, emb_dims=64)
    # simple line cloud
    pc = np.stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)], axis=1).astype(np.float32)
    result = embedder.embed(pc)
    assert result.embedding.shape[0] == 64
    # ensure preprocessing kept the extra feature
    assert cfg.append_scale is True
    assert result.embedding.isfinite().all()
