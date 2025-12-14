import numpy as np

from src.pc_preprocess import PreprocessConfig, DEFAULT_POINTS
from src.reid_embedder import PointReID


def test_embedder_handles_scale_feature():
    cfg = PreprocessConfig(
        normalization="center_and_scale_with_scale_feature",
        append_scale=True,
        fps_points=32,
    )
    embedder = PointReID(preprocess_config=cfg, emb_dims=64, model_name="light")
    # simple line cloud
    pc = np.stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)], axis=1).astype(np.float32)
    result = embedder.embed(pc)
    assert result.embedding.shape[0] == 64
    # ensure preprocessing kept the extra feature
    assert cfg.append_scale is True
    assert result.embedding.isfinite().all()


def test_heavy_backbone_accepts_extra_features():
    cfg = PreprocessConfig(normalization="center_only", append_scale=False, fps_points=32)
    embedder = PointReID(preprocess_config=cfg, emb_dims=512, model_name="heavy")
    pc = np.random.rand(64, 3).astype(np.float32)
    result = embedder.embed(pc)
    assert result.embedding.shape[0] == embedder.model.emb_dims
    assert float(result.embedding.norm()) > 0


def test_xheavy_transformer_backbone():
    cfg = PreprocessConfig(normalization="center_only", append_scale=False, fps_points=16)
    embedder = PointReID(preprocess_config=cfg, emb_dims=256, model_name="xheavy")
    pc = np.random.rand(32, 3).astype(np.float32)
    result = embedder.embed(pc)
    assert result.embedding.shape[0] == embedder.model.emb_dims
    assert float(result.embedding.norm()) > 0
