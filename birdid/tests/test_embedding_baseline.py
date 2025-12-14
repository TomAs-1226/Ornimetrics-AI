import numpy as np
from birdid.embedding_baseline import compute_baseline_embedding


def test_baseline_deterministic():
    depth = np.ones((4, 4), dtype=np.float32) * 0.4
    mask = np.zeros_like(depth, dtype=bool)
    mask[1:3, 1:3] = True
    emb1 = compute_baseline_embedding(depth, mask)
    emb2 = compute_baseline_embedding(depth, mask)
    np.testing.assert_allclose(emb1, emb2)
    assert np.isclose(np.linalg.norm(emb1), 1.0)
