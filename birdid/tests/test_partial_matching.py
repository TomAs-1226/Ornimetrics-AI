import numpy as np

from birdid import embedding_baseline
from birdid.db import BirdIDDatabase
from birdid.validate_3d import validate_tracklet


def test_partial_depth_still_matches(tmp_path):
    db = BirdIDDatabase(tmp_path / "birdid.sqlite")
    base = np.ones((20, 20), dtype=np.float32) * 0.4
    gradient = np.linspace(0, 0.01, 20, dtype=np.float32)
    depth = base + gradient[None, :]
    mask_full = np.ones_like(depth, dtype=bool)
    emb_full = embedding_baseline.compute_baseline_embedding(depth, mask_full)
    indiv = db.add_individual("sparrow", emb_full)

    mask_partial = np.zeros_like(mask_full)
    mask_partial[:10, :10] = True
    emb_partial = embedding_baseline.compute_baseline_embedding(depth, mask_partial)
    match = db.match("sparrow", emb_partial, threshold=0.8)
    assert match.individual_id == indiv
    assert match.distance < 0.8

    validation = validate_tracklet([(depth, mask_partial)], type("cfg", (), {
        "validation_min_size_m": 0.01,
        "validation_max_size_m": 1.0,
        "validation_min_valid_ratio": 0.05,
        "validation_min_points": 5,
        "validation_min_thickness": 0.0,
        "validation_max_thickness": 1.0,
        "validation_planar_ratio": 0.0,
        "validation_max_centroid_jitter": 1.0,
    }))
    assert validation.is_valid
    assert validation.quality_score < 1.0
