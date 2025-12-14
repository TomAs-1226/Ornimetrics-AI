import numpy as np
from src.gallery import Gallery


def test_gallery_threshold_new_identity():
    gallery = Gallery(default_threshold=0.3)
    emb1 = np.ones(256)
    gallery.update("bird", "id1", emb1)
    emb_far = np.zeros(256)
    match_id, dist = gallery.match("bird", emb_far)
    assert gallery.needs_new_identity("bird", dist)
