import numpy as np
from src.gallery import Gallery, cosine_distance


def test_gallery_threshold_new_identity():
    gallery = Gallery(default_threshold=0.3)
    emb1 = np.ones(4)
    gallery.update("bird", "id1", emb1)
    emb_far = np.zeros(4)
    _, dist, _ = gallery.match("bird", emb_far)
    assert gallery.needs_new_identity("bird", dist)


def test_cosine_distance_direction():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    dist = cosine_distance(a, b)
    assert dist > 0.9
    assert cosine_distance(a, a) < 1e-6


def test_gallery_per_species():
    gallery = Gallery(default_threshold=0.3)
    gallery.update("bird", "id1", np.ones(3))
    gallery.update("cat", "id2", np.array([0.1, 0.2, 0.3]))
    best_bird, _, _ = gallery.match("bird", np.ones(3))
    best_cat, _, _ = gallery.match("cat", np.array([0.1, 0.2, 0.3]))
    assert best_bird == "id1"
    assert best_cat == "id2"


def test_geometry_guard_creates_new_identity():
    gallery = Gallery(default_threshold=0.3, geometry_guard=0.05)
    base_emb = np.ones(3)
    ref_points = np.zeros((10, 3))
    gallery.update("bird", "id1", base_emb, geometry_points=ref_points)

    candidate_points = np.ones((10, 3)) * 10.0  # far away geometry
    _, dist, scored = gallery.match("bird", base_emb, candidate_points=candidate_points)
    assert scored[0]["geometry"] > gallery.geometry_guard
    assert gallery.needs_new_identity("bird", dist, geometry=scored[0]["geometry"])
