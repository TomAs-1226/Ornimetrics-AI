import numpy as np

from birdid.matching import cosine_distance, topk_matches


def test_cosine_distance_distinguishes_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([1.0, 0.1, 0.0])
    assert cosine_distance(a, b) > cosine_distance(a, c)


def test_topk_matches_returns_sorted_neighbors():
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prototypes = [
        (1, np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0),
        (2, np.array([0.0, 1.0, 0.0], dtype=np.float32), 1.0),
    ]
    neighbors = topk_matches(embedding, prototypes, k=2)
    assert [n.individual_id for n in neighbors] == [1, 2]
