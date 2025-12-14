"""Embedding-based matching helpers with top-k diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Neighbor:
    individual_id: int
    distance: float


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / denom)


def topk_matches(
    embedding: np.ndarray,
    prototypes: Sequence[Tuple[int, np.ndarray, float]],
    k: int = 5,
) -> List[Neighbor]:
    """Return top-k nearest prototypes by cosine distance (weight-aware)."""

    neighbors: List[Neighbor] = []
    for individual_id, proto_vec, weight in prototypes:
        dist = cosine_distance(embedding, proto_vec)
        dist /= max(weight, 1e-6)
        neighbors.append(Neighbor(individual_id=int(individual_id), distance=float(dist)))
    neighbors.sort(key=lambda n: n.distance)
    return neighbors[:k]

