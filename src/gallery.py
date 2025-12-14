from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    return 1 - float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


@dataclass
class IdentityEntry:
    individual_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    name: Optional[str] = None
    geometry_ref: Optional[np.ndarray] = None

    def add_embedding(self, emb: np.ndarray, max_keep: int = 20, geometry: Optional[np.ndarray] = None):
        self.embeddings.append(emb)
        if len(self.embeddings) > max_keep:
            self.embeddings = self.embeddings[-max_keep:]
        if geometry is not None and geometry.size > 0:
            # Keep a small deterministic slice for geometry checks
            self.geometry_ref = geometry.astype(np.float32)

    @property
    def representative(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0,), dtype=np.float32)
        stacked = np.stack(self.embeddings)
        return np.median(stacked, axis=0)

    def geometry_distance(self, points: Optional[np.ndarray]) -> float:
        if points is None or self.geometry_ref is None or self.geometry_ref.size == 0 or points.size == 0:
            return float("inf")
        a = self.geometry_ref[:, :3]
        b = points[:, :3]
        tree_a = cKDTree(a)
        tree_b = cKDTree(b)
        dist_ba, _ = tree_b.query(a, k=1)
        dist_ab, _ = tree_a.query(b, k=1)
        return float((dist_ab.mean() + dist_ba.mean()) / 2.0)


class Gallery:
    def __init__(self, default_threshold: float = 0.05, margin_guard: float = 0.02, geometry_guard: float = 0.02):
        self.galleries: Dict[str, Dict[str, IdentityEntry]] = {}
        self.thresholds: Dict[str, float] = {}
        self.default_threshold = default_threshold
        self.margin_guard = margin_guard
        self.geometry_guard = geometry_guard

    def register_class(self, class_name: str, threshold: float = None):
        if class_name not in self.galleries:
            self.galleries[class_name] = {}
        if threshold is not None:
            self.thresholds[class_name] = threshold

    def get_threshold(self, class_name: str) -> float:
        return self.thresholds.get(class_name, self.default_threshold)

    def match(
        self, class_name: str, embedding: np.ndarray, candidate_points: Optional[np.ndarray] = None
    ) -> Tuple[Optional[str], float, List[Dict[str, float]]]:
        class_gallery = self.galleries.get(class_name, {})
        if not class_gallery:
            return None, 1.0, []
        scored: List[Dict[str, float]] = []
        for indiv_id, entry in class_gallery.items():
            dist = cosine_distance(embedding, entry.representative)
            geom = entry.geometry_distance(candidate_points)
            scored.append({"id": indiv_id, "dist": dist, "geometry": geom})
        scored.sort(key=lambda x: x["dist"])
        best = scored[0]
        return best["id"], best["dist"], scored[:5]

    def needs_new_identity(self, class_name: str, distance: float, geometry: float = float("inf")) -> bool:
        thr = self.get_threshold(class_name)
        if geometry < float("inf") and geometry > self.geometry_guard:
            return True
        return distance > thr

    def update(
        self,
        class_name: str,
        indiv_id: str,
        embedding: np.ndarray,
        allow: bool = True,
        name: Optional[str] = None,
        geometry_points: Optional[np.ndarray] = None,
    ):
        self.register_class(class_name)
        if indiv_id not in self.galleries[class_name]:
            self.galleries[class_name][indiv_id] = IdentityEntry(individual_id=indiv_id, name=name)
        if allow:
            geom = None
            if geometry_points is not None and geometry_points.size > 0:
                geom = geometry_points.astype(np.float32)
            self.galleries[class_name][indiv_id].add_embedding(embedding, geometry=geom)

    def should_update_identity(self, class_name: str, dist: float) -> bool:
        thr = self.get_threshold(class_name)
        return dist < (thr - self.margin_guard)


__all__ = ["Gallery", "cosine_distance", "IdentityEntry"]
