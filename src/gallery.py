from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.linalg import norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    return 1 - float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


@dataclass
class IdentityEntry:
    individual_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    name: Optional[str] = None

    def add_embedding(self, emb: np.ndarray, max_keep: int = 20):
        self.embeddings.append(emb)
        if len(self.embeddings) > max_keep:
            self.embeddings = self.embeddings[-max_keep:]

    @property
    def representative(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0,), dtype=np.float32)
        stacked = np.stack(self.embeddings)
        return np.median(stacked, axis=0)


class Gallery:
    def __init__(self, default_threshold: float = 0.3, margin_guard: float = 0.05):
        self.galleries: Dict[str, Dict[str, IdentityEntry]] = {}
        self.thresholds: Dict[str, float] = {}
        self.default_threshold = default_threshold
        self.margin_guard = margin_guard

    def register_class(self, class_name: str, threshold: float = None):
        if class_name not in self.galleries:
            self.galleries[class_name] = {}
        if threshold is not None:
            self.thresholds[class_name] = threshold

    def get_threshold(self, class_name: str) -> float:
        return self.thresholds.get(class_name, self.default_threshold)

    def match(self, class_name: str, embedding: np.ndarray) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        class_gallery = self.galleries.get(class_name, {})
        if not class_gallery:
            return None, 1.0, []
        scored = []
        for indiv_id, entry in class_gallery.items():
            dist = cosine_distance(embedding, entry.representative)
            scored.append((indiv_id, dist))
        scored.sort(key=lambda x: x[1])
        best_id, best_dist = scored[0]
        return best_id, best_dist, scored[:5]

    def needs_new_identity(self, class_name: str, distance: float) -> bool:
        thr = self.get_threshold(class_name)
        return distance > thr

    def update(self, class_name: str, indiv_id: str, embedding: np.ndarray, allow: bool = True, name: Optional[str] = None):
        self.register_class(class_name)
        if indiv_id not in self.galleries[class_name]:
            self.galleries[class_name][indiv_id] = IdentityEntry(individual_id=indiv_id, name=name)
        if allow:
            self.galleries[class_name][indiv_id].add_embedding(embedding)

    def should_update_identity(self, class_name: str, dist: float) -> bool:
        thr = self.get_threshold(class_name)
        return dist < (thr - self.margin_guard)


__all__ = ["Gallery", "cosine_distance", "IdentityEntry"]
