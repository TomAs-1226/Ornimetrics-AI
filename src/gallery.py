from typing import Dict, List, Tuple
import numpy as np
from numpy.linalg import norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    return 1 - float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


class Gallery:
    def __init__(self, default_threshold: float = 0.3):
        self.galleries: Dict[str, Dict[str, np.ndarray]] = {}
        self.thresholds: Dict[str, float] = {}
        self.default_threshold = default_threshold

    def register_class(self, class_name: str, threshold: float = None):
        if class_name not in self.galleries:
            self.galleries[class_name] = {}
        if threshold is not None:
            self.thresholds[class_name] = threshold

    def match(self, class_name: str, embedding: np.ndarray) -> Tuple[str, float]:
        class_gallery = self.galleries.get(class_name, {})
        if not class_gallery:
            return None, 1.0
        best_id, best_dist = None, float("inf")
        for indiv_id, emb in class_gallery.items():
            dist = cosine_distance(embedding, emb)
            if dist < best_dist:
                best_dist = dist
                best_id = indiv_id
        return best_id, best_dist

    def update(self, class_name: str, indiv_id: str, embedding: np.ndarray):
        self.register_class(class_name)
        class_gallery = self.galleries[class_name]
        if indiv_id in class_gallery:
            class_gallery[indiv_id] = 0.5 * class_gallery[indiv_id] + 0.5 * embedding
        else:
            class_gallery[indiv_id] = embedding

    def needs_new_identity(self, class_name: str, distance: float) -> bool:
        thr = self.thresholds.get(class_name, self.default_threshold)
        return distance > thr

