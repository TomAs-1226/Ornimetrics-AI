from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def create_kf(x, y, w, h):
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )
    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]
    )
    kf.R *= 10
    kf.P *= 10
    kf.x[:4] = np.array([[x], [y], [w], [h]])
    return kf


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    embedding_history: List[np.ndarray] = field(default_factory=list)
    kf: KalmanFilter = None
    time_since_update: int = 0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.kf = create_kf((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox, embedding=None, smooth_window: int = 5):
        x1, y1, x2, y2 = bbox
        z = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]])
        self.kf.update(z)
        self.time_since_update = 0
        self.bbox = bbox
        if embedding is not None:
            self.embedding_history.append(embedding)
            if len(self.embedding_history) > smooth_window:
                self.embedding_history = self.embedding_history[-smooth_window:]

    @property
    def smoothed_embedding(self):
        if not self.embedding_history:
            return None
        stacked = np.stack(self.embedding_history)
        return stacked.mean(axis=0)

    def current_bbox(self):
        cx, cy, w, h = self.kf.x[:4].reshape(-1)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class Tracker:
    def __init__(self, alpha: float = 0.5, max_age: int = 10, smooth_window: int = 5):
        self.alpha = alpha
        self.max_age = max_age
        self.smooth_window = smooth_window
        self.tracks: List[Track] = []
        self.next_id = 1

    def _appearance_cost(self, track: Track, det_emb: np.ndarray):
        if track.smoothed_embedding is None or det_emb is None:
            return 1.0
        te = track.smoothed_embedding
        denom = (np.linalg.norm(te) * np.linalg.norm(det_emb) + 1e-8)
        return 1 - float(np.dot(te, det_emb) / denom)

    def _bbox_cost(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter + 1e-6
        iou = inter / union
        return 1 - iou

    def _cost_matrix(self, detections: List[Dict[str, Any]]):
        if not self.tracks or not detections:
            return np.zeros((0, 0))
        cost = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                bbox_cost = self._bbox_cost(track.current_bbox(), det["bbox"])
                emb_cost = self._appearance_cost(track, det.get("embedding"))
                cost[i, j] = self.alpha * bbox_cost + (1 - self.alpha) * emb_cost
        return cost

    def update(self, detections: List[Dict[str, Any]]):
        for track in self.tracks:
            track.predict()
        cost = self._cost_matrix(detections)
        if cost.size == 0:
            unmatched_dets = list(range(len(detections)))
            matched_indices = []
        else:
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_indices = list(zip(row_ind, col_ind))
            unmatched_dets = [i for i in range(len(detections)) if i not in col_ind]
        used_tracks = set()
        for r, c in matched_indices:
            track = self.tracks[r]
            det = detections[c]
            motion_cost = self._bbox_cost(track.current_bbox(), det["bbox"])
            appearance_cost = self._appearance_cost(track, det.get("embedding"))
            combined = self.alpha * motion_cost + (1 - self.alpha) * appearance_cost
            track.update(det["bbox"], det.get("embedding"), self.smooth_window)
            det["track_id"] = track.track_id
            det["tracker_debug"] = {
                "motion_cost": float(motion_cost),
                "appearance_cost": float(appearance_cost),
                "combined_cost": float(combined),
            }
            used_tracks.add(r)
        # Create new tracks for unmatched detections
        for idx in unmatched_dets:
            det = detections[idx]
            track = Track(self.next_id, det["bbox"])
            track.update(det["bbox"], det.get("embedding"), self.smooth_window)
            self.tracks.append(track)
            det["track_id"] = self.next_id
            det["tracker_debug"] = {
                "motion_cost": 1.0,
                "appearance_cost": 1.0,
                "combined_cost": 1.0,
            }
            self.next_id += 1
        # Prune old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return detections


__all__ = ["Tracker", "Track"]
