import numpy as np
import sys
import types

sys.modules.setdefault("cv2", types.SimpleNamespace(imread=lambda *args, **kwargs: None, IMREAD_UNCHANGED=0))
class _DummyKDTree:
    def __init__(self, data):
        self.data = data

    def query(self, points, k=1):
        vals = np.ones(len(points)) * 10.0
        return vals, None


scipy_spatial = types.SimpleNamespace(cKDTree=_DummyKDTree)
sys.modules.setdefault("scipy", types.SimpleNamespace(spatial=scipy_spatial))
sys.modules.setdefault("scipy.spatial", scipy_spatial)
sys.modules.setdefault("src.tracker", types.SimpleNamespace(Tracker=type("DummyTrackerCls", (), {"__init__": lambda self, *args, **kwargs: None, "tracks": []})))
sys.modules.setdefault("src.yolo_detect", types.SimpleNamespace(YOLODetector=object, load_rgb_image=lambda *args, **kwargs: None))
sys.modules.setdefault("src.depth_to_points", types.SimpleNamespace(backproject_depth=lambda depth, intrinsics, bbox: (depth, None), BackprojectStats=type("_BS", (), {"__init__": lambda self, *args, **kwargs: None})))
sys.modules.setdefault("src.naming", types.SimpleNamespace(generate_name=lambda: "name"))
sys.modules.setdefault("src.db", types.SimpleNamespace(IdentityDB=object))
sys.modules.setdefault("src.pc_preprocess", types.SimpleNamespace(PreprocessConfig=object, DEFAULT_POINTS=2048, _get_o3d=lambda: None))
dummy_reid = types.SimpleNamespace(
    PointReID=type(
        "DummyPointReID",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "embed": lambda self, points: None,
            "model": type("m", (), {"emb_dims": 4})(),
        },
    )
)
sys.modules.setdefault("src.reid_embedder", dummy_reid)

from scripts.infer_stream import process_frame
from src.gallery import Gallery


class DummyDetModel:
    def detect(self, _rgb):
        return [
            {
                "bbox": [0, 0, 1, 1],
                "class_name": "bird",
                "class_id": 0,
            }
        ]


class DummyTensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class DummyEmbedResult:
    def __init__(self, embedding, points):
        self.embedding = embedding
        self.points = points
        self.preprocess_stats = {}


class DummyEmbedder:
    def __init__(self, emb_dims=4, embed_value=1.0):
        self.model = type("m", (), {"emb_dims": emb_dims})
        self._embed_value = embed_value

    def embed(self, points):
        emb = DummyTensor(np.ones(self.model.emb_dims, dtype=np.float32) * self._embed_value)
        return DummyEmbedResult(emb, points.astype(np.float32))


class DummyTracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        return detections


class DummyDB:
    def __init__(self, name="known"):
        self._name = name

    def upsert_individual(self, *_args, **_kwargs):
        return None

    def get_identity(self, _indiv_id):
        return (self._name, "bird")


def test_geometry_guard_rejects_dispense_when_geometry_far():
    gallery = Gallery(default_threshold=0.3, geometry_guard=0.05)
    ref_points = np.zeros((8, 3), dtype=np.float32)
    gallery.update("bird", "id1", np.ones(4), geometry_points=ref_points)

    # Candidate geometry far from reference to trigger guard
    candidate_points = np.ones((8, 3), dtype=np.float32) * 2.0

    outputs = process_frame(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=candidate_points,
        det_model=DummyDetModel(),
        embedder=DummyEmbedder(),
        gallery=gallery,
        tracker=DummyTracker(),
        db=DummyDB(),
        intrinsics=[1, 1, 0, 0],
        threshold=0.3,
        debug=False,
        bird_extractor=None,
    )

    assert outputs[0]["action"] == "rejected"
    assert outputs[0]["decision_state"] == "rejected"
    assert outputs[0]["decision_reason"] == "geometry_guard"
