import numpy as np

from src.multi_instance_depth import MultiInstanceConfig, MultiInstanceDepthProcessor


INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}


def _make_depth_with_clusters():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[10:20, 10:20] = 1.0
    depth[60:70, 60:70] = 2.0
    return depth


def test_assigns_unique_clusters():
    depth = _make_depth_with_clusters()
    cfg = MultiInstanceConfig(enabled=True, use_radius_clustering=True, radius=0.05, min_points=5)
    proc = MultiInstanceDepthProcessor(cfg)
    detections = [
        {"bbox": (8, 8, 25, 25)},
        {"bbox": (58, 58, 75, 75)},
    ]
    assignments, debug_info = proc.assign_clusters(depth, INTRINSICS, detections)
    assert len(assignments) == 2
    ids = {assignments[i].get("cluster_id") for i in assignments}
    assert None not in ids
    assert len(ids) == 2
    assert debug_info["num_clusters"] >= 2


def test_mixed_instance_rejection():
    depth = _make_depth_with_clusters()
    cfg = MultiInstanceConfig(
        enabled=True,
        use_radius_clustering=True,
        radius=0.05,
        min_points=5,
        min_iou=0.01,
        max_cost=10.0,
        split_birdness=0.0,
        split_iou_threshold=0.01,
    )
    proc = MultiInstanceDepthProcessor(cfg)
    detections = [{"bbox": (5, 5, 80, 80)}]
    assignments, _ = proc.assign_clusters(depth, INTRINSICS, detections)
    assert assignments[0].get("deny_reason") == "mixed_instances"


def test_disabled_returns_empty():
    depth = _make_depth_with_clusters()
    cfg = MultiInstanceConfig(enabled=False)
    proc = MultiInstanceDepthProcessor(cfg)
    detections = [{"bbox": (8, 8, 25, 25)}]
    assignments, debug_info = proc.assign_clusters(depth, INTRINSICS, detections)
    assert assignments == {}
    assert debug_info == {}
