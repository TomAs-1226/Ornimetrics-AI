from pathlib import Path

from src.replay import compare_outputs, record_outputs, replay


def test_record_and_replay_roundtrip(tmp_path: Path):
    outputs = [
        {"track_id": 1, "individual_id": "a", "min_dist": 0.1},
        {"track_id": 2, "individual_id": "b", "min_dist": 0.2},
    ]
    record_path = tmp_path / "record.json"
    record_outputs(outputs, record_path)
    loaded = replay(record_path)
    assert compare_outputs(outputs, loaded)


def test_compare_outputs_tolerance(tmp_path: Path):
    outputs = [{"track_id": 1, "min_dist": 0.1}]
    alt = [{"track_id": 1, "min_dist": 0.1001}]
    assert compare_outputs(outputs, alt, tolerance=1e-3)
    assert not compare_outputs(outputs, alt, tolerance=1e-7)
