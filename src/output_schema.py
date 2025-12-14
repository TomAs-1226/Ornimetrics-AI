from typing import Dict, Any
import json


def format_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "species": det.get("class_name"),
        "track_id": det.get("track_id"),
        "individual_id": det.get("individual_id"),
        "individual_name": det.get("individual_name"),
        "min_dist": det.get("distance"),
        "margin": det.get("margin"),
        "quality_score": det.get("quality", 1.0),
        "frames_used": det.get("frames_used", 1),
    }


def to_json(det: Dict[str, Any]) -> str:
    return json.dumps(format_detection(det))


__all__ = ["format_detection", "to_json"]

