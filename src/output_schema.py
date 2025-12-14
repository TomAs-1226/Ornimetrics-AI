from typing import Dict, Any
import json


def format_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "species": det.get("class_name"),
        "track_id": det.get("track_id"),
        "individual_id": det.get("individual_id"),
        "individual_name": det.get("individual_name"),
        "min_dist": det.get("min_dist"),
        "second_best_dist": det.get("second_best_dist"),
        "margin": det.get("margin"),
        "quality_score": det.get("quality", 1.0),
        "frames_used": det.get("frames_used", 1),
        "deny_reason": det.get("deny_reason"),
        "decision_state": det.get("decision_state"),
        "decision_reason": det.get("decision_reason"),
        "action": det.get("action"),
    }


def to_json(det: Dict[str, Any]) -> str:
    return json.dumps(format_detection(det))


__all__ = ["format_detection", "to_json"]

