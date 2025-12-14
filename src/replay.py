"""Deterministic record/replay helpers for regression safety."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence


@dataclass
class ReplayFrame:
    rgb_path: Path
    depth_path: Path
    detections_path: Path


def record_outputs(outputs: Sequence[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(outputs), f, indent=2, sort_keys=True)


def replay(outputs_path: Path) -> List[Dict[str, Any]]:
    with open(outputs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_outputs(recorded: Sequence[Dict[str, Any]], fresh: Sequence[Dict[str, Any]], tolerance: float = 1e-6) -> bool:
    if len(recorded) != len(fresh):
        return False
    for rec, new in zip(recorded, fresh):
        if rec.keys() != new.keys():
            return False
        for key in rec:
            if isinstance(rec[key], float):
                if abs(rec[key] - new[key]) >= tolerance:
                    return False
            else:
                if rec[key] != new[key]:
                    return False
    return True


__all__ = ["record_outputs", "replay", "compare_outputs", "ReplayFrame"]
