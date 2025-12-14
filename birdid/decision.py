"""Decision layer for dispense/enroll/deny outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .db import BirdIDDatabase, MatchResult
from .validate_3d import ValidationResult


@dataclass
class DecisionResult:
    decision: str
    individual_id: Optional[int]
    margin: float
    deny_reason: Optional[str]
    cooldown_remaining: float


def decide(
    *,
    species: str,
    track_id: int,
    embedding,
    match: MatchResult,
    validation: ValidationResult,
    frames_used: int,
    config,
    db: BirdIDDatabase,
    now_ts: float,
) -> DecisionResult:
    if not validation.is_valid:
        return DecisionResult("DENY", match.individual_id, 0.0, validation.reason or "invalid_depth_or_spoof", 0.0)

    if match.has_prototypes:
        margin = float(match.second_best - match.distance)
    else:
        margin = float("inf")
    cooldown_remaining = 0.0

    if not match.has_prototypes or not match.is_match:
        new_id = db.add_individual(species, embedding)
        db.record_attempt(new_id, now_ts)
        if config.allow_first_seen_dispense and validation.quality_score >= config.min_quality_score_for_dispense and frames_used >= config.min_frames_for_dispense:
            return DecisionResult("DISPENSE", new_id, margin, None, 0.0)
        return DecisionResult("ENROLL", new_id, margin, None, 0.0)

    individual_id = match.individual_id
    stats = db.get_stats(individual_id, now_ts)
    cooldown_remaining = max(0.0, config.cooldown_seconds - (now_ts - stats["last_dispense_ts"]))
    if cooldown_remaining > 0:
        db.record_attempt(individual_id, now_ts)
        return DecisionResult("DENY", individual_id, margin, "cooldown", cooldown_remaining)
    if stats["dispense_count_today"] >= config.max_dispenses_per_day:
        db.record_attempt(individual_id, now_ts)
        return DecisionResult("DENY", individual_id, margin, "daily_limit", 0.0)

    meets_dispense = (
        match.distance < config.dispense_threshold
        and margin > config.margin_threshold
        and frames_used >= config.min_frames_for_dispense
        and validation.quality_score >= config.min_quality_score_for_dispense
    )
    if not meets_dispense:
        db.record_attempt(individual_id, now_ts)
        return DecisionResult("DENY", individual_id, margin, "low_confidence", 0.0)

    db.record_dispense(individual_id, now_ts)
    return DecisionResult("DISPENSE", individual_id, margin, None, 0.0)
