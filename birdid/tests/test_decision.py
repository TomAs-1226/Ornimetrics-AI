import numpy as np

from birdid.decision import decide
from birdid.db import BirdIDDatabase, MatchResult
from birdid.engine import BirdIDConfig
from birdid.validate_3d import ValidationResult


def test_cooldown_blocks_dispense(tmp_path):
    cfg = BirdIDConfig()
    cfg.cooldown_seconds = 60
    db = BirdIDDatabase(tmp_path / "birdid.sqlite")
    emb = np.ones(4, dtype=np.float32)
    ind_id = db.add_individual("sparrow", emb)
    now = 1000.0
    db.record_dispense(ind_id, now)
    match = MatchResult(individual_id=ind_id, distance=0.1, confidence=1.0, second_best=0.2, is_match=True, has_prototypes=True)
    validation = ValidationResult(True, 0.9)
    decision = decide(
        species="sparrow",
        track_id=1,
        embedding=emb,
        match=match,
        validation=validation,
        frames_used=5,
        config=cfg,
        db=db,
        now_ts=now + 10,
    )
    assert decision.decision == "DENY"
    assert decision.deny_reason == "cooldown"
    db.close()


def test_first_seen_enroll_no_dispense(tmp_path):
    cfg = BirdIDConfig()
    cfg.allow_first_seen_dispense = False
    db = BirdIDDatabase(tmp_path / "birdid.sqlite")
    emb = np.ones(4, dtype=np.float32)
    match = db.match("sparrow", emb, threshold=0.3)
    validation = ValidationResult(True, 0.8)
    decision = decide(
        species="sparrow",
        track_id=1,
        embedding=emb,
        match=match,
        validation=validation,
        frames_used=5,
        config=cfg,
        db=db,
        now_ts=100.0,
    )
    assert decision.decision == "ENROLL"
    db.close()


def test_dispense_requires_high_margin(tmp_path):
    cfg = BirdIDConfig()
    cfg.margin_threshold = 0.05
    cfg.dispense_threshold = 0.3
    emb = np.ones(4, dtype=np.float32)
    db = BirdIDDatabase(tmp_path / "birdid.sqlite")
    ind_id = db.add_individual("sparrow", emb)
    match = MatchResult(individual_id=ind_id, distance=0.28, confidence=1.0, second_best=0.30, is_match=True, has_prototypes=True)
    validation = ValidationResult(True, 0.9)
    decision = decide(
        species="sparrow",
        track_id=1,
        embedding=emb,
        match=match,
        validation=validation,
        frames_used=cfg.min_frames_for_dispense,
        config=cfg,
        db=db,
        now_ts=500.0,
    )
    assert decision.decision == "DENY"
    assert decision.deny_reason == "low_confidence"
    db.close()
