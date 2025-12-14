import numpy as np

from birdid.decision import decide
from birdid.db import BirdIDDatabase, MatchResult
from birdid.engine import BirdIDConfig
from birdid.validate_3d import ValidationResult


def test_decision_determinism(tmp_path):
    cfg = BirdIDConfig()
    cfg.dispense_threshold = 0.3
    cfg.margin_threshold = 0.05
    emb = np.ones(4, dtype=np.float32)

    def _run_once(db_path):
        db = BirdIDDatabase(db_path)
        ind = db.add_individual("sparrow", emb)
        match = MatchResult(individual_id=ind, distance=0.2, confidence=1.0, second_best=0.35, is_match=True, has_prototypes=True)
        validation = ValidationResult(True, 0.9)
        result = decide(
            species="sparrow",
            track_id=1,
            embedding=emb,
            match=match,
            validation=validation,
            frames_used=cfg.min_frames_for_dispense,
            config=cfg,
            db=db,
            now_ts=1000.0,
        )
        db.close()
        return result

    r1 = _run_once(tmp_path / "one.sqlite")
    r2 = _run_once(tmp_path / "two.sqlite")
    assert r1.decision == r2.decision == "DISPENSE"
    assert r1.margin == r2.margin
