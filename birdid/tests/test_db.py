import numpy as np
from birdid.db import BirdIDDatabase


def test_open_set_and_match(tmp_path):
    db = BirdIDDatabase(tmp_path / "birdid.sqlite")
    vec1 = np.ones(8, dtype=np.float32)
    res1 = db.match("sparrow", vec1, threshold=0.3)
    assert res1.individual_id > 0
    vec2 = vec1 + 0.001
    res2 = db.match("sparrow", vec2, threshold=0.3)
    assert res2.individual_id == res1.individual_id
    db.close()
