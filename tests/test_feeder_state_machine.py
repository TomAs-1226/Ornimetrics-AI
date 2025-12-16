import datetime as dt

from src.status.feeder_state import FeederStateMachine, SensorSample, Thresholds


def make_sample(val: float | None, ok: bool = True, ts: dt.datetime | None = None) -> SensorSample:
    return SensorSample(value_mm=val, ok=ok, timestamp=ts or dt.datetime.utcnow())


def test_hysteresis_and_debounce_food_empty():
    machine = FeederStateMachine("feeder", Thresholds(empty_on_mm=300, empty_off_mm=200), debounce_count=2, median_window=3)
    now = dt.datetime.utcnow()
    for _ in range(2):
        res = machine.update(make_sample(350, ts=now), make_sample(100, ts=now))
    assert res.food_empty is True
    assert res.state == "FOOD_EMPTY"
    # drop back below off threshold with debounce
    for _ in range(3):
        res = machine.update(make_sample(120, ts=now), make_sample(100, ts=now))
    assert res.food_empty is False
    assert res.state == "OK"


def test_priority_error_over_clog():
    machine = FeederStateMachine("feeder", Thresholds())
    now = dt.datetime.utcnow()
    # Mark sensors bad
    res = machine.update(make_sample(None, ok=False, ts=now), make_sample(None, ok=False, ts=now))
    res = machine.update(make_sample(None, ok=False, ts=now + dt.timedelta(seconds=6)), make_sample(None, ok=False, ts=now + dt.timedelta(seconds=6)))
    assert res.state == "ERROR"
    assert res.reason == "sensor_unavailable"


def test_cleaning_due_flag():
    cleaned = dt.datetime.utcnow() - dt.timedelta(days=20)
    machine = FeederStateMachine("feeder", Thresholds(), last_cleaned_at=cleaned, cleaning_days=14)
    now = dt.datetime.utcnow()
    res = machine.update(make_sample(100, ts=now), make_sample(100, ts=now))
    assert res.needs_cleaning is True
    assert res.state == "NEEDS_CLEANING"
