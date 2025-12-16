"""Feeder status state machine with hysteresis, debounce, and cleaning reminders."""

from __future__ import annotations

import datetime as dt
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class Thresholds:
    empty_on_mm: float = 300.0
    empty_off_mm: float = 200.0
    clog_on_mm: float = 40.0
    clog_off_mm: float = 80.0


@dataclass
class SensorSample:
    value_mm: Optional[float]
    ok: bool
    timestamp: dt.datetime


@dataclass
class StateResult:
    feeder_id: str
    state: str
    food_empty: bool
    clogged: bool
    needs_cleaning: bool
    tof_food_mm: Optional[float]
    tof_clog_mm: Optional[float]
    thresholds: Thresholds
    confidence: float
    reason: str
    updated_at: dt.datetime
    revision: int
    sensor_health: Dict[str, Dict[str, Optional[dt.datetime]]]
    last_cleaned_at: Optional[dt.datetime]


class FeederStateMachine:
    """State machine that fuses ToF sensors into feeder status.

    Priority: ERROR > CLOGGED > FOOD_EMPTY > NEEDS_CLEANING > OK. The
    NEEDS_CLEANING flag is also emitted when a higher-priority state is active
    so the app can display reminders without losing critical alerts.
    """

    def __init__(
        self,
        feeder_id: str,
        thresholds: Thresholds,
        median_window: int = 5,
        debounce_count: int = 3,
        poll_interval_s: float = 0.5,
        error_grace_s: float = 5.0,
        cleaning_days: int = 14,
        cleaning_feed_events: int = 200,
        cleaning_clog_events: int = 5,
        initial_revision: int = 0,
        last_cleaned_at: Optional[dt.datetime] = None,
    ) -> None:
        self.feeder_id = feeder_id
        self.thresholds = thresholds
        self.median_window = median_window
        self.debounce_count = debounce_count
        self.poll_interval_s = poll_interval_s
        self.error_grace = dt.timedelta(seconds=error_grace_s)
        self.cleaning_days = cleaning_days
        self.cleaning_feed_events = cleaning_feed_events
        self.cleaning_clog_events = cleaning_clog_events
        self.last_cleaned_at = last_cleaned_at
        self.revision = initial_revision

        self.food_samples: Deque[SensorSample] = deque(maxlen=median_window)
        self.clog_samples: Deque[SensorSample] = deque(maxlen=median_window)
        self.food_on_streak = 0
        self.food_off_streak = 0
        self.clog_on_streak = 0
        self.clog_off_streak = 0
        self.food_empty = False
        self.clogged = False
        self.needs_cleaning_flag = False
        self.feed_event_count = 0
        self.clog_event_count = 0

    def mark_cleaned(self, when: Optional[dt.datetime] = None) -> None:
        self.last_cleaned_at = when or dt.datetime.utcnow()
        self.feed_event_count = 0
        self.clog_event_count = 0

    def register_feed_event(self) -> None:
        self.feed_event_count += 1

    def _median_or_none(self, samples: Deque[SensorSample]) -> Optional[float]:
        vals = [s.value_mm for s in samples if s.ok and s.value_mm is not None]
        if not vals:
            return None
        return statistics.median(vals)

    def _sensor_health(self, samples: Deque[SensorSample]) -> Dict[str, Optional[dt.datetime]]:
        last_ok = next((s.timestamp for s in reversed(samples) if s.ok), None)
        return {"ok": bool(last_ok), "last_read_ok_at": last_ok}

    def update(
        self,
        food_sample: SensorSample,
        clog_sample: SensorSample,
        now: Optional[dt.datetime] = None,
    ) -> StateResult:
        now = now or dt.datetime.utcnow()
        self.food_samples.append(food_sample)
        self.clog_samples.append(clog_sample)

        food_median = self._median_or_none(self.food_samples)
        clog_median = self._median_or_none(self.clog_samples)

        if food_median is not None:
            if food_median > self.thresholds.empty_on_mm:
                self.food_on_streak += 1
                self.food_off_streak = 0
            elif food_median < self.thresholds.empty_off_mm:
                self.food_off_streak += 1
                self.food_on_streak = 0
            if self.food_on_streak >= self.debounce_count:
                if not self.food_empty:
                    self.revision += 1
                self.food_empty = True
                self.food_on_streak = 0
            if self.food_off_streak >= self.debounce_count:
                if self.food_empty:
                    self.revision += 1
                self.food_empty = False
                self.food_off_streak = 0

        if clog_median is not None:
            if clog_median < self.thresholds.clog_on_mm:
                self.clog_on_streak += 1
                self.clog_off_streak = 0
            elif clog_median > self.thresholds.clog_off_mm:
                self.clog_off_streak += 1
                self.clog_on_streak = 0
            if self.clog_on_streak >= self.debounce_count:
                if not self.clogged:
                    self.revision += 1
                    self.clog_event_count += 1
                self.clogged = True
                self.clog_on_streak = 0
            if self.clog_off_streak >= self.debounce_count:
                if self.clogged:
                    self.revision += 1
                self.clogged = False
                self.clog_off_streak = 0

        self._update_cleaning_flag(now)

        sensor_health = {
            "food_sensor": self._sensor_health(self.food_samples),
            "clog_sensor": self._sensor_health(self.clog_samples),
        }
        both_bad = not sensor_health["food_sensor"]["ok"] and not sensor_health["clog_sensor"]["ok"]
        last_ok = [v["last_read_ok_at"] for v in sensor_health.values() if v["last_read_ok_at"]]
        oldest_ok = min(last_ok) if last_ok else None
        error = False
        reason = ""
        if both_bad:
            if oldest_ok is None or (now - oldest_ok) > self.error_grace:
                error = True
                reason = "sensor_unavailable"

        state = "OK"
        if error:
            state = "ERROR"
        elif self.clogged:
            state = "CLOGGED"
            reason = "clog_detected"
        elif self.food_empty:
            state = "FOOD_EMPTY"
            reason = "food_below_threshold"
        elif self.needs_cleaning_flag:
            state = "NEEDS_CLEANING"
            reason = "cleaning_due"
        else:
            reason = "normal"

        confidence = 0.0
        for samples in (self.food_samples, self.clog_samples):
            vals = [s for s in samples if s.ok]
            if vals:
                confidence += min(1.0, len(vals) / self.median_window)
        confidence = min(1.0, confidence / 2.0)

        result = StateResult(
            feeder_id=self.feeder_id,
            state=state,
            food_empty=self.food_empty,
            clogged=self.clogged,
            needs_cleaning=self.needs_cleaning_flag,
            tof_food_mm=food_median,
            tof_clog_mm=clog_median,
            thresholds=self.thresholds,
            confidence=confidence,
            reason=reason,
            updated_at=now,
            revision=self.revision,
            sensor_health=sensor_health,
            last_cleaned_at=self.last_cleaned_at,
        )
        return result

    def _update_cleaning_flag(self, now: dt.datetime) -> None:
        due_time = False
        if self.last_cleaned_at:
            days_since = (now - self.last_cleaned_at).days
            due_time = days_since >= self.cleaning_days
        due_events = (
            self.feed_event_count >= self.cleaning_feed_events
            or self.clog_event_count >= self.cleaning_clog_events
        )
        new_flag = due_time or due_events
        if new_flag != self.needs_cleaning_flag:
            self.revision += 1
        self.needs_cleaning_flag = new_flag

