"""Feeder status service that polls sensors and publishes live Firebase status."""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict
from typing import Optional

from src.status.feeder_state import FeederStateMachine, SensorSample, Thresholds
from src.status.firebase_status_store import append_history_event, read_last_status, write_current_status


class StatusService:
    def __init__(
        self,
        feeder_id: str,
        food_sensor,
        clog_sensor,
        thresholds: Thresholds,
        poll_interval_s: float = 0.5,
        heartbeat_s: float = 30.0,
        cleaning_days: int = 14,
        cleaning_feed_events: int = 200,
        cleaning_clog_events: int = 5,
        debug: bool = False,
    ) -> None:
        last_status = read_last_status(feeder_id) or {}
        last_cleaned_at = None
        if "last_cleaned_at" in last_status:
            try:
                last_cleaned_at = dt.datetime.fromisoformat(last_status["last_cleaned_at"])
            except Exception:
                last_cleaned_at = None

        self.state_machine = FeederStateMachine(
            feeder_id=feeder_id,
            thresholds=thresholds,
            poll_interval_s=poll_interval_s,
            cleaning_days=cleaning_days,
            cleaning_feed_events=cleaning_feed_events,
            cleaning_clog_events=cleaning_clog_events,
            initial_revision=int(last_status.get("revision", 0)),
            last_cleaned_at=last_cleaned_at,
        )
        self.feeder_id = feeder_id
        self.food_sensor = food_sensor
        self.clog_sensor = clog_sensor
        self.poll_interval_s = poll_interval_s
        self.heartbeat_s = heartbeat_s
        self.debug = debug
        self._last_publish = dt.datetime.utcnow()
        self._last_revision = self.state_machine.revision

    def _sample_sensor(self, sensor) -> SensorSample:
        ts = dt.datetime.utcnow()
        value = None
        ok = True
        try:
            value = sensor.read_mm()
            ok = value is not None
        except Exception:
            ok = False
        return SensorSample(value_mm=value, ok=ok, timestamp=ts)

    def _status_to_payload(self, result) -> dict:
        payload = {
            "feeder_id": result.feeder_id,
            "state": result.state,
            "food_empty": result.food_empty,
            "clogged": result.clogged,
            "needs_cleaning": result.needs_cleaning,
            "tof_food_mm": result.tof_food_mm,
            "tof_clog_mm": result.tof_clog_mm,
            "thresholds": asdict(result.thresholds),
            "confidence": result.confidence,
            "reason": result.reason,
            "updated_at": result.updated_at.isoformat(),
            "revision": result.revision,
            "sensor_health": {
                k: {"ok": v["ok"], "last_read_ok_at": v["last_read_ok_at"].isoformat() if v["last_read_ok_at"] else None}
                for k, v in result.sensor_health.items()
            },
            "last_cleaned_at": result.last_cleaned_at.isoformat() if result.last_cleaned_at else None,
        }
        return payload

    def publish(self, payload: dict) -> None:
        write_current_status(self.feeder_id, payload)
        append_history_event(self.feeder_id, payload)
        if self.debug:
            print(json.dumps(payload, indent=2))

    def run_once(self, now: Optional[dt.datetime] = None) -> dict:
        result = self.state_machine.update(self._sample_sensor(self.food_sensor), self._sample_sensor(self.clog_sensor), now=now)
        payload = self._status_to_payload(result)
        self.publish(payload)
        self._last_publish = result.updated_at
        self._last_revision = result.revision
        return payload

    def loop(self) -> None:
        while True:
            result = self.state_machine.update(self._sample_sensor(self.food_sensor), self._sample_sensor(self.clog_sensor))
            payload = self._status_to_payload(result)
            should_publish = False
            if (dt.datetime.utcnow() - self._last_publish).total_seconds() >= self.heartbeat_s:
                should_publish = True
            if result.revision != self._last_revision:
                should_publish = True
            if should_publish:
                self.publish(payload)
                self._last_publish = dt.datetime.utcnow()
                self._last_revision = result.revision
            time.sleep(self.poll_interval_s)

