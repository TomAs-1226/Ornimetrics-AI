"""Run the feeder status service with real or mock sensors."""

from __future__ import annotations

import argparse

from src.sensors.tof_interface import AdafruitVL53L0XSensor, BaseToFSensor, make_mock_scenario
from src.status.feeder_state import Thresholds
from src.status.status_service import StatusService


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run feeder status publisher")
    ap.add_argument("--feeder_id", required=True)
    ap.add_argument("--pi", action="store_true", help="Use hardware sensors")
    ap.add_argument("--mock", action="store_true", help="Use mock scripted sensors")
    ap.add_argument("--mock_scenario", default="stable_full")
    ap.add_argument("--empty_on", type=float, default=300.0)
    ap.add_argument("--empty_off", type=float, default=200.0)
    ap.add_argument("--clog_on", type=float, default=40.0)
    ap.add_argument("--clog_off", type=float, default=80.0)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--cleaning_days", type=int, default=14)
    ap.add_argument("--cleaning_feed_events", type=int, default=200)
    ap.add_argument("--cleaning_clog_events", type=int, default=5)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def build_sensor(args: argparse.Namespace) -> BaseToFSensor:
    if args.mock:
        return make_mock_scenario(args.mock_scenario)
    if args.pi:
        return AdafruitVL53L0XSensor()
    # default to mock for PC convenience
    return make_mock_scenario(args.mock_scenario)


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        empty_on_mm=args.empty_on,
        empty_off_mm=args.empty_off,
        clog_on_mm=args.clog_on,
        clog_off_mm=args.clog_off,
    )
    food_sensor = build_sensor(args)
    clog_sensor = build_sensor(args)
    service = StatusService(
        feeder_id=args.feeder_id,
        food_sensor=food_sensor,
        clog_sensor=clog_sensor,
        thresholds=thresholds,
        poll_interval_s=args.poll,
        heartbeat_s=args.heartbeat,
        cleaning_days=args.cleaning_days,
        cleaning_feed_events=args.cleaning_feed_events,
        cleaning_clog_events=args.cleaning_clog_events,
        debug=args.debug,
    )
    if args.debug:
        print("Starting status service with thresholds", thresholds)
    service.loop()


if __name__ == "__main__":
    main()

