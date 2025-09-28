#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_smoketest_ch1.py
Adafruit PCA9685 (Servo HAT) smoketest for a single servo (default channel 1).

- Sets frequency to 50 Hz
- Sets pulse width range (default 1000–2000 us)
- Moves to closed -> open -> closed for N cycles
- Optional sweep across angles
- Prints clear diagnostics; safe to Ctrl+C

Run:
  python3 servo_smoketest_ch1.py
  # or override options:
  python3 servo_smoketest_ch1.py --channel 1 --freq 50 --min-us 1000 --max-us 2000 --closed 5 --open 60 --cycles 5 --hold 1.2 --pause 0.8 --sweep
"""

import time
import argparse
import sys

try:
    from adafruit_servokit import ServoKit
except Exception as e:
    print("ERROR: adafruit-circuitpython-servokit not installed:", e, file=sys.stderr)
    print("Try: pip install adafruit-circuitpython-servokit")
    sys.exit(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=1, help="Servo channel (0-15). Default: 1")
    ap.add_argument("--freq", type=int, default=50, help="PWM frequency Hz. Default: 50")
    ap.add_argument("--min-us", type=int, default=1000, help="Min pulse width in microseconds")
    ap.add_argument("--max-us", type=int, default=2000, help="Max pulse width in microseconds")
    ap.add_argument("--closed", type=float, default=5.0, help="Closed angle")
    ap.add_argument("--open", type=float, default=60.0, help="Open angle")
    ap.add_argument("--cycles", type=int, default=3, help="Open/close cycles")
    ap.add_argument("--hold", type=float, default=1.0, help="Hold time at open (s)")
    ap.add_argument("--pause", type=float, default=0.8, help="Pause between moves (s)")
    ap.add_argument("--sweep", action="store_true", help="Sweep 0..180..0 at the end")
    ap.add_argument("--i2c-addr", type=lambda x:int(x,0), default=0x40, help="PCA9685 I2C address (hex like 0x40)")
    args = ap.parse_args()

    print("=== Servo HAT Smoketest ===")
    print(f"I2C addr: 0x{args.i2c_addr:02X}  |  freq: {args.freq} Hz  |  channel: {args.channel}")
    print(f"Pulse range: {args.min_us}-{args.max_us} us  |  closed: {args.closed} deg  |  open: {args.open} deg")
    print(f"Cycles: {args.cycles}  |  hold: {args.hold}s  |  pause: {args.pause}s")
    print("Safety: ensure external 5–6 V on V+ screw terminal, and Pi GND ↔ HAT GND tied.\n")

    try:
        kit = ServoKit(channels=16, address=args.i2c_addr, frequency=args.freq)
    except Exception as e:
        print("ERROR: Could not init ServoKit (I2C?).", e, file=sys.stderr)
        print("Hints:")
        print("  - Enable I2C (raspi-config) and reboot")
        print("  - Check wiring; run: sudo i2cdetect -y 1  (should see 0x40)")
        sys.exit(2)

    try:
        s = kit.servo[args.channel]
    except Exception as e:
        print(f"ERROR: Invalid channel {args.channel} (0-15).", e, file=sys.stderr)
        sys.exit(2)

    try:
        s.set_pulse_width_range(args.min_us, args.max_us)
    except Exception as e:
        print("WARNING: set_pulse_width_range failed, continuing:", e)

    def set_angle(a, note=""):
        try:
            s.angle = float(a)
            print(f"[MOVE] angle -> {a:.1f}  {note}")
        except Exception as e:
            print("ERROR: write angle failed:", e)

    try:
        # Center-ish to start (avoid end-stops)
        set_angle(args.closed, "(closed start)")
        time.sleep(max(0.6, args.pause))

        for i in range(1, args.cycles + 1):
            print(f"\n-- Cycle {i}/{args.cycles} --")
            set_angle(args.open, "(open)")
            time.sleep(max(0.2, args.hold))
            set_angle(args.closed, "(closed)")
            time.sleep(max(0.4, args.pause))

        if args.sweep:
            print("\n-- Sweep 0..180..0 --")
            for a in range(0, 181, 15):
                set_angle(a)
                time.sleep(0.15)
            for a in range(180, -1, -15):
                set_angle(a)
                time.sleep(0.15)

        print("\nSmoketest complete.")
        print("If movements were weak/jittery: check external 5–6 V on V+, add 470–1000 uF across V+ and GND, and keep signal wiring short.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            # Optionally release the servo (comment out if you prefer to hold position)
            # s.angle = None
            pass
        except Exception:
            pass

if __name__ == "__main__":
    main()
