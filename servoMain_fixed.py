# servoMain_fixed.py
# Per-species cooldowns + jitter-safe writes + monotonic timers
# NEW: non-blocking ramp (slew rate) so moves happen smoothly under load
#      configurable pulse widths (min_us/max_us) and slew_deg_per_s

import time
from time import monotonic as now_mono
from adafruit_servokit import ServoKit

def _norm_label(s):
    s = (s or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    return s

class ServoTrap:
    def __init__(self, channel=0, closed_angle=5.0, open_angle=60.0,
                 min_inter_trigger_sec=1.0, pwm_freq=50, min_us=1000, max_us=2000,
                 i2c_addr=0x40, debug=True, slew_deg_per_s=240.0):
        # PCA9685 setup
        self.kit = ServoKit(channels=16, address=i2c_addr, frequency=pwm_freq)
        self.servo = self.kit.servo[int(channel)]
        try:
            self.servo.set_pulse_width_range(int(min_us), int(max_us))
        except Exception:
            pass

        # Config
        self.channel = int(channel)
        self.closed = float(closed_angle)
        self.open = float(open_angle)
        self.min_inter_trigger = float(min_inter_trigger_sec)
        self.debug = bool(debug)
        self.slew = float(slew_deg_per_s) if slew_deg_per_s is not None else 0.0

        # Anti-jitter
        self.deadband_deg = 2.5
        self.min_cmd_interval = 0.06 if self.slew > 0 else 0.12  # allow slightly faster updates during ramp

        # State
        self.last_angle = None
        self.last_write_ts = 0.0
        self.target_angle = None
        self._last_tick_t = now_mono()

        # Latching
        self.hold_until = 0.0
        self.busy_until = 0.0
        self.cooldown_until_per = {}  # per-species cooldown
        self._first_trigger_done = False

        # Start closed
        self._request_angle(self.closed, reason="init-closed")

    # ---- helpers ----
    def _should_write(self, angle: float):
        now = now_mono()
        if self.last_angle is None or abs(angle - self.last_angle) >= self.deadband_deg:
            return True
        if (now - self.last_write_ts) < self.min_cmd_interval:
            return False
        return True

    def _write_angle(self, angle: float, reason: str = "") -> bool:
        angle = float(angle)
        if not self._should_write(angle):
            if self.debug:
                print(f"[Servo] skip write (deadband/rate) angle={angle:.1f} last={self.last_angle} reason={reason}")
            return False
        try:
            self.servo.angle = angle
            self.last_angle = angle
            self.last_write_ts = now_mono()
            if self.debug:
                print(f"[Servo] angle->{angle:.1f} ch={self.channel} reason={reason}")
            return True
        except Exception as e:
            if self.debug:
                print(f"[Servo] WRITE ERROR: {e}")
            return False

    def _request_angle(self, angle: float, reason: str = ""):
        """Non-blocking: set a target and let tick() ramp to it. If slew==0, write immediately."""
        self.target_angle = float(angle)
        if self.slew <= 0:
            self._write_angle(self.target_angle, reason=reason)

    # ---- diagnostics ----
    def can_trigger(self, label=None, force=False):
        """Return (ok, reason, species_cd_left, global_busy_left)."""
        now = now_mono()
        lab = _norm_label(label)
        if not self._first_trigger_done:
            return True, "first_trigger", 0.0, 0.0
        gb = max(0.0, self.busy_until - now)
        if not force and gb > 0:
            return False, "global_busy", 0.0, gb
        sp = max(0.0, self.cooldown_until_per.get(lab, 0.0) - now)
        if not force and sp > 0:
            return False, "species_cooldown", sp, 0.0
        return True, "ok", 0.0, 0.0

    def get_state(self):
        now = now_mono()
        cd_map = {k: max(0.0, v - now) for k, v in self.cooldown_until_per.items()}
        return {
            "last_angle": self.last_angle,
            "target_angle": self.target_angle,
            "hold_left": max(0.0, self.hold_until - now),
            "global_busy_left": max(0.0, self.busy_until - now),
            "species_cd_left": cd_map
        }

    # ---- API ----
    def trigger(self, open_duration=1.5, cooldown_duration=8.0,
                label=None, confidence=None, min_conf=0.0, force=False) -> bool:
        now = now_mono()
        lab = _norm_label(label)

        if not self._first_trigger_done:
            force = True

        ok, reason, sp_left, gb_left = self.can_trigger(label=lab, force=force)
        if not ok:
            if self.debug:
                if reason == "global_busy":
                    print(f"[Servo] trigger BLOCKED: global_busy {gb_left:.2f}s lab={lab}")
                elif reason == "species_cooldown":
                    print(f"[Servo] trigger BLOCKED: species_cd {sp_left:.2f}s lab={lab}")
                else:
                    print(f"[Servo] trigger BLOCKED: {reason} lab={lab}")
            return False

        # Command open (non-blocking ramp)
        self._request_angle(self.open, reason=f"trigger-open lab={lab} conf={confidence}")
        self.hold_until = now + float(open_duration)
        self.busy_until = self.hold_until + self.min_inter_trigger
        self.cooldown_until_per[lab] = now + float(cooldown_duration)
        self._first_trigger_done = True

        if self.debug:
            print(f"[Servo] TRIGGERED lab={lab} open_dur={open_duration:.2f}s "
                  f"species_cooldown={(self.cooldown_until_per[lab]-now):.2f}s "
                  f"global_quiet={(self.busy_until-now):.2f}s")
        return True

    def tick(self):
        """Advance ramp and handle auto-close when hold elapses."""
        t = now_mono()

        # Auto-close when hold window ends
        if self.hold_until and t >= self.hold_until:
            self._request_angle(self.closed, reason="auto-close")
            self.hold_until = 0.0

        # Ramp towards target if needed
        if self.target_angle is not None and self.last_angle is not None and self.slew > 0:
            dt = max(0.0, t - self._last_tick_t)
            max_step = self.slew * dt
            delta = self.target_angle - self.last_angle
            if abs(delta) <= max_step:
                self._write_angle(self.target_angle, reason="ramp-final")
                self.target_angle = None
            else:
                step = max_step if delta > 0 else -max_step
                self._write_angle(self.last_angle + step, reason="ramp")
        elif self.target_angle is not None and self.last_angle is None and self.slew > 0:
            # First write when we didn't have a last angle yet
            self._write_angle(self.target_angle, reason="ramp-init")
            self.target_angle = None

        self._last_tick_t = t

    # Manual helpers
    def open_now(self):  return self._request_angle(self.open, reason="manual-open")
    def close_now(self): return self._request_angle(self.closed, reason="manual-close")
