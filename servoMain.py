import time
from adafruit_servokit import ServoKit

class ServoTrap:
    def __init__(self, channel=1, closed_angle=0, open_angle=90, log_file="trap_log.txt"):
        self.kit = ServoKit(channels=16)
        self.channel = channel
        self.closed_angle = closed_angle
        self.open_angle = open_angle
        self.last_trigger_times = {}  # key = class label, value = last time triggered
        self.log_file = log_file
        self.kit.servo[self.channel].angle = self.closed_angle
        print("Trap initialized and closed.")

    def trigger(self, open_duration=2, cooldown_duration=5, label="unknown"):
        current_time = time.time()

        # Check last trigger time for this label
        last_trigger_time = self.last_trigger_times.get(label, 0)

        if current_time - last_trigger_time >= cooldown_duration:
            print(f"Opening trap for {open_duration} seconds (triggered by {label}).")
            self.kit.servo[self.channel].angle = self.open_angle
            time.sleep(open_duration)
            print("Closing trap.")
            self.kit.servo[self.channel].angle = self.closed_angle
            self.last_trigger_times[label] = current_time  # Update only THIS label's last trigger time
            self.log_event(label, open_duration, cooldown_duration)
        else:
            cooldown_remaining = cooldown_duration - (current_time - last_trigger_time)
            print(f"Cooldown active for {label} ({cooldown_remaining:.1f} seconds left). Not opening trap.")

    def log_event(self, label, open_duration, cooldown_duration):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (f"[{timestamp}] Trap triggered by '{label}'. "
                     f"Open for {open_duration}s, cooldown {cooldown_duration}s.\n")
        with open(self.log_file, "a") as f:
            f.write(log_entry)
