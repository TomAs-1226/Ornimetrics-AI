"""Interfaces for ToF distance sensors used by the feeder status service.

The implementation keeps hardware-specific imports optional so the same code
can run on a PC in mock mode or on a Raspberry Pi with CircuitPython
drivers installed. The primary abstraction is :class:`BaseToFSensor` which
exposes a ``read_mm`` method returning a distance in millimetres or ``None``
when a reading fails.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol


class BaseToFSensor(Protocol):
    """Minimal protocol for distance sensors."""

    def read_mm(self) -> Optional[float]:
        """Return the current distance in millimetres or ``None`` on failure."""


@dataclass
class MockToFSensor:
    """A deterministic mock sensor for PC simulations.

    The sensor cycles through provided scripted distances (in mm). Optional
    Gaussian noise can be added to emulate measurement jitter.
    """

    script: Iterable[float]
    noise_std: float = 0.0
    drop_every: int = 0
    _values: list[float] = field(init=False)
    _idx: int = field(default=0, init=False)
    _read_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._values = list(self.script)
        if not self._values:
            self._values = [0.0]

    def read_mm(self) -> Optional[float]:
        self._read_count += 1
        if self.drop_every and self._read_count % self.drop_every == 0:
            return None
        value = self._values[self._idx % len(self._values)]
        self._idx += 1
        if self.noise_std > 0:
            value = random.gauss(value, self.noise_std)
        return value


class AdafruitVL53L0XSensor:
    """Adafruit VL53L0X ToF sensor wrapper.

    Imports are performed lazily inside ``__init__`` to avoid hard
    dependencies when running in environments without hardware.
    """

    def __init__(self, i2c_factory: Optional[Callable[[], object]] = None, address: int = 0x29):
        if i2c_factory is None:
            try:
                import board  # type: ignore
                import busio  # type: ignore

                def i2c_factory() -> object:  # type: ignore
                    return busio.I2C(board.SCL, board.SDA)

            except Exception as exc:  # pragma: no cover - requires hardware
                raise RuntimeError("I2C bus not available") from exc

        try:
            import adafruit_vl53l0x  # type: ignore
        except Exception as exc:  # pragma: no cover - requires hardware
            raise RuntimeError("adafruit_vl53l0x library missing") from exc

        i2c = i2c_factory()
        self._sensor = adafruit_vl53l0x.VL53L0X(i2c, address=address)

    def read_mm(self) -> Optional[float]:  # pragma: no cover - hardware dependent
        try:
            return float(self._sensor.range)
        except Exception:
            return None


def make_mock_scenario(name: str) -> MockToFSensor:
    """Factory for common mock scenarios used in tests and PC demos."""

    scenarios: dict[str, list[float]] = {
        "stable_full": [50.0] * 20,
        "empty_then_full": [400.0] * 5 + [60.0] * 20,
        "clog_then_clear": [20.0] * 5 + [120.0] * 10,
    }
    if name not in scenarios:
        raise ValueError(f"Unknown mock scenario: {name}")
    return MockToFSensor(script=scenarios[name], noise_std=2.0 if "stable" not in name else 0.5)

