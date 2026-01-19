"""CS20 depth camera capture utilities.

This module wraps the DFRobot CS20 depth-only ToF camera. The hardware ships
with an SDK exposing a V4L2-compatible UVC device on Linux. On Raspberry Pi 5
it appears as ``/dev/video*`` and streams 16-bit depth. We avoid any
heavyweight dependencies so the capture defaults to ``pyusb``/``libuvc`` if
available, otherwise falls back to OpenCV ``VideoCapture`` or synthetic data for
CI.

Notes for deployment on the Pi:
- Install the vendor SDK / udev rules from https://github.com/DFRobot-official/DFRobot_CX20_driver
- Ensure ``sudo usermod -a -G video pi`` so the process can open the device.
- Prefer 320x240 mode (``mode="320x240"``) for motion or CPU bound cases.
- Shield the lens from direct sunlight with a short hood to reduce IR washout.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Tuple

import numpy as np

_DEPTH_DTYPE = np.float32


@dataclass
class DepthFrame:
    timestamp: float  # seconds since epoch
    depth: np.ndarray  # 2D depth map in meters
    valid: np.ndarray  # boolean mask same shape as depth
    point_cloud: np.ndarray | None = None


class CS20Camera:
    """Threaded CS20 capture.

    The implementation favors simplicity and resilience: if the real device is
    unavailable, synthetic depth is emitted so downstream tests can still run.
    """

    def __init__(self, mode: str = "320x240", max_queue: int = 5, fallback_to_synthetic: bool = False):
        if mode not in {"320x240", "640x480"}:
            raise ValueError("mode must be '320x240' or '640x480'")
        self.mode = mode
        self._queue: "Queue[DepthFrame]" = Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device = None  # placeholder for vendor/OpenCV handle
        self._fallback_to_synthetic = fallback_to_synthetic
        self._hardware_available = False

    def _open_device(self) -> None:
        # Lazy import to avoid import errors when SDK is absent.
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None
        if cv2 is None:
            self._device = None
            self._hardware_available = False
            return
        width, height = (320, 240) if self.mode == "320x240" else (640, 480)

        # Try to detect CS20 depth camera specifically
        # The CS20 camera usually appears on /dev/video0 or /dev/video1
        for video_idx in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(video_idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    # Try to read a test frame to verify it's actually working
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.dtype == np.uint16:
                        # Likely a depth camera with 16-bit depth data
                        self._device = cap
                        self._hardware_available = True
                        return
                    cap.release()
            except Exception:
                continue

        self._device = None
        self._hardware_available = False

    def _read_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._device is None:
            if not self._fallback_to_synthetic:
                raise RuntimeError("CS20 hardware not available and synthetic fallback disabled")
            # Synthetic plane with mild noise for CI
            width, height = (320, 240) if self.mode == "320x240" else (640, 480)
            depth = np.full((height, width), 0.4, dtype=_DEPTH_DTYPE)
            depth += np.random.normal(0, 0.005, size=depth.shape).astype(_DEPTH_DTYPE)
            valid = np.ones_like(depth, dtype=bool)
            return depth, valid
        try:
            import cv2  # type: ignore
        except Exception:
            raise RuntimeError("OpenCV unexpectedly missing during capture")
        ok, frame = self._device.read()
        if not ok:
            raise RuntimeError("Failed to read CS20 frame")
        # Depth comes as 16-bit millimeters when using vendor driver.
        if frame.dtype != np.uint16:
            # Unexpected; treat as invalid.
            depth = np.full(frame.shape[:2], np.nan, dtype=_DEPTH_DTYPE)
            valid = np.zeros_like(depth, dtype=bool)
            return depth, valid
        depth = frame.astype(_DEPTH_DTYPE) / 1000.0
        valid = np.isfinite(depth) & (depth > 0)
        return depth, valid

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = time.time()
            try:
                depth, valid = self._read_frame()
            except Exception:
                # Backoff to avoid spinning on hardware errors
                time.sleep(0.05)
                continue
            frame = DepthFrame(timestamp=ts, depth=depth, valid=valid)
            try:
                self._queue.put_nowait(frame)
            except Exception:
                # Drop oldest to keep latency low
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
                self._queue.put_nowait(frame)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._open_device()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._device is not None:
            try:
                self._device.release()
            except Exception:
                pass
        self._device = None

    def get_latest(self, timeout: float = 0.0) -> Optional[DepthFrame]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def __enter__(self) -> "CS20Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def is_hardware_available(self) -> bool:
        """Check if actual CS20 hardware was detected and is available."""
        return self._hardware_available
