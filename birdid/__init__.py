"""BirdID: depth-assisted individual bird re-identification.

This package is designed for Raspberry Pi 5 deployments that pair an RGB camera
running YOLO with a USB dToF depth camera (DFRobot CS20). It focuses on
low-latency, robust identification using depth cues and lightweight embedding
backends.
"""

__all__ = [
    "engine",
    "db",
    "segment",
    "embedding_baseline",
    "embedding_tflite",
    "sync",
]
