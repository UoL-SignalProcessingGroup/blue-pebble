"""Detector package public API."""

from .algorithms import (
    CACFARDetector,
    DetectionAlgorithm,
    OSCFARDetector,
    PeakDetector,
    ThresholdDetector,
)
from .passive import PassiveSonarDetector
from .metrics import SweepResult, SweepSpec, sweep_detection_parameter

__all__ = [
    "CACFARDetector",
    "DetectionAlgorithm",
    "OSCFARDetector",
    "PeakDetector",
    "PassiveSonarDetector",
    "SweepResult",
    "SweepSpec",
    "ThresholdDetector",
]
