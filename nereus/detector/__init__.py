"""Detector package public API."""

from .algorithms import (
    CACFARDetector,
    DetectionAlgorithm,
    OSCFARDetector,
    PeakDetector,
    ThresholdDetector,
)
from .metrics import SweepResult, SweepSpec
from .passive import PassiveSonarDetector

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
