"""Detector package public API."""

from .algorithms import (
    CACFARDetector,
    DetectionAlgorithm,
    OSCFARDetector,
    PeakDetector,
    ThresholdDetector,
)
from .passive import PassiveSonarDetector

__all__ = [
    "CACFARDetector",
    "DetectionAlgorithm",
    "OSCFARDetector",
    "PeakDetector",
    "PassiveSonarDetector",
    "ThresholdDetector",
]
