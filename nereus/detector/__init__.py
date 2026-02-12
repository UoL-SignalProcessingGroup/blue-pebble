"""Detector package public API."""

from .algorithms import (
    CFARDetector,
    DetectionAlgorithm,
    PeakDetector,
    ThresholdDetector,
)
from .passive import PassiveSonarDetector

__all__ = [
    "CFARDetector",
    "DetectionAlgorithm",
    "PeakDetector",
    "PassiveSonarDetector",
    "ThresholdDetector",
]
