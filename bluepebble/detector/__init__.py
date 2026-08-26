"""Detector package public API."""

from .active import ActiveSonarDetectorOmni, matched_filter
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
    "ActiveSonarDetectorOmni",
    "CACFARDetector",
    "DetectionAlgorithm",
    "matched_filter",
    "OSCFARDetector",
    "PeakDetector",
    "PassiveSonarDetector",
    "SweepResult",
    "SweepSpec",
    "ThresholdDetector",
]
