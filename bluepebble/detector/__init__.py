"""Detector package public API."""

from .active import ActiveSonarDetectorOmni, bistatic_range, matched_filter
from .algorithms import (
    CACFARDetector,
    DetectionAlgorithm,
    DetectorChain,
    OSCFARDetector,
    PeakDetector,
    ThresholdDetector,
)
from .metrics import SweepResult, SweepSpec
from .passive import PassiveSonarDetector

__all__ = [
    "ActiveSonarDetectorOmni",
    "bistatic_range",
    "CACFARDetector",
    "DetectionAlgorithm",
    "DetectorChain",
    "matched_filter",
    "OSCFARDetector",
    "PeakDetector",
    "PassiveSonarDetector",
    "SweepResult",
    "SweepSpec",
    "ThresholdDetector",
]
