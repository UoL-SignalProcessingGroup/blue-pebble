"""Detector package public API."""

from .algorithms import (
    CACFARDetector,
    DetectionAlgorithm,
    OSCFARDetector,
    PeakDetector,
    ThresholdDetector,
    run_detection_chain,
)
from .metrics import (
    SweepResult,
    SweepSpec,
    sweep_detection_parameter,
    sweep_detection_parameter_multiband,
)
from .passive import (
    BandDetector,
    MultibandPassiveSonarDetector,
    PassiveSonarDetector,
    snr_from_beamformed_data,
)

__all__ = [
    "BandDetector",
    "CACFARDetector",
    "DetectionAlgorithm",
    "MultibandPassiveSonarDetector",
    "OSCFARDetector",
    "PeakDetector",
    "PassiveSonarDetector",
    "SweepResult",
    "SweepSpec",
    "ThresholdDetector",
    "run_detection_chain",
    "snr_from_beamformed_data",
    "sweep_detection_parameter",
    "sweep_detection_parameter_multiband",
]
