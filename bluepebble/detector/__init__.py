"""Detector package public API."""

from .algorithms import (
    CACFARDetector,
    DetectionAlgorithm,
    OSCFARDetector,
)
from .fluctuation_models import (
    FluctuationModel,
    NonFluctuating,
    RayleighFluctuation,
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
    "FluctuationModel",
    "MultibandPassiveSonarDetector",
    "NonFluctuating",
    "OSCFARDetector",
    "PassiveSonarDetector",
    "RayleighFluctuation",
    "SweepResult",
    "SweepSpec",
    "snr_from_beamformed_data",
    "sweep_detection_parameter",
    "sweep_detection_parameter_multiband",
]
