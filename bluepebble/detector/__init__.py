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
    ca_cfar_roc,
    estimate_effective_looks_per_frame,
    os_cfar_roc,
    snr_linear_from_ground_truth_bearing,
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
    "ca_cfar_roc",
    "estimate_effective_looks_per_frame",
    "os_cfar_roc",
    "snr_from_beamformed_data",
    "snr_linear_from_ground_truth_bearing",
    "sweep_detection_parameter",
    "sweep_detection_parameter_multiband",
]
