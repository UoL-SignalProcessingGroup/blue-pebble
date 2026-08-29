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


# Names removed by the single-detector refactor, mapped to what replaces them. Raising
# ImportError (not AttributeError) here is deliberate: `from bluepebble.detector import X`
# discards an AttributeError's message and reports its own generic one, so the guidance
# below would never reach the caller.
_REMOVED = {
    "PeakDetector": (
        "Peak consolidation is built into OSCFARDetector -- configure it with "
        "peak_distance / peak_prominence instead of chaining a separate detector."
    ),
    "ThresholdDetector": (
        "A fixed threshold is a CFAR detector with a fixed alpha. Use CACFARDetector or "
        "OSCFARDetector with target_pfa, which calibrates the threshold for you."
    ),
    "run_detection_chain": (
        "Detectors are no longer chained. Call detector.detect(beamformed_data) directly "
        "on raw (num_beams, num_frames) data."
    ),
}


def __getattr__(name: str):
    """Point callers of the removed chain API at its replacement."""
    if name in _REMOVED:
        raise ImportError(
            f"{name} was removed from bluepebble.detector. {_REMOVED[name]} "
            f"See the 'replace detection chains with self-calibrating CFAR detectors' "
            f"commit for the full migration."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
