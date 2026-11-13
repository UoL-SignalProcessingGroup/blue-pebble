"""Signal processing package public API."""

from .adaptive import MinimumVarianceDistortionlessResponseBeamformer
from .base import Beamformer, FrequencyBand, MirrorPlan
from .conventional import DelayAndSumBeamformer
from .matchedfilter import MatchedFilter
from .resolution import beams_per_mainlobe, cfar_window_for_mainlobe
from .steering import FULL_CIRCLE, PORT, STARBOARD, SteeringCalculator
from .utils import parabolic_peak_offset

__all__ = [
    "FULL_CIRCLE",
    "PORT",
    "STARBOARD",
    "Beamformer",
    "DelayAndSumBeamformer",
    "FrequencyBand",
    "MatchedFilter",
    "MinimumVarianceDistortionlessResponseBeamformer",
    "MirrorPlan",
    "SteeringCalculator",
    "beams_per_mainlobe",
    "cfar_window_for_mainlobe",
    "parabolic_peak_offset",
]
