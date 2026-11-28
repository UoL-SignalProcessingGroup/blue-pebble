"""Signal package public API."""

from .active import ActiveSignal, CWSignal, LFMSignal
from .anthropogenic import (
    AnthropogenicSignal,
    ContinuousWaveSignal,
    RecordedAnthropogenicSignal,
    SpectrumLevelSignal,
    SyntheticAnthropogenicSignal,
)
from .base import Signal, SourceSignal
from .biological import (
    AlphaStableAmplitudes,
    AmplitudeDistribution,
    BiologicalSignal,
    LognormalAmplitudes,
    SnappingShrimpColonySignal,
    snap_rate_from_temperature,
)
from .random import ColouredNoiseSignal, RandomSignal, WhiteNoiseSignal
from .utils import analytic_signal, apply_fade_in, apply_fade_out, compute_stft, inverse_stft

__all__ = [
    "Signal",
    "SourceSignal",
    "ActiveSignal",
    "CWSignal",
    "LFMSignal",
    "AmplitudeDistribution",
    "AlphaStableAmplitudes",
    "LognormalAmplitudes",
    "BiologicalSignal",
    "SnappingShrimpColonySignal",
    "snap_rate_from_temperature",
    "RandomSignal",
    "WhiteNoiseSignal",
    "ColouredNoiseSignal",
    "AnthropogenicSignal",
    "SyntheticAnthropogenicSignal",
    "RecordedAnthropogenicSignal",
    "SpectrumLevelSignal",
    "ContinuousWaveSignal",
    "analytic_signal",
    "compute_stft",
    "inverse_stft",
    "apply_fade_in",
    "apply_fade_out",
]
