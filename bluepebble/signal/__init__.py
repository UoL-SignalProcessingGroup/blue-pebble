"""Signal package public API."""

from .active import ActiveSignal, CWSignal, LFMSignal
from .anthropogenic import (
    AnthropogenicSignal,
    RecordedAnthropogenicSignal,
    SyntheticAnthropogenicSignal,
)
from .base import Signal, SourceSignal
from .biological import (
    BiologicalSignal,
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
    "BiologicalSignal",
    "SnappingShrimpColonySignal",
    "snap_rate_from_temperature",
    "RandomSignal",
    "WhiteNoiseSignal",
    "ColouredNoiseSignal",
    "AnthropogenicSignal",
    "SyntheticAnthropogenicSignal",
    "RecordedAnthropogenicSignal",
    "analytic_signal",
    "compute_stft",
    "inverse_stft",
    "apply_fade_in",
    "apply_fade_out",
]
