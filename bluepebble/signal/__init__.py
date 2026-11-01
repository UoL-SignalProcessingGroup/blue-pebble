"""Signal package public API."""

from .active import ActiveSignal, CWSignal, LFMSignal
from .anthropogenic import (
    AnthropogenicSignal,
    RecordedAnthropogenicSignal,
    SyntheticAnthropogenicSignal,
)
from .base import Signal
from .biological import BiologicalSignal
from .random import ColouredNoiseSignal, RandomSignal, WhiteNoiseSignal
from .utils import apply_fade_in, apply_fade_out, compute_stft, inverse_stft

__all__ = [
    "Signal",
    "ActiveSignal",
    "CWSignal",
    "LFMSignal",
    "BiologicalSignal",
    "RandomSignal",
    "WhiteNoiseSignal",
    "ColouredNoiseSignal",
    "AnthropogenicSignal",
    "SyntheticAnthropogenicSignal",
    "RecordedAnthropogenicSignal",
    "compute_stft",
    "inverse_stft",
    "apply_fade_in",
    "apply_fade_out",
]
