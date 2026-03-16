"""Signal package public API."""

from .random import RandomSignal, ColouredNoise, WhiteNoise
from .anthropogenic import (
    Anthropogenic,
    AnthropogenicSignal,
    Recorded,
    RecordedSignal,
    Synthetic,
    SyntheticSignal,
)
from .base import Signal
from .biological import Biological
from .utils import apply_fade_in, apply_fade_out, compute_stft, inverse_stft

__all__ = [
    # New canonical names
    "Signal",
    "Biological",
    "RandomSignal",
    "WhiteNoise",
    "ColouredNoise",
    "Anthropogenic",
    "Synthetic",
    "Recorded",
    # Deprecated aliases — remove after next release cycle
    "AnthropogenicSignal",
    "SyntheticSignal",
    "RecordedSignal",
    # Utilities
    "compute_stft",
    "inverse_stft",
    "apply_fade_in",
    "apply_fade_out",
]
