"""Signal package public API."""

from .ambient import Ambient, AmbientNoise, ColouredNoise, WhiteNoise
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
    "Ambient",
    "WhiteNoise",
    "ColouredNoise",
    "Anthropogenic",
    "Synthetic",
    "Recorded",
    # Deprecated aliases — remove after next release cycle
    "AmbientNoise",
    "AnthropogenicSignal",
    "SyntheticSignal",
    "RecordedSignal",
    # Utilities
    "compute_stft",
    "inverse_stft",
    "apply_fade_in",
    "apply_fade_out",
]
