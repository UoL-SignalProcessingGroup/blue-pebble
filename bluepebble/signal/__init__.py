"""Signal package public API."""

from .ambient import AmbientNoise, ColouredNoise, WhiteNoise
from .anthropogenic import AnthropogenicSignal, RecordedSignal, SyntheticSignal
from .base import Signal
from .biological import (
    CallEvent,
    DiffuseSnappingShrimpSignal,
    PointSourceSnappingShrimpSignal,
    WhaleCallSignal,
)
from .effects import Effect, Reverb
from .utils import apply_fade_in, apply_fade_out, compute_stft, inverse_stft

__all__ = [
    "Signal",
    "AmbientNoise",
    "WhiteNoise",
    "ColouredNoise",
    "AnthropogenicSignal",
    "SyntheticSignal",
    "RecordedSignal",
    "CallEvent",
    "PointSourceSnappingShrimpSignal",
    "DiffuseSnappingShrimpSignal",
    "WhaleCallSignal",
    "Effect",
    "Reverb",
    "compute_stft",
    "inverse_stft",
    "apply_fade_in",
    "apply_fade_out",
]
