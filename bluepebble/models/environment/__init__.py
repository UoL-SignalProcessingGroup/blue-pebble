"""Environment models public API."""

from .bathymetry import (
    Bathymetry,
    FlatBathymetry,
    GEBCOBathymetry,
    SeamountBathymetry,
    WedgeBathymetry,
)
from .sound_speed_profile import (
    NPL,
    UNESCO,
    Arctan,
    Constant,
    CopernicusSoundSpeedProfile,
    Coppens,
    DelGrosso,
    EmpiricalSoundSpeedProfile,
    Linear,
    Mackenzie,
    Munk,
    SoundSpeedProfile,
    SoundSpeedRangeWarning,
)

__all__ = [
    "NPL",
    "UNESCO",
    "Arctan",
    "Bathymetry",
    "Constant",
    "CopernicusSoundSpeedProfile",
    "Coppens",
    "DelGrosso",
    "EmpiricalSoundSpeedProfile",
    "FlatBathymetry",
    "GEBCOBathymetry",
    "Linear",
    "Mackenzie",
    "Munk",
    "SeamountBathymetry",
    "SoundSpeedProfile",
    "SoundSpeedRangeWarning",
    "WedgeBathymetry",
]

# Removed names, mapped to their replacements. ImportError rather than AttributeError, because
# `from ... import X` discards an AttributeError's message.
_REMOVED = {
    "LeroyCopernicusSoundSpeedProfile": (
        "Use CopernicusSoundSpeedProfile(equation_cls=NPL), naming its arguments; NPL is the "
        "equation of Leroy, Robinson and Goldsmith (2008) the old name referred to."
    ),
}


def __getattr__(name: str):
    """Point callers of a removed name at its replacement."""
    if name in _REMOVED:
        raise ImportError(
            f"{name} was removed from bluepebble.models.environment in v0.6.0. {_REMOVED[name]}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
