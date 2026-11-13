"""Platform models public API."""

from ..arrays.base import PlatformState
from .base import SonarPlatform
from .sonobuoy import Sonobuoy

__all__ = ["PlatformState", "SonarPlatform", "Sonobuoy"]

# Removed names, mapped to their replacements. ImportError rather than AttributeError, because
# `from ... import X` discards an AttributeError's message.
_REMOVED = {
    "TowedArrayPlatform": (
        "Create a SonarPlatform and mount a bluepebble.arrays.TowedArray on it with "
        "add_sensor(); the tow point is the array's mounting_offset."
    ),
}


def __getattr__(name: str):
    """Point callers of a removed name at its replacement."""
    if name in _REMOVED:
        raise ImportError(
            f"{name} was removed from bluepebble.platform in v0.8.0. {_REMOVED[name]}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
