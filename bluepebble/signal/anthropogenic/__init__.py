"""Public anthropogenic signal API."""

from .base import BroadbandStftSignalBase, NarrowbandSignalBase, NarrowbandStatefulSignalBase
from .continuous import BroadbandRecordedSignal, BroadbandSyntheticSignal
from .discrete import (
    NarrowbandBlendedTonalSignal,
    NarrowbandOverlapAddTonalSignal,
    NarrowbandTonalSignal,
)

__all__ = [
    "NarrowbandSignalBase",
    "NarrowbandStatefulSignalBase",
    "BroadbandStftSignalBase",
    "NarrowbandTonalSignal",
    "NarrowbandBlendedTonalSignal",
    "NarrowbandOverlapAddTonalSignal",
    "BroadbandSyntheticSignal",
    "BroadbandRecordedSignal",
]
