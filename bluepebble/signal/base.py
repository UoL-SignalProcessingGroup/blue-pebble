"""Base signal properties and methods for signal models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Base, Property

if TYPE_CHECKING:
    from stonesoup.types.state import State

ComplexArray: TypeAlias = NDArray[np.complex128]


def _get_source_metadata(source: "State") -> Mapping[str, object]:
    """Validate and return the metadata mapping from a source state.

    Parameters
    ----------
    source : State
        Source state exposing a ``metadata`` attribute.

    Returns
    -------
    Mapping[str, object]
        The validated metadata mapping.

    Raises
    ------
    ValueError
        If ``metadata`` is absent or not mapping-like.

    """
    metadata = getattr(source, "metadata", None)
    if not isinstance(metadata, Mapping):
        msg = "Source state metadata must be mapping-like"
        raise ValueError(msg)
    return metadata


class Signal(Base):
    """Shared sampling parameter contract for all signal and noise models.

    ``Signal`` is the public unified root of the signal hierarchy.  All
    concrete signal types — :class:`~bluepebble.signal.biological.Biological`,
    :class:`~bluepebble.signal.anthropogenic.Anthropogenic`, and
    :class:`~bluepebble.signal.random.RandomSignal` — inherit from it as siblings.

    This class carries only the shared ``duration_s`` / ``sampling_rate_hz``
    parameter contract and the derived ``num_samples`` property.  Each branch
    defines its own generation interface independently.

    Parameters
    ----------
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    duration_s: float = Property(doc="Duration of the signal in seconds")
    sampling_rate_hz: int = Property(doc="Sampling rate in Hertz")

    @property
    def num_samples(self) -> int:
        """Calculate the number of samples based on duration and sampling rate.

        Returns
        -------
        int
            The number of samples in the signal snapshot.

        """
        return int(self.duration_s * self.sampling_rate_hz)


# ---------------------------------------------------------------------------
# Private alias — preserved so that the per-timestep body can be written on
# Biological without re-importing the root under a different name.
# ---------------------------------------------------------------------------
_SignalBase = Signal
