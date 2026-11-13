"""Target scattering (reflectivity) models for active sonar simulation."""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.float64]


class TargetScatteringModel(ABC, Base):
    """Abstract base class for target scattering (reflectivity) models.

    Maps frequency to a linear (not dB) scattering gain, applied once per round trip to the
    echo between the incident and scattered propagation legs.
    """

    @abstractmethod
    def scatter(self, frequencies_hz: ArrayLike) -> ComplexArray | FloatArray:
        """Return the linear scattering gain at each requested frequency.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the scattering response.

        Returns
        -------
        ComplexArray or FloatArray
            Linear (not dB) scattering gain, shape matching ``frequencies_hz``.

        """
        ...


class ConstantTargetStrength(TargetScatteringModel):
    r"""Frequency- and aspect-independent target strength.

    Models the target as a point scatterer with one target strength :math:`TS`, giving the
    same linear pressure gain at every frequency:

    .. math::

        g = \sqrt{10^{TS/10}} = 10^{TS/20}

    Parameters
    ----------
    target_strength_db : float
        Target strength in dB re 1 m (ratio of re-radiated to incident intensity, referenced
        at 1 m from the target).

    Assumptions
    -----------
    - The target scatters equally in every direction and at every frequency, so aspect,
      shape and resonances are ignored.

    """

    target_strength_db: float = Property(doc="Target strength in dB re 1 m")

    def scatter(self, frequencies_hz: ArrayLike) -> FloatArray:
        """Return the same linear target-strength gain at every requested frequency."""
        frequencies = np.asarray(frequencies_hz, dtype=float)
        gain = float(np.sqrt(10.0 ** (self.target_strength_db / 10.0)))
        return np.full(frequencies.shape, gain, dtype=np.float64)
