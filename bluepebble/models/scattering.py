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

    Maps frequency to a linear (not dB) scattering gain applied once per round trip to the
    combined propagation transfer function during two-hop Fourier-synthesis active sonar
    simulation (see :class:`~bluepebble.simulator.active_rtrs.RtrsActiveSonarSimulatorOmni`).
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
    """Frequency- and aspect-independent target strength.

    Models the target as an isotropic point scatterer with a single scalar target strength,
    matching the convention used by the Bellhop-eigenray active sonar path
    (:class:`~bluepebble.simulator.active_bellhop.BellhopActiveSonarSimulatorOmni`): a constant
    scalar gain
    ``sqrt(10**(target_strength_db / 10))`` applied once per round trip.

    Parameters
    ----------
    target_strength_db : float
        Target strength in dB (ratio of re-radiated to incident intensity, referenced at 1 m).

    """

    target_strength_db: float = Property(doc="Target strength in dB re 1 m")

    def scatter(self, frequencies_hz: ArrayLike) -> FloatArray:
        """Return the same linear target-strength gain at every requested frequency."""
        frequencies = np.asarray(frequencies_hz, dtype=float)
        gain = float(np.sqrt(10.0 ** (self.target_strength_db / 10.0)))
        return np.full(frequencies.shape, gain, dtype=np.float64)
