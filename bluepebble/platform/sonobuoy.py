"""Defines a sonobuoy platform for active sonar simulation."""

from stonesoup.base import Property
from stonesoup.platform import FixedPlatform
from stonesoup.types.array import StateVector, StateVectors

from ..sensors.base import ArrayState


class OmniSonobuoyPlatform(FixedPlatform):
    """A stationary omnidirectional sonobuoy platform for monostatic active sonar simulation.

    A sonobuoy floats at the ocean surface with a hydrophone suspended at a fixed depth.
    It acts as both transmitter and receiver (monostatic) and transmits/receives equally
    in all directions (omnidirectional).

    The platform's horizontal position (x, y) is defined by the initial state vector.
    The hydrophone depth is a fixed property separate from the state vector.

    Parameters
    ----------
    hydrophone_depth_m : float
        Depth of the hydrophone below the surface in metres.
    states : list[State]
        Initial state(s) of the platform. State vector should contain [x, y] with
        ``position_mapping=[0, 1]``.

    """

    hydrophone_depth_m: float = Property(
        doc="Depth of the hydrophone below the surface in metres"
    )

    @property
    def position_3d(self) -> StateVector:
        """Return the hydrophone position as ``[x, y, hydrophone_depth_m]``."""
        xy = self.position
        return StateVector([float(xy[0, 0]), float(xy[1, 0]), self.hydrophone_depth_m])

    @property
    def array(self) -> ArrayState:
        """Return a single-element array state at the hydrophone position.

        Satisfies the ``platform.array`` interface (``state_vector`` /
        ``ref_state_vector``) that :class:`~bluepebble.models.propagation.AcousticPropagationModel`
        methods expect, so ``OmniSonobuoyPlatform`` can be used directly as a receiver with the
        shared propagation models (e.g. ``rtrsAcousticPropagationModel.propagate_spectrum``).
        """
        position = self.position_3d
        return ArrayState(
            num_sensors=1,
            state_vector=StateVectors(position),
            ref_state_vector=position,
        )
