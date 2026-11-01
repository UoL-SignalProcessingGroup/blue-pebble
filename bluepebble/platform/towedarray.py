"""Deprecated combined platform and towed array.

Mount a :class:`~bluepebble.arrays.TowedArray` on a :class:`~bluepebble.platform.SonarPlatform`
instead.
"""

import warnings
from datetime import datetime

from stonesoup.base import Property
from stonesoup.movable.movable import MovingMovable
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import State

from ..arrays.base import ArrayState, FloatArray
from ..arrays.towedarray import HostState, PlatformState, TowedArray
from .base import SonarPlatform

__all__ = ["ArrayState", "HostState", "PlatformState", "TowedArrayPlatform"]


class TowedArrayPlatform(SonarPlatform):
    """A sonar platform with a towed array built in.

    .. deprecated:: 0.6.0
        Create a :class:`~bluepebble.platform.SonarPlatform` and mount a
        :class:`~bluepebble.arrays.TowedArray` on it with ``add_sensor``.
        ``TowedArrayPlatform`` will be removed in a later release.

    This is a thin wrapper: it mounts a ``TowedArray`` with the tow point at the platform's
    reference point and forwards the array queries to it, so it behaves as it did before the
    platform and array were split.

    Parameters
    ----------
    num_sensors : int
        Number of sensors in the array.
    cable_length_m : float
        Length of the main tow cable in metres (distance from platform to first sensor).
    sensor_spacing_m : float
        Spacing between subsequent sensors in metres.
    array_depth_m : float
        Depth at which the array is towed in metres.
    reference_sensor_idx : int, optional
        Index of the reference sensor.

    """

    num_sensors: int = Property(doc="Number of sensors in the array")
    cable_length_m: float = Property(doc="Length of the main tow cable in metres")
    sensor_spacing_m: float = Property(doc="Spacing between sensors in metres")
    array_depth_m: float = Property(doc="Depth at which the array is towed in metres")
    reference_sensor_idx: int = Property(default=0, doc="Index of the reference sensor")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the platform and its towed array.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        warnings.warn(
            "TowedArrayPlatform is deprecated; create a bluepebble.platform.SonarPlatform and "
            "mount a bluepebble.arrays.TowedArray on it with add_sensor() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        self._towed_array = TowedArray(
            num_sensors=self.num_sensors,
            cable_length_m=self.cable_length_m,
            sensor_spacing_m=self.sensor_spacing_m,
            array_depth_m=self.array_depth_m,
            reference_sensor_idx=self.reference_sensor_idx,
        )
        self.add_sensor(self._towed_array)

    @property
    def towed_array(self) -> TowedArray:
        """The towed array this platform builds and mounts."""
        return self._towed_array

    @property
    def towed_sensors(self) -> list[MovingMovable]:
        """The simulated sensor elements trailing the platform."""
        return self._towed_array.towed_sensors

    @property
    def platform_history(self) -> list[PlatformState]:
        """The platform's composite state at each timestamp."""
        return self._towed_array.platform_history

    @property
    def array(self) -> ArrayState:
        """Return the array state at the platform's latest timestamp."""
        return self._towed_array.array

    def get_platform_state_at(self, timestamp: datetime) -> PlatformState | None:
        """Get the platform state at a specific timestamp; see `TowedArray`."""
        return self._towed_array.get_platform_state_at(timestamp)

    def get_host_state_at(self, timestamp: datetime) -> State | None:
        """Get the platform's own state at a specific timestamp; see `TowedArray`."""
        return self._towed_array.get_host_state_at(timestamp)

    def get_sensor_states_at(self, timestamp: datetime) -> list[GroundTruthState] | None:
        """Get the states of all towed sensors at a specific timestamp; see `TowedArray`."""
        return self._towed_array.get_sensor_states_at(timestamp)

    @property
    def host_path(self) -> FloatArray | None:
        """Get the complete path of the platform; see `TowedArray`."""
        return self._towed_array.host_path

    @property
    def sensor_paths(self) -> list[FloatArray]:
        """Get sensor position paths as a list of arrays; see `TowedArray`."""
        return self._towed_array.sensor_paths

    def __repr__(self) -> str:
        """Return string representation of the platform."""
        return (
            f"TowedArrayPlatform(num_sensors={self.num_sensors}, "
            f"cable_length_m={self.cable_length_m}, "
            f"sensor_spacing_m={self.sensor_spacing_m}, "
            f"array_depth_m={self.array_depth_m})"
        )
