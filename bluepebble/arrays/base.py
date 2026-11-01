"""Shared contract for sensor arrays mounted on or towed behind a platform."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.sensor.base import PlatformMountable
from stonesoup.types.array import StateVector, StateVectors

if TYPE_CHECKING:
    from stonesoup.movable.movable import Movable
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class ArrayState:
    """A container for a sensor array's world-frame state at one instant.

    Parameters
    ----------
    num_sensors : int
        The total number of sensors in the array.
    state_vector : FloatArray
        The sensor positions stacked column-wise, shape ``(3, num_sensors)``.
    ref_state_vector : StateVector
        The world-frame position of the reference sensor.
    sensor_normals : StateVectors or None, optional
        World-frame unit vectors, one per sensor, pointing out of the face each sensor is
        mounted on (e.g. outward from the hull for a flank panel). ``None`` means the sensors
        are omnidirectional and unbaffled.
    baffle_attenuation_db : float, optional
        Attenuation, in dB, of sound arriving from or sent towards the back of a sensor's
        mounting face (the side its normal points away from). Applies on receive and on
        transmit. Only used when ``sensor_normals`` is set.
    axis : StateVector or None, optional
        World-frame unit vector of the array's own reference axis, which steering fans and
        bearings off the array are measured from: forward along the hull for hull-mounted
        arrays, forward along the line (aft sensor -> forward sensor) for a towed array.
        ``None`` for arrays with no natural axis, such as a single hydrophone.

    """

    num_sensors: int
    state_vector: FloatArray
    ref_state_vector: StateVector
    sensor_normals: StateVectors | None = None
    baffle_attenuation_db: float = 0.0
    axis: "StateVector | None" = None

    @property
    def axis_azimuth_rad(self) -> float:
        """World-frame azimuth of :attr:`axis`, in radians (anticlockwise from +x).

        Raises
        ------
        ValueError
            If the array has no ``axis``.

        """
        if self.axis is None:
            raise ValueError("This array has no axis to take an azimuth from.")
        axis = np.asarray(self.axis, dtype=float).ravel()
        return float(np.arctan2(axis[1], axis[0]))

    def sensor_gains(self, direction: ArrayLike) -> NDArray[np.float64]:
        """Return each sensor's baffle amplitude gain towards a world-frame direction.

        A sensor whose mounting face points away from ``direction`` (normal . direction < 0)
        gets ``10 ** (-baffle_attenuation_db / 20)``; every other sensor gets 1. The same gain
        applies whether the sensor is receiving sound from that direction or transmitting
        towards it.

        Parameters
        ----------
        direction : ArrayLike
            World-frame vector from the array towards the target, shape ``(3,)``. Only its
            direction matters.

        Returns
        -------
        NDArray[np.float64]
            Amplitude gains, shape ``(num_sensors,)``. All ones when the array has no
            ``sensor_normals`` (omnidirectional, unbaffled sensors).

        """
        gains = np.ones(self.num_sensors)
        if self.sensor_normals is None:
            return gains
        facing = np.asarray(self.sensor_normals).T @ np.asarray(direction, dtype=float).ravel()
        gains[facing < 0] = 10.0 ** (-self.baffle_attenuation_db / 20.0)
        return gains

    def transmit_gain(self, direction: ArrayLike) -> float:
        """Return the array's transmit amplitude gain towards a world-frame direction.

        The ping is modelled as coming from the array reference point, with every sensor
        driven equally, so the level sent towards ``direction`` is the mean of the sensor
        gains: 1 when every sensor faces that way, the baffle level when none does (e.g. a bow
        dome pinging a target astern), and in between for an array whose sensors face several
        ways (e.g. a two-sided flank array).

        Parameters
        ----------
        direction : ArrayLike
            World-frame vector from the array towards the target, shape ``(3,)``.

        Returns
        -------
        float
            Transmit amplitude gain, in ``(0, 1]``.

        """
        return float(self.sensor_gains(direction).mean())


class SensorArray(PlatformMountable):
    """An array of sensors mounted on or towed behind a platform.

    Arrays use Stone Soup's mounting: mount one with the platform's ``add_sensor``, and it
    shares the platform's :attr:`movement_controller` and takes its ``mounting_offset`` and
    ``rotation_offset`` relative to the platform's reference point and direction of motion. The
    sensors are the array's elements, described by :class:`ArrayState`.

    Subclasses implement :attr:`array` and :meth:`get_platform_state_at`. Arrays whose sensors
    have their own dynamics (e.g. a towed array) also override :meth:`_on_mounted` and
    :meth:`_on_platform_moved`, which a :class:`~bluepebble.platform.SonarPlatform` calls.
    """

    @property
    def array(self) -> ArrayState:
        """Return the current world-frame array state. Must be overridden by subclasses."""
        raise NotImplementedError

    def get_platform_state_at(self, timestamp: datetime) -> object | None:
        """Return this array's state at ``timestamp``. Must be overridden by subclasses."""
        raise NotImplementedError

    @property
    def _controller(self) -> "Movable":
        """Return the platform's movement controller, raising if the array is not mounted."""
        controller = self.movement_controller
        if controller is None:
            raise ValueError(
                f"{type(self).__name__} is not mounted; mount it with platform.add_sensor() "
                "first."
            )
        return controller

    @property
    def _velocity_mapping(self) -> list[int]:
        """Return the platform's velocity mapping (Stone Soup fills in position + 1 if unset)."""
        controller = self._controller
        velocity_mapping = getattr(controller, "velocity_mapping", None)
        if velocity_mapping is None:
            return [p + 1 for p in controller.position_mapping]
        return list(velocity_mapping)

    def _platform_state_at(self, timestamp: datetime) -> "State | None":
        """Return the platform's recorded state at ``timestamp``, or None if absent."""
        # Called straight after the platform moves, so check the latest state first.
        states = self._controller.states
        if states and states[-1].timestamp == timestamp:
            return states[-1]
        for state in states:
            if state.timestamp == timestamp:
                return state
        return None

    def _on_mounted(self) -> None:
        """Handle having just been mounted on a platform. No-op for rigid arrays."""

    def _on_platform_moved(self, timestamp: datetime, **kwargs: object) -> None:
        """Handle the platform having just moved to ``timestamp``. No-op for rigid arrays.

        Parameters
        ----------
        timestamp : datetime
            The timestamp the platform was just moved to.
        **kwargs : dict
            Whatever was passed to the platform's ``move()`` call.

        """
