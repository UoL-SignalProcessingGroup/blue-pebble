"""Shared contract for sensor arrays mounted on or towed behind a platform."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.functions import build_rotation_matrix
from stonesoup.movable.movable import _get_rotation_matrix
from stonesoup.sensor.base import PlatformMountable
from stonesoup.types.angle import mod_bearing
from stonesoup.types.array import StateVector, StateVectors

if TYPE_CHECKING:
    from stonesoup.movable.movable import Movable
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]

# Nominal sound speed for the grating-lobe spacing check only; propagation uses the
# environment's sound speed.
NOMINAL_SOUND_SPEED_MPS = 1500.0


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



@dataclass
class HostState:
    """A container for the platform's state.

    Parameters
    ----------
    state : State
        The state of the platform carrying the array.
    heading_rad : float
        The platform's direction of travel, in radians anticlockwise from +x, in
        ``[-pi, pi)``.
    position_mapping : tuple of int, optional
        Indices of x, y and z in ``state.state_vector``.

    """

    state: "State"
    heading_rad: float
    position_mapping: tuple[int, ...] = (0, 2, 4)


@dataclass
class PlatformState:
    """A data class to hold the state of the platform and an array at one timestamp.

    Parameters
    ----------
    timestamp : datetime
        The time at which this state is valid.
    host : HostState
        The state container for the platform carrying the array.
    array : ArrayState
        The state container for the sensor array.

    """

    timestamp: datetime
    host: HostState
    array: ArrayState

    @property
    def position(self) -> StateVector:
        """Return the position of the platform carrying the array.

        Returns
        -------
        StateVector
            The 3D position vector (x, y, z) of the platform.

        """
        return StateVector(self.host.state.state_vector[list(self.host.position_mapping)])


def _check_sensor_spacing(spacing_m: float, size_m: float, freq_max_hz: float) -> None:
    """Check that sensors do not touch and are spaced within the grating-lobe limit.

    Raises
    ------
    ValueError
        If ``spacing_m <= size_m``, or if ``spacing_m`` exceeds half a wavelength at
        ``freq_max_hz`` for a nominal 1500 m/s sound speed.

    """
    if spacing_m <= size_m:
        raise ValueError(
            f"sensor_spacing_m ({spacing_m}m) must be strictly greater than sensor_size_m "
            f"({size_m}m) so sensors do not touch."
        )
    max_spacing_m = NOMINAL_SOUND_SPEED_MPS / (2 * freq_max_hz)
    if spacing_m > max_spacing_m:
        raise ValueError(
            f"sensor_spacing_m ({spacing_m}m) exceeds the grating-lobe limit of lambda/2 "
            f"({max_spacing_m}m) at freq_max_hz ({freq_max_hz}Hz) for a nominal "
            f"{NOMINAL_SOUND_SPEED_MPS}m/s sound speed."
        )

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

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the array, storing ``mounting_offset`` as a column ``StateVector``.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        # Stone Soup's mounting maths needs a column vector; a plain list would broadcast.
        if self.mounting_offset is not None and not isinstance(self.mounting_offset, StateVector):
            self.mounting_offset = StateVector(self.mounting_offset)

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
    def _velocity_mapping(self) -> list[int] | None:
        """Return the platform's velocity mapping, or None for a fixed platform.

        Stone Soup's moving movables default it to ``position_mapping + 1``; its fixed
        movables have none.
        """
        velocity_mapping = getattr(self._controller, "velocity_mapping", None)
        return None if velocity_mapping is None else list(velocity_mapping)

    def _heading_rad(self, state: "State") -> float:
        """Return the platform's direction of travel in ``state``, in ``[-pi, pi)``.

        A fixed platform has no velocity, so its heading is the yaw of its orientation.
        """
        velocity_mapping = self._velocity_mapping
        if velocity_mapping is None:
            return float(mod_bearing(float(self._controller.orientation[2, 0])))
        velocity_xy = state.state_vector[velocity_mapping[:2]]
        # arctan2 returns +pi for due -x; mod_bearing wraps it to [-pi, pi) like Bearing.
        return float(mod_bearing(np.arctan2(velocity_xy[1, 0], velocity_xy[0, 0])))

    def _host_state(self, state: "State") -> HostState:
        """Wrap a recorded platform state with its heading and position mapping."""
        return HostState(
            state=state,
            heading_rad=self._heading_rad(state),
            position_mapping=tuple(self._controller.position_mapping),
        )

    def _pose_at(self, state: "State") -> tuple[FloatArray, FloatArray]:
        """Return the platform position and body-to-world rotation in a recorded state.

        The rotation is the one Stone Soup applies to a ``mounting_offset``
        (:meth:`~stonesoup.movable.movable.Movable._get_rotated_offset`): yaw and pitch from
        the velocity, identity when the platform is not moving. Using the same rotation keeps
        an array's reference point equal to Stone Soup's :attr:`position`.

        Returns
        -------
        tuple of FloatArray
            The platform position, shape ``(3, 1)``, and the rotation, shape ``(3, 3)``.

        """
        controller = self._controller
        position = np.asarray(state.state_vector[controller.position_mapping], dtype=float)
        velocity_mapping = self._velocity_mapping
        if velocity_mapping is None:
            return position, np.eye(3)
        velocity = state.state_vector[velocity_mapping]
        if not np.any(velocity != 0):
            return position, np.eye(3)
        return position, np.asarray(_get_rotation_matrix(StateVector(velocity)), dtype=float)

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

    def _on_mounted(self, platform: object) -> None:
        """Handle having just been mounted on ``platform``. No-op by default.

        Parameters
        ----------
        platform : object
            The platform the array was mounted on.

        """

    def _on_platform_moved(self, timestamp: datetime, **kwargs: object) -> None:
        """Handle the platform having just moved to ``timestamp``. No-op for rigid arrays.

        Parameters
        ----------
        timestamp : datetime
            The timestamp the platform was just moved to.
        **kwargs : dict
            Whatever was passed to the platform's ``move()`` call.

        """


class RigidArray(SensorArray):
    r"""An array whose sensors are fixed relative to the platform, e.g. hull-mounted.

    Subclasses describe the sensors in the array's own frame: :attr:`_local_offsets` from the
    array's reference point, and optionally :attr:`_local_normals` and :attr:`_local_axis`.
    The array frame is turned by Stone Soup's ``rotation_offset`` into the platform's body
    frame (+x forward along the direction of motion, +z up) and placed at
    ``mounting_offset``, so sensor ``i`` sits at

    .. math::

        \mathbf{p}_i = \mathbf{p}_\text{platform}
            + R_\text{platform} \left(\mathbf{m} + R_\text{offset} \mathbf{l}_i\right)

    where :math:`\mathbf{m}` is the mounting offset and :math:`\mathbf{l}_i` the local offset.

    Unlike a :class:`~bluepebble.arrays.TowedArray`, the geometry at any time follows from the
    platform's recorded state at that time, so the array keeps no history of its own.

    Assumptions
    -----------
    - The platform's rotation follows its velocity (yaw and pitch, no roll), as in Stone Soup;
      a stationary or fixed platform is not rotated.
    """

    baffle_attenuation_db: float = 0.0
    _local_offsets: FloatArray
    _local_normals: FloatArray | None = None
    _local_axis: FloatArray | None = None

    @property
    def num_sensors(self) -> int:
        """Number of sensors in the array."""
        return int(self._local_offsets.shape[1])

    def _offset_rotation(self) -> FloatArray:
        """Return the array-to-body rotation given by ``rotation_offset``."""
        rotation_offset = self.rotation_offset
        if rotation_offset is None:
            return np.eye(3)
        return np.asarray(build_rotation_matrix(StateVector(rotation_offset)), dtype=float).T

    def _array_state(self, state: "State") -> ArrayState:
        """Return the world-frame array state for one recorded platform state."""
        position, rotation = self._pose_at(state)
        offset_rotation = self._offset_rotation()
        mounting = np.asarray(
            self.mounting_offset if self.mounting_offset is not None else np.zeros((3, 1)),
            dtype=float,
        ).reshape(3, 1)
        body_offsets = mounting + offset_rotation @ self._local_offsets
        world_positions = rotation @ body_offsets + position

        normals = None
        if self._local_normals is not None:
            normals = StateVectors(rotation @ (offset_rotation @ self._local_normals))
        axis = None
        if self._local_axis is not None:
            axis = StateVector(rotation @ (offset_rotation @ self._local_axis.reshape(3, 1)))
        return ArrayState(
            num_sensors=self.num_sensors,
            state_vector=world_positions,
            ref_state_vector=StateVector(rotation @ mounting + position),
            sensor_normals=normals,
            baffle_attenuation_db=self.baffle_attenuation_db,
            axis=axis,
        )

    @property
    def array(self) -> ArrayState:
        """Return the array state at the platform's latest recorded state."""
        return self._array_state(self._controller.state)

    def get_platform_state_at(self, timestamp: datetime) -> PlatformState | None:
        """Return the platform and array state at ``timestamp``.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        PlatformState or None
            The state if the platform has been moved to ``timestamp``, otherwise ``None``.

        """
        state = self._platform_state_at(timestamp)
        if state is None:
            return None
        return PlatformState(
            timestamp=timestamp, host=self._host_state(state), array=self._array_state(state)
        )
