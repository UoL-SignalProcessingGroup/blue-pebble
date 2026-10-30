"""Shared contract for sensors mounted on or towed behind a HostPlatform."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property
from stonesoup.types.array import StateVector, StateVectors

if TYPE_CHECKING:
    from ..platform.base import HostPlatform


@dataclass
class ArrayState:
    """A container for a sensor array's world-frame state at one instant.

    Parameters
    ----------
    num_sensors : int
        The total number of elements in the array.
    state_vector : StateVectors
        The combined world-frame position of all elements.
    ref_state_vector : StateVector
        The world-frame position of the reference element.
    element_normals : StateVectors or None, optional
        World-frame unit vectors, one per element, pointing out of the face each element
        is mounted on (e.g. outward from the hull for a flank panel). ``None`` (the
        default) means the elements are omnidirectional and unbaffled.
    baffle_attenuation_db : float, optional
        Attenuation, in dB, of sound arriving from or sent towards the back of an element's
        mounting face (the side its normal points away from). Applies on receive and on
        transmit. Only used when ``element_normals`` is set. Defaults to ``0.0`` (no
        baffling).
    axis : StateVector or None, optional
        World-frame unit vector of the array's own reference axis, which steering fans and
        bearings off the array are measured from: forward along the hull for hull-mounted
        arrays, forward along the line (aft element -> forward element) for a towed array.
        ``None`` (the default) for arrays with no natural axis, such as a single
        hydrophone.

    """

    num_sensors: int
    state_vector: StateVectors
    ref_state_vector: StateVector
    element_normals: StateVectors | None = None
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

    def element_gains(self, direction: ArrayLike) -> NDArray[np.float64]:
        """Return each element's baffle amplitude gain towards a world-frame direction.

        An element whose mounting face points away from ``direction`` (normal . direction
        < 0) gets ``10 ** (-baffle_attenuation_db / 20)``; every other element gets 1. The
        same gain applies whether the element is receiving sound from that direction or
        transmitting towards it.

        Parameters
        ----------
        direction : ArrayLike
            World-frame vector from the array towards the target, shape ``(3,)``. Only its
            direction matters.

        Returns
        -------
        NDArray[np.float64]
            Amplitude gains, shape ``(num_sensors,)``. All ones when the array has no
            ``element_normals`` (omnidirectional, unbaffled elements).

        """
        gains = np.ones(self.num_sensors)
        if self.element_normals is None:
            return gains
        facing = np.asarray(self.element_normals).T @ np.asarray(direction, dtype=float).ravel()
        gains[facing < 0] = 10.0 ** (-self.baffle_attenuation_db / 20.0)
        return gains

    def transmit_gain(self, direction: ArrayLike) -> float:
        """Return the array's transmit amplitude gain towards a world-frame direction.

        The ping is modelled as coming from the array reference point, with every element
        driven equally, so the level sent towards ``direction`` is the mean of the element
        gains: 1 when every element faces that way, the baffle level when none does (e.g.
        a bow dome pinging a target astern), and in between for an array whose elements
        face several ways (e.g. a two-sided flank array).

        Parameters
        ----------
        direction : ArrayLike
            World-frame vector from the array towards the target, shape ``(3,)``.

        Returns
        -------
        float
            Transmit amplitude gain, in ``(0, 1]``.

        """
        return float(self.element_gains(direction).mean())


class Sensor(Base):
    """Something mounted on or towed behind a :class:`~bluepebble.platform.HostPlatform`.

    Concrete sensors (:class:`~bluepebble.sensors.bow_array.BowArraySensor`,
    :class:`~bluepebble.sensors.towed_array.TowedArraySensor`) derive their geometry from
    ``host``'s position/velocity/history rather than tracking platform motion themselves.

    ``move()``/``state`` are provided as delegates to ``host`` so that code written against
    the old combined platform classes (``self.platform.move(timestamp)``,
    ``self.platform.state.timestamp``) keeps working unchanged against a sensor object.
    """

    host: "HostPlatform" = Property(doc="The platform this sensor is mounted on or towed by")

    @property
    def array(self) -> ArrayState:
        """Return the current world-frame array state. Must be overridden by subclasses."""
        raise NotImplementedError

    def get_platform_state_at(self, timestamp: datetime) -> object | None:
        """Return this sensor's state at ``timestamp``. Must be overridden by subclasses."""
        raise NotImplementedError

    def move(self, timestamp: datetime, **kwargs: object) -> None:
        """Advance the host (and, via its cascade, every sensor attached to it).

        Convenience delegate so callers holding only a sensor reference can drive the
        scenario without reaching into ``sensor.host`` directly.

        Parameters
        ----------
        timestamp : datetime
            The new timestamp to move the host to.
        **kwargs : dict
            Additional arguments passed to the host's transition model.

        """
        self.host.move(timestamp, **kwargs)

    @property
    def state(self):
        """Delegate to the host's current (most recently recorded) state."""
        return self.host.state

    def _on_host_moved(self, timestamp: datetime, **kwargs: object) -> None:
        """Handle the host having just moved to ``timestamp``.

        No-op by default -- a rigid sensor (e.g. :class:`~.BowArraySensor`) needs nothing
        beyond the host's new position/velocity, which it reads live. Sensors with their own
        dynamics (e.g. :class:`~.TowedArraySensor`'s follower chain) override this.

        Do not call this directly; it is invoked by :meth:`HostPlatform.move`. Calling
        :meth:`move` on the sensor instead re-enters ``host.move()`` correctly without
        recursing back into this hook a second time.

        Parameters
        ----------
        timestamp : datetime
            The timestamp the host was just moved to.
        **kwargs : dict
            Whatever was passed to the host's ``move()`` call.

        """
