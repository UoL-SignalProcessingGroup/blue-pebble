"""Shared contract for sensors mounted on or towed behind a HostPlatform."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

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

    """

    num_sensors: int
    state_vector: StateVectors
    ref_state_vector: StateVector


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
