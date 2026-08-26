"""Defines the host platform that sensors can be mounted on or towed behind."""

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from stonesoup.base import Property
from stonesoup.platform.base import MultiTransitionMovingPlatform
from stonesoup.types.state import State

if TYPE_CHECKING:
    from ..sensors.base import Sensor


class HostPlatform(MultiTransitionMovingPlatform):
    """The hull -- ship or submarine -- that sensors mount on or tow behind.

    Depth is not a distinct concept from position: it is whatever z the 3D
    ``position_mapping`` and transition model produce. A surface ship is a
    ``HostPlatform`` whose z stays at (or near) 0; a submarine is one whose z is free
    to change over time, e.g. via a dive/surface transition model.

    Attached sensors (:class:`~bluepebble.sensors.base.Sensor` subclasses, e.g.
    :class:`~bluepebble.sensors.bow_array.BowArraySensor`,
    :class:`~bluepebble.sensors.towed_array.TowedArraySensor`) are advanced automatically
    whenever the host moves -- see :meth:`attach` and :meth:`move`.
    """

    attached_sensors: list["Sensor"] = Property(
        default=None,
        doc="Sensors mounted on or towed by this host. Populated via attach(), not "
        "normally passed directly to the constructor.",
    )
    velocity_mapping: Sequence[int] | None = Property(
        default=None,
        doc="Indices for velocity in the state vector, used by attached sensors that need "
        "the host's heading/velocity (e.g. to place a towed array). If not set, defaults "
        "to position_mapping indices + 1.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the HostPlatform.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        if self.attached_sensors is None:
            self.attached_sensors = []

    def resolved_velocity_mapping(self) -> Sequence[int]:
        """Return a guaranteed velocity mapping sequence.

        Falls back to ``position_mapping + 1`` when ``velocity_mapping`` is unset.
        """
        if self.velocity_mapping is None:
            return [p + 1 for p in self.position_mapping]
        return self.velocity_mapping

    @property
    def depth(self) -> float:
        """Return the current z-component of position.

        Requires a 3D ``position_mapping``. Sign convention (positive up vs. positive
        down) is whatever the caller's state vectors use.
        """
        return float(self.position[2, 0])

    def get_state_at(self, timestamp: datetime) -> State | None:
        """Return this host's own recorded state at ``timestamp``, or None if absent.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        State or None
            The matching state if the host has already been moved to that timestamp,
            otherwise ``None``.

        """
        for state in self.states:
            if state.timestamp == timestamp:
                return state
        return None

    def attach(self, sensor: "Sensor") -> "Sensor":
        """Attach a sensor to this host, so it is advanced on every :meth:`move`.

        Parameters
        ----------
        sensor : Sensor
            The sensor to attach. Its ``host`` should already reference this platform.

        Returns
        -------
        Sensor
            The same sensor, for convenient chaining.

        """
        self.attached_sensors.append(sensor)
        return sensor

    def move(self, timestamp: datetime, **kwargs: object) -> None:
        """Advance the host, then let every attached sensor react to the new state.

        Parameters
        ----------
        timestamp : datetime
            The new timestamp to move the host to.
        **kwargs : dict
            Additional arguments passed to the host's transition model.

        """
        self.movement_controller.move(timestamp, **kwargs)
        for sensor in self.attached_sensors:
            sensor._on_host_moved(timestamp, **kwargs)
