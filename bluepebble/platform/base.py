"""Defines the sonar platform that sensor arrays are mounted on or towed behind."""

from datetime import datetime
from typing import TYPE_CHECKING

from stonesoup.platform.base import MultiTransitionMovingPlatform

from ..arrays.base import SensorArray

if TYPE_CHECKING:
    from stonesoup.sensor.base import PlatformMountable


class SonarPlatform(MultiTransitionMovingPlatform):
    """The hull, ship or submarine, that sonar arrays mount on or tow behind.

    Arrays are mounted with Stone Soup's :meth:`add_sensor` (or the ``sensors`` argument), take
    their position and orientation from the platform's movement controller, and are listed by
    :attr:`arrays`. The one thing this adds to Stone Soup's
    :class:`~stonesoup.platform.base.MultiTransitionMovingPlatform` is that :meth:`move` also
    advances every mounted array, which a :class:`~bluepebble.arrays.TowedArray` needs because
    its sensors have their own dynamics rather than a fixed mounting offset. Rigidly mounted
    arrays need nothing from it and work on any Stone Soup platform.

    Depth is not a distinct concept from position: it is whatever z the 3D
    ``position_mapping`` and transition model produce. A surface ship is a
    ``SonarPlatform`` whose z stays at (or near) 0; a submarine is one whose z is free
    to change over time, e.g. via a dive/surface transition model.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the platform and any arrays passed in ``sensors``.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        for array in self.arrays:
            array._on_mounted()

    @property
    def arrays(self) -> tuple[SensorArray, ...]:
        """The sensor arrays mounted on this platform, in mounting order."""
        return tuple(sensor for sensor in self.sensors if isinstance(sensor, SensorArray))

    @property
    def depth(self) -> float:
        """Return the current z-component of position.

        Requires a 3D ``position_mapping``. Sign convention (positive up vs. positive
        down) is whatever the caller's state vectors use.
        """
        return float(self.position[2, 0])

    def add_sensor(self, sensor: "PlatformMountable") -> None:
        """Mount a sensor or sensor array on this platform.

        Parameters
        ----------
        sensor : PlatformMountable
            The Stone Soup sensor or Blue Pebble array to mount.

        """
        # Stone Soup annotates add_sensor with Sensor, but it only needs a PlatformMountable.
        super().add_sensor(sensor)  # pyrefly: ignore[bad-argument-type]
        if isinstance(sensor, SensorArray):
            sensor._on_mounted()

    def move(self, timestamp: datetime, **kwargs) -> None:
        """Advance the platform, then let every mounted array react to the new state.

        Parameters
        ----------
        timestamp : datetime
            The new timestamp to move the platform to.
        **kwargs : dict
            Additional arguments passed to the platform's transition model.

        """
        self.movement_controller.move(timestamp, **kwargs)
        for array in self.arrays:
            array._on_platform_moved(timestamp, **kwargs)
