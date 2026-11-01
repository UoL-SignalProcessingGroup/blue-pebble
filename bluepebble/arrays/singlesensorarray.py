"""Defines a single-sensor array: one sensor, such as a sonobuoy hydrophone."""

import numpy as np

from .base import RigidArray


class SingleSensorArray(RigidArray):
    """An array of one sensor, at the array's mounting point.

    Mount it with the platform's ``add_sensor``; ``mounting_offset`` is the sensor's position
    relative to the platform's reference point. A single point has no axis, and the sensor has
    no baffle, so it receives and transmits equally in every direction. A
    :class:`~bluepebble.platform.Sonobuoy` builds and mounts one as its hydrophone.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the array with its one sensor at the mounting point.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        self._local_offsets = np.zeros((3, 1))
