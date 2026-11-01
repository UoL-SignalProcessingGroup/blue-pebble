"""Defines a sonobuoy: a fixed platform with a hydrophone suspended below it."""

from stonesoup.base import Property
from stonesoup.platform.base import FixedPlatform
from stonesoup.types.array import StateVector

from ..arrays.singlesensorarray import SingleSensorArray
from .base import _ArrayHost


class Sonobuoy(_ArrayHost, FixedPlatform):
    """A stationary buoy with a hydrophone suspended at a fixed depth.

    The buoy floats at its state's position (z normally 0, the surface); the hydrophone is a
    :class:`~bluepebble.arrays.SingleSensorArray` mounted ``hydrophone_depth_m`` below it, so it
    sits at z = -``hydrophone_depth_m`` relative to the buoy, matching Blue Pebble's convention
    of negative z below the surface. Move the buoy through each timestamp you simulate, as for any
    platform, so its hydrophone has a state at those times.

    Parameters
    ----------
    hydrophone_depth_m : float
        Depth of the hydrophone below the buoy, in metres; must be non-negative.

    Raises
    ------
    ValueError
        If ``hydrophone_depth_m`` is negative.

    """

    hydrophone_depth_m: float = Property(
        doc="Depth of the hydrophone below the buoy, in metres (positive is deeper)"
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the buoy and mount its hydrophone.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        if self.hydrophone_depth_m < 0:
            raise ValueError(
                f"hydrophone_depth_m ({self.hydrophone_depth_m}m) must be non-negative; give "
                "the depth below the buoy as a positive number."
            )
        hydrophone_offset = StateVector([0.0, 0.0, -self.hydrophone_depth_m])
        self.add_sensor(SingleSensorArray(mounting_offset=hydrophone_offset))

    @property
    def hydrophone(self) -> SingleSensorArray:
        """The buoy's hydrophone."""
        return next(array for array in self.arrays if isinstance(array, SingleSensorArray))
