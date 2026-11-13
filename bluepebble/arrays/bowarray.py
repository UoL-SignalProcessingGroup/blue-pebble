"""Defines a bow array: a curved, forward-facing dome of sensors on a platform's bow."""

import numpy as np
from stonesoup.base import Property

from .base import FloatArray, RigidArray, _check_sensor_spacing


def _dome_sensor_offsets(
    dome_radius_m: float,
    azimuth_extent_rad: float,
    elevation_extent_rad: float,
    sensor_spacing_m: float,
) -> FloatArray:
    """Compute sensor positions on a spherical-patch dome, relative to the dome centre.

    The dome is a patch of a sphere of radius ``dome_radius_m`` centred on the array's
    boresight (local +x). Rows are laid out in elevation and, within each row, sensors in
    azimuth, both spaced no closer than ``sensor_spacing_m`` of arc along the sphere.

    Parameters
    ----------
    dome_radius_m : float
        Radius of curvature of the dome, in metres.
    azimuth_extent_rad : float
        Angular width of the patch in azimuth, in radians.
    elevation_extent_rad : float
        Angular width of the patch in elevation, in radians.
    sensor_spacing_m : float
        Minimum centre-to-centre spacing between adjacent sensors, in metres.

    Returns
    -------
    FloatArray
        Sensor offsets with shape ``(3, num_sensors)`` in the array frame (+x forward, +z up),
        ordered by row (lowest first), then by azimuth.

    """
    num_rows = int(np.floor(elevation_extent_rad * dome_radius_m / sensor_spacing_m)) + 1
    elevation_angles = np.linspace(-elevation_extent_rad / 2, elevation_extent_rad / 2, num_rows)

    offsets = []
    for phi in elevation_angles:
        row_span_m = azimuth_extent_rad * dome_radius_m * np.cos(phi)
        sensors_in_row = int(np.floor(row_span_m / sensor_spacing_m)) + 1
        azimuth_angles = np.linspace(
            -azimuth_extent_rad / 2, azimuth_extent_rad / 2, sensors_in_row
        )

        x = dome_radius_m * np.cos(phi) * np.cos(azimuth_angles)
        y = dome_radius_m * np.cos(phi) * np.sin(azimuth_angles)
        z = np.full_like(azimuth_angles, dome_radius_m * np.sin(phi))
        offsets.append(np.vstack([x, y, z]))

    return np.hstack(offsets)


class BowArray(RigidArray):
    """A curved, forward-facing dome of sensors mounted rigidly on a platform's bow.

    Mount it with the platform's ``add_sensor``. Its ``mounting_offset`` is the dome centre
    relative to the platform's reference point, e.g. ``[0, 0, -3]`` for a dome 3 m below the
    bow, so propagation is computed at the dome's depth. The hull behind the dome baffles
    every sensor, so all of them face forward, and the array axis points along the heading.

    Parameters
    ----------
    dome_radius_m : float
        Radius of curvature of the dome, in metres.
    azimuth_extent_rad : float
        Angular width of the forward-facing patch in azimuth, in radians.
    elevation_extent_rad : float
        Angular width of the forward-facing patch in elevation, in radians.
    sensor_spacing_m : float
        Centre-to-centre spacing between adjacent sensors, in metres.
    sensor_size_m : float
        Physical diameter of one sensor, in metres; must be less than ``sensor_spacing_m``.
    freq_max_hz : float
        Highest frequency the array is used at, in Hz, for the grating-lobe spacing check.
    baffle_attenuation_db : float, optional
        Attenuation of sound arriving from behind the dome face (aft of the bow), in dB.

    Raises
    ------
    ValueError
        If sensors would touch, or are spaced beyond half a wavelength at ``freq_max_hz``.

    """

    dome_radius_m: float = Property(doc="Radius of curvature of the bow dome, in metres")
    azimuth_extent_rad: float = Property(
        doc="Angular width of the forward-facing dome patch in azimuth, in radians"
    )
    elevation_extent_rad: float = Property(
        doc="Angular width of the forward-facing dome patch in elevation, in radians"
    )
    sensor_spacing_m: float = Property(
        doc="Centre-to-centre spacing between adjacent sensors, in metres"
    )
    sensor_size_m: float = Property(doc="Physical diameter of one sensor, in metres")
    freq_max_hz: float = Property(
        doc="Highest frequency the array is used at, in Hz, for the grating-lobe check"
    )
    baffle_attenuation_db: float = Property(
        default=40.0,
        doc="Attenuation of sound arriving from behind the dome face (aft of the bow), in dB. "
        "Real hull baffles typically give 15-30 dB, less at low frequency.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the BowArray and lay out its sensors.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        _check_sensor_spacing(self.sensor_spacing_m, self.sensor_size_m, self.freq_max_hz)
        self._local_offsets = _dome_sensor_offsets(
            self.dome_radius_m,
            self.azimuth_extent_rad,
            self.elevation_extent_rad,
            self.sensor_spacing_m,
        )
        forward = np.array([[1.0], [0.0], [0.0]])
        self._local_normals = np.tile(forward, (1, self.num_sensors))
        self._local_axis = forward
