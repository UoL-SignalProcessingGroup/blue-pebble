"""Defines a bow array: sensors on a platform's bow, on a spherical dome or a cylinder."""

import warnings
from typing import Literal

import numpy as np
from stonesoup.base import Property

from .base import FloatArray, RigidArray, _check_sensor_spacing

# Positional order of BowArray's arguments up to 0.7.0. Stone Soup moves properties with a
# default after those without, so giving ``elevation_extent_rad`` a default moved it past
# ``freq_max_hz``. Calls with three or more positional arguments are still read in this order,
# with a DeprecationWarning; remove no earlier than the release after the one that first warns.
_LEGACY_POSITIONAL_ORDER = (
    "dome_radius_m",
    "azimuth_extent_rad",
    "elevation_extent_rad",
    "sensor_spacing_m",
    "sensor_size_m",
    "freq_max_hz",
    "rotation_offset",
    "mounting_offset",
    "movement_controller",
    "baffle_attenuation_db",
)

_SHAPES = ("spherical", "cylindrical")

# Keeps a stave on the edge of a sector, and a length that is a whole number of spacings,
# from falling on the wrong side of a comparison through rounding.
_TOLERANCE = 1e-9


def _legacy_keywords(args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
    """Map positional arguments given in the 0.7.0 order to keywords, with a warning."""
    if len(args) > len(_LEGACY_POSITIONAL_ORDER):
        raise TypeError(
            f"BowArray takes at most {len(_LEGACY_POSITIONAL_ORDER)} positional arguments "
            f"({len(args)} given)."
        )
    positional = dict(zip(_LEGACY_POSITIONAL_ORDER, args, strict=False))
    repeated = sorted(positional.keys() & kwargs.keys())
    if repeated:
        raise TypeError(f"BowArray got multiple values for {', '.join(repeated)}.")
    warnings.warn(
        "Passing BowArray arguments after azimuth_extent_rad positionally is deprecated; "
        "pass them by keyword.",
        DeprecationWarning,
        stacklevel=3,
    )
    return {**positional, **kwargs}


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


def _num_cells(length_m: float, max_pitch_m: float) -> int:
    """Return the fewest equal cells, each at most ``max_pitch_m`` long, that cover a length."""
    return max(1, int(np.ceil(length_m / max_pitch_m - _TOLERANCE)))


def _ring_stave_azimuths(
    radius_m: float, azimuth_extent_rad: float, sensor_spacing_m: float
) -> tuple[FloatArray, int]:
    r"""Compute the azimuths of a cylinder's staves, for a full ring or a sector of one.

    The full ring has :math:`n = \lceil 2 \pi r / d \rceil` staves at
    :math:`\theta_k = 2 \pi k / n`, with :math:`k = 0` on boresight (local +x), so the arc
    between adjacent staves is at most the spacing :math:`d`. A sector keeps the staves with
    :math:`|\theta_k| \le` ``azimuth_extent_rad / 2``, so it is a subset of the full ring.

    Parameters
    ----------
    radius_m : float
        Radius of the cylinder, in metres.
    azimuth_extent_rad : float
        Arc the staves cover, centred on boresight, in radians; :math:`2 \pi` is a full ring.
    sensor_spacing_m : float
        Largest allowed arc between adjacent staves, in metres.

    Returns
    -------
    tuple of (FloatArray, int)
        The kept staves' azimuths in ascending order, within :math:`(-\pi, \pi]`, and the
        number of staves in the full ring.

    """
    num_ring = _num_cells(2 * np.pi * radius_m, sensor_spacing_m)
    stave_index = np.arange(-((num_ring - 1) // 2), num_ring // 2 + 1)
    azimuths = 2 * np.pi * stave_index / num_ring
    return azimuths[np.abs(azimuths) <= azimuth_extent_rad / 2 + _TOLERANCE], num_ring


def _cylinder_sensor_offsets(
    radius_m: float, stave_azimuths_rad: FloatArray, height_m: float, sensor_spacing_m: float
) -> FloatArray:
    r"""Compute sensor positions on a cylinder's staves, relative to the cylinder's centre.

    Each stave has :math:`m = \lceil h / d \rceil` sensors at the centres of equal cells over
    the height :math:`h`, :math:`z_j = -h / 2 + (j + 1/2) h / m`, so the row pitch
    :math:`h / m` is at most the spacing :math:`d`.

    Parameters
    ----------
    radius_m : float
        Radius of the cylinder, in metres.
    stave_azimuths_rad : FloatArray
        Azimuth of each stave from boresight (local +x), in radians.
    height_m : float
        Height of a stave, in metres.
    sensor_spacing_m : float
        Largest allowed pitch between adjacent rows, in metres.

    Returns
    -------
    FloatArray
        Sensor offsets with shape ``(3, num_sensors)`` in the array frame (+x forward, +z up),
        ordered by row (lowest first), then by azimuth.

    """
    num_rows = _num_cells(height_m, sensor_spacing_m)
    row_heights = (np.arange(num_rows) + 0.5) * height_m / num_rows - height_m / 2
    x = radius_m * np.cos(stave_azimuths_rad)
    y = radius_m * np.sin(stave_azimuths_rad)
    return np.vstack(
        [
            np.tile(x, num_rows),
            np.tile(y, num_rows),
            np.repeat(row_heights, stave_azimuths_rad.size),
        ]
    )


class BowArray(RigidArray):
    r"""Sensors mounted rigidly on a platform's bow, on a spherical dome or a cylinder.

    Mount it with the platform's ``add_sensor``. Its ``mounting_offset`` is the centre of the
    sphere or cylinder relative to the platform's reference point, e.g. ``[0, 0, -3]`` for an
    array 3 m below the bow, so propagation is computed at the array's depth. The array axis
    points along the heading for both shapes.

    ``shape="spherical"`` is a forward-facing patch of a sphere, ``azimuth_extent_rad`` wide and
    ``elevation_extent_rad`` high. The hull behind the dome baffles every sensor, so all of
    them face forward.

    ``shape="cylindrical"`` is a ring of vertical staves (a stave is a vertical column of
    sensors) around a vertical axis, as on most surface-ship hull sonars. With radius
    :math:`r`, height :math:`h` and spacing :math:`d`, the full ring has
    :math:`n = \lceil 2 \pi r / d \rceil` staves at azimuths :math:`\theta_k = 2 \pi k / n`
    from boresight, each with :math:`m = \lceil h / d \rceil` sensors at heights
    :math:`z_j = -h / 2 + (j + 1/2) h / m`. An ``azimuth_extent_rad`` below :math:`2 \pi`
    keeps the staves with :math:`|\theta_k|` up to half of it. Each stave faces outwards, with
    normal :math:`(\cos \theta_k, \sin \theta_k, 0)`, and is baffled separately.

    Parameters
    ----------
    dome_radius_m : float
        Radius of the sphere or cylinder, in metres.
    azimuth_extent_rad : float
        Angular width of the array in azimuth, centred on boresight, in radians. For
        ``"cylindrical"``, :math:`2 \pi` gives a full ring.
    sensor_spacing_m : float
        Centre-to-centre spacing between adjacent sensors, in metres. ``"cylindrical"`` treats
        it as the largest allowed pitch, around the ring (as an arc) and down each stave.
    sensor_size_m : float
        Physical diameter of one sensor, in metres; must be less than the sensors' pitch.
    freq_max_hz : float
        Highest frequency the array is used at, in Hz, for the grating-lobe spacing check.
    elevation_extent_rad : float, optional
        Angular width of the dome patch in elevation, in radians. Required for
        ``"spherical"``; must be ``None`` for ``"cylindrical"``.
    baffle_attenuation_db : float, optional
        Attenuation of sound arriving from behind a sensor's face, in dB.
    shape : {"spherical", "cylindrical"}, optional
        Surface the sensors lie on.
    height_m : float, optional
        Height of a stave, in metres. Required for ``"cylindrical"``; must be ``None`` for
        ``"spherical"``.

    Raises
    ------
    ValueError
        If ``shape`` is unknown, a property belongs to the other shape or one this shape needs
        is missing, or sensors would touch or are spaced beyond half a wavelength at
        ``freq_max_hz``. ``"spherical"`` checks ``sensor_spacing_m``; ``"cylindrical"`` checks
        the pitches it lays out, the chord between adjacent staves and the row pitch.

    Assumptions
    -----------
    - Staves are straight, vertical in the array frame and identical, with evenly spaced rows.
    - A sensor hears a direction at full gain within 90 degrees of its normal and at
      ``baffle_attenuation_db`` beyond it; there is no other element directivity. Only the
      active simulator applies this. The passive simulators ignore baffling, so a full ring
      there receives on every stave.
    - The transmit gain is the mean of the sensor gains, so a full ring transmits about 6 dB
      down in every direction. That is an artefact of the mean, not a property of a real
      cylinder.
    - Beamformers apply one set of sensor weights to every beam, so a beam is not formed from
      only the staves that face it.

    """

    dome_radius_m: float = Property(doc="Radius of the sphere or cylinder, in metres")
    azimuth_extent_rad: float = Property(
        doc="Angular width of the array in azimuth, centred on boresight, in radians. For "
        '"cylindrical", 2 pi gives a full ring.'
    )
    elevation_extent_rad: float | None = Property(
        default=None,
        doc="Angular width of the dome patch in elevation, in radians. Required for "
        '"spherical"; must be None for "cylindrical".',
    )
    sensor_spacing_m: float = Property(
        doc="Centre-to-centre spacing between adjacent sensors, in metres. "
        '"cylindrical" treats it as the largest allowed pitch.'
    )
    sensor_size_m: float = Property(doc="Physical diameter of one sensor, in metres")
    freq_max_hz: float = Property(
        doc="Highest frequency the array is used at, in Hz, for the grating-lobe check"
    )
    baffle_attenuation_db: float = Property(
        default=40.0,
        doc="Attenuation of sound arriving from behind a sensor's face, in dB. "
        "Real hull baffles typically give 15-30 dB, less at low frequency.",
    )
    shape: Literal["spherical", "cylindrical"] = Property(
        default="spherical", doc="Surface the sensors lie on"
    )
    height_m: float | None = Property(
        default=None,
        doc='Height of a stave, in metres. Required for "cylindrical"; must be None for '
        '"spherical".',
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the BowArray and lay out its sensors.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass. Pass at most ``dome_radius_m`` and
            ``azimuth_extent_rad`` this way; more are deprecated and read in the order
            ``BowArray`` took them up to 0.7.0.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        if len(args) > 2:
            args, kwargs = (), _legacy_keywords(args, kwargs)
        super().__init__(*args, **kwargs)
        if self.shape == "spherical":
            self._lay_out_dome()
        elif self.shape == "cylindrical":
            self._lay_out_cylinder()
        else:
            raise ValueError(f"shape ({self.shape!r}) must be one of {_SHAPES}.")
        self._local_axis = np.array([[1.0], [0.0], [0.0]])

    def _lay_out_dome(self) -> None:
        """Lay sensors on the spherical patch, all facing forward."""
        elevation_extent_rad = self.elevation_extent_rad
        if elevation_extent_rad is None:
            raise ValueError('elevation_extent_rad is required for shape="spherical".')
        if self.height_m is not None:
            raise ValueError('height_m must be None for shape="spherical".')
        _check_sensor_spacing(self.sensor_spacing_m, self.sensor_size_m, self.freq_max_hz)
        self._local_offsets = _dome_sensor_offsets(
            self.dome_radius_m,
            self.azimuth_extent_rad,
            elevation_extent_rad,
            self.sensor_spacing_m,
        )
        forward = np.array([[1.0], [0.0], [0.0]])
        self._local_normals = np.tile(forward, (1, self.num_sensors))

    def _lay_out_cylinder(self) -> None:
        """Lay sensors on the cylinder's staves, each stave facing outwards."""
        height_m = self.height_m
        if height_m is None:
            raise ValueError('height_m is required for shape="cylindrical".')
        if height_m <= 0:
            raise ValueError(f"height_m ({height_m}m) must be positive.")
        if self.elevation_extent_rad is not None:
            raise ValueError('elevation_extent_rad must be None for shape="cylindrical".')
        if not 0 < self.azimuth_extent_rad <= 2 * np.pi + _TOLERANCE:
            raise ValueError(
                f"azimuth_extent_rad ({self.azimuth_extent_rad} rad) must be in (0, 2 pi] for "
                'shape="cylindrical".'
            )

        azimuths, num_ring = _ring_stave_azimuths(
            self.dome_radius_m, self.azimuth_extent_rad, self.sensor_spacing_m
        )
        self._local_offsets = _cylinder_sensor_offsets(
            self.dome_radius_m, azimuths, height_m, self.sensor_spacing_m
        )
        num_rows = self.num_sensors // azimuths.size

        # The layout's pitches are at most sensor_spacing_m, so the nominal spacing can fail
        # the grating-lobe check, or pass the touching check, when the sensors as laid out
        # would not. Adjacent staves are a chord apart, not an arc.
        if azimuths.size > 1:
            ring_pitch_m = 2 * self.dome_radius_m * np.sin(np.pi / num_ring)
            _check_sensor_spacing(ring_pitch_m, self.sensor_size_m, self.freq_max_hz, "ring pitch")
        if num_rows > 1:
            _check_sensor_spacing(
                height_m / num_rows, self.sensor_size_m, self.freq_max_hz, "row pitch"
            )

        outwards = np.vstack([np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)])
        self._local_normals = np.tile(outwards, (1, num_rows))
