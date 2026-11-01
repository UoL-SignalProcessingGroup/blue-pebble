"""Defines a flank array: a flat panel of sensors on one side of a platform's hull."""

import numpy as np
from stonesoup.base import Property
from stonesoup.types.array import StateVector

from .base import FloatArray, RigidArray, _check_sensor_spacing

# Yaw of each side's panel normal from the platform's forward axis (+y is port).
_SIDE_YAW_RAD = {"port": np.pi / 2, "starboard": -np.pi / 2}


def _panel_sensor_offsets(
    aperture_length_m: float, aperture_height_m: float, sensor_spacing_m: float
) -> FloatArray:
    """Compute a flat panel's sensor positions in the platform's body frame, about its centre.

    The panel is a uniform rectangular grid in the hull's side: columns run along the hull
    (body x) and rows in depth (body z), both exactly ``sensor_spacing_m`` apart, with as many
    as fit inside the aperture.

    Parameters
    ----------
    aperture_length_m : float
        Maximum centre-to-centre extent of the panel along the hull, in metres.
    aperture_height_m : float
        Maximum centre-to-centre extent of the panel in depth, in metres (0 for one row).
    sensor_spacing_m : float
        Centre-to-centre spacing between adjacent sensors, in metres.

    Returns
    -------
    FloatArray
        Offsets with shape ``(3, num_sensors)`` from the panel centre (body y is 0), ordered by
        row (top first), then column (forward first).

    """
    # Small tolerance so an aperture that is an exact multiple of the spacing keeps its last
    # sensor despite floating-point round-off.
    num_cols = int(np.floor(aperture_length_m / sensor_spacing_m + 1e-9)) + 1
    num_rows = int(np.floor(aperture_height_m / sensor_spacing_m + 1e-9)) + 1
    x_cols = -sensor_spacing_m * (np.arange(num_cols) - (num_cols - 1) / 2)
    z_rows = -sensor_spacing_m * (np.arange(num_rows) - (num_rows - 1) / 2)
    z_grid, x_grid = np.meshgrid(z_rows, x_cols, indexing="ij")
    x_flat = x_grid.ravel()
    return np.vstack([x_flat, np.zeros_like(x_flat), z_grid.ravel()]).astype(np.float64)


class FlankArray(RigidArray):
    """A flat panel of sensors mounted rigidly on one side of a platform's hull.

    Mount one per side with the platform's ``add_sensor``. ``mounting_offset`` is the panel's
    centre relative to the platform's reference point (the bow), e.g. ``[-25, 5, -4]`` for a
    port panel centred 25 m aft and 4 m down on a 10 m beam; its y must be positive for port
    and negative for starboard. ``side`` sets Stone Soup's ``rotation_offset`` to a yaw of
    +90 degrees (port) or -90 degrees (starboard), so every sensor faces straight out of its
    side of the hull, and the array axis points forward along the hull.

    Parameters
    ----------
    side : str
        ``"port"`` or ``"starboard"``.
    aperture_length_m : float
        Maximum centre-to-centre extent of the panel along the hull, in metres.
    sensor_spacing_m : float
        Centre-to-centre spacing between adjacent sensors, in metres.
    sensor_size_m : float
        Physical diameter of one sensor, in metres; must be less than ``sensor_spacing_m``.
    freq_max_hz : float
        Highest frequency the array is used at, in Hz, for the grating-lobe spacing check.
    aperture_height_m : float, optional
        Maximum centre-to-centre extent of the panel in depth, in metres. 0 gives a single
        horizontal line of sensors.
    baffle_attenuation_db : float, optional
        Attenuation of sound arriving from across the hull (behind the panel's face), in dB.

    Raises
    ------
    ValueError
        If ``side`` is unknown, ``mounting_offset`` is missing or on the wrong side, a given
        ``rotation_offset`` disagrees with ``side``, an aperture is negative, or the sensor
        spacing is invalid. When mounted on a platform, also if the panel's footprint is not
        between the bow and the tow point of any towed array already mounted.

    """

    side: str = Property(doc='Which side of the hull the panel is on: "port" or "starboard".')
    aperture_length_m: float = Property(
        doc="Maximum centre-to-centre extent of the panel along the hull, in metres"
    )
    sensor_spacing_m: float = Property(
        doc="Centre-to-centre spacing between adjacent sensors, in metres"
    )
    sensor_size_m: float = Property(doc="Physical diameter of one sensor, in metres")
    freq_max_hz: float = Property(
        doc="Highest frequency the array is used at, in Hz, for the grating-lobe check"
    )
    aperture_height_m: float = Property(
        default=0.0,
        doc="Maximum centre-to-centre extent of the panel in depth, in metres. 0 gives a "
        "single horizontal line of sensors.",
    )
    baffle_attenuation_db: float = Property(
        default=40.0,
        doc="Attenuation of sound arriving from across the hull (behind the panel's face), "
        "in dB. Real hull baffles typically give 15-30 dB, less at low frequency.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the FlankArray, check its geometry and lay out its sensors.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        if self.side not in _SIDE_YAW_RAD:
            raise ValueError(f"side ({self.side!r}) must be one of {sorted(_SIDE_YAW_RAD)}.")
        if self.mounting_offset is None:
            raise ValueError(
                "FlankArray needs a mounting_offset: the panel centre relative to the "
                "platform's reference point, e.g. [-25, 5, -4] for a port panel."
            )
        side_sign = 1.0 if self.side == "port" else -1.0
        if not float(self.mounting_offset[1, 0]) * side_sign > 0:
            raise ValueError(
                f"mounting_offset y ({float(self.mounting_offset[1, 0])}m) must be "
                f"{'positive' if self.side == 'port' else 'negative'} for a {self.side} panel."
            )

        expected = StateVector([0.0, 0.0, _SIDE_YAW_RAD[self.side]])
        if self.rotation_offset is not None and not np.allclose(
            np.asarray(self.rotation_offset, dtype=float), expected
        ):
            raise ValueError(
                f"rotation_offset is set by side; a {self.side} panel needs "
                f"[0, 0, {_SIDE_YAW_RAD[self.side]:.6f}], not "
                f"{np.asarray(self.rotation_offset, dtype=float).ravel().tolist()}."
            )
        self.rotation_offset = expected

        for name in ("aperture_length_m", "aperture_height_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} ({getattr(self, name)}m) must be non-negative.")
        _check_sensor_spacing(self.sensor_spacing_m, self.sensor_size_m, self.freq_max_hz)

        # Lay the panel out in the body frame, then express it in the array frame, whose +x is
        # the panel's outward normal.
        body_to_array = self._offset_rotation().T
        self._local_offsets = body_to_array @ _panel_sensor_offsets(
            self.aperture_length_m, self.aperture_height_m, self.sensor_spacing_m
        )
        self._local_normals = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, self.num_sensors))
        self._local_axis = body_to_array @ np.array([[1.0], [0.0], [0.0]])

    def _footprint_x_extent_m(self) -> tuple[float, float]:
        """Return the body-x of the panel's aft and forward footprint edges, in metres."""
        centre_x = float(self.mounting_offset[0, 0])
        num_cols = int(np.floor(self.aperture_length_m / self.sensor_spacing_m + 1e-9)) + 1
        half_extent = self.sensor_spacing_m * (num_cols - 1) / 2 + self.sensor_size_m / 2
        return centre_x - half_extent, centre_x + half_extent

    def _on_mounted(self, platform: object) -> None:
        """Check the panel fits between the bow and any mounted towed array's tow point.

        Mount the towed array first for its tow point to be checked.

        Raises
        ------
        ValueError
            If the panel's footprint reaches forward of the platform's reference point (the
            bow), or aft of the tow point of a :class:`~bluepebble.arrays.TowedArray` already
            mounted on the platform.

        """
        from .towedarray import TowedArray  # noqa: PLC0415  (towedarray imports base)

        aft_edge, forward_edge = self._footprint_x_extent_m()
        if forward_edge > 0:
            raise ValueError(
                f"The {self.side} flank panel's footprint reaches {forward_edge}m forward of "
                "the bow (the platform's reference point); move its mounting_offset aft."
            )
        for array in getattr(platform, "arrays", ()):
            if isinstance(array, TowedArray) and array.mounting_offset is not None:
                tow_point_x = float(array.mounting_offset[0, 0])
                if aft_edge < tow_point_x:
                    raise ValueError(
                        f"The {self.side} flank panel's footprint reaches {aft_edge}m, aft of "
                        f"the towed array's tow point at {tow_point_x}m; shorten the aperture "
                        "or move the panel forward."
                    )
