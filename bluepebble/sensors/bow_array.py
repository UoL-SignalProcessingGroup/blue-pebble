"""Defines a bow array sensor: a curved forward-facing dome mounted rigidly on a host."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.movable.movable import _get_rotation_matrix
from stonesoup.types.array import StateVector, StateVectors

from .base import ArrayState, Sensor

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class BowArraySnapshot:
    """A single-field wrapper exposing ``.array`` for a historical dome state.

    Returned by :meth:`BowArraySensor.get_platform_state_at` when the host has a
    recorded state at the requested timestamp, so callers get an ``ArrayState`` for
    that specific timestamp rather than always the host's current/live one.
    """

    array: ArrayState


def _compute_dome_element_offsets(
    dome_radius_m: float,
    azimuth_extent_rad: float,
    elevation_extent_rad: float,
    element_spacing_m: float,
    array_depth_offset_m: float,
) -> FloatArray:
    """Compute local body-frame element positions for a spherical-patch bow dome.

    The dome is treated as a patch of a sphere of radius ``dome_radius_m`` centred on the
    array's boresight (local +x, straight ahead). Rows are laid out in elevation and, within
    each row, elements are laid out in azimuth, both spaced no closer than
    ``element_spacing_m`` (measured as arc length along the sphere). The whole patch is then
    lowered by ``array_depth_offset_m`` so it sits below the host mounting point.

    Parameters
    ----------
    dome_radius_m : float
        Radius of curvature of the dome, in metres.
    azimuth_extent_rad : float
        Angular width of the forward-facing patch in azimuth, in radians.
    elevation_extent_rad : float
        Angular width of the forward-facing patch in elevation, in radians.
    element_spacing_m : float
        Minimum centre-to-centre spacing between adjacent elements, in metres.
    array_depth_offset_m : float
        Distance the dome is mounted below the host reference point, in metres
        (positive is deeper; the body frame is +x forward, +z up).

    Returns
    -------
    FloatArray
        Local element offsets with shape ``(3, num_elements)``, in the body frame (local +x
        forward, +z up), relative to the host mounting point.

    """
    num_rows = int(np.floor(elevation_extent_rad * dome_radius_m / element_spacing_m)) + 1
    elevation_angles = np.linspace(
        -elevation_extent_rad / 2, elevation_extent_rad / 2, num_rows
    )

    offsets = []
    for phi in elevation_angles:
        row_span_m = azimuth_extent_rad * dome_radius_m * np.cos(phi)
        elements_in_row = int(np.floor(row_span_m / element_spacing_m)) + 1
        azimuth_angles = np.linspace(
            -azimuth_extent_rad / 2, azimuth_extent_rad / 2, elements_in_row
        )

        x = dome_radius_m * np.cos(phi) * np.cos(azimuth_angles)
        y = dome_radius_m * np.cos(phi) * np.sin(azimuth_angles)
        z = np.full_like(azimuth_angles, dome_radius_m * np.sin(phi)) - array_depth_offset_m
        offsets.append(np.vstack([x, y, z]))

    return np.hstack(offsets)


class BowArraySensor(Sensor):
    """A curved forward-facing dome of elements mounted rigidly on a host's bow.

    Being rigidly mounted, the dome has no dynamics of its own: its world-frame geometry is
    always a direct function of the host's current position and velocity-derived orientation
    (see :meth:`array`), so this sensor keeps no per-timestep history and needs no
    ``_on_host_moved`` hook (unlike :class:`~bluepebble.sensors.towed_array.TowedArraySensor`).
    """

    dome_radius_m: float = Property(doc="Radius of curvature of the bow dome, in metres")
    azimuth_extent_rad: float = Property(
        doc="Angular width of the forward-facing dome patch in azimuth, in radians"
    )
    elevation_extent_rad: float = Property(
        doc="Angular width of the forward-facing dome patch in elevation, in radians"
    )
    element_spacing_m: float = Property(
        doc="Fixed centre-to-centre spacing between adjacent elements, in metres"
    )
    element_size_m: float = Property(
        doc="Physical diameter/footprint of a single element, in metres. Must be strictly "
        "less than element_spacing_m so elements do not touch."
    )
    freq_max_hz: float = Property(
        doc="Maximum frequency of the transmit pulse (LFM sweep or CW tone), in Hz. "
        "Used to check the grating-lobe spacing limit."
    )

    array_depth_offset_m: float = Property(
        default=0.0,
        doc="Distance the dome is mounted below the host reference point, in metres. "
        "Lowers the elements and the array reference point together, so the "
        "source/receiver depth used for propagation is the dome's depth, not the host's "
        "(a surface ship at z = 0 still transmits from below the hull).",
    )
    baffle_attenuation_db: float = Property(
        default=40.0,
        doc="Attenuation of echoes arriving from behind the dome face (aft of the bow), "
        "in dB. Set by the hull baffle; real values are typically 15-30 dB, lower at low "
        "frequency.",
    )

    NOMINAL_SOUND_SPEED = 1500.0

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the BowArraySensor.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        Raises
        ------
        ValueError
            If ``element_spacing_m`` does not strictly exceed ``element_size_m``, meaning
            elements would touch or overlap.
        ValueError
            If ``element_spacing_m`` exceeds ``lambda_min / 2`` at ``freq_max_hz`` (using
            ``NOMINAL_SOUND_SPEED``), risking grating lobes.

        """
        super().__init__(*args, **kwargs)

        if self.element_spacing_m <= self.element_size_m:
            raise ValueError(
                f"element_spacing_m ({self.element_spacing_m}m) must be strictly greater "
                f"than element_size_m ({self.element_size_m}m) so elements do not touch."
            )

        max_spacing_m = self.NOMINAL_SOUND_SPEED / (2 * self.freq_max_hz)
        if self.element_spacing_m > max_spacing_m:
            raise ValueError(
                f"element_spacing_m ({self.element_spacing_m}m) exceeds the grating-lobe "
                f"limit of lambda/2 ({max_spacing_m}m) at freq_max_hz ({self.freq_max_hz}Hz) "
                f"using NOMINAL_SOUND_SPEED_MPS ({self.NOMINAL_SOUND_SPEED}m/s)."
            )

        self._element_offsets_m = _compute_dome_element_offsets(
            self.dome_radius_m,
            self.azimuth_extent_rad,
            self.elevation_extent_rad,
            self.element_spacing_m,
            self.array_depth_offset_m,
        )
        # Reference point: the dome centre, directly below the host mounting point.
        self._reference_offset_m = np.array([[0.0], [0.0], [-self.array_depth_offset_m]])

    def _array_from_position_velocity(
        self, position: StateVector, velocity: StateVector
    ) -> ArrayState:
        """Compute dome geometry for an explicit host position/velocity pair.

        Shared by :attr:`array` (the host's current position/velocity) and
        :meth:`get_platform_state_at` (a specific historical position/velocity), so both
        report the dome geometry the same way -- see the note on :meth:`array` about why
        this can't just always be "live".
        """
        is_moving = bool(np.any(velocity != 0))
        rotation = _get_rotation_matrix(velocity) if is_moving else np.eye(3)

        world_positions = rotation @ self._element_offsets_m + position
        # The hull behind the dome baffles every element, so all of them face forward.
        forward = rotation @ np.array([[1.0], [0.0], [0.0]])

        return ArrayState(
            num_sensors=world_positions.shape[1],
            state_vector=StateVectors(world_positions),
            ref_state_vector=StateVector(rotation @ self._reference_offset_m + position),
            element_normals=StateVectors(np.tile(forward, (1, world_positions.shape[1]))),
            baffle_attenuation_db=self.baffle_attenuation_db,
            axis=StateVector(forward),
        )

    @property
    def array(self) -> ArrayState:
        """Return the world-frame array state for the dome's elements, live.

        Rotates the local body-frame element layout by the host's *current*
        velocity-derived orientation (yaw and pitch; roll is not modelled, matching
        :class:`~stonesoup.movable.movable.MovingMovable`) and translates it by the host's
        current position. The reference point is the dome centre: the host position lowered
        by ``array_depth_offset_m``, so propagation is computed at the dome's depth.

        This always reflects the host's latest recorded state -- if the host has already
        been advanced past the timestamp you actually care about (e.g. a caller drives it
        through every ping timestamp before running a simulator), use
        :meth:`get_platform_state_at` instead so you get the geometry for the timestamp
        you asked for, not wherever the host ended up.

        Returns
        -------
        ArrayState
            Array state with ``num_sensors`` equal to the number of dome elements,
            per-element world positions in ``state_vector``, ``ref_state_vector`` at
            the dome centre (``array_depth_offset_m`` below the host), and
            ``element_normals`` all pointing along the host's heading (the dome is
            baffled by the hull behind it), with
            ``baffle_attenuation_db`` copied from the sensor.

        """
        return self._array_from_position_velocity(self.host.position, self.host.velocity)

    def get_platform_state_at(self, timestamp: datetime) -> "BowArraySensor | BowArraySnapshot":
        """Return an object exposing ``.array`` for the dome's geometry at ``timestamp``.

        Looks up the host's own recorded state at ``timestamp`` (via
        :meth:`~bluepebble.platform.base.HostPlatform.get_state_at`) and computes the dome
        geometry from that specific position/velocity, so a caller that has already driven
        the host through several timestamps (e.g. every ping in a scenario) gets the
        correct historical geometry for each one -- not wherever the host is *now*, which
        is what plain :attr:`array` would give if the host has already moved on.

        Falls back to the live :attr:`array` when the host has no recorded state at
        ``timestamp`` (e.g. a stationary installation that is never explicitly moved, so
        its one initial state is valid at every timestamp).

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        BowArraySensor or BowArraySnapshot
            This sensor itself (live) if the host has no recorded state at ``timestamp``,
            otherwise a snapshot exposing ``.array`` for that historical state.

        """
        host_state = self.host.get_state_at(timestamp)
        if host_state is None:
            return self

        position = host_state.state_vector[self.host.position_mapping]
        velocity = host_state.state_vector[self.host.resolved_velocity_mapping()]
        return BowArraySnapshot(array=self._array_from_position_velocity(position, velocity))
