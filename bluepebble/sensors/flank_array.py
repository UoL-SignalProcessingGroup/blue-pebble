"""Defines a flank array sensor: conformal hull-mounted panels along the host's sides."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.movable.movable import _get_rotation_matrix
from stonesoup.types.array import StateVector, StateVectors

from .base import ArrayState, Sensor
from .towed_array import TowedArraySensor

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]

_SIDE_SIGNS = {"port": (1.0,), "starboard": (-1.0,), "both": (1.0, -1.0)}


@dataclass
class FlankArraySnapshot:
    """A single-field wrapper exposing ``.array`` for a historical flank array state.

    Returned by :meth:`FlankArraySensor.get_platform_state_at` when the host has a
    recorded state at the requested timestamp, so callers get an ``ArrayState`` for
    that specific timestamp rather than always the host's current/live one.
    """

    array: ArrayState


def _compute_flank_element_offsets(
    hull_beam_m: float,
    flank_bow_offset_m: float,
    aperture_length_m: float,
    aperture_height_m: float,
    element_spacing_m: float,
    element_size_m: float,
    array_depth_offset_m: float,
    sides: str,
) -> FloatArray:
    """Compute local body-frame element positions for one or two flat flank panels.

    Each panel is a uniform rectangular grid lying in the hull's side (local x-z plane at
    ``y = +/- hull_beam_m / 2``). Columns run aft along the hull starting with the forward
    element's footprint edge ``flank_bow_offset_m`` behind the host reference point; rows are
    centred vertically on ``array_depth_offset_m`` below that point. Elements are placed at
    exactly ``element_spacing_m`` in both directions, so the grid is a true uniform
    rectangular array (as many columns/rows as fit inside the requested aperture).

    Parameters
    ----------
    hull_beam_m : float
        Full width of the hull at the flank panels, in metres.
    flank_bow_offset_m : float
        Distance aft of the host reference point (the bow) to the forward edge of the
        forward-most element's footprint, in metres.
    aperture_length_m : float
        Maximum centre-to-centre extent of each panel along the hull, in metres.
    aperture_height_m : float
        Maximum centre-to-centre extent of each panel in depth, in metres (0 for one row).
    element_spacing_m : float
        Centre-to-centre spacing between adjacent elements, in metres.
    element_size_m : float
        Physical diameter/footprint of a single element, in metres.
    array_depth_offset_m : float
        Distance the panel centres are mounted below the host reference point, in metres
        (positive is deeper; the body frame is +x forward, +y port, +z up).
    sides : str
        Which side(s) of the hull carry a panel: ``"port"``, ``"starboard"`` or ``"both"``.

    Returns
    -------
    FloatArray
        Local element offsets with shape ``(3, num_elements)``, in the body frame, relative to
        the host reference point. Ordered by side (port first), then row (top first), then
        column (forward first).

    """
    # Small tolerance so an aperture that is an exact multiple of the spacing keeps its
    # last element despite floating-point round-off.
    num_cols = int(np.floor(aperture_length_m / element_spacing_m + 1e-9)) + 1
    num_rows = int(np.floor(aperture_height_m / element_spacing_m + 1e-9)) + 1

    x_cols = -(flank_bow_offset_m + element_size_m / 2) - element_spacing_m * np.arange(num_cols)
    z_rows = -array_depth_offset_m - element_spacing_m * (np.arange(num_rows) - (num_rows - 1) / 2)

    z_grid, x_grid = np.meshgrid(z_rows, x_cols, indexing="ij")
    x_flat = x_grid.ravel()
    z_flat = z_grid.ravel()

    panels = [
        np.vstack([x_flat, np.full_like(x_flat, sign * hull_beam_m / 2), z_flat])
        for sign in _SIDE_SIGNS[sides]
    ]
    return np.hstack(panels)


class FlankArraySensor(Sensor):
    """Flat conformal panels of elements mounted rigidly along the host's port/starboard hull.

    The panels sit on the hull section between the bow (the host reference point, where
    :class:`~bluepebble.sensors.bow_array.BowArraySensor`'s dome projects forward from) and
    the tow point (``towed_array_bow_offset_m`` aft of the bow, where
    :class:`~bluepebble.sensors.towed_array.TowedArraySensor`'s cable starts). Construction
    fails if the requested aperture's element footprints would not fit inside that section.

    Like the bow array, it is rigidly mounted, so its world-frame geometry is always a direct
    function of the host's position and velocity-derived orientation (see :meth:`array`); it
    keeps no per-timestep history and needs no ``_on_host_moved`` hook.
    """

    hull_beam_m: float = Property(
        doc="Full width of the hull at the flank panels, in metres. Port/starboard elements "
        "sit at +/- hull_beam_m / 2 from the centreline."
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
    towed_array_bow_offset_m: float | None = Property(
        default=None,
        doc="Distance from the host reference point (bow) to the tow point, in metres -- the "
        "aft limit of the hull section the panels must fit within. If None, taken from a "
        "TowedArraySensor already attached to the host.",
    )
    flank_bow_offset_m: float = Property(
        default=0.0,
        doc="Distance aft of the host reference point (bow) to the forward edge of the "
        "forward-most element's footprint, in metres. Keeps the panels clear of the bow dome.",
    )
    aperture_length_m: float | None = Property(
        default=None,
        doc="Maximum centre-to-centre extent of each panel along the hull, in metres. If None, "
        "the panels fill the whole hull section between flank_bow_offset_m and the tow point.",
    )
    aperture_height_m: float = Property(
        default=0.0,
        doc="Maximum centre-to-centre extent of each panel in depth, in metres. 0 gives a "
        "single horizontal line of elements per side.",
    )
    array_depth_offset_m: float = Property(
        default=0.0,
        doc="Distance the panel centres are mounted below the host reference point, in "
        "metres. Lowers the elements and the array reference point together, so "
        "propagation is computed at the panels' depth (matching BowArraySensor).",
    )
    sides: str = Property(
        default="both",
        doc='Which side(s) of the hull carry a panel: "port", "starboard" or "both".',
    )
    baffle_attenuation_db: float = Property(
        default=40.0,
        doc="Attenuation of echoes arriving from across the hull (behind the panel's face), "
        "in dB. Set by the hull baffle; real values are typically 15-30 dB, lower at low "
        "frequency.",
    )

    NOMINAL_SOUND_SPEED = 1500.0

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the FlankArraySensor.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        Raises
        ------
        ValueError
            If the element spacing is invalid (elements touching, or beyond the
            grating-lobe limit), the geometry parameters are out of range, no tow-point
            offset can be determined, or the aperture does not fit on the hull section
            between the bow and the tow point.

        """
        super().__init__(*args, **kwargs)

        if self.sides not in _SIDE_SIGNS:
            raise ValueError(
                f"sides ({self.sides!r}) must be one of {sorted(_SIDE_SIGNS)}."
            )
        if self.hull_beam_m <= 0:
            raise ValueError(f"hull_beam_m ({self.hull_beam_m}m) must be positive.")
        if self.flank_bow_offset_m < 0:
            raise ValueError(
                f"flank_bow_offset_m ({self.flank_bow_offset_m}m) must be non-negative so "
                "the panels stay aft of the bow."
            )
        if self.aperture_height_m < 0:
            raise ValueError(
                f"aperture_height_m ({self.aperture_height_m}m) must be non-negative."
            )

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

        if self.towed_array_bow_offset_m is None:
            self.towed_array_bow_offset_m = self._towed_array_bow_offset_from_host()

        # Element centres can span at most the hull section minus one footprint, so that
        # the forward and aft elements' footprints stay between the bow and the tow point.
        available_length_m = (
            self.towed_array_bow_offset_m - self.flank_bow_offset_m - self.element_size_m
        )
        if available_length_m < 0:
            raise ValueError(
                f"No room for flank elements: flank_bow_offset_m ({self.flank_bow_offset_m}m) "
                f"plus one element_size_m ({self.element_size_m}m) exceeds the tow point at "
                f"towed_array_bow_offset_m ({self.towed_array_bow_offset_m}m)."
            )
        if self.aperture_length_m is None:
            self.aperture_length_m = available_length_m
        elif not 0 <= self.aperture_length_m <= available_length_m:
            raise ValueError(
                f"aperture_length_m ({self.aperture_length_m}m) does not fit on the hull: "
                f"between flank_bow_offset_m ({self.flank_bow_offset_m}m) and the tow point "
                f"({self.towed_array_bow_offset_m}m) there is at most {available_length_m}m "
                f"of element-centre span for element_size_m ({self.element_size_m}m)."
            )

        self._element_offsets_m = _compute_flank_element_offsets(
            self.hull_beam_m,
            self.flank_bow_offset_m,
            self.aperture_length_m,
            self.aperture_height_m,
            self.element_spacing_m,
            self.element_size_m,
            self.array_depth_offset_m,
            self.sides,
        )

        # Each element faces straight out of the hull side it sits on (+y port, -y starboard).
        num_elements = self._element_offsets_m.shape[1]
        self._element_normals = np.zeros((3, num_elements))
        self._element_normals[1] = np.sign(self._element_offsets_m[1])

        # Reference point: the panels' centroid, i.e. at the panels' depth (rows are centred
        # on -array_depth_offset_m), so propagation uses the array's depth, not the host's.
        self._reference_offset_m = np.mean(self._element_offsets_m, axis=1, keepdims=True)

    def _towed_array_bow_offset_from_host(self) -> float:
        """Return the tow-point offset of a TowedArraySensor already attached to the host.

        Raises
        ------
        ValueError
            If the host has no attached towed array to take the offset from.

        """
        for sensor in self.host.attached_sensors:
            if isinstance(sensor, TowedArraySensor):
                return sensor.towed_array_bow_offset_m
        raise ValueError(
            "towed_array_bow_offset_m was not given and the host has no attached "
            "TowedArraySensor to take it from; pass it explicitly or create the towed "
            "array first."
        )

    def _array_from_position_velocity(
        self, position: StateVector, velocity: StateVector
    ) -> ArrayState:
        """Compute flank geometry for an explicit host position/velocity pair.

        Shared by :attr:`array` (the host's current position/velocity) and
        :meth:`get_platform_state_at` (a specific historical position/velocity).
        """
        is_moving = bool(np.any(velocity != 0))
        rotation = _get_rotation_matrix(velocity) if is_moving else np.eye(3)

        world_positions = rotation @ self._element_offsets_m + position

        return ArrayState(
            num_sensors=world_positions.shape[1],
            state_vector=StateVectors(world_positions),
            ref_state_vector=StateVector(rotation @ self._reference_offset_m + position),
            element_normals=StateVectors(rotation @ self._element_normals),
            baffle_attenuation_db=self.baffle_attenuation_db,
            axis=StateVector(rotation @ np.array([[1.0], [0.0], [0.0]])),
        )

    @property
    def array(self) -> ArrayState:
        """Return the world-frame array state for the flank elements, live.

        Rotates the local body-frame layout by the host's *current* velocity-derived
        orientation (yaw and pitch; roll is not modelled) and translates it by the host's
        current position. If the host has already been advanced past the timestamp you care
        about, use :meth:`get_platform_state_at` instead.

        Returns
        -------
        ArrayState
            Array state with ``num_sensors`` equal to the number of flank elements (both
            sides), per-element world positions in ``state_vector``,
            ``ref_state_vector`` at the panels' centroid (at the panels' depth), and
            ``element_normals`` pointing out of the hull side each element is on, with
            ``baffle_attenuation_db`` copied from the sensor.

        """
        return self._array_from_position_velocity(self.host.position, self.host.velocity)

    def get_platform_state_at(
        self, timestamp: datetime
    ) -> "FlankArraySensor | FlankArraySnapshot":
        """Return an object exposing ``.array`` for the flank geometry at ``timestamp``.

        Falls back to the live :attr:`array` (returning this sensor) when the host has no
        recorded state at ``timestamp``.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to query.

        Returns
        -------
        FlankArraySensor or FlankArraySnapshot
            This sensor itself (live) if the host has no recorded state at ``timestamp``,
            otherwise a snapshot exposing ``.array`` for that historical state.

        """
        host_state = self.host.get_state_at(timestamp)
        if host_state is None:
            return self

        position = host_state.state_vector[self.host.position_mapping]
        velocity = host_state.state_vector[self.host.resolved_velocity_mapping()]
        return FlankArraySnapshot(array=self._array_from_position_velocity(position, velocity))

    def panel_states(
        self, timestamp: datetime | None = None
    ) -> dict[str, tuple[IntArray, FlankArraySnapshot]]:
        """Split the array into its port and starboard panels, one array state each.

        Each hull side is a separate aperture facing a different way, so it is normally
        beamformed on its own. Every panel gets its own reference point at its centroid
        and keeps its elements' normals, baffle level and the hull axis.

        Parameters
        ----------
        timestamp : datetime or None, optional
            Time of the geometry, as for :meth:`get_platform_state_at`. ``None`` (the
            default) uses the live :attr:`array`.

        Returns
        -------
        dict[str, tuple[IntArray, FlankArraySnapshot]]
            ``{"port": ..., "starboard": ...}`` (only the sides this sensor has). Each value
            is ``(element_indices, snapshot)``: the panel's rows in the full array (e.g. to
            pick its channels out of the received waveform) and a snapshot exposing
            ``.array`` for the panel alone, usable anywhere a sensor snapshot is (e.g.
            ``SteeringCalculator.calculate``).

        """
        full = self.array if timestamp is None else self.get_platform_state_at(timestamp).array
        positions = np.asarray(full.state_vector)
        normals = np.asarray(full.element_normals)
        on_port = self._element_normals[1] > 0  # body frame: +y is port

        panels = {}
        for side, mask in (("port", on_port), ("starboard", ~on_port)):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            panels[side] = (idx, FlankArraySnapshot(array=ArrayState(
                num_sensors=len(idx),
                state_vector=StateVectors(positions[:, idx]),
                ref_state_vector=StateVector(positions[:, idx].mean(axis=1, keepdims=True)),
                element_normals=StateVectors(normals[:, idx]),
                baffle_attenuation_db=full.baffle_attenuation_db,
                axis=full.axis,
            )))
        return panels
