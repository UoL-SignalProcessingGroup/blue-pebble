"""Steering-delay geometry for horizontal sensor arrays."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from stonesoup.base import Base, Property

from ..models.environment import SoundSpeedProfile
from .base import BoolArray, FloatArray, IntArray, MirrorPlan

if TYPE_CHECKING:
    from ..platform.towedarray import PlatformState

# Angles closer than this (radians) count as equal: sector ends meeting in a full circle, and a
# sine sector's end touching endfire.
_ANGLE_TOLERANCE_RAD = 1e-9

# In the world frame a sine-spaced grid is built from the array axis at the first scan. The axis
# may move by at most this fraction of the grid's narrowest beam spacing afterwards; beyond that
# the grid is no longer evenly spaced in the sine of the angle from broadside.
_SINE_AXIS_TOLERANCE_FRACTION = 0.1

# Parameters removed when the grid moved into the calculator, mapped to what replaces them.
_REMOVED_STEERING_KWARGS = {
    "steering_azimuths_rad": (
        "The calculator now builds its own grid: pass steering_sector_rad=(start, end), "
        "num_beams and spacing, e.g. steering_sector_rad=(-np.pi, np.pi) for a full circle. "
        "Read the grid back with steering_bearings()."
    ),
}


def _wrap_to_pi(angles_rad: FloatArray) -> FloatArray:
    """Wrap angles into ``[-pi, pi)``, leaving values already in range bit-for-bit unchanged."""
    angles = np.asarray(angles_rad, dtype=np.float64)
    out_of_range = (angles < -np.pi) | (angles >= np.pi)
    return np.where(out_of_range, (angles + np.pi) % (2 * np.pi) - np.pi, angles)


def _normalise_sector(start_rad: float, end_rad: float) -> tuple[float, float]:
    """Return a sector's wrapped start and its anticlockwise width.

    Endpoints that differ numerically but point the same way, such as ``(pi, -pi)`` or
    ``(0, 2 * pi)``, are a full circle and are returned as starting at ``-pi``, so that
    beam 0 lies on the array axis as mirror pairing requires.

    Raises
    ------
    ValueError
        If the endpoints are equal, or further apart than a full circle.

    """
    difference = end_rad - start_rad
    if abs(difference) < _ANGLE_TOLERANCE_RAD:
        raise ValueError(
            f"steering_sector_rad ({start_rad}, {end_rad}) is empty; give a full circle as "
            "(-np.pi, np.pi)."
        )
    if abs(difference) > 2 * np.pi + _ANGLE_TOLERANCE_RAD:
        raise ValueError(
            f"steering_sector_rad ({start_rad}, {end_rad}) spans more than a full circle."
        )
    if abs(abs(difference) - 2 * np.pi) <= _ANGLE_TOLERANCE_RAD:
        return -np.pi, 2 * np.pi
    width = difference if difference > 0 else difference + 2 * np.pi
    return float(_wrap_to_pi(np.array(start_rad))), float(width)


class SteeringCalculator(Base):
    r"""Compute steering delays for a horizontal sensor array over a sector of bearings.

    The calculator builds its steering grid itself: ``num_beams`` angles :math:`\psi_k`
    across ``steering_sector_rad``, spaced as ``spacing`` selects. A partial sector includes
    both endpoints; a full circle does not repeat its start, since ``-pi`` and ``pi`` point
    the same way.

    ``frame`` sets what the angles are measured from. In the ``"array"`` frame it is the
    array's forward direction :math:`\alpha(t)`, from its last sensor towards sensor 0, and
    each scan's bearings are

    .. math::

        \phi_k(t) = \alpha(t) + \psi_k,

    so the grid turns with the array and every beam keeps its angle to the array axis. In the
    ``"world"`` frame they are measured from +x and :math:`\phi_k = \psi_k` stays fixed while
    the array turns. :meth:`steering_bearings` returns the bearings for a scan.

    With ``spacing="uniform"`` the beams are evenly spaced in angle, starting at sector
    start :math:`\psi_0` with width :math:`W`:

    .. math::

        \psi_k = \psi_0 + k \frac{W}{N - 1} \;\text{(partial sector)}, \qquad
        \psi_k = -\pi + k \frac{2\pi}{N} \;\text{(full circle)}.

    With ``spacing="sine"`` they are evenly spaced in :math:`u = \sin\theta`, the sine of the
    angle from broadside :math:`\beta` (the direction perpendicular to the array axis, on the
    sector's side of the array):

    .. math::

        u_k = u_0 + k \frac{u_{N-1} - u_0}{N - 1}, \qquad \psi_k = \beta + \arcsin u_k.

    A line array's beam pattern depends on direction only through :math:`u`, so on a sine grid
    every beam's mainlobe spans the same number of beams. Neighbouring beams are then equally
    correlated at every bearing, so a CFAR (constant false alarm rate) detector's noise
    statistics are the same for every beam. On a uniform grid the mainlobe widens as
    :math:`1 / \cos\theta` towards endfire, and beams there become near-copies of each other.
    This is the u-space description of [1]_ (Ch. 2).

    Assumptions
    -----------
    - The array is a straight horizontal line; delays use each sensor's actual position, but
      the array frame, the sine grid and mirror pairing take the array axis from the two end
      sensors. Mid-turn this approximates a bent array's orientation.
    - Sensor 0 is the front of the array, nearest the tow point.
    - The sensor positions are known exactly: the delays use the platform's true geometry.
    - For ``spacing="sine"`` in the ``"world"`` frame, the array heading is constant: the grid
      is built from the axis at the first call and a later axis change raises.
    - For ``spacing="sine"``, the sector lies within one side of the array (at most endfire to
      endfire).

    References
    ----------
    .. [1] Van Trees, H. L. "Optimum Array Processing: Part IV of Detection, Estimation, and
           Modulation Theory." Wiley-Interscience, New York, 2002. ISBN 978-0-471-09390-9.

    """

    ssp: SoundSpeedProfile = Property(
        doc="Sound speed profile for calculating delays",
    )
    steering_sector_rad: tuple[float, float] = Property(
        doc="Sector to steer, (start, end) in radians, running anticlockwise from start to "
        "end. In the 'array' frame the angles are anticlockwise from the array's forward "
        "direction, so port is (0, np.pi) and starboard is (-np.pi, 0); in the 'world' frame "
        "they are anticlockwise from +x. Endpoints that point the same way but differ "
        "numerically, such as (-np.pi, np.pi), give a full circle.",
    )
    num_beams: int = Property(
        doc="Number of beams across the sector, at least 2.",
    )
    frame: Literal["array", "world"] = Property(
        default="array",
        doc="What steering_sector_rad is measured from: 'array' uses the array's forward "
        "direction, so the grid turns with the array; 'world' uses +x, so the grid stays "
        "fixed while the array turns.",
    )
    spacing: Literal["uniform", "sine"] = Property(
        default="uniform",
        doc="'uniform' spaces beams evenly in angle; 'sine' spaces them evenly in the sine "
        "of the angle from broadside, for a sector within one side of the array.",
    )
    mirror_half_plane: bool = Property(
        default=False,
        doc="Exploit a linear array's mirror symmetry about its axis: the steering delays are "
        "identical either side of it, so only one half-plane needs to be steered directly. "
        "When True, calculate returns delays for about half the grid, and mirror_plan returns "
        "the bookkeeping to expand a beamformer's output back to the full grid via "
        "'Beamformer.expand_mirrored'. Requires a full-circle sector with uniform spacing. "
        "In the 'array' frame the expanded output is exact while the array is straight. In "
        "the 'world' frame it is exact only while the array is straight and its axis falls on "
        "the grid (a multiple of 2*pi/num_beams). Otherwise every beam is steered at its "
        "reported bearing plus the axis's offset from the nearest grid bearing, up to half a "
        "beam spacing. In either frame a bent array (e.g. during a turn) adds a further "
        "approximation.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Validate the sector and build the grid and any mirror bookkeeping.

        Raises
        ------
        TypeError
            If a removed parameter (``steering_azimuths_rad``) is passed.
        ValueError
            If the sector is empty or wider than a full circle, ``num_beams`` is below 2,
            ``spacing`` or ``frame`` is unknown, a sine-spaced sector is wider than a
            half-plane or, in the ``"array"`` frame, crosses the array axis, or
            ``mirror_half_plane`` is set without a uniform full circle.

        """
        # Caught before Stone Soup's Base sees them: an unknown kwarg there surfaces as
        # "unexpected keyword argument", which names the mistake but not its replacement.
        for removed, guidance in _REMOVED_STEERING_KWARGS.items():
            if removed in kwargs:
                raise TypeError(
                    f"{removed} is no longer a parameter of SteeringCalculator. {guidance}"
                )

        super().__init__(*args, **kwargs)

        if len(self.steering_sector_rad) != 2:
            raise ValueError(
                f"steering_sector_rad must be (start, end), got {self.steering_sector_rad!r}"
            )
        if self.num_beams < 2:
            raise ValueError(f"num_beams ({self.num_beams}) must be >= 2")
        if self.spacing not in ("uniform", "sine"):
            raise ValueError(f"spacing ({self.spacing!r}) must be 'uniform' or 'sine'")
        if self.frame not in ("array", "world"):
            raise ValueError(f"frame ({self.frame!r}) must be 'array' or 'world'")

        start_rad, end_rad = (float(angle) for angle in self.steering_sector_rad)
        self._sector_start_rad, self._sector_width_rad = _normalise_sector(start_rad, end_rad)
        is_full_circle = self._sector_width_rad == 2 * np.pi

        if self.spacing == "sine" and self._sector_width_rad > np.pi + _ANGLE_TOLERANCE_RAD:
            raise ValueError(
                "A sine-spaced sector must lie within one side of the array, so it can be at "
                f"most pi wide; steering_sector_rad {self.steering_sector_rad} is "
                f"{self._sector_width_rad:.4g} rad wide."
            )
        if self.mirror_half_plane and not (is_full_circle and self.spacing == "uniform"):
            raise ValueError(
                "mirror_half_plane requires a full-circle steering_sector_rad, e.g. "
                "(-np.pi, np.pi), with spacing='uniform'."
            )

        # The grid of angles in the calculator's own frame. In the array frame the axis is 0 by
        # definition, so even a sine grid is fixed now; in the world frame it waits for the
        # array axis.
        self._grid: FloatArray | None = None
        if self.spacing == "uniform":
            self._grid = self._uniform_grid(is_full_circle)
        elif self.frame == "array":
            self._grid = self._sine_grid(0.0)
        self._sine_axis_rad: float | None = None

        self._mirror_idx: IntArray | None = None
        self._primary_mask: BoolArray | None = None
        self._beam_spacing_rad: float | None = None
        if self.mirror_half_plane:
            n = self.num_beams
            self._beam_spacing_rad = 2 * np.pi / n
            self._mirror_idx = (-np.arange(n)) % n
            self._primary_mask = np.arange(n) <= self._mirror_idx

    def _uniform_grid(self, is_full_circle: bool) -> FloatArray:
        """Bearings evenly spaced across the sector."""
        if is_full_circle:
            return np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False)
        start = self._sector_start_rad
        return _wrap_to_pi(np.linspace(start, start + self._sector_width_rad, self.num_beams))

    def _sine_grid(self, axis_rad: float) -> FloatArray:
        """Bearings evenly spaced in the sine of the angle from broadside, for this axis.

        Raises
        ------
        ValueError
            If the sector crosses the array axis.

        """
        start = self._sector_start_rad
        width = self._sector_width_rad
        midpoint = start + width / 2
        # Broadside on the sector's side: whichever normal to the axis is nearer its middle.
        broadside = min(
            (axis_rad + np.pi / 2, axis_rad - np.pi / 2),
            key=lambda normal: abs(float(_wrap_to_pi(np.array(midpoint - normal)))),
        )
        start_from_broadside = float(_wrap_to_pi(np.array(start - broadside)))
        end_from_broadside = start_from_broadside + width
        limit = np.pi / 2 + _ANGLE_TOLERANCE_RAD
        if start_from_broadside < -limit or end_from_broadside > limit:
            raise ValueError(
                f"steering_sector_rad {self.steering_sector_rad} crosses the array axis "
                f"({axis_rad:.4g} rad); a sine-spaced sector must lie within one side of it."
            )
        u = np.linspace(np.sin(start_from_broadside), np.sin(end_from_broadside), self.num_beams)
        # The tolerance lets u stray just past +-1 at endfire; arcsin would return NaN there.
        return _wrap_to_pi(broadside + np.arcsin(np.clip(u, -1.0, 1.0)))

    def steering_bearings(self, platform: "PlatformState | None" = None) -> FloatArray:
        """Return a scan's steering bearings, in radians anticlockwise from +x, in ``[-pi, pi)``.

        Parameters
        ----------
        platform : PlatformState, optional
            The platform state for the scan. Needed in the ``"array"`` frame, whose bearings
            follow the array's forward direction, and for a ``"world"``-frame sine grid, which
            the first call builds from the array axis; later calls check the axis has not
            moved.

        Returns
        -------
        FloatArray
            ``num_beams`` bearings, in beam order.

        Raises
        ------
        ValueError
            If ``platform`` is missing when needed. For a ``"world"``-frame sine grid, also
            if the sector crosses the array axis or the axis has moved since the grid was
            built.

        """
        if self.frame == "world" and self._grid is not None and self.spacing == "uniform":
            return self._grid.copy()
        if platform is None:
            reason = (
                "frame='array' needs a platform to read the array's forward direction from"
                if self.frame == "array"
                else "spacing='sine' needs a platform to read the array axis from"
            )
            raise ValueError(f"{reason}; pass the platform state for the scan.")
        if self.frame == "array" and self._grid is not None:
            return _wrap_to_pi(self._array_forward_rad(platform) + self._grid)

        axis_rad = self._array_axis_rad(platform)
        if self._grid is None or self._sine_axis_rad is None:
            self._grid = self._sine_grid(axis_rad)
            self._sine_axis_rad = axis_rad
            return self._grid.copy()

        # The axis direction is ambiguous by pi (see _array_axis_rad), so compare modulo pi.
        moved = abs(float(_wrap_to_pi(np.array(2 * (axis_rad - self._sine_axis_rad))))) / 2
        tolerance = _SINE_AXIS_TOLERANCE_FRACTION * float(np.min(np.abs(np.diff(self._grid))))
        if moved > tolerance:
            raise ValueError(
                f"The array axis has turned by {moved:.3g} rad since the sine-spaced grid was "
                f"built (tolerance {tolerance:.3g} rad). spacing='sine' assumes a constant "
                "heading; use spacing='uniform' for a turning array."
            )
        return self._grid.copy()

    @staticmethod
    def _array_axis_rad(platform: "PlatformState") -> float:
        """Return the array's instantaneous line orientation, from its two end sensors.

        Wrapped to ``[-pi, pi)``; since the mirror reflection ``2 * axis - theta`` is
        unaffected by a 180 deg ambiguity in which "end" of the array is used, either
        sensor ordering gives an equally valid axis.
        """
        sensor_positions = platform.array.state_vector
        endpoints_xy = sensor_positions[:2, [0, -1]]
        dx, dy = endpoints_xy[:, 1] - endpoints_xy[:, 0]
        # arctan2 alone gives (-pi, pi]; the modulo moves +pi to -pi.
        return float((np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _array_forward_rad(platform: "PlatformState") -> float:
        """Return the array's forward direction, from its last sensor towards sensor 0.

        This is the reference every ``"array"``-frame angle is measured from. Unlike
        :meth:`_array_axis_rad` it has no pi ambiguity, which matters here because
        reversing it would swap port and starboard.
        """
        sensor_positions = platform.array.state_vector
        dx, dy = sensor_positions[:2, 0] - sensor_positions[:2, -1]
        return float(_wrap_to_pi(np.array(np.arctan2(dy, dx))))

    def _delays_for_azimuths(
        self, platform: "PlatformState", azimuths_rad: FloatArray
    ) -> FloatArray:
        """Compute steering delays for an explicit, arbitrary set of azimuths.

        This method assumes the platform has an `array` attribute which is an object with
        `state_vector` and `ref_state_vector` attributes, such as the one configured by
        `TowedArrayPlatform`.

        Parameters
        ----------
        platform : PlatformState
            The platform containing the sensor array.
        azimuths_rad : FloatArray
            Azimuth angles to steer, in radians.

        Returns
        -------
        FloatArray
            Steering-delay matrix in seconds with shape
            ``(len(azimuths_rad), num_sensors)``.

        """
        # Get sensor positions - these are 3D positions [x, y, z] for each sensor
        sensor_positions = platform.array.state_vector  # Shape: (3, num_sensors)

        # Center the array relative to the reference sensor
        reference_position = platform.array.ref_state_vector  # Shape: (3, 1)
        sensor_positions_relative = sensor_positions - reference_position

        # Calculate the 2D direction vectors for each steering direction
        # Elevation = 0 for horizontal array, so only x-y components
        direction_vectors = np.array(
            [
                np.cos(azimuths_rad),  # x component
                np.sin(azimuths_rad),  # y component
                np.zeros_like(azimuths_rad),  # z component (always 0)
            ]
        )  # Shape: (3, num_directions)

        # Calculate the projection of each sensor position onto each direction vector
        # This gives the distance along the direction of arrival for each sensor
        distances = np.dot(
            direction_vectors.T, sensor_positions_relative
        )  # Shape: (num_directions, num_sensors)

        # Get average sound speed at array depth
        array_depth = sensor_positions[2, 0]  # z-coordinate of first sensor
        sound_speed = self.ssp.calculate(array_depth)

        # Convert distances to time delays
        # Negative sign because we want delays to ADD to make signals arrive in-phase
        return -distances / sound_speed

    def calculate(self, platform: "PlatformState") -> FloatArray:
        """Calculate per-direction per-sensor steering delays.

        This method assumes the platform has an `array` attribute which is an object with
        `state_vector` and `ref_state_vector` attributes, such as the one configured by
        `TowedArrayPlatform`.

        Parameters
        ----------
        platform : PlatformState
            The platform containing the sensor array.

        Returns
        -------
        FloatArray
            Steering-delay matrix in seconds. Shape ``(num_directions, num_sensors)``,
            where ``num_directions`` is ``num_beams`` normally, or roughly half of it when
            ``mirror_half_plane`` is set. Use with :meth:`mirror_plan` and
            ``Beamformer.expand_mirrored`` to recover the full grid.

        Raises
        ------
        ValueError
            As :meth:`steering_bearings`, when it raises for this scan.

        """
        bearings = self.steering_bearings(platform)
        if not self.mirror_half_plane or self._primary_mask is None:
            return self._delays_for_azimuths(platform, bearings)
        if self.frame == "array":
            # Beam 0 points aft along the axis, so each beam's mirror partner is its exact
            # reflection and no roll back onto the grid is needed.
            return self._delays_for_azimuths(platform, bearings[self._primary_mask])

        axis_rad = self._array_axis_rad(platform)
        axis_centred_grid = (axis_rad + bearings + np.pi) % (2 * np.pi) - np.pi
        return self._delays_for_azimuths(platform, axis_centred_grid[self._primary_mask])

    def mirror_plan(self, platform: "PlatformState") -> MirrorPlan:
        """Return the mirror bookkeeping to expand a half-plane beamformer output.

        Call this once per timestep alongside :meth:`calculate`, using the same
        ``platform`` -- the array's axis is recomputed from its current geometry each
        time, since a bent (mid-turn) array's axis is only a straight-line approximation.

        Parameters
        ----------
        platform : PlatformState
            The platform containing the sensor array.

        Returns
        -------
        MirrorPlan
            Bookkeeping for ``Beamformer.expand_mirrored``.

        Raises
        ------
        RuntimeError
            If ``mirror_half_plane`` is not set on this calculator.

        """
        if (
            self._primary_mask is None
            or self._mirror_idx is None
            or self._beam_spacing_rad is None
        ):
            raise RuntimeError(
                "mirror_plan() requires mirror_half_plane=True on this SteeringCalculator"
            )

        roll_shift = 0
        if self.frame == "world":
            axis_rad = self._array_axis_rad(platform)
            roll_shift = int(np.round(axis_rad / self._beam_spacing_rad))
        return MirrorPlan(
            primary_mask=self._primary_mask,
            mirror_idx=self._mirror_idx,
            roll_shift=roll_shift,
        )
