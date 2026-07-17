"""Steering-delay geometry for horizontal sensor arrays."""

from typing import TYPE_CHECKING

import numpy as np
from stonesoup.base import Base, Property

from ..models.environment import SoundSpeedProfile
from .base import BoolArray, FloatArray, IntArray, MirrorPlan

if TYPE_CHECKING:
    from stonesoup.platform.base import Platform


class SteeringCalculator(Base):
    """Compute steering delays for a horizontal sensor array."""

    ssp: SoundSpeedProfile = Property(
        doc="Sound speed profile for calculating delays",
    )
    steering_azimuths_rad: FloatArray = Property(
        doc="Azimuth angles for steering, in radians",
    )
    mirror_half_plane: bool = Property(
        default=False,
        doc="Exploit a linear array's mirror symmetry along array axis."
        "for a linear array the steering delays are identical about this axis,"
        "so only one half-plane needs to be steered directly. When True, "
        "calculate returns delays for aprox half of steering_azimuths_rad,"
        "and mirror_plan returns the bookkeeping to expand a beamformer's "
        "output back to the full grid via 'Beamformer.expand_mirrored'. "
        "Requires 'steering_azimuths_rad' to be a uniform, full-circle grid,"
        "e.g. np.linspace(-pi, pi, N, endpoint=False). This is Exact while "
        "the array is linear (straight) and only approximate while it bends "
        "(e.g. during a turn).",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the steering calculator and precompute mirror bookkeeping if needed.

        Raises
        ------
        ValueError
            If ``mirror_half_plane`` is set and ``steering_azimuths_rad`` is not a
            uniform, full-circle grid.

        """
        super().__init__(*args, **kwargs)

        self._mirror_idx: IntArray | None = None
        self._primary_mask: BoolArray | None = None
        self._beam_spacing_rad: float | None = None

        if self.mirror_half_plane:
            self._validate_uniform_grid(self.steering_azimuths_rad)
            n = len(self.steering_azimuths_rad)
            self._beam_spacing_rad = 2 * np.pi / n
            self._mirror_idx = (-np.arange(n)) % n
            self._primary_mask = np.arange(n) <= self._mirror_idx

    @staticmethod
    def _validate_uniform_grid(azimuths_rad: FloatArray) -> None:
        """Check that ``azimuths_rad`` is a uniform grid spanning the full circle.

        Raises
        ------
        ValueError
            If there are fewer than 2 directions, or the directions are not evenly
            spaced around the full circle (e.g. ``np.linspace(-pi, pi, N,
            endpoint=False)``), regardless of input order.

        """
        n = len(azimuths_rad)
        if n < 2:
            raise ValueError("mirror_half_plane requires at least 2 steering directions")

        sorted_azimuths = np.sort(azimuths_rad)
        gaps = np.diff(sorted_azimuths)
        wrap_gap = (sorted_azimuths[0] + 2 * np.pi) - sorted_azimuths[-1]
        all_gaps = np.append(gaps, wrap_gap)
        expected_gap = 2 * np.pi / n

        if not np.allclose(all_gaps, expected_gap, atol=1e-9):
            raise ValueError(
                "mirror_half_plane requires a uniform, full-circle steering grid (e.g. "
                "np.linspace(-pi, pi, N, endpoint=False)); steering_azimuths_rad is not "
                "evenly spaced around the full circle."
            )

    @staticmethod
    def _array_axis_rad(platform: "Platform") -> float:
        """Return the array's instantaneous line orientation, from its two end sensors.

        Wrapped to ``[-pi, pi)``; since the mirror reflection ``2 * axis - theta`` is
        unaffected by a 180 deg ambiguity in which "end" of the array is used, either
        sensor ordering gives an equally valid axis.
        """
        sensor_positions = platform.array.state_vector
        endpoints_xy = sensor_positions[:2, [0, -1]]
        dx, dy = endpoints_xy[:, 1] - endpoints_xy[:, 0]
        return float(np.arctan2(dy, dx))

    def _delays_for_azimuths(self, platform: "Platform", azimuths_rad: FloatArray) -> FloatArray:
        """Compute steering delays for an explicit, arbitrary set of azimuths.

        This method assumes the platform has an `array` attribute which is an object with
        `state_vector` and `ref_state_vector` attributes, such as the one configured by
        `TowedArrayPlatform`.

        Parameters
        ----------
        platform : Platform
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

    def calculate(self, platform: "Platform") -> FloatArray:
        """Calculate per-direction per-sensor steering delays.

        This method assumes the platform has an `array` attribute which is an object with
        `state_vector` and `ref_state_vector` attributes, such as the one configured by
        `TowedArrayPlatform`.

        Parameters
        ----------
        platform : Platform
            The platform containing the sensor array.

        Returns
        -------
        FloatArray
            Steering-delay matrix in seconds. Shape ``(num_directions, num_sensors)``,
            where ``num_directions`` is ``len(steering_azimuths_rad)`` normally, or
            roughly half of it when ``mirror_half_plane`` is set. Use with
            :meth:`mirror_plan` and ``Beamformer.expand_mirrored`` to recover the full
            grid.

        """
        if not self.mirror_half_plane or self._primary_mask is None:
            return self._delays_for_azimuths(platform, self.steering_azimuths_rad)

        axis_rad = self._array_axis_rad(platform)
        axis_centred_grid = (axis_rad + self.steering_azimuths_rad + np.pi) % (2 * np.pi) - np.pi
        return self._delays_for_azimuths(platform, axis_centred_grid[self._primary_mask])

    def mirror_plan(self, platform: "Platform") -> MirrorPlan:
        """Return the mirror bookkeeping to expand a half-plane beamformer output.

        Call this once per timestep alongside :meth:`calculate`, using the same
        ``platform`` -- the array's axis is recomputed from its current geometry each
        time, since a bent (mid-turn) array's axis is only a straight-line approximation.

        Parameters
        ----------
        platform : Platform
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

        axis_rad = self._array_axis_rad(platform)
        roll_shift = int(np.round(axis_rad / self._beam_spacing_rad))
        return MirrorPlan(
            primary_mask=self._primary_mask,
            mirror_idx=self._mirror_idx,
            roll_shift=roll_shift,
        )
