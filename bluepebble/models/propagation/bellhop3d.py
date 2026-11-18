"""Eigenray propagation with Bellhop3D in arrivals mode: 3D (``--3D``) or N x 2D (``--Nx2D``)."""

import math
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from stonesoup.base import Property

from .bellhop import _DEPTH_FORMAT, BellhopAcousticPropagationModel, Eigenray, FloatArray, _warn

#: Bellhop3D's codes for joining the bathymetry grid, by ``bathymetry_interpolation``.
_GRID_INTERPOLATIONS = {"curvilinear": "C", "linear": "R"}


class Bellhop3DAcousticPropagationModel(BellhopAcousticPropagationModel):
    r"""Ray propagation computed by Bellhop3D, with every sensor at its own bearing.

    A subclass of :class:`~bluepebble.models.propagation.BellhopAcousticPropagationModel` that
    runs ``bellhopcxx`` in 3D mode instead of 2D. Rays are traced in three dimensions, so they
    can leave their launch bearing, for example by reflecting off a seabed that slopes across
    it, which no vertical-plane model can follow. Each run writes an environment file
    with the source at its world ``(x, y)``, a receiver grid of every distinct bearing, depth
    and horizontal range from the source, and a gridded bathymetry (``.bty``) covering the
    square of half-width the beam box around the source, sampled ``n_bathy_samples`` times
    along each axis. Transfer functions, :meth:`propagate` and :meth:`propagate_spectrum`
    behave as in the 2D model.

    Parameters
    ----------
    far_field : bool, optional
        If True, trace to the array reference point only and add plane-wave delays across
        the array, as in the 2D model. By default (False) every sensor is a receiver in one
        run, at its own bearing, range and depth.
    separate_planes : bool, optional
        Not applicable: every sensor's rays are traced at its own bearing. True is ignored,
        with a warning.
    receiver_geometry : str, optional
        Deprecated; use ``far_field``. ``"reference"`` and ``"per_sensor"`` are the default,
        and ``"far_field"`` is ``far_field=True``.
    bearing_beam_spacing_deg : float, optional
        Spacing of the beams in the horizontal (bearing) fan, in degrees.
        Geometric beams put a receiver that falls between two bearings slightly early, by
        roughly the square of the spacing: against the 2D model over a flat bottom at
        1.5-2.5 km, first arrivals were up to 12 us early at 0.25 degrees, 2 us at 0.1 and
        0.6 us at 0.05. The run time grows in inverse proportion to the spacing.
    bearing_margin_deg : float, optional
        The bearing fan spans the receivers' bearings widened by this much on
        each side, so paths that leave their launch bearing still reach them.
    bearing_resolution_deg : float, optional
        Receiver bearings are rounded to this step, which limits the size of the receiver
        grid. A sensor keeps its exact range and depth, so the rounding only changes which
        environment it sees, by at most half a step sideways.
    nx2d : bool, optional
        If True, run Bellhop3D's N x 2D mode (``--Nx2D``) instead of 3D: one 2D trace in the
        vertical plane of each receiver bearing, over the seabed along that bearing, sliced
        from the bathymetry grid. Rays stay in their plane, as in the 2D model with
        ``separate_planes``, so paths that leave it are missed; the bearing fan is unused.
        Default False.

    Other parameters are those of
    :class:`~bluepebble.models.propagation.BellhopAcousticPropagationModel`, with
    ``num_beams`` and ``beam_angle_limits_deg`` setting the elevation fan.

    Assumptions
    -----------
    - The sound speed varies with depth only, as in the 2D model; only the bathymetry varies
      with ``(x, y)``.
    - The bathymetry is a rectilinear grid with spacing ``2 * box / (n_bathy_samples - 1)``;
      features smaller than that are smoothed. With ``bathymetry_interpolation="curvilinear"``
      (the default) Bellhop3D takes the seabed's curvature into account between grid points;
      with ``"linear"`` it joins them with flat facets, which over a curved seabed such as a
      seamount doesn't converge as points are added.
    - The bearing fan has to be dense enough to bracket every receiver: geometric beams only
      reach receivers that lie between neighbouring rays.

    """

    bearing_beam_spacing_deg: float = Property(
        default=0.1, doc="Spacing of the beams in the bearing fan, in degrees"
    )
    bearing_margin_deg: float = Property(
        default=5.0,
        doc="How far the bearing fan extends beyond the receivers' bearings, in degrees",
    )
    bearing_resolution_deg: float = Property(
        default=1e-3, doc="Step to which receiver bearings are rounded, in degrees"
    )
    nx2d: bool = Property(
        default=False,
        doc="Run Bellhop3D's N x 2D mode: one 2D trace per receiver bearing, instead of 3D",
    )

    _legacy_geometries = {
        "reference": (False, False),
        "per_sensor": (False, False),
        "far_field": (True, False),
    }

    def _resolve_planes(self) -> None:
        """Ignore ``separate_planes``: every sensor is traced at its own bearing in 3D."""
        if self.separate_planes:
            _warn(
                "separate_planes does not apply to Bellhop3D: every sensor's rays are traced "
                "at its own bearing. Ignoring separate_planes=True.",
                UserWarning,
            )
            self.separate_planes = False

    def _run_arrivals(
        self,
        source_position: ArrayLike,
        receiver_positions: FloatArray,
        plane_towards_xy: FloatArray,
        frequency_hz: float,
    ) -> list[list[Eigenray]]:
        """Run Bellhop3D once for one source and several receivers.

        Parameters
        ----------
        source_position : ArrayLike
            World-frame source position, z negative below the surface.
        receiver_positions : FloatArray
            World-frame receiver positions, shape ``(3, num_receivers)``.
        plane_towards_xy : FloatArray
            Unused: each receiver is at its own bearing.
        frequency_hz : float
            Frequency of the run in Hz.

        Returns
        -------
        list[list[Eigenray]]
            The arrivals at each receiver, in order.

        """
        source = np.asarray(source_position, dtype=float).ravel()
        receivers = np.asarray(receiver_positions, dtype=float).reshape(3, -1)
        source_depth_m = -source[2]
        receiver_depths_m = -receivers[2]
        if source_depth_m <= 0 or np.any(receiver_depths_m <= 0):
            raise ValueError(
                "Source and receivers must be below the surface (z < 0); got source z = "
                f"{source[2]} and receiver z = {receivers[2].tolist()}."
            )
        dx, dy = receivers[0] - source[0], receivers[1] - source[1]
        ranges_m = np.hypot(dx, dy)
        if np.any(ranges_m <= 0):
            raise ValueError("Source and receivers must be at different horizontal positions.")

        bearings_deg = _contiguous_bearings_deg(dx, dy)
        step = self.bearing_resolution_deg
        grid_bearings_deg, bearing_index = np.unique(
            np.round(bearings_deg / step) * step, return_inverse=True
        )
        grid_depths_m, depth_index = np.unique(np.round(receiver_depths_m, 4), return_inverse=True)
        grid_ranges_km, range_index = np.unique(np.round(ranges_m / 1000, 6), return_inverse=True)

        box_m = 1.05 * float(ranges_m.max())
        x_m, y_m, bottom_depth_m = self._bathymetry_grid(source[:2], box_m)
        water_depth_m = float(bottom_depth_m.max())

        exe = self._resolve_exe()
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "bp"
            self._write_env_3d(
                stem,
                source,
                grid_depths_m,
                grid_ranges_km,
                grid_bearings_deg,
                box_m,
                water_depth_m,
                frequency_hz,
            )
            self._write_bty_3d(
                stem,
                x_m,
                y_m,
                bottom_depth_m,
                _GRID_INTERPOLATIONS[self.bathymetry_interpolation],
            )
            self._run_bellhopcxx(exe, stem, self.memory, "--Nx2D" if self.nx2d else "--3D")
            blocks = self._parse_arr_3d(stem.with_suffix(".arr"))
        # Blocks run over bearings, then depths, then ranges (fastest).
        num_depths, num_ranges = grid_depths_m.size, grid_ranges_km.size
        return [
            blocks[(b * num_depths + d) * num_ranges + r]
            for b, d, r in zip(
                bearing_index.ravel(), depth_index.ravel(), range_index.ravel(), strict=True
            )
        ]

    def _bathymetry_grid(
        self, source_xy_m: FloatArray, half_width_m: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Sample the water depth on a square grid centred on the source.

        Returns
        -------
        tuple of FloatArray
            The grid's x and y coordinates in metres, and positive water depths in metres
            with shape ``(num_y, num_x)``.

        """
        offsets = np.linspace(-half_width_m, half_width_m, self.n_bathy_samples)
        x_m = source_xy_m[0] + offsets
        y_m = source_xy_m[1] + offsets
        depth_m = np.array(
            [[abs(float(self.bathymetry.get_depth(float(x), float(y)))) for x in x_m] for y in y_m]
        )
        return x_m, y_m, depth_m

    def _write_env_3d(
        self,
        stem: Path,
        source_m: FloatArray,
        receiver_depths_m: FloatArray,
        receiver_ranges_km: FloatArray,
        receiver_bearings_deg: FloatArray,
        box_m: float,
        water_depth_m: float,
        frequency_hz: float,
    ) -> None:
        """Write a Bellhop3D arrivals-mode environment file for a polar receiver grid."""
        low_angle, high_angle = self.beam_angle_limits_deg
        low_bearing, high_bearing = self._bearing_fan_deg(receiver_bearings_deg)
        box_km = box_m / 1000.0
        env = "\n".join(
            [
                "'BellhopArrivals3D'",
                f"{frequency_hz:.4f}",
                *self._medium_lines(water_depth_m),
                "1",
                f"{source_m[0] / 1000:.9f} /",
                "1",
                f"{source_m[1] / 1000:.9f} /",
                "1",
                f"{-source_m[2]:.4f} /",
                str(receiver_depths_m.size),
                " ".join(f"{d:.4f}" for d in receiver_depths_m) + " /",
                str(receiver_ranges_km.size),
                " ".join(f"{r:.6f}" for r in receiver_ranges_km) + " /",
                str(receiver_bearings_deg.size),
                " ".join(f"{b:.6f}" for b in receiver_bearings_deg) + " /",
                # ASCII arrivals, geometric hat beams in Cartesian coordinates, point source,
                # rectilinear receiver grid, then 3D (3) or N x 2D (2) in the sixth column.
                "'AG   2'" if self.nx2d else "'AG   3'",
                str(self.num_beams),
                f"{low_angle} {high_angle} /",
                str(math.ceil((high_bearing - low_bearing) / self.bearing_beam_spacing_deg) + 1),
                f"{low_bearing:.6f} {high_bearing:.6f} /",
                f"0.0 {box_km:.6f} {box_km:.6f} {water_depth_m * 1.05:.2f}",
            ]
        )
        stem.with_suffix(".env").write_text(env)

    def _bearing_fan_deg(self, receiver_bearings_deg: FloatArray) -> tuple[float, float]:
        """Return the bearing fan: the receivers' bearings widened by the margin, at most 360."""
        low = float(receiver_bearings_deg.min()) - self.bearing_margin_deg
        high = float(receiver_bearings_deg.max()) + self.bearing_margin_deg
        if high - low >= 360.0:
            centre = (low + high) / 2
            return centre - 180.0, centre + 180.0
        return low, high

    @staticmethod
    def _write_bty_3d(
        stem: Path, x_m: FloatArray, y_m: FloatArray, depth_m: FloatArray, interpolation: str = "C"
    ) -> None:
        """Write a gridded bathymetry file, one row of depths per y.

        ``interpolation`` is Bellhop3D's code for joining the grid: ``"C"`` curvilinear or
        ``"R"`` rectilinear (flat facets).
        """
        lines = [
            f"'{interpolation}'",
            str(x_m.size),
            " ".join(f"{x / 1000:.9f}" for x in x_m),
            str(y_m.size),
            " ".join(f"{y / 1000:.9f}" for y in y_m),
        ]
        # To the micrometre, as in the 2D file: coarser rounding kinks a sloped seabed.
        lines += [" ".join(f"{d:{_DEPTH_FORMAT}}" for d in row) for row in depth_m]
        stem.with_suffix(".bty").write_text("\n".join(lines) + "\n")

    @staticmethod
    def _parse_arr_3d(arr_path: Path) -> list[list[Eigenray]]:
        """Parse a 3D ASCII arrivals file for one source and a polar receiver grid.

        After the ``'3D'`` line the file holds the frequency; the source x, y and depth
        coordinates and the receiver depths, ranges and bearings, each count first; a
        maximum-arrivals hint; then for every bearing, depth and range (range fastest) a count
        of arrivals followed by one row per arrival::

            amplitude  phase_deg  delay_s  delay_imag_s  src_decl  src_azim  rcv_decl
            rcv_azim  n_top  n_bottom

        Returns
        -------
        list[list[Eigenray]]
            The arrivals at each grid point, in file order.

        """
        with open(arr_path) as f:
            f.readline()  # '3D'
            tokens = f.read().split()
        position = 0

        def take(n: int) -> list[str]:
            nonlocal position
            values = tokens[position : position + n]
            position += n
            return values

        take(1)  # frequency
        counts = []
        for _ in range(6):  # source x, y, depth; receiver depths, ranges, bearings
            count = int(take(1)[0])
            take(count)
            counts.append(count)
        num_sources = counts[0] * counts[1] * counts[2]
        num_points = counts[3] * counts[4] * counts[5]
        receivers: list[list[Eigenray]] = []
        for _ in range(num_sources):
            take(1)  # maximum number of arrivals
            for _ in range(num_points):
                arrivals = []
                for _ in range(int(take(1)[0])):
                    row = take(10)
                    arrivals.append(
                        Eigenray(
                            amplitude=float(row[0]),
                            phase_rad=float(np.radians(float(row[1]))),
                            delay_s=float(row[2]),
                            src_angle_deg=float(row[4]),
                            rcv_angle_deg=float(row[6]),
                            n_surface_bounces=int(float(row[8])),
                            n_bottom_bounces=int(float(row[9])),
                            src_azimuth_deg=float(row[5]),
                            rcv_azimuth_deg=float(row[7]),
                        )
                    )
                receivers.append(arrivals)
        return receivers


def _contiguous_bearings_deg(dx: FloatArray, dy: FloatArray) -> FloatArray:
    """Return bearings anticlockwise from +x, unwrapped around their circular mean.

    Receivers either side of -x then have neighbouring bearings (e.g. 179 and 181 degrees)
    rather than ones 358 degrees apart.
    """
    bearings = np.degrees(np.arctan2(dy, dx))
    centre = np.degrees(np.arctan2(np.sum(dy / np.hypot(dx, dy)), np.sum(dx / np.hypot(dx, dy))))
    return np.asarray(centre + (bearings - centre + 180.0) % 360.0 - 180.0, dtype=float)
