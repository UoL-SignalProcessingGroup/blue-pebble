"""Defines a collection of sound speed profile (SSP) models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

FloatArray: TypeAlias = NDArray[np.float64]
Range1D: TypeAlias = tuple[float, float]
Grid3DResult: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
DepthInput: TypeAlias = float | ArrayLike
SpeedOutput: TypeAlias = float | FloatArray


class SoundSpeedProfile(ABC, Base):
    """Abstract base class for sound speed profile models."""

    @abstractmethod
    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate the sound speed at a given depth.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. Can be positive (oceanographic convention, measured downward from
            surface) or negative (3D coordinate system where surface == 0 and underwater is
            negative z).

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        ...

    def get_3d_grid(
        self,
        x_range: Range1D,
        y_range: Range1D,
        z_range: Range1D,
        x_res: float = 5000.0,
        y_res: float = 5000.0,
        z_res: float = 100.0,
    ) -> Grid3DResult:
        """Get a 3D grid representation of the sound speed profile.

        Parameters
        ----------
        x_range : Range1D
            Tuple of (x_min, x_max) in meters.
        y_range : Range1D
            Tuple of (y_min, y_max) in meters.
        z_range : Range1D
            Tuple of (z_min, z_max) in meters (negative depths).
        x_res : float, optional
            Grid resolution in x direction in meters (default 5000.0).
        y_res : float, optional
            Grid resolution in y direction in meters (default 5000.0).
        z_res : float, optional
            Grid resolution in z direction in meters (default 100.0).

        Returns
        -------
        Grid3DResult
            ``(x_grid, y_grid, z_grid, c_grid)`` where
            - ``x_grid`` : 1D array of x coordinates
            - ``y_grid`` : 1D array of y coordinates
            - ``z_grid`` : 1D array of z coordinates (negative depths)
            - ``c_grid`` : 3D array of sound speeds, flattened in C order

        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # Create grid points
        x_points = int((x_max - x_min) / x_res) + 1
        y_points = int((y_max - y_min) / y_res) + 1
        z_points = int(abs(z_max - z_min) / z_res) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))
        z_grid = np.linspace(z_min, z_max, max(2, z_points))

        # Calculate sound speed at each depth (z values are negative)
        c_at_depths = np.array([self.calculate(z) for z in z_grid])

        # Create 3D grid by tiling the 1D profile across x and y
        # Shape: (nx, ny, nz)
        c_grid_3d = np.tile(c_at_depths, (len(x_grid), len(y_grid), 1))

        # Flatten in C order (row-major) as expected by rtrs
        c_grid_flat = c_grid_3d.flatten(order="C")

        return x_grid, y_grid, z_grid, c_grid_flat

    def _calc_temperature(self, depth: DepthInput) -> SpeedOutput:
        """Calculate ocean temperature based on vertical variation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Temperature in degrees Celsius.

        """
        depth_array = np.asarray(depth, dtype=float)
        temperature = 10 * (1 - np.tanh((depth_array - 100.0) / 50.0)) + 2.0
        if depth_array.ndim == 0:
            return float(temperature)
        return np.asarray(temperature, dtype=float)

    def _calc_salinity(self, depth: DepthInput) -> SpeedOutput:
        """Calculate ocean salinity model based on vertical variation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Salinity in practical salinity units (PSU).

        """
        depth_array = np.asarray(depth, dtype=float)
        salinity = 0.5 * (1 - np.tanh((depth_array - 200.0) / 100.0)) + 35.0
        if depth_array.ndim == 0:
            return float(salinity)
        return np.asarray(salinity, dtype=float)


class Constant(SoundSpeedProfile):
    """Constant sound speed profile model.

    This model assumes a uniform sound speed throughout the water column.

    Attributes
    ----------
    speed : float
        Constant sound speed in m/s.

    """

    speed: float = Property(default=1500.0, doc="Constant sound speed in m/s")

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Return the constant sound speed.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (not used in this model).

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        depth_array = np.asarray(depth)
        if depth_array.ndim == 0:
            return float(self.speed)
        return np.full(depth_array.shape, self.speed, dtype=float)


class Linear(SoundSpeedProfile):
    """Linear sound speed profile model.

    This model assumes that the sound speed varies linearly with depth.

    Attributes
    ----------
    surface_speed : float
        Sound speed at the surface in m/s.
    gradient : float
        Sound speed gradient in s^-1 (change per meter).

    """

    surface_speed: float = Property(default=1500.0, doc="Sound speed at the surface in m/s")
    gradient: float = Property(
        default=0.017, doc="Sound speed gradient in s^-1 (change per meter)"
    )

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using a linear profile.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)
        c = self.surface_speed + self.gradient * depth_positive
        return c


class Arctan(SoundSpeedProfile):
    """Arctan sound speed profile model.

    This model describes the sound speed profile using an arctangent function,
    which can represent a smooth transition in sound speed with depth.

    Attributes
    ----------
    surface_speed : float
        The speed of sound at the surface in m/s. Defaults to 1500.0 m/s.
    mid_depth : float
        The depth at which the sound speed transition occurs in meters. Defaults to 1000.0 m.
    steepness : float
        The steepness of the transition. Higher values result in a sharper transition.
        Defaults to 0.005.

    """

    surface_speed: float = Property(default=1500.0, doc="Speed of sound at the surface in m/s")
    mid_depth: float = Property(
        default=1000.0, doc="Depth at which sound speed transition occurs in meters"
    )
    steepness: float = Property(default=0.005, doc="Steepness of the transition")

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the arctan profile.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)
        c = self.surface_speed + 50.0 * np.arctan(
            self.steepness * (depth_positive - self.mid_depth)
        )
        return c


class Munk(SoundSpeedProfile):
    """Munk sound speed profile model.

    This model describes the sound speed profile using an analytical equation proposed by Walter
    Munk. It is characterised by a deep sound channel axis and is widely used in ocean acoustics.

    Attributes
    ----------
    surface_speed : float
        The speed of sound at the surface in m/s. Defaults to 1500.0 m/s.

    """

    surface_speed: float = Property(default=1500.0, doc="Speed of sound at the surface in m/s")

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the Munk equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)

        zt = 2.0 * (depth_positive - 1300.0) / 1300.0
        c = self.surface_speed * (1.0 + 0.00737 * (zt - 1.0 + np.exp(-zt)))
        return c


class Mackenzie(SoundSpeedProfile):
    """Mackenzie sound speed profile model.

    This model calculates the sound speed using the nine-term Mackenzie equation, which is an
    empirical formula based on temperature, salinity, and depth. This implementation uses internal
    models for temperature and salinity as a function of depth.
    """

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the Mackenzie nine-term equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)

        temp = self._calc_temperature(depth_positive)
        salt = self._calc_salinity(depth_positive)
        c = (
            1448.96
            + 4.591 * temp
            - 5.304e-2 * temp**2
            + 2.374e-4 * temp**3
            + 1.340 * (salt - 35)
            + 1.630e-2 * depth_positive
            + 1.675e-7 * depth_positive**2
            - 1.025e-2 * temp * (salt - 35)
            - 7.139e-13 * temp * depth_positive**3
        )
        return c


class LeroyCopernicusSoundSpeedProfile(SoundSpeedProfile):
    """SSP model built from Copernicus temperature/salinity using the Leroy equation.

    Notes
    -----
    - Copernicus depth is expected in oceanographic convention (``+z`` downward).
    - ``calculate`` accepts Nereus depth inputs and internally uses ``abs(depth)``.
    - ``get_3d_grid`` returns ``z_grid`` in RTRS convention (``+z`` downward).

    """

    temperature_file_path: str = Property(doc="Path to Copernicus temperature NetCDF file")
    salinity_file_path: str = Property(doc="Path to Copernicus salinity NetCDF file")
    reference_lat_deg: float | None = Property(
        default=None,
        doc="Reference latitude for local x/y conversion. Defaults to dataset midpoint.",
    )
    reference_lon_deg: float | None = Property(
        default=None,
        doc="Reference longitude for local x/y conversion. Defaults to dataset midpoint.",
    )
    fill_speed_m_s: float = Property(
        default=1480.0,
        doc="Fallback fill value for columns with no finite Copernicus values.",
    )

    def _load_data(self) -> None:
        """Load Copernicus T/S fields and precompute Leroy sound speed on native grids."""
        temp_path = Path(self.temperature_file_path)
        sal_path = Path(self.salinity_file_path)
        if not temp_path.exists():
            raise FileNotFoundError(f"Copernicus temperature file not found: {temp_path}")
        if not sal_path.exists():
            raise FileNotFoundError(f"Copernicus salinity file not found: {sal_path}")

        try:
            import netCDF4 as nc
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for LeroyCopernicusSoundSpeedProfile. "
                "Install with `pip install netCDF4`."
            ) from exc

        with nc.Dataset(temp_path, "r") as ds_t:
            t = self._to_float_with_nan(ds_t.variables["thetao"][0, :, :, :])
            self._z_m = np.asarray(ds_t.variables["depth"][:], dtype=float)
            self._lat_deg = np.asarray(ds_t.variables["latitude"][:], dtype=float)
            self._lon_deg = np.asarray(ds_t.variables["longitude"][:], dtype=float)

        with nc.Dataset(sal_path, "r") as ds_s:
            s = self._to_float_with_nan(ds_s.variables["so"][0, :, :, :])

        if t.shape != s.shape:
            raise ValueError(
                "Copernicus temperature and salinity arrays must have matching shapes."
            )
        if t.ndim != 3:
            raise ValueError("Copernicus arrays must have shape (depth, lat, lon).")
        if self._z_m.ndim != 1 or self._lat_deg.ndim != 1 or self._lon_deg.ndim != 1:
            raise ValueError("Copernicus depth/latitude/longitude coordinates must be 1D.")
        if np.any(np.diff(self._z_m) <= 0.0) or self._z_m[0] < 0.0:
            raise ValueError("Copernicus depth axis must be non-negative and strictly increasing.")

        lat0 = self.reference_lat_deg
        lon0 = self.reference_lon_deg
        if lat0 is None:
            lat0 = float(0.5 * (self._lat_deg.min() + self._lat_deg.max()))
        if lon0 is None:
            lon0 = float(0.5 * (self._lon_deg.min() + self._lon_deg.max()))

        self.reference_lat_deg = float(lat0)
        self.reference_lon_deg = float(lon0)

        self._x_m, _ = self._latlon_to_xy_m(
            np.full_like(self._lon_deg, self.reference_lat_deg),
            self._lon_deg,
            self.reference_lat_deg,
            self.reference_lon_deg,
        )
        _, self._y_m = self._latlon_to_xy_m(
            self._lat_deg,
            np.full_like(self._lat_deg, self.reference_lon_deg),
            self.reference_lat_deg,
            self.reference_lon_deg,
        )

        # Ensure monotonic increasing x/y for interpolation.
        if np.any(np.diff(self._x_m) < 0.0):
            x_order = np.argsort(self._x_m)
            self._x_m = self._x_m[x_order]
            self._lon_deg = self._lon_deg[x_order]
            t = t[:, :, x_order]
            s = s[:, :, x_order]
        if np.any(np.diff(self._y_m) < 0.0):
            y_order = np.argsort(self._y_m)
            self._y_m = self._y_m[y_order]
            self._lat_deg = self._lat_deg[y_order]
            t = t[:, y_order, :]
            s = s[:, y_order, :]

        self._c_zyx = self._leroy_sound_speed(self._z_m, t, s, self._lat_deg)

        # Representative 1D profile for pointwise speed requests in existing interfaces.
        valid_counts = np.sum(np.isfinite(self._c_zyx), axis=(1, 2))
        summed = np.nansum(self._c_zyx, axis=(1, 2))
        self._c_z_mean = np.divide(
            summed,
            valid_counts,
            out=np.full(len(self._z_m), np.nan, dtype=float),
            where=valid_counts > 0,
        )
        nan_mask = ~np.isfinite(self._c_z_mean)
        if np.any(nan_mask):
            finite_mask = np.isfinite(self._c_z_mean)
            if not np.any(finite_mask):
                self._c_z_mean = np.full_like(self._z_m, self.fill_speed_m_s, dtype=float)
            else:
                self._c_z_mean[nan_mask] = np.interp(
                    self._z_m[nan_mask],
                    self._z_m[finite_mask],
                    self._c_z_mean[finite_mask],
                )
        self._is_loaded = True

    def __post_init__(self) -> None:
        """Attempt eager load; public methods also support lazy loading."""
        self._is_loaded = False
        self._load_data()

    def _ensure_loaded(self) -> None:
        """Ensure cached Copernicus arrays are loaded."""
        if getattr(self, "_is_loaded", False):
            return
        self._load_data()

    @staticmethod
    def _to_float_with_nan(var_data: ArrayLike) -> FloatArray:
        arr = np.ma.array(var_data)
        arr = np.ma.filled(arr, np.nan)
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > 1.0e4] = np.nan
        return arr

    @staticmethod
    def _latlon_to_xy_m(
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        lat0_deg: float,
        lon0_deg: float,
    ) -> tuple[FloatArray, FloatArray]:
        lat_array = np.asarray(lat_deg, dtype=float)
        lon_array = np.asarray(lon_deg, dtype=float)
        lat0_rad = np.deg2rad(lat0_deg)
        dlon_rad = np.deg2rad(lon_array - lon0_deg)
        dlat_rad = np.deg2rad(lat_array - lat0_deg)

        a = 6_378_137.0
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)

        sin_lat0 = np.sin(lat0_rad)
        w = np.sqrt(1.0 - e2 * sin_lat0**2)
        n = a / w
        m = a * (1.0 - e2) / (w**3)

        x = dlon_rad * n * np.cos(lat0_rad)
        y = dlat_rad * m
        return x, y

    @staticmethod
    def _leroy_sound_speed(
        z_m: ArrayLike,
        temp_zyx: ArrayLike,
        sal_zyx: ArrayLike,
        lat_deg: ArrayLike,
    ) -> FloatArray:
        z_array = np.asarray(z_m, dtype=float)
        temp_array = np.asarray(temp_zyx, dtype=float)
        sal_array = np.asarray(sal_zyx, dtype=float)
        z4 = z_array[:, None, None]
        lat4 = np.asarray(lat_deg, dtype=float)[None, :, None]
        c = (
            1402.5
            + 5.0 * temp_array
            - 5.44e-2 * temp_array**2
            + 2.1e-4 * temp_array**3
            + 1.33 * sal_array
            - 1.23e-2 * sal_array * temp_array
            + 8.7e-5 * sal_array * temp_array**2
            + 1.56e-2 * z4
            + 2.55e-7 * z4**2
            - 7.3e-12 * z4**3
            + 1.2e-6 * z4 * (lat4 - 45.0)
            - 9.5e-13 * temp_array * z4**3
            + 3e-7 * temp_array**2 * z4
            + 1.43e-5 * sal_array * z4
        )
        return np.asarray(c, dtype=float)

    @staticmethod
    def _fill_nans_2d(arr_2d: ArrayLike) -> FloatArray:
        out = np.array(arr_2d, dtype=float, copy=True)
        ny, nx = out.shape

        for row_idx in range(ny):
            row = out[row_idx, :]
            valid = np.isfinite(row)
            if np.any(valid):
                out[row_idx, :] = np.interp(np.arange(nx), np.where(valid)[0], row[valid])

        for col_idx in range(nx):
            col = out[:, col_idx]
            valid = np.isfinite(col)
            if np.any(valid):
                out[:, col_idx] = np.interp(np.arange(ny), np.where(valid)[0], col[valid])

        return out

    @staticmethod
    def _interp_2d_regular(
        z_old_yx: ArrayLike,
        x_old: ArrayLike,
        y_old: ArrayLike,
        x_new: ArrayLike,
        y_new: ArrayLike,
    ) -> FloatArray:
        z_old_yx = np.asarray(z_old_yx, dtype=float)
        x_old_array = np.asarray(x_old, dtype=float)
        y_old_array = np.asarray(y_old, dtype=float)
        x_new_array = np.asarray(x_new, dtype=float)
        y_new_array = np.asarray(y_new, dtype=float)
        z_x = np.vstack([np.interp(x_new_array, x_old_array, row) for row in z_old_yx])
        z_xy = np.vstack(
            [np.interp(y_new_array, y_old_array, z_x[:, i]) for i in range(z_x.shape[1])]
        ).T
        return z_xy

    @staticmethod
    def _interp_3d_horizontal(
        c_zyx: ArrayLike,
        x_old: ArrayLike,
        y_old: ArrayLike,
        x_new: ArrayLike,
        y_new: ArrayLike,
    ) -> FloatArray:
        c_array = np.asarray(c_zyx, dtype=float)
        x_old_array = np.asarray(x_old, dtype=float)
        y_old_array = np.asarray(y_old, dtype=float)
        x_new_array = np.asarray(x_new, dtype=float)
        y_new_array = np.asarray(y_new, dtype=float)
        nz = c_array.shape[0]
        out = np.empty((nz, len(y_new_array), len(x_new_array)), dtype=float)
        for k in range(nz):
            layer = LeroyCopernicusSoundSpeedProfile._fill_nans_2d(c_array[k, :, :])
            out[k, :, :] = LeroyCopernicusSoundSpeedProfile._interp_2d_regular(
                layer,
                x_old_array,
                y_old_array,
                x_new_array,
                y_new_array,
            )
        return out

    @staticmethod
    def _extrapolate_columns_to_depth(
        c_zyx: ArrayLike,
        z_in: ArrayLike,
        z_out: ArrayLike,
        c_fill: float,
    ) -> FloatArray:
        c_array = np.asarray(c_zyx, dtype=float)
        z_in_array = np.asarray(z_in, dtype=float)
        z_out_array = np.asarray(z_out, dtype=float)
        ny, nx = c_array.shape[1], c_array.shape[2]
        out = np.full((len(z_out_array), ny, nx), c_fill, dtype=float)
        for j in range(ny):
            for i in range(nx):
                col = c_array[:, j, i]
                valid = np.isfinite(col)
                if np.sum(valid) == 0:
                    continue
                if np.sum(valid) == 1:
                    out[:, j, i] = col[valid][0]
                    continue

                zv = z_in_array[valid]
                cv = col[valid]
                out[:, j, i] = np.interp(z_out_array, zv, cv)

                deep = z_out_array > zv[-1]
                slope = (cv[-1] - cv[-2]) / (zv[-1] - zv[-2])
                out[deep, j, i] = cv[-1] + slope * (z_out_array[deep] - zv[-1])

        return out

    @staticmethod
    def _normalise_z_range_to_positive(z_range: tuple[float, float]) -> tuple[float, float]:
        z0, z1 = float(z_range[0]), float(z_range[1])
        if z0 <= 0.0 and z1 <= 0.0:
            # Nereus convention input (-z underwater).
            return min(abs(z0), abs(z1)), max(abs(z0), abs(z1))

        if z0 >= 0.0 and z1 >= 0.0:
            # Already oceanographic/RTRS (+z downward).
            return min(z0, z1), max(z0, z1)

        return min(abs(z0), abs(z1)), max(abs(z0), abs(z1))

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate representative sound speed at ``depth`` using a domain-mean profile."""
        self._ensure_loaded()
        depth_arr = np.asarray(depth, dtype=float)
        depth_pos = np.abs(depth_arr)
        depth_pos = np.clip(depth_pos, self._z_m[0], self._z_m[-1])
        c = np.interp(depth_pos, self._z_m, self._c_z_mean)
        if np.isscalar(depth):
            return float(c)
        return c

    def get_3d_grid(
        self,
        x_range: Range1D,
        y_range: Range1D,
        z_range: Range1D,
        x_res: float = 5000.0,
        y_res: float = 5000.0,
        z_res: float = 100.0,
    ) -> Grid3DResult:
        """Get a regular SSP cube resampled from Copernicus data for RTRS."""
        self._ensure_loaded()
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_points = int((x_max - x_min) / x_res) + 1
        y_points = int((y_max - y_min) / y_res) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        z_min_pos, z_max_pos = self._normalise_z_range_to_positive(z_range)
        z_points = int((z_max_pos - z_min_pos) / z_res) + 1
        z_grid = np.linspace(z_min_pos, z_max_pos, max(2, z_points))

        c_reg_h = self._interp_3d_horizontal(self._c_zyx, self._x_m, self._y_m, x_grid, y_grid)
        c_reg = self._extrapolate_columns_to_depth(
            c_reg_h,
            self._z_m,
            z_grid,
            c_fill=float(self.fill_speed_m_s),
        )

        # Convert (nz, ny, nx) -> (nx, ny, nz), then flatten C-order for RTRS.
        c_grid_flat = np.transpose(c_reg, (2, 1, 0)).flatten(order="C")

        return x_grid, y_grid, z_grid, c_grid_flat
