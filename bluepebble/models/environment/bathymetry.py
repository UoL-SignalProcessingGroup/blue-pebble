"""Defines bathymetry models for representing seafloor topography."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]
Range1D: TypeAlias = tuple[float, float]
GridResult: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]


class Bathymetry(ABC, Base):
    """Abstract base class for bathymetry models."""

    resolution: float = Property(
        default=1000.0, doc="Default grid resolution in meters for gridded outputs"
    )

    @abstractmethod
    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth at a given (x, y) position.

        Parameters
        ----------
        x : float
            X coordinate in meters.
        y : float
            Y coordinate in meters.

        Returns
        -------
        float
            Depth in meters (positive value below surface).

        Notes
        -----
        Implementations must override this abstract method.

        """
        ...

    @abstractmethod
    def get_grid(self, x_range: Range1D, y_range: Range1D) -> GridResult:
        """Get a gridded representation of the bathymetry.

        Parameters
        ----------
        x_range : Range1D
            Tuple of (x_min, x_max) in meters.
        y_range : Range1D
            Tuple of (y_min, y_max) in meters.
        resolution : float, optional
            Grid resolution in meters (defaults to ``resolution``).

        Returns
        -------
        GridResult
            ``(x_grid, y_grid, z_grid)`` where
            - ``x_grid`` : 1D array of x coordinates
            - ``y_grid`` : 1D array of y coordinates
            - ``z_grid`` : 2D array of depths (positive, below surface)

        Notes
        -----
        Implementations must override this abstract method.

        """
        ...


class FlatBathymetry(Bathymetry):
    """A flat seafloor bathymetry model.

    Attributes
    ----------
    depth : float
        The constant depth of the seafloor in meters. Must be negative
        (below surface, Blue Pebble ``-z`` convention). Defaults to -5000.0 m.

    """

    depth: float = Property(default=-5000.0, doc="Constant depth of the seafloor in meters")

    def __post_init__(self) -> None:
        """Validate the depth after initialization."""
        if self.depth >= 0:
            raise ValueError("Depth must be negative for Blue Pebble -z convention.")

    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth (constant everywhere).

        Parameters
        ----------
        x : float
            X coordinate in meters (ignored).
        y : float
            Y coordinate in meters (ignored).

        Returns
        -------
        float
            Constant depth in meters.

        """
        return self.depth

    def get_grid(self, x_range: Range1D, y_range: Range1D) -> GridResult:
        """Get a gridded representation of the flat bathymetry.

        Parameters
        ----------
        x_range : Range1D
            Tuple of (x_min, x_max) in meters.
        y_range : Range1D
            Tuple of (y_min, y_max) in meters.
        resolution : float, optional
            Grid resolution in meters (defaults to ``resolution``).

        Returns
        -------
        GridResult
            ``(x_grid, y_grid, z_grid)`` where ``z_grid`` is constant.

        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid points
        x_points = int((x_max - x_min) / self.resolution) + 1
        y_points = int((y_max - y_min) / self.resolution) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        # Create constant depth grid
        z_grid = np.full((len(x_grid), len(y_grid)), self.depth)

        return x_grid, y_grid, z_grid


class WedgeBathymetry(Bathymetry):
    """A linearly sloping seafloor bathymetry model.

    Attributes
    ----------
    depth_at_origin : float
        Depth at the origin (0, 0) in meters. Defaults to -1000.0 m.
    x_gradient : float
        Depth gradient in the x direction (m/m). Defaults to 0.0 (no slope in x).
    y_gradient : float
        Depth gradient in the y direction (m/m). Defaults to 0.001 (1 m increase per 1000 m in y).

    """

    depth_at_origin: float = Property(default=-1000.0, doc="Depth at the origin (0, 0) in meters")
    x_gradient: float = Property(default=0.0, doc="Depth gradient in the x direction (m/m)")
    y_gradient: float = Property(default=0.001, doc="Depth gradient in the y direction (m/m)")

    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth at a given position.

        Parameters
        ----------
        x : float
            X coordinate in meters.
        y : float
            Y coordinate in meters.

        Returns
        -------
        float
            Depth in meters at (x, y).

        """
        depth = self.depth_at_origin + self.x_gradient * x + self.y_gradient * y
        return min(0.0, depth)  # Ensure depth is non-positive in Blue Pebble -z convention

    def get_grid(self, x_range: Range1D, y_range: Range1D) -> GridResult:
        """Get a gridded representation of the sloping bathymetry.

        Parameters
        ----------
        x_range : Range1D
            Tuple of (x_min, x_max) in meters.
        y_range : Range1D
            Tuple of (y_min, y_max) in meters.
        resolution : float, optional
            Grid resolution in meters (defaults to ``resolution``).

        Returns
        -------
        GridResult
            ``(x_grid, y_grid, z_grid)`` with linearly varying depths.

        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid points
        x_points = int((x_max - x_min) / self.resolution) + 1
        y_points = int((y_max - y_min) / self.resolution) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        # Create meshgrid and calculate depths
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        z_grid = self.depth_at_origin + self.x_gradient * X + self.y_gradient * Y
        z_grid = np.minimum(z_grid, 0.0)  # Ensure non-positive in Blue Pebble -z convention

        return x_grid, y_grid, z_grid


class SeamountBathymetry(Bathymetry):
    """A bathymetry model with an idealized seamount feature.

    Attributes
    ----------
    summit_position : tuple
        ``(x, y, z)`` coordinates of the summit in meters. Defaults to
        ``(25000.0, 25000.0, -1000.0)``.
    radius : float
        Radius of the seamount in meters. Defaults to 15000.0 m.
    plateau_depth : float
        Depth at the surrounding plateau in meters. Defaults to -5000.0 m.

    """

    summit_position: tuple[float, float, float] = Property(
        default=(25000.0, 25000.0, -1000.0),
        doc="(x, y, z) coordinates of the summit in meters",
    )
    radius: float = Property(default=15000.0, doc="Radius of the seamount in meters")
    plateau_depth: float = Property(
        default=-5000.0, doc="Depth at the surrounding plateau in meters"
    )

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.radius <= 0:
            raise ValueError("Radius must be positive.")
        if self.plateau_depth >= 0:
            raise ValueError("Plateau depth must be negative for Blue Pebble -z convention.")
        summit_z = self.summit_position[2]
        if summit_z > 0 or summit_z <= self.plateau_depth:
            raise ValueError(
                "Summit depth must be <= 0 and shallower (less negative) than plateau depth."
            )

    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth at a given position.

        Parameters
        ----------
        x : float
            X coordinate in meters.
        y : float
            Y coordinate in meters.

        Returns
        -------
        float
            Depth in meters at (x, y).

        """
        summit_x, summit_y, summit_z = self.summit_position
        r = np.sqrt((x - summit_x) ** 2 + (y - summit_y) ** 2)

        if r <= self.radius:
            depth = summit_z + (self.plateau_depth - summit_z) * (r / self.radius)
        else:
            depth = self.plateau_depth

        return min(0.0, depth)

    def get_grid(self, x_range: Range1D, y_range: Range1D) -> GridResult:
        """Get a gridded representation of the seamount bathymetry.

        Parameters
        ----------
        x_range : Range1D
            Tuple of (x_min, x_max) in meters.
        y_range : Range1D
            Tuple of (y_min, y_max) in meters.
        resolution : float, optional
            Grid resolution in meters (defaults to ``resolution``).

        Returns
        -------
        GridResult
            ``(x_grid, y_grid, z_grid)`` with seamount topography.

        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid points
        x_points = int((x_max - x_min) / self.resolution) + 1
        y_points = int((y_max - y_min) / self.resolution) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        # Create meshgrid and calculate depths
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

        summit_x, summit_y, summit_z = self.summit_position
        r = np.sqrt((X - summit_x) ** 2 + (Y - summit_y) ** 2)

        z_grid = np.where(
            r <= self.radius,
            summit_z + (self.plateau_depth - summit_z) * (r / self.radius),
            self.plateau_depth,
        )

        z_grid = np.minimum(z_grid, 0.0)

        return x_grid, y_grid, z_grid


class GEBCOBathymetry(Bathymetry):
    """Bathymetry model backed by a GEBCO NetCDF dataset.

    Notes
    -----
    - Coordinates are converted from lat/lon to local Cartesian meters.
    - Internally and at output, depth follows the Nereus convention (``-z`` underwater).

    """

    file_path: str = Property(doc="Path to GEBCO NetCDF file")
    reference_lat_deg: float | None = Property(
        default=None,
        doc="Reference latitude for local x/y conversion. Defaults to dataset midpoint.",
    )
    reference_lon_deg: float | None = Property(
        default=None,
        doc="Reference longitude for local x/y conversion. Defaults to dataset midpoint.",
    )

    def _load_data(self) -> None:
        """Load and cache GEBCO bathymetry data."""
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"GEBCO bathymetry file not found: {path}")

        try:
            import netCDF4 as nc
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for GEBCOBathymetry. Install with `pip install netCDF4`."
            ) from exc

        with nc.Dataset(path, "r") as ds:
            self._lat_deg = np.asarray(ds.variables["lat"][:], dtype=float)
            self._lon_deg = np.asarray(ds.variables["lon"][:], dtype=float)
            elev_m = np.asarray(ds.variables["elevation"][:], dtype=float)

        if self._lat_deg.ndim != 1 or self._lon_deg.ndim != 1 or elev_m.ndim != 2:
            raise ValueError("GEBCO bathymetry must provide 1D lat/lon and 2D elevation arrays.")

        if elev_m.shape != (len(self._lat_deg), len(self._lon_deg)):
            raise ValueError("GEBCO elevation shape must match (lat, lon).")

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

        if np.any(np.diff(self._x_m) <= 0.0) or np.any(np.diff(self._y_m) <= 0.0):
            raise ValueError("Converted GEBCO x/y axes must be strictly increasing.")

        # GEBCO elevation: underwater is negative. Nereus convention is -z underwater.
        self._z_grid_yx = np.minimum(elev_m, 0.0)
        self._is_loaded = True

    def __post_init__(self) -> None:
        """Attempt eager load; get_depth/get_grid also support lazy loading."""
        self._is_loaded = False
        self._load_data()

    def _ensure_loaded(self) -> None:
        """Ensure cached GEBCO arrays are loaded."""
        if getattr(self, "_is_loaded", False):
            return
        self._load_data()

    @staticmethod
    def _latlon_to_xy_m(
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        lat0_deg: float,
        lon0_deg: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Convert geodetic coordinates to local tangent-plane x/y in meters."""
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
    def _nearest_indices(old_axis: FloatArray, new_axis: FloatArray) -> IntArray:
        """Map target coordinates to nearest indices on a monotonic source axis."""
        idx = np.searchsorted(old_axis, new_axis)
        idx = np.clip(idx, 1, len(old_axis) - 1)
        left = old_axis[idx - 1]
        right = old_axis[idx]
        choose_left = np.abs(new_axis - left) <= np.abs(right - new_axis)
        return np.where(choose_left, idx - 1, idx)

    def get_depth(self, x: float, y: float) -> float:
        """Get nearest-neighbour GEBCO depth at ``(x, y)`` in meters (Nereus ``-z``)."""
        self._ensure_loaded()
        ix = int(self._nearest_indices(self._x_m, np.asarray([x], dtype=float))[0])
        iy = int(self._nearest_indices(self._y_m, np.asarray([y], dtype=float))[0])
        return float(min(0.0, self._z_grid_yx[iy, ix]))

    def get_grid(self, x_range: Range1D, y_range: Range1D) -> GridResult:
        """Get a regular GEBCO bathymetry grid for the requested x/y extent."""
        self._ensure_loaded()
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_points = int((x_max - x_min) / self.resolution) + 1
        y_points = int((y_max - y_min) / self.resolution) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        ix = self._nearest_indices(self._x_m, x_grid)
        iy = self._nearest_indices(self._y_m, y_grid)

        z_yx = self._z_grid_yx[np.ix_(iy, ix)]
        z_grid = z_yx.T
        z_grid = np.minimum(z_grid, 0.0)

        return x_grid, y_grid, z_grid
