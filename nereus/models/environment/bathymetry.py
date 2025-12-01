"""Defines bathymetry models for representing seafloor topography.

© Copyright 2025 Joshua J. Wakefield.
© Copyright 2025 Finley Boulton.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np
from stonesoup.base import Base, Property


class Bathymetry(ABC, Base):
    """Abstract base class for bathymetry models."""

    @abstractmethod
    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth at a given (x, y) position.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.

        Returns:
            Depth in meters (positive value below surface).

        Note:
            This method must be implemented by subclasses.

        """
        pass

    @abstractmethod
    def get_grid(self, x_range: tuple, y_range: tuple, resolution: float = 1000.0):
        """Get a gridded representation of the bathymetry.

        Args:
            x_range: Tuple of (x_min, x_max) in meters.
            y_range: Tuple of (y_min, y_max) in meters.
            resolution: Grid resolution in meters. Defaults to 1000.0.

        Returns:
            Tuple of (x_grid, y_grid, z_grid) where:
                - x_grid: 1D array of x coordinates
                - y_grid: 1D array of y coordinates
                - z_grid: 2D array of depths (positive, below surface)

        Note:
            This method must be implemented by subclasses.

        """
        pass


class FlatBathymetry(Bathymetry):
    """A flat seafloor bathymetry model.

    Attributes:
        depth (float): The constant depth of the seafloor in meters.
            Must be positive (below surface). Defaults to 5000.0 m.

    """

    depth: float = Property(
        default=5000.0, doc="Constant depth of the seafloor in meters"
    )

    def __post_init__(self):
        """Validate the depth after initialization."""
        if self.depth <= 0:
            raise ValueError("Depth must be positive.")

    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth (constant everywhere).

        Args:
            x: X coordinate in meters (ignored).
            y: Y coordinate in meters (ignored).

        Returns:
            Constant depth in meters.

        """
        return self.depth

    def get_grid(self, x_range: tuple, y_range: tuple, resolution: float = 1000.0):
        """Get a gridded representation of the flat bathymetry.

        Args:
            x_range: Tuple of (x_min, x_max) in meters.
            y_range: Tuple of (y_min, y_max) in meters.
            resolution: Grid resolution in meters. Defaults to 1000.0.

        Returns:
            Tuple of (x_grid, y_grid, z_grid) where z_grid is constant.

        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid points
        x_points = int((x_max - x_min) / resolution) + 1
        y_points = int((y_max - y_min) / resolution) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        # Create constant depth grid
        z_grid = np.full((len(x_grid), len(y_grid)), self.depth)

        return x_grid, y_grid, z_grid


class SlopingBathymetry(Bathymetry):
    """A linearly sloping seafloor bathymetry model.

    Attributes:
        depth_at_origin (float): Depth at the origin (0, 0) in meters.
            Defaults to 1000.0 m.
        x_gradient (float): Depth gradient in the x direction (m/m).
            Defaults to 0.0 (no slope in x).
        y_gradient (float): Depth gradient in the y direction (m/m).
            Defaults to 0.001 (1 m increase per 1000 m in y).

    """

    depth_at_origin: float = Property(
        default=1000.0, doc="Depth at the origin (0, 0) in meters"
    )
    x_gradient: float = Property(
        default=0.0, doc="Depth gradient in the x direction (m/m)"
    )
    y_gradient: float = Property(
        default=0.001, doc="Depth gradient in the y direction (m/m)"
    )

    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth at a given position.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.

        Returns:
            Depth in meters at (x, y).

        """
        depth = self.depth_at_origin + self.x_gradient * x + self.y_gradient * y
        return max(0.0, depth)  # Ensure depth is non-negative

    def get_grid(self, x_range: tuple, y_range: tuple, resolution: float = 1000.0):
        """Get a gridded representation of the sloping bathymetry.

        Args:
            x_range: Tuple of (x_min, x_max) in meters.
            y_range: Tuple of (y_min, y_max) in meters.
            resolution: Grid resolution in meters. Defaults to 1000.0.

        Returns:
            Tuple of (x_grid, y_grid, z_grid) with linearly varying depths.

        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid points
        x_points = int((x_max - x_min) / resolution) + 1
        y_points = int((y_max - y_min) / resolution) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        # Create meshgrid and calculate depths
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        z_grid = (
            self.depth_at_origin + self.x_gradient * X + self.y_gradient * Y
        )
        z_grid = np.maximum(z_grid, 0.0)  # Ensure non-negative

        return x_grid, y_grid, z_grid


class SeamountBathymetry(Bathymetry):
    """A bathymetry model with an idealized seamount feature.

    Attributes:
        summit_position (tuple): (x, y, z) coordinates of the summit in meters.
            Defaults to (25000.0, 25000.0, 1000.0).
        radius (float): Radius of the seamount in meters. Defaults to 15000.0 m.
        plateau_depth (float): Depth at the surrounding plateau in meters.
            Defaults to 5000.0 m.

    """

    summit_position: tuple = Property(
        default=(25000.0, 25000.0, 1000.0),
        doc="(x, y, z) coordinates of the summit in meters",
    )
    radius: float = Property(default=15000.0, doc="Radius of the seamount in meters")
    plateau_depth: float = Property(
        default=5000.0, doc="Depth at the surrounding plateau in meters"
    )

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.radius <= 0:
            raise ValueError("Radius must be positive.")
        if self.plateau_depth <= 0:
            raise ValueError("Plateau depth must be positive.")
        if self.summit_position[2] < 0 or self.summit_position[2] >= self.plateau_depth:
            raise ValueError(
                "Summit depth must be non-negative and less than plateau depth."
            )

    def get_depth(self, x: float, y: float) -> float:
        """Get the seafloor depth at a given position.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.

        Returns:
            Depth in meters at (x, y).

        """
        summit_x, summit_y, summit_z = self.summit_position
        r = np.sqrt((x - summit_x) ** 2 + (y - summit_y) ** 2)

        if r <= self.radius:
            depth = summit_z + (self.plateau_depth - summit_z) * (r / self.radius)
        else:
            depth = self.plateau_depth

        return depth

    def get_grid(self, x_range: tuple, y_range: tuple, resolution: float = 1000.0):
        """Get a gridded representation of the seamount bathymetry.

        Args:
            x_range: Tuple of (x_min, x_max) in meters.
            y_range: Tuple of (y_min, y_max) in meters.
            resolution: Grid resolution in meters. Defaults to 1000.0.

        Returns:
            Tuple of (x_grid, y_grid, z_grid) with seamount topography.

        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid points
        x_points = int((x_max - x_min) / resolution) + 1
        y_points = int((y_max - y_min) / resolution) + 1

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

        return x_grid, y_grid, z_grid
