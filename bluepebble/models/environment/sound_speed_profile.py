"""Defines a collection of sound speed profile (SSP) models."""

from abc import ABC, abstractmethod

import numpy as np
from stonesoup.base import Base, Property


class SoundSpeedProfile(ABC, Base):
    """Abstract base class for sound speed profile models."""

    @abstractmethod
    def calculate(self, depth: float) -> float:
        """Calculate the sound speed at a given depth.

        Parameters
        ----------
        depth : float
            Depth in meters. Can be positive (oceanographic convention, measured downward from
            surface) or negative (3D coordinate system where surface == 0 and underwater is
            negative z).

        Returns
        -------
        float
            Sound speed in m/s.

        """
        pass

    def get_3d_grid(
        self,
        x_range: tuple,
        y_range: tuple,
        z_range: tuple,
        x_res: float = 5000.0,
        y_res: float = 5000.0,
        z_res: float = 100.0,
    ):
        """Get a 3D grid representation of the sound speed profile.

        Parameters
        ----------
        x_range : tuple
            Tuple of (x_min, x_max) in meters.
        y_range : tuple
            Tuple of (y_min, y_max) in meters.
        z_range : tuple
            Tuple of (z_min, z_max) in meters (negative depths).
        x_res : float, optional
            Grid resolution in x direction in meters (default 5000.0).
        y_res : float, optional
            Grid resolution in y direction in meters (default 5000.0).
        z_res : float, optional
            Grid resolution in z direction in meters (default 100.0).

        Returns
        -------
        tuple
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

    def _calc_temperature(self, depth: float) -> float:
        """Calculate ocean temperature based on vertical variation.

        Parameters
        ----------
        depth : float
            Depth in meters (positive, below surface).

        Returns
        -------
        float
            Temperature in degrees Celsius.

        """
        return 10 * (1 - np.tanh((depth - 100) / 50)) + 2

    def _calc_salinity(self, depth: float) -> float:
        """Calculate ocean salinity model based on vertical variation.

        Parameters
        ----------
        depth : float
            Depth in meters (positive, below surface).

        Returns
        -------
        float
            Salinity in practical salinity units (PSU).

        """
        return 0.5 * (1 - np.tanh((depth - 200) / 100)) + 35


class Constant(SoundSpeedProfile):
    """Constant sound speed profile model.

    This model assumes a uniform sound speed throughout the water column.

    Attributes
    ----------
    speed : float
        Constant sound speed in m/s.

    """

    speed: float = Property(default=1500.0, doc="Constant sound speed in m/s")

    def calculate(self, depth: float) -> float:
        """Return the constant sound speed.

        Parameters
        ----------
        depth : float
            Depth in meters (not used in this model).

        Returns
        -------
        float
            Sound speed in m/s.

        """
        return self.speed


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

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using a linear profile.

        Parameters
        ----------
        depth : float
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        float
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

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the arctan profile.

        Parameters
        ----------
        depth : float
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        float
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

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the Munk equation.

        Parameters
        ----------
        depth : float
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        float
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

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the Mackenzie nine-term equation.

        Parameters
        ----------
        depth : float
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        float
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
