"""Defines a collection of sound speed profile (SSP) models.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np
from stonesoup.base import Base, Property


class SoundSpeedProfile(ABC, Base):
    """Abstract base class for sound speed profile models."""

    @abstractmethod
    def calculate(self, depth: float) -> float:
        """Calculate the sound speed at a given depth.

        Args:
            depth: Depth in meters. Can be positive (oceanographic convention,
                  measured downward from surface) or negative (3D coordinate
                  system where surface=0, underwater is negative z).

        Returns:
            Sound speed in m/s.

        Note:
            This method must be implemented by subclasses.

        """
        pass

    def _calc_temperature(self, depth: float) -> float:
        """Calculate ocean temperature based on vertical variation.

        Args:
            depth: Depth in meters (positive, below surface).

        Returns:
            Temperature in degrees Celsius.

        """
        return 10 * (1 - np.tanh((depth - 100) / 50)) + 2

    def _calc_salinity(self, depth: float) -> float:
        """Calculate ocean salinity model based on vertical variation.

        Args:
            depth: Depth in meters (positive, below surface).

        Returns:
            Salinity in practical salinity units (PSU).

        """
        return 0.5 * (1 - np.tanh((depth - 200) / 100)) + 35


class Munk(SoundSpeedProfile):
    """Munk sound speed profile model.

    This model describes the sound speed profile using an analytical equation
    proposed by Walter Munk. It is characterized by a deep sound channel axis
    and is widely used in ocean acoustics.

    Attributes:
        surface_speed (float): The speed of sound at the surface in m/s.
            Defaults to 1500.0 m/s.
    """

    surface_speed: float = Property(
        default=1500.0, doc="Speed of sound at the surface in m/s"
    )

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the Munk equation.

        Args:
            depth: Depth in meters. If negative (z-coordinate), converts to
                  positive depth below surface for calculation.

        Returns:
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)

        zt = 2.0 * (depth_positive - 1300.0) / 1300.0
        c = self.surface_speed * (1.0 + 0.00737 * (zt - 1.0 + np.exp(-zt)))
        return c


class Mackenzie(SoundSpeedProfile):
    """Mackenzie sound speed profile model.

    This model calculates the sound speed using the nine-term Mackenzie
    equation, which is an empirical formula based on temperature, salinity,
    and depth. This implementation uses internal models for temperature and
    salinity as a function of depth.
    """

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the Mackenzie nine-term equation.

        Args:
            depth: Depth in meters. If negative (z-coordinate), converts to
                  positive depth below surface for calculation.

        Returns:
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
