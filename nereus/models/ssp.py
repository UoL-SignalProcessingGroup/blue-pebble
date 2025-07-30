"""Defines a collection of sound speed profile (SSP) models.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np


class SoundSpeedProfile(ABC):
    """Abstract base class for sound speed profile models."""

    @abstractmethod
    def calculate(self, depth: float) -> float:
        """Calculate the sound speed at a given depth.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _calc_temperature(self, depth: float) -> float:
        """Calculate ocean temperature based on vertical variation."""
        return 10 * (1 - np.tanh((depth - 100) / 50)) + 2

    def _calc_salinity(self, depth: float) -> float:
        """Calculate ocean salinity model based on vertical variation."""
        return 0.5 * (1 - np.tanh((depth - 200) / 100)) + 35


class Munk(SoundSpeedProfile):
    """Munk sound speed profile model."""

    def __init__(self, surface_speed: float = 1500.0):
        """Initialise the Munk model.

        Args:
            surface_speed (float, optional): The speed of sound at the surface in m/s.
                Defaults to 1500.0.

        """
        self.surface_speed = surface_speed

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the Munk equation."""
        zt = 2.0 * (depth - 1300.0) / 1300.0
        c = self.surface_speed * (1.0 + 0.00737 * (zt - 1.0 + np.exp(-zt)))
        return c


class Mackenzie(SoundSpeedProfile):
    """Mackenzie sound speed profile model."""

    def calculate(self, depth: float) -> float:
        """Calculate sound speed using the Mackenzie nine-term equation."""
        temp = self._calc_temperature(depth)
        salt = self._calc_salinity(depth)
        c = (
            1448.96
            + 4.591 * temp
            - 5.304e-2 * temp**2
            + 2.374e-4 * temp**3
            + 1.340 * (salt - 35)
            + 1.630e-2 * depth
            + 1.675e-7 * depth**2
            - 1.025e-2 * temp * (salt - 35)
            - 7.139e-13 * temp * depth**3
        )
        return c
