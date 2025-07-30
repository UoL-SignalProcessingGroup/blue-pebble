"""Defines source model classes for representing sound sources.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from datetime import datetime

import numpy as np

from nereus.types.states import GroundTruthState


class AcousticPointSource(GroundTruthState):
    """Represents a single point source that emits multiple tonal frequencies.

    Attributes:
        state_vector (np.ndarray): The state vector representing the physical
            state of the source (e.g., position, velocity).
        timestamp (datetime): The timestamp associated with the state.
        amplitude (np.ndarray): A 1D array of linear amplitudes for each tonal.
        frequency (np.ndarray): A 1D array of frequencies for each tonal in Hertz.
        phase (np.ndarray): A 1D array of initial phases for each tonal in radians.
        num_tonals (int): The number of tonal components for this source.

    """

    def __init__(
        self,
        state_vector: np.ndarray,
        amplitude: np.ndarray | list,
        frequency: np.ndarray | list,
        phase_deg: np.ndarray | list,
        timestamp: datetime = None,
    ):
        """Initialise the AcousticPointSource.

        Args:
            state_vector (np.ndarray): The state vector of the source.
            amplitude (np.ndarray | list): A 1D array of amplitudes for each tonal.
            frequency (np.ndarray | list): A 1D array of frequencies in Hertz.
            phase_deg (np.ndarray | list): A 1D array of initial phases in degrees.
            timestamp (datetime, optional): The timestamp of the state.
                Defaults to None.

        """
        super().__init__(state_vector, timestamp)
        self.amplitude = np.array(amplitude)
        self.frequency = np.array(frequency)
        self.phase = np.deg2rad(np.array(phase_deg))
        self.num_tonals = len(self.frequency)
