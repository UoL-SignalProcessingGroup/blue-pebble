"""Defines beamforming algorithms for processing signals from an array of sensors.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange, types

from nereus.models.sensor import SensorPlatform
from nereus.models.ssp import SoundSpeedProfile


class Beamformer(ABC):
    """Abstract base class for beamformers."""

    @abstractmethod
    def beamform(
        self, signal_array: np.ndarray, steering_delays: np.ndarray
    ) -> np.ndarray:
        """Process sensor signals to form beams in specified directions.

        Args:
            signal_array (np.ndarray): An array of sensor signals with shape
                (num_sensors, num_samples).
            steering_delays (np.ndarray): An array of time delays for each
                sensor and steering direction, with shape (num_directions, num_sensors).

        Returns:
            np.ndarray: An array of beamformed signals with shape
                (num_directions, num_samples).

        """
        raise NotImplementedError


class DelayAndSumBeamformer(Beamformer):
    """A Delay-and-Sum (DAS) beamformer.

    This class implements the DAS algorithm in either the time or frequency
    domain. It steers an array of sensors by applying time delays to the
    received signals and summing them to enhance the signal from desired
    directions.
    """

    def __init__(
        self,
        sampling_frequency: float,
        shading: np.ndarray | None = None,
        domain: str = "time",
    ):
        """Initialise the DelayAndSumBeamformer.

        Args:
            sampling_frequency (float): The sampling frequency of the sensor
                signals, in Hz.
            shading (np.ndarray | None, optional): An array of shading weights
                to apply to each sensor. If None, uniform weights are used.
                Defaults to None.
            domain (str, optional): The domain for beamforming, either 'time'
                or 'frequency'. Defaults to "time".

        Raises:
            ValueError: If the specified domain is not 'time' or 'frequency'.

        """
        self.sampling_frequency = sampling_frequency

        if shading is not None:
            self.shading = shading / np.sum(shading)
        else:
            self.shading = None

        if domain not in ["time", "frequency"]:
            raise ValueError(
                "Invalid beamforming domain. Must be 'time' or 'frequency'"
            )
        self.domain = domain

    def beamform(
        self, signal_array: np.ndarray, steering_delays: np.ndarray
    ) -> np.ndarray:
        """Process sensor signals to form beams in specified directions.

        Args:
            signal_array (np.ndarray): An array of sensor signals with shape
                (num_sensors, num_samples).
            steering_delays (np.ndarray): An array of time delays for each
                sensor and steering direction, with shape
                (num_directions, num_sensors).

        Returns:
            np.ndarray: An array of beamformed signals with shape
                (num_directions, num_samples).

        Raises:
            ValueError: If the number of sensors in signal_array does not
                match the number of sensors in steering_delays.

        """
        num_sensors, _ = signal_array.shape

        if num_sensors != steering_delays.shape[1]:
            raise ValueError(
                "Number of sensors must match the number of steering delays"
            )

        shading_weights = (
            np.ones(num_sensors) / num_sensors if self.shading is None else self.shading
        )

        if self.domain == "time":
            return _time_das(
                signal_array,
                steering_delays,
                shading_weights,
                self.sampling_frequency,
            )
        else:
            return _frequency_das(
                signal_array, steering_delays, shading_weights, self.sampling_frequency
            )


@njit(
    (
        types.Array(types.complex128, 2, "C"),
        types.Array(types.float64, 2, "F"),
        types.Array(types.float64, 1, "C"),
        types.float64,
    ),
    cache=True,
    parallel=True,
    fastmath=True,
)
def _time_das(
    signal_array: np.ndarray,
    steering_delays: np.ndarray,
    shading_weights: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """Perform Delay-and-Sum beamforming in the time domain."""
    num_directions = steering_delays.shape[0]
    num_sensors, num_samples = signal_array.shape
    delays = np.round(-steering_delays * sampling_frequency).astype(np.int64)

    beamformed_signals = np.zeros((num_directions, num_samples), dtype=np.complex128)

    for i in prange(num_directions):
        for j in range(num_sensors):
            d = delays[i, j]
            beamformed_signals[i] += shading_weights[j] * np.roll(signal_array[j], d)

    # Crop the signals to remove circular shift artifacts
    # Note: Numba does not support np.max/min on the entire 2D delays array in parallel
    # loops so we compute it outside the parallel section.
    left = 0
    right = 0
    if delays.size > 0:
        left = np.max(delays)
        right = -np.min(delays)

    if right > 0:
        return beamformed_signals[:, left:-right]
    else:
        return beamformed_signals[:, left:]


@njit(
    (
        types.Array(types.complex128, 2, "C"),
        types.Array(types.float64, 2, "C"),
        types.Array(types.float64, 1, "C"),
        types.float64,
    ),
    cache=True,
    parallel=True,
    fastmath=True,
)
def _frequency_das(
    signal_array: np.ndarray,
    steering_delays: np.ndarray,
    shading_weights: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """Perform Delay-and-Sum beamforming in the frequency domain."""
    num_directions, num_sensors = steering_delays.shape
    num_samples = signal_array.shape[1]

    # 1. Perform FFT on all sensor signals
    signals_f = np.fft.fft(signal_array, axis=1)

    # 2. Compute frequency bins
    frequency_bins = np.fft.fftfreq(num_samples, d=1.0 / sampling_frequency)

    # 3. Initialise beamformed signals in the frequency domain
    beamformed_f = np.zeros((num_directions, num_samples), dtype=np.complex128)

    # 4. Apply phase shifts and sum in the frequency domain
    for i in prange(num_directions):
        for j in range(num_sensors):
            # Calculate phase shift for this sensor and direction
            phase_shift = np.exp(
                -1j * 2 * np.pi * frequency_bins * steering_delays[i, j]
            )
            # Apply shading and phase shift, and accumulate
            beamformed_f[i] += signals_f[j] * phase_shift * shading_weights[j]

    # 5. Perform IFFT once on the final result
    beamformed_signals = np.fft.ifft(beamformed_f, axis=1)

    return beamformed_signals


def calculate_directional_power(beamformed_signals: np.ndarray) -> np.ndarray:
    """Calculate the total power for each beamformed signal direction.

    Args:
        beamformed_signals (np.ndarray): An array of beamformed signals with
            shape (num_directions, num_samples).

    Returns:
        np.ndarray: An array of power values (linear scale) for each
            direction.

    """
    power_linear = np.mean(np.abs(beamformed_signals) ** 2, axis=1)
    return power_linear


def calculate_steering_delays(
    sensor_array: SensorPlatform,
    steering_azimuths: np.ndarray,
    steering_elevations: np.ndarray,
    ssp: SoundSpeedProfile,
) -> np.ndarray:
    """Calculate the geometric time delays for steering the sensor array in 3D.

    Args:
        sensor_array (SensorPlatform): The sensor array to be steered.
        steering_azimuths (np.ndarray): An array of steering azimuth
            directions in radians.
        steering_elevations (np.ndarray): An array of steering elevation
            directions in radians.
        ssp (SoundSpeedProfile): The sound speed profile used
            to calculate signal travel time.

    Returns:
        np.ndarray: An array of steering delays with shape
            (num_directions, num_sensors).

    """
    sensor_positions = sensor_array.state_vector

    # Center the array by subtracting the mean position of all sensors.
    # The mean is taken along axis 1, which corresponds to the sensors.
    sensor_positions = sensor_positions - sensor_positions.mean(axis=1, keepdims=True)

    # Create 3D direction vectors from azimuth and elevation.
    # Shape: (3, num_directions)
    direction_vector = np.array(
        [
            np.cos(steering_elevations) * np.cos(steering_azimuths),
            np.cos(steering_elevations) * np.sin(steering_azimuths),
            np.sin(steering_elevations),
        ]
    )

    # Project sensor positions onto the direction vectors.
    # (num_directions, 3) @ (3, num_sensors) -> (num_directions, num_sensors)
    distances = np.dot(direction_vector.T, sensor_positions)

    # NOTE: This assumes a constant sound speed across the array aperture.
    # For very large vertical arrays, a per-sensor sound speed might be needed.
    sound_speed = ssp.calculate(sensor_array.origin[2])

    return distances / sound_speed
