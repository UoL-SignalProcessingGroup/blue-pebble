"""Defines beamforming algorithms for processing signals from an array of sensors.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange, types
from stonesoup.platform.base import Platform

from nereus.stonesoup.sound_speed_profiles import SoundSpeedProfile


class Beamformer(ABC):
    """Abstract base class for beamformers."""

    @abstractmethod
    def beamform(
        self, sensor_signals: np.ndarray, steering_delays: np.ndarray
    ) -> np.ndarray:
        """Process sensor signals to form beams in specified directions.

        Args:
            sensor_signals (np.ndarray): An array of sensor signals with shape
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
        sampling_rate_hz: float,
        shading: np.ndarray | None = None,
        domain: str = "time",
    ):
        """Initialise the DelayAndSumBeamformer.

        Args:
            sampling_rate_hz (float): The sampling frequency of the sensor
                signals, in Hz.
            shading (np.ndarray | None, optional): An array of shading weights
                to apply to each sensor. If None, uniform weights are used.
                Defaults to None.
            domain (str, optional): The domain for beamforming, either 'time'
                or 'frequency'. Defaults to "time".

        Raises:
            ValueError: If the specified domain is not 'time' or 'frequency'.

        """
        self.sampling_rate_hz = sampling_rate_hz

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
        self, sensor_signals: np.ndarray, steering_delays_s: np.ndarray
    ) -> np.ndarray:
        """Process sensor signals to form beams in specified directions.

        Args:
            sensor_signals (np.ndarray): An array of sensor signals with shape
                (num_sensors, num_samples).
            steering_delays_s (np.ndarray): An array of time delays for each
                sensor and steering direction, with shape
                (num_directions, num_sensors).

        Returns:
            np.ndarray: An array of beamformed signals with shape
                (num_directions, num_samples).

        Raises:
            ValueError: If the number of sensors in signal_array does not
                match the number of sensors in steering_delays.

        """
        num_sensors, _ = sensor_signals.shape

        if num_sensors != steering_delays_s.shape[1]:
            raise ValueError(
                "Number of sensors must match the number of steering delays"
            )

        shading_weights = (
            np.ones(num_sensors) / num_sensors if self.shading is None else self.shading
        )

        if self.domain == "time":
            return _time_das(
                sensor_signals,
                steering_delays_s,
                shading_weights,
                self.sampling_rate_hz,
            )
        else:
            return _frequency_das(
                sensor_signals,
                steering_delays_s,
                shading_weights,
                self.sampling_rate_hz,
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
    sensor_signals: np.ndarray,
    steering_delays_s: np.ndarray,
    shading_weights: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Perform Delay-and-Sum beamforming in the time domain.

    Args:
        sensor_signals (np.ndarray): An array of sensor signals with shape
            (num_sensors, num_samples).
        steering_delays_s (np.ndarray): An array of time delays for each
            sensor and steering direction, with shape
            (num_directions, num_sensors).
        shading_weights (np.ndarray): An array of shading weights for each
            sensor, with shape (num_sensors,).
        sampling_rate_hz (float): The sampling frequency of the sensor
            signals, in Hz.

    Returns:
        np.ndarray: An array of beamformed signals with shape
            (num_directions, num_samples).

    """
    num_directions = steering_delays_s.shape[0]
    num_sensors, num_samples = sensor_signals.shape
    delays = np.round(-steering_delays_s * sampling_rate_hz).astype(np.int64)

    beamformed_signals = np.zeros((num_directions, num_samples), dtype=np.complex128)

    for i in prange(num_directions):
        for j in range(num_sensors):
            d = delays[i, j]
            beamformed_signals[i] += shading_weights[j] * np.roll(sensor_signals[j], d)

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
    sensor_signals: np.ndarray,
    steering_delays_s: np.ndarray,
    shading_weights: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Perform Delay-and-Sum beamforming in the frequency domain.

    Args:
        sensor_signals (np.ndarray): An array of sensor signals with shape
            (num_sensors, num_samples).
        steering_delays_s (np.ndarray): An array of time delays for each
            sensor and steering direction, with shape
            (num_directions, num_sensors).
        shading_weights (np.ndarray): An array of shading weights for each
            sensor, with shape (num_sensors,).
        sampling_rate_hz (float): The sampling frequency of the sensor
            signals, in Hz.

    """
    num_directions, num_sensors = steering_delays_s.shape
    num_samples = sensor_signals.shape[1]

    # 1. Perform FFT on all sensor signals
    signals_f = np.fft.fft(sensor_signals, axis=1)

    # 2. Compute frequency bins
    frequency_bins = np.fft.fftfreq(num_samples, d=1.0 / sampling_rate_hz)

    # 3. Initialise beamformed signals in the frequency domain
    beamformed_f = np.zeros((num_directions, num_samples), dtype=np.complex128)

    # 4. Apply phase shifts and sum in the frequency domain
    for i in prange(num_directions):
        for j in range(num_sensors):
            # Calculate phase shift for this sensor and direction
            phase_shift = np.exp(
                -1j * 2 * np.pi * frequency_bins * steering_delays_s[i, j]
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


class SteeringCalculator:
    """Calculates the geometric time delays for steering a sensor array in 3D."""

    # Parameters required for 3D steering
    steering_azimuths_rad: np.ndarray = None
    steering_elevations_rad: np.ndarray = None
    ssp: SoundSpeedProfile = None

    def __init__(
        self,
        ssp: SoundSpeedProfile,
        steering_azimuths_rad: np.ndarray = None,
        steering_elevations_rad: np.ndarray = None,
    ):
        """Initialise the SteeringCalculator.

        Args:
            ssp (SoundSpeedProfile): The sound speed profile to use for
                calculating delays.
            steering_azimuths_rad (np.ndarray, optional): The azimuth angles
                for steering, in radians.
            steering_elevations_rad (np.ndarray, optional): The elevation angles
                for steering, in radians.

        """
        self.steering_azimuths_rad = steering_azimuths_rad
        self.steering_elevations_rad = steering_elevations_rad
        self.ssp = ssp

        # Ensure that at least one steering angle is provided
        if self.steering_azimuths_rad is None and self.steering_elevations_rad is None:
            raise ValueError(
                "Must provide at least one of steering_azimuths_rad or "
                "steering_elevations_rad"
            )

    def calculate(self, platform: Platform) -> np.ndarray:
        """Calculate steering delays for the current 3D array geometry.

        Args:
            platform (Platform): The platform containing the sensor array.

        Returns:
            np.ndarray: An array of steering delays for each sensor relative to the
            steering direction, with shape (num_sensors,).

        """
        # Default to 0 for the angle that is not provided
        azimuths = (
            self.steering_azimuths_rad if self.steering_azimuths_rad is not None else 0
        )
        elevations = (
            self.steering_elevations_rad
            if self.steering_elevations_rad is not None
            else 0
        )

        # Build the 3D sensor position array from the platform's sensors list
        sensor_positions = np.array([sensor.position for sensor in platform.sensors]).T

        # Get positions relative to the array's geometric center
        sensor_positions_relative = sensor_positions - sensor_positions.mean(
            axis=1, keepdims=True
        )

        # Calculate the 3D direction vector for steering
        direction_vector = np.array(
            [
                np.cos(elevations) * np.cos(azimuths),
                np.cos(elevations) * np.sin(azimuths),
                np.sin(elevations),
            ]
        )

        # Calculate distances from each sensor to the steering direction
        distances = np.dot(direction_vector.T, sensor_positions_relative)

        # Calculate sound speed at each individual sensor's depth
        sensor_depths = sensor_positions[2, :]
        sound_speeds = self.ssp.calculate(sensor_depths)

        return distances / sound_speeds
