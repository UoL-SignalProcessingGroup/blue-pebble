"""Defines beamforming algorithms for processing signals from an array of sensors.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange, types
from stonesoup.base import Base, Property
from stonesoup.platform.base import Platform

from nereus.stonesoup.models.environment import SoundSpeedProfile


class Beamformer(Base, ABC):
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
        pass


class DelayAndSumBeamformer(Beamformer):
    """A Delay-and-Sum (DAS) beamformer.

    This class implements the DAS algorithm in either the time or frequency
    domain. It steers an array of sensors by applying time delays to the
    received signals and summing them to enhance the signal from desired
    directions.
    """

    sampling_rate_hz = Property(
        float, doc="The sampling frequency of the sensor signals, in Hz"
    )
    shading = Property(
        np.ndarray,
        default=None,
        doc="An array of shading weights to apply to each sensor. "
        "If None, uniform weights are used.",
    )
    domain = Property(
        str,
        default="time",
        doc="The domain for beamforming, either 'time' or 'frequency'",
    )

    def __init__(self, *args, **kwargs):
        """Initialise the DelayAndSumBeamformer.

        Args:
            *args: Positional arguments to pass to the parent class.
            **kwargs: Keyword arguments to pass to the parent class.
                Specifically expects:
                - sampling_rate_hz (float): The sampling frequency of the sensor
                  signals, in Hz.
                - shading (np.ndarray | None, optional): An array of shading weights
                  to apply to each sensor. If None, uniform weights are used.
                - domain (str, optional): The domain for beamforming, either 'time'
                  or 'frequency'. Defaults to "time".

        Raises:
            ValueError: If the specified domain is not 'time' or 'frequency'.

        """
        super().__init__(*args, **kwargs)

        if self.shading is not None:
            self.shading = self.shading / np.sum(self.shading)

        # Store number of sensors for consistent shading
        self._num_sensors = None

        if self.domain not in ["time", "frequency"]:
            raise ValueError(
                "Invalid beamforming domain. Must be 'time' or 'frequency'"
            )

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

        # Use consistent shading normalization
        if self.shading is None:
            shading_weights = np.ones(num_sensors) / num_sensors
        else:
            if len(self.shading) != num_sensors:
                raise ValueError(
                    f"Shading length ({len(self.shading)}) must match "
                    f"number of sensors ({num_sensors})"
                )
            # Shading is already normalized in __init__
            shading_weights = self.shading

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
        types.Array(types.float64, 2, "C"),
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
    delays = np.round(steering_delays_s * sampling_rate_hz).astype(np.int64)

    beamformed_signals = np.zeros((num_directions, num_samples), dtype=np.complex128)

    for i in prange(num_directions):
        for j in range(num_sensors):
            d = delays[i, j]
            # Apply negative delay to align signals (compensate for propagation delays)
            beamformed_signals[i] += shading_weights[j] * np.roll(sensor_signals[j], -d)

    # Crop the signals to remove circular shift artifacts
    # Note: Numba does not support np.max/min on the entire 2D delays array in parallel
    # loops so we compute it outside the parallel section.
    if delays.size > 0:
        max_positive_delay = np.max(np.maximum(delays, 0))
        max_negative_delay = -np.min(np.minimum(delays, 0))

        # Only crop what we can afford to crop
        left = min(max_positive_delay, num_samples // 4)  # Don't crop more than 25%
        right = min(max_negative_delay, num_samples // 4)  # Don't crop more than 25%

        # Ensure we have at least some signal left
        total_crop = left + right
        if total_crop >= num_samples:
            # If delays are too large, just crop minimally to avoid edge effects
            left = min(10, num_samples // 2)
            right = min(10, num_samples // 2 - left)
    else:
        left = 0
        right = 0

    # Apply cropping if beneficial
    if right > 0 and left + right < num_samples:
        return beamformed_signals[:, left:-right]
    elif left > 0 and left < num_samples:
        return beamformed_signals[:, left:]
    else:
        return beamformed_signals


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
                1j * 2 * np.pi * frequency_bins * steering_delays_s[i, j]
            )
            # Apply shading and phase shift, and accumulate
            beamformed_f[i] += signals_f[j] * phase_shift * shading_weights[j]

    # 5. Perform IFFT once on the final result
    beamformed_signals = np.fft.ifft(beamformed_f, axis=1)

    return beamformed_signals


class SteeringCalculator(Base):
    """Computes time delays for beamforming with a horizontal sensor array."""

    ssp = Property(SoundSpeedProfile, doc="Sound speed profile for calculating delays")
    steering_azimuths_rad = Property(
        np.ndarray, doc="Azimuth angles for steering, in radians"
    )

    def calculate(self, platform: Platform) -> np.ndarray:
        """Calculate steering delays for the current horizontal array geometry.

        Args:
            platform (Platform): The platform containing the sensor array.

        Returns:
            np.ndarray: An array of steering delays for each sensor relative to the
            steering direction, with shape (num_directions, num_sensors,).

        """
        # Get sensor positions - these are 3D positions [x, y, z] for each sensor
        sensor_positions = platform.array.state_vector  # Shape: (3, num_sensors)

        # Center the array relative to the reference sensor
        reference_position = platform.array.ref_state_vector  # Shape: (3, 1)
        sensor_positions_relative = sensor_positions - reference_position

        # Calculate the 2D direction vectors for each steering direction
        # Elevation = 0 for horizontal array, so only x-y components
        direction_vectors = np.array(
            [
                np.cos(self.steering_azimuths_rad),  # x component
                np.sin(self.steering_azimuths_rad),  # y component
                np.zeros_like(self.steering_azimuths_rad),  # z component (always 0)
            ]
        )  # Shape: (3, num_directions)

        # Calculate the projection of each sensor position onto each direction vector
        # This gives the distance along the direction of arrival for each sensor
        distances = np.dot(
            direction_vectors.T, sensor_positions_relative
        )  # Shape: (num_directions, num_sensors)

        # Get average sound speed at array depth
        array_depth = sensor_positions[2, 0]  # z-coordinate of first sensor
        sound_speed = self.ssp.calculate(array_depth)

        # Convert distances to time delays
        # Negative sign because we want delays to ADD to make signals arrive in-phase
        return -distances / sound_speed
