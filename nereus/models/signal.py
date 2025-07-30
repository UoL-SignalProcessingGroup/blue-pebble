"""Defines models for generating acoustic signals for sensor arrays.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np


class AcousticSignalModel:
    """Generates a complete, noiseless signal array for a given source.

    This class models the source's waveform, transmission loss, propagation
    delays across the array, and sensor hardware imperfections.
    """

    def generate(
        self,
        platform,
        source,
        sensor_delays: np.ndarray,
        tloss_db: float,
        propagation_time: float,
        signal_config,
    ) -> np.ndarray:
        """Generate the signal received across all sensors from a single source.

        Args:
            platform (SensorPlatform): The sensor platform.
            source (AcousticPointSource): The source model.
            sensor_delays (np.ndarray): The time delays for each sensor in the array.
            tloss_db (float): The transmission loss in decibels.
            propagation_time (float): The time taken for the signal to propagate to the
                sensors.
            signal_config (SignalConfig): Configuration parameters for the signal
                generation.

        Returns:
            np.ndarray: An array of complex signals received by the sensors, with shape
                (num_sensors, num_samples).

        """
        num_samples = int(signal_config.duration_s * signal_config.sampling_rate_hz)
        time_array = np.arange(num_samples) / signal_config.sampling_rate_hz

        # Calculate the received amplitude after transmission loss
        received_amplitude = source.amplitude * 10 ** (-tloss_db / 20.0)

        # Reshape arrays for broadcasting:
        # time_array:       (num_samples,) -> (1, 1, num_samples)
        # sensor_delays:    (num_sensors,) -> (num_sensors, 1, 1)
        # source.frequency: (num_tonals,)  -> (1, num_tonals, 1)
        # source.phase:     (num_tonals,)  -> (1, num_tonals, 1)
        time_reshaped = time_array[np.newaxis, np.newaxis, :]
        delays_reshaped = sensor_delays[:, np.newaxis, np.newaxis]
        freq_reshaped = source.frequency[np.newaxis, :, np.newaxis]
        phase_reshaped = source.phase[np.newaxis, :, np.newaxis]

        # Calculate phase for all sensors, tonals, and samples at once
        phase = (
            2
            * np.pi
            * freq_reshaped
            * (time_reshaped - propagation_time - delays_reshaped)
            + phase_reshaped
        )

        # Create complex signal components for all tonals
        # received_amplitude: (num_tonals,) -> (1, num_tonals, 1)
        amp_reshaped = received_amplitude[np.newaxis, :, np.newaxis]
        all_components = amp_reshaped * np.exp(1j * phase)

        # Sum the tonal components along the tonal axis (axis=1)
        signal_array = np.sum(all_components, axis=1)

        # Apply hardware imperfections if they exist
        if platform.has_gain_phase_errors:
            # Reshape imperfections for broadcasting: (num_sensors,) -> (num_sensors, 1)
            signal_array *= platform.complex_imperfections[:, np.newaxis]

        return signal_array


class NoiseModel(ABC):
    """Abstract base class for noise models."""

    @abstractmethod
    def generate(
        self, noise_level_db: float, num_sensors: int, num_samples: int
    ) -> np.ndarray:
        """Generate a noise array."""
        raise NotImplementedError


class WhiteNoise(NoiseModel):
    """Generates complex white Gaussian noise with a flat power spectrum."""

    def generate(
        self, noise_level_db: float, num_sensors: int, num_samples: int
    ) -> np.ndarray:
        """Generate a complex white Gaussian noise array.

        Args:
            noise_level_db (float): The noise level in decibels.
            num_sensors (int): The number of sensors in the array.
            num_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: A complex array of white noise with shape
                (num_sensors, num_samples).

        """
        # Convert dB to linear amplitude
        noise_amplitude = 10 ** (noise_level_db / 20.0)

        # Generate standard complex white noise with unit power
        white_noise = (
            np.random.randn(num_sensors, num_samples)
            + 1j * np.random.randn(num_sensors, num_samples)
        ) / np.sqrt(2)

        return noise_amplitude * white_noise


class ColouredNoise(NoiseModel):
    """Generates complex coloured noise using FFT filtering.

    The spectral shape is defined by a power-law exponent. For example:
    - Pink Noise: spectral_exponent = -1 (power ~ 1/f)
    - Red/Brownian Noise: spectral_exponent = -2 (power ~ 1/f^2)
    """

    def __init__(self, spectral_exponent: float):
        """Initialise the coloured noise model with a spectral exponent."""
        self.spectral_exponent = spectral_exponent

    def generate(
        self, noise_level_db: float, num_sensors: int, num_samples: int
    ) -> np.ndarray:
        """Generate a complex coloured noise array.

        Args:
            noise_level_db (float): The noise level in decibels.
            num_sensors (int): The number of sensors in the array.
            num_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: A complex array of coloured noise with shape
                (num_sensors, num_samples).

        """
        # 1. Generate the base white noise
        white_noise = (
            np.random.randn(num_sensors, num_samples)
            + 1j * np.random.randn(num_sensors, num_samples)
        ) / np.sqrt(2)

        # 2. Get the corresponding frequencies for the FFT
        freqs = np.fft.fftfreq(num_samples)

        # 3. Create the frequency-domain filter
        # Avoid division by zero at the DC component (f=0)
        with np.errstate(divide="ignore"):
            # The filter exponent is half the power exponent because we are
            # filtering amplitude, not power.
            filter_gain = np.abs(freqs) ** (self.spectral_exponent / 2.0)

        filter_gain[freqs == 0] = 1  # Set DC gain to 1 (0 for no DC)

        # 4. Apply the filter in the frequency domain
        fft_white_noise = np.fft.fft(white_noise, axis=1)
        fft_coloured_noise = fft_white_noise * filter_gain

        # 5. Transform back to the time domain
        coloured_noise = np.fft.ifft(fft_coloured_noise, axis=1)

        # 6. Normalise to the desired power and return
        # First, normalise the generated coloured noise to have unit power
        power = np.mean(np.abs(coloured_noise) ** 2)
        normalised_coloured_noise = coloured_noise / np.sqrt(power)

        # Then, scale to the desired amplitude
        target_amplitude = 10 ** (noise_level_db / 20.0)

        return target_amplitude * normalised_coloured_noise
