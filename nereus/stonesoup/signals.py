"""Defines models for generating acoustic signals for sensor arrays.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod

import numpy as np


class AcousticSignalModel:
    """Generates a complete, noiseless signal array for a given source."""

    duration_s: float
    sampling_rate_hz: int
    num_samples: int

    def __init__(self, duration_s: float, sampling_rate_hz: int, **kwargs):
        """Initialise the signal model.

        Args:
            duration_s (float): The duration of the signal snapshot in seconds.
            sampling_rate_hz (int): The sampling rate in Hertz.
            **kwargs: Additional keyword arguments.

        """
        self.duration_s = duration_s
        self.sampling_rate_hz = sampling_rate_hz
        self.num_samples = int(duration_s * sampling_rate_hz)

    def generate(
        self, source, sensor_delays_s, tloss_db, propagation_time_s
    ) -> np.ndarray:
        """Generate the signal received across all sensors from a single source.

        Args:
            source (GroundTruthState): The source state. Must contain `amplitude_upa`,
                `frequency_hz`, and `phase_rad` in its metadata dictionary.
            sensor_delays_s (np.ndarray): The relative time delay for each sensor
                in the array.
            tloss_db (float): The transmission loss in decibels.
            propagation_time_s (float): The time in seconds for the signal to
                propagate from the source to the array's origin.

        Returns:
            np.ndarray: An array of complex signals received by the sensors,
                with shape (num_sensors, num_samples).

        """
        # Create a 1D array representing the time vector for the signal snapshot
        time_array_s = np.arange(self.num_samples) / self.sampling_rate_hz

        # Attenuate the source amplitude(s) based on transmission loss
        received_amplitude_upa = source.amplitudes_upa * 10 ** (-tloss_db / 20.0)

        # --- Use NumPy broadcasting to perform calculations efficiently ---
        # Reshape arrays to dimensions: (sensors, tonals, samples)
        time_reshaped = time_array_s[np.newaxis, np.newaxis, :]
        delays_reshaped = sensor_delays_s[:, np.newaxis, np.newaxis]
        freq_reshaped = source.frequencies_hz[np.newaxis, :, np.newaxis]
        phase_reshaped = source.phases_rad[np.newaxis, :, np.newaxis]

        # Calculate the instantaneous phase for every sensor, for every tonal,
        # at every point in time.
        total_phase = (
            2
            * np.pi
            * freq_reshaped
            * (time_reshaped - propagation_time_s - delays_reshaped)
            + phase_reshaped
        )

        # Create the complex signal components using Euler's formula
        amp_reshaped = received_amplitude_upa[np.newaxis, :, np.newaxis]
        all_tonal_components = amp_reshaped * np.exp(1j * total_phase)

        # Sum the components along the tonal axis (axis=1) to get the final
        # signal at each sensor.
        sensor_signals = np.sum(all_tonal_components, axis=1)

        return sensor_signals


class NoiseModel(ABC):
    """Abstract base class for noise models."""

    def _generate_unit_white_noise(
        self, num_sensors: int, num_samples: int
    ) -> np.ndarray:
        """Generate standard complex white noise with unit power.

        Args:
            num_sensors (int): The number of sensors in the array.
            num_samples (int): The number of samples in the signal snapshot.

        Returns:
            np.ndarray: A complex array of shape (num_sensors, num_samples) with
                unit power.

        """
        # Generate real and imaginary parts from a standard normal distribution
        # and scale by 1/sqrt(2) to ensure the total power is 1.
        return (
            np.random.randn(num_sensors, num_samples)
            + 1j * np.random.randn(num_sensors, num_samples)
        ) / np.sqrt(2)

    @abstractmethod
    def generate(self, num_sensors: int) -> np.ndarray:
        """Generate a noise array. This must be implemented by subclasses."""
        raise NotImplementedError


class WhiteNoise(NoiseModel):
    """Generates complex white Gaussian noise with a flat power spectrum."""

    amplitude_upa: float
    duration_s: float
    sampling_rate_hz: int
    num_samples: int

    def __init__(
        self, amplitude_upa: float, duration_s: float, sampling_rate_hz: int, **kwargs
    ):
        """Initialise the white noise model.

        Args:
            amplitude_upa (float): The noise amplitude (e.g., in µPa).
            duration_s (float): The duration of the signal snapshot in seconds.
            sampling_rate_hz (int): The sampling rate in Hertz.
            **kwargs: Additional keyword arguments.

        """
        self.amplitude_upa = amplitude_upa
        self.duration_s = duration_s
        self.sampling_rate_hz = sampling_rate_hz
        self.num_samples = int(duration_s * sampling_rate_hz)

    def generate(self, num_sensors: int) -> np.ndarray:
        """Generate a complex white Gaussian noise array.

        Args:
            num_sensors (int): The number of sensors in the array.

        Returns:
            np.ndarray: A complex array of white noise.

        """
        # Generate the base noise with unit power
        white_noise = self._generate_unit_white_noise(num_sensors, self.num_samples)

        # Scale the unit-power noise to the target amplitude
        return self.amplitude_upa * white_noise


class ColouredNoise(NoiseModel):
    """Generates complex coloured noise using FFT filtering."""

    spectral_exponent: float
    amplitude_upa: float
    duration_s: float
    sampling_rate_hz: int
    num_samples: int

    def __init__(
        self,
        spectral_exponent: float,
        amplitude_upa: float,
        duration_s: float,
        sampling_rate_hz: int,
        **kwargs,
    ):
        """Initialise the coloured noise model.

        Args:
            spectral_exponent (float): The power-law exponent for the noise spectrum
                (e.g., -1 for pink noise, -2 for red noise, -3 for brown noise,
                -4 for violet noise, -5 for grey noise).
            amplitude_upa (float): The noise amplitude.
            duration_s (float): The duration of the signal snapshot in seconds.
            sampling_rate_hz (int): The sampling rate in Hertz.
            **kwargs: Additional keyword arguments.

        """
        self.spectral_exponent = spectral_exponent
        self.amplitude_upa = amplitude_upa
        self.duration_s = duration_s
        self.sampling_rate_hz = sampling_rate_hz
        self.num_samples = int(self.duration_s * self.sampling_rate_hz)

    def generate(self, num_sensors: int) -> np.ndarray:
        """Generate a complex coloured noise array.

        Args:
            num_sensors (int): The number of sensors in the array.

        Returns:
            np.ndarray: A complex array of coloured noise.

        """
        # 1. Generate the base white noise with a flat spectrum
        white_noise = self._generate_unit_white_noise(num_sensors, self.num_samples)

        # 2. Get the corresponding frequencies for the FFT
        freqs = np.fft.fftfreq(self.num_samples)

        # 3. Create a frequency-domain filter based on the spectral exponent
        with np.errstate(divide="ignore"):
            # The filter exponent is half the power exponent because we are
            # filtering amplitude, not power (Power ∝ Amplitude^2).
            filter_gain = np.abs(freqs) ** (self.spectral_exponent / 2.0)

        # Avoid division by zero at the DC component (frequency = 0)
        filter_gain[freqs == 0] = 1

        # 4. Apply the filter by multiplying in the frequency domain
        fft_white_noise = np.fft.fft(white_noise, axis=1)
        fft_coloured_noise = fft_white_noise * filter_gain

        # 5. Transform back to the time domain via Inverse FFT
        coloured_noise = np.fft.ifft(fft_coloured_noise, axis=1)

        # 6. Normalise the generated noise to have unit power
        power = np.mean(np.abs(coloured_noise) ** 2)
        normalised_coloured_noise = coloured_noise / np.sqrt(power)

        # 7. Scale to the desired amplitude
        return self.amplitude_upa * normalised_coloured_noise
