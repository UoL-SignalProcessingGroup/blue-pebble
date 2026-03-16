"""Statistical ambient noise models for sensor arrays."""

from abc import abstractmethod

import numpy as np
from stonesoup.base import Property

from .base import ComplexArray, Signal

__all__ = ["AmbientNoise", "WhiteNoise", "ColouredNoise"]


class AmbientNoise(Signal):
    """Abstract base class for ambient noise models.

    These models generate non-propagating background noise that is present across the entire sensor
    array.

    Parameters
    ----------
    amplitude_upa : float
        The noise amplitude (e.g., in µPa).
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    amplitude_upa: float = Property(doc="The noise amplitude (e.g., in µPa)")

    @property
    def num_samples(self) -> int:
        """Calculate the number of samples based on duration and sampling rate.

        Returns
        -------
        int
            The number of samples in the signal snapshot.

        """
        return int(self.duration_s * self.sampling_rate_hz)

    def _generate_unit_white_noise(
        self, num_sensors: int, num_samples: int | None = None
    ) -> ComplexArray:
        """Generate standard complex white noise with unit power.

        Parameters
        ----------
        num_sensors : int
            The number of sensors in the array.
        num_samples : int or None, optional
            Number of samples to generate. Defaults to ``self.num_samples``.

        Returns
        -------
        ComplexArray
            Complex array of shape ``(num_sensors, num_samples)`` with unit
            power.

        """
        n = num_samples if num_samples is not None else self.num_samples
        # Generate real and imaginary parts from a standard normal distribution
        # and scale by 1/sqrt(2) to ensure the total power is 1.
        return (np.random.randn(num_sensors, n) + 1j * np.random.randn(num_sensors, n)) / np.sqrt(
            2
        )

    @abstractmethod
    def generate(self, num_sensors: int = 1, num_samples: int | None = None) -> ComplexArray:
        """Generate a noise array. This must be implemented by subclasses.

        Parameters
        ----------
        num_sensors : int, optional
            The number of sensors in the array. Defaults to 1.
        num_samples : int or None, optional
            Number of samples to generate. Defaults to ``self.num_samples`` when ``None``.

        Returns
        -------
        ComplexArray
            Noise matrix of shape ``(num_sensors, num_samples)``.

        """


class WhiteNoise(AmbientNoise):
    """Generates complex white Gaussian noise with a flat power spectrum.

    Parameters
    ----------
    amplitude_upa : float
        The noise amplitude (e.g., in µPa).
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    def generate(self, num_sensors: int = 1, num_samples: int | None = None) -> ComplexArray:
        """Generate a complex white Gaussian noise array.

        Parameters
        ----------
        num_sensors : int, optional
            The number of sensors in the array. Defaults to 1.
        num_samples : int or None, optional
            Number of samples to generate. Defaults to ``self.num_samples`` when ``None``.

        Returns
        -------
        ComplexArray
            Complex white-noise matrix of shape ``(num_sensors, num_samples)``.

        """
        white_noise = self._generate_unit_white_noise(num_sensors, num_samples)
        return self.amplitude_upa * white_noise


class ColouredNoise(AmbientNoise):
    """Generates complex coloured noise using FFT filtering.

    This class generates noise with a power spectral density proportional to 1/f^alpha.

    Parameters
    ----------
    spectral_exponent : float
        The power-law exponent for the noise spectrum (e.g., -1 for pink noise, -2 for red/brownian
        noise).
    amplitude_upa : float
        The noise amplitude (e.g., in µPa).
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    spectral_exponent: float = Property(
        doc="The power-law exponent for the noise spectrum (e.g., -1 for pink noise, -2 for "
        "red/brownian noise)."
    )

    def generate(self, num_sensors: int = 1, num_samples: int | None = None) -> ComplexArray:
        """Generate a complex coloured noise array.

        Parameters
        ----------
        num_sensors : int, optional
            The number of sensors in the array. Defaults to 1.
        num_samples : int or None, optional
            Number of samples to generate. Defaults to ``self.num_samples`` when ``None``.

        Returns
        -------
        ComplexArray
            Complex coloured-noise matrix of shape ``(num_sensors, num_samples)``,
            normalised to the specified amplitude.

        """
        n = num_samples if num_samples is not None else self.num_samples
        # 1. Generate the base white noise with a flat spectrum
        white_noise = self._generate_unit_white_noise(num_sensors, n)

        # 2. Get the corresponding frequencies for the FFT
        freqs = np.fft.fftfreq(n)

        # 3. Create a frequency-domain filter based on the spectral exponent
        with np.errstate(divide="ignore"):
            # The filter exponent is half the power exponent because we are filtering amplitude,
            # not power (Power ∝ Amplitude^2).
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
