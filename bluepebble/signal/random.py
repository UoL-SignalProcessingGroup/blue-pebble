"""Stochastic signal generation models for sensor arrays."""

from abc import ABC, abstractmethod

import numpy as np
from stonesoup.base import Property

from .base import ComplexArray, Signal


class RandomSignal(Signal, ABC):
    """Abstract base class for stochastic signal generation models.

    These models generate random signals that can be applied either as spatially
    distributed signals across the entire sensor array, or as part of a localized
    source's radiated signature. The base class is agnostic to the deployment
    mode—subclasses implement the signal generation, and the simulator determines
    spatial distribution.

    Parameters
    ----------
    amplitude_upa : float
        The signal amplitude (e.g., in µPa).
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.
    seed : int or None, optional
        Seed for the random number generator. When ``None`` (default), the RNG
        is seeded non-deterministically. Provide an integer for reproducible
        signal realisations across runs.

    """

    amplitude_upa: float = Property(doc="The signal amplitude (e.g., in µPa)")
    seed: int | None = Property(
        default=None,
        doc="Seed for the random number generator. ``None`` gives non-deterministic output; "
        "an integer makes signal realisations reproducible across runs.",
    )

    def __init__(self, *args, **kwargs) -> None:
        """Initialise and seed the random number generator."""
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(self.seed)

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
        return (
            self._rng.standard_normal((num_sensors, n))
            + 1j * self._rng.standard_normal((num_sensors, n))
        ) / np.sqrt(2)

    @abstractmethod
    def generate(self, num_sensors: int = 1, num_samples: int | None = None) -> ComplexArray:
        """Generate a signal array. This must be implemented by subclasses.

        Parameters
        ----------
        num_sensors : int, optional
            The number of sensors in the array. Defaults to 1.
        num_samples : int or None, optional
            Number of samples to generate. Defaults to ``self.num_samples`` when ``None``.

        Returns
        -------
        ComplexArray
            Signal matrix of shape ``(num_sensors, num_samples)``.

        """


class WhiteNoise(RandomSignal):
    """Generates complex white Gaussian noise with a flat power spectrum.

    Parameters
    ----------
    amplitude_upa : float
        The signal amplitude (e.g., in µPa).
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


class ColouredNoise(RandomSignal):
    """Generates complex coloured noise using FFT filtering.

    This class generates noise with a power spectral density proportional to 1/f^alpha.

    Parameters
    ----------
    spectral_exponent : float
        The power-law exponent for the noise spectrum (e.g., -1 for pink noise, -2 for red/brownian
        noise).
    amplitude_upa : float
        The signal amplitude (e.g., in µPa).
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

        # 3. Create a frequency-domain filter based on the spectral exponent.
        #    The filter exponent is half the power exponent because we are filtering amplitude,
        #    not power (Power ∝ Amplitude^2).
        #    np.errstate suppresses the RuntimeWarning from 0.0 ** negative_exponent at the DC
        #    bin; the resulting inf/nan is corrected in the next step.
        with np.errstate(divide="ignore", invalid="ignore"):
            filter_gain = np.abs(freqs) ** (self.spectral_exponent / 2.0)

        # The DC bin (f=0) evaluates to 0^(alpha/2), which is 0 for positive alpha and inf/nan
        # for negative alpha. Either way the power-law shape is undefined at DC. We set the DC
        # gain to 1 so the bin passes through unattenuated — a pragmatic convention that avoids
        # a singularity. The consequence is that the DC component of the output is not shaped
        # by spectral_exponent and will retain the amplitude of the underlying white noise after
        # normalisation. This is generally negligible for acoustic signals.
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
