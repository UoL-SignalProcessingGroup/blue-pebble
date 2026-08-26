"""Active sonar signal models: CW and LFM waveforms."""

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property

from .base import Signal

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]


class ActiveSignal(Signal, ABC):
    """Abstract base class for active sonar transmit waveforms.

    Subclasses generate a single pulse as a complex analytic time-series
    suitable for eigenray-based active sonar simulation.
    """

    source_level_db: float = Property(doc="Source level in dB re 1 µPa")
    phase_rad: float = Property(default=0.0, doc="Carrier phase offset in radians")

    @property
    def amplitude(self) -> float:
        """Linear amplitude derived from source level in dB re 1 µPa."""
        return float(10 ** (self.source_level_db / 20.0))

    @property
    @abstractmethod
    def carrier_frequency_hz(self) -> float:
        """Centre frequency used for fractional time-delay phase shifts."""
        ...

    @abstractmethod
    def generate(self) -> ComplexArray:
        """Generate the transmit pulse as a complex analytic signal.

        Returns
        -------
        ComplexArray
            Complex waveform of shape ``(num_samples,)``.

        """
        ...


class CWSignal(ActiveSignal):
    """Continuous wave (single-tone) active sonar pulse.

    Generates a complex sinusoid at a fixed frequency for the full pulse duration.

    Parameters
    ----------
    frequency_hz : float
        Carrier frequency in Hz.
    source_level_db : float
        Source level in dB re 1 µPa.
    phase_rad : float, optional
        Carrier phase offset in radians. Default is 0.0.
    duration_s : float
        Pulse duration in seconds.
    sampling_rate_hz : int
        Sampling rate in Hz.

    """

    frequency_hz: float = Property(doc="Carrier frequency in Hz")

    @property
    def carrier_frequency_hz(self) -> float:
        return self.frequency_hz

    def generate(self) -> ComplexArray:
        """Generate a CW pulse.

        Returns
        -------
        ComplexArray
            Complex sinusoid of shape ``(num_samples,)``.

        """
        t = np.arange(self.num_samples) / self.sampling_rate_hz
        return np.asarray(
            self.amplitude * np.exp(2j * np.pi * self.frequency_hz * t + 1j * self.phase_rad),
            dtype=np.complex128,
        )


class LFMSignal(ActiveSignal):
    """Linear frequency modulated (chirp) active sonar pulse.

    Generates a complex LFM waveform with a sinusoidal amplitude taper at the
    leading and trailing edges to suppress spectral sidelobes.

    Parameters
    ----------
    freq_min_hz : float
        Start frequency of the sweep in Hz.
    freq_max_hz : float
        End frequency of the sweep in Hz.
    rise_time_s : float
        Duration of the sinusoidal taper at each pulse edge in seconds.
    source_level_db : float
        Source level in dB re 1 µPa.
    phase_rad : float, optional
        Carrier phase offset in radians. Default is 0.0.
    duration_s : float
        Pulse duration in seconds.
    sampling_rate_hz : int
        Sampling rate in Hz.

    """

    freq_min_hz: float = Property(doc="Start frequency of the LFM sweep in Hz")
    freq_max_hz: float = Property(doc="End frequency of the LFM sweep in Hz")
    rise_time_s: float = Property(doc="Sinusoidal taper duration at each pulse edge in seconds")

    @property
    def carrier_frequency_hz(self) -> float:
        return (self.freq_min_hz + self.freq_max_hz) / 2.0

    def _envelope(self, t: FloatArray) -> FloatArray:
        """Compute the sinusoidal rise/fall amplitude envelope.

        Parameters
        ----------
        t : FloatArray
            Time axis in seconds, shape ``(num_samples,)``.

        Returns
        -------
        FloatArray
            Envelope values in [0, 1], same shape as ``t``.

        """
        d = self.duration_s
        r = self.rise_time_s
        envelope = np.ones_like(t)
        envelope = np.where(t <= r, np.sin(np.pi / 2 * t / r), envelope)
        envelope = np.where(t > d - r, np.sin(np.pi / 2 * (t - (d - 2 * r)) / r), envelope)
        envelope = np.where(t > d, 0.0, envelope)
        return np.asarray(envelope, dtype=np.float64)

    def generate(self) -> ComplexArray:
        """Generate a shaped LFM pulse.

        The instantaneous phase is the integral of the linearly-swept frequency,
        producing a quadratic phase (chirp). The sinusoidal envelope suppresses
        spectral sidelobes at the pulse edges.

        Returns
        -------
        ComplexArray
            Complex LFM waveform of shape ``(num_samples,)``.

        """
        t = np.arange(self.num_samples) / self.sampling_rate_hz
        sweep_rate = (self.freq_max_hz - self.freq_min_hz) / self.duration_s
        phase = 2 * np.pi * (self.freq_min_hz * t + 0.5 * sweep_rate * t**2)
        chirp = np.exp(1j * (phase + self.phase_rad))
        envelope = self._envelope(t)
        return np.asarray(self.amplitude * envelope * chirp, dtype=np.complex128)
