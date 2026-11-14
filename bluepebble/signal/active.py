"""Active sonar transmit waveforms: CW (continuous wave) and LFM (linear frequency modulated)."""

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property

from .base import Signal

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]


class ActiveSignal(Signal, ABC):
    r"""Abstract base class for active sonar transmit pulses.

    A pulse is generated once, as a complex analytic time series at the carrier frequency, and
    a simulator delays and scales copies of it along each propagation path.

    With ``baseband`` set, the pulse is generated as its complex envelope about the carrier
    frequency :math:`f_c` instead,

    .. math::

        s_\text{bb}(t) = s(t) \, e^{-j 2 \pi f_c t},

    so the sampling rate only has to cover the pulse's bandwidth, not its highest frequency.

    Parameters
    ----------
    source_level_db : float
        Source level in dB re 1 µPa at 1 m: the pulse's plateau amplitude is
        :math:`A = 10^{SL/20}` µPa.
    phase_rad : float, optional
        Carrier phase offset in radians.
    baseband : bool, optional
        Generate the complex envelope about ``carrier_frequency_hz`` rather than the pulse at
        its carrier.
    duration_s : float
        Pulse duration in seconds.
    sampling_rate_hz : int
        Sampling rate in Hz.

    Raises
    ------
    ValueError
        If ``duration_s`` is not positive. At passband, if the sampling rate does not exceed
        the highest frequency in the pulse. At baseband, if the sampling rate does not exceed
        twice the pulse's largest offset from the carrier, or half of it is not below the
        carrier.

    """

    source_level_db: float = Property(doc="Source level in dB re 1 µPa at 1 m")
    phase_rad: float = Property(default=0.0, doc="Carrier phase offset in radians")
    baseband: bool = Property(
        default=False,
        doc="Generate the complex envelope about the carrier frequency, not the pulse at its "
        "carrier",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the pulse and check it can be sampled.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        if self.duration_s <= 0:
            raise ValueError(f"duration_s ({self.duration_s}s) must be positive.")
        if self.baseband:
            self._check_baseband_sampling()
        elif self.sampling_rate_hz <= self.highest_frequency_hz:
            # A complex analytic signal needs fs > f_max, not 2 f_max, to avoid aliasing.
            raise ValueError(
                f"sampling_rate_hz ({self.sampling_rate_hz}Hz) must exceed the pulse's highest "
                f"frequency ({self.highest_frequency_hz}Hz)."
            )

    def _check_baseband_sampling(self) -> None:
        """Raise unless the sampling rate holds the envelope and maps to positive frequencies."""
        carrier_hz = self.carrier_frequency_hz
        largest_offset_hz = max(
            self.highest_frequency_hz - carrier_hz, carrier_hz - self.lowest_frequency_hz
        )
        if self.sampling_rate_hz <= 2 * largest_offset_hz:
            raise ValueError(
                f"sampling_rate_hz ({self.sampling_rate_hz}Hz) must exceed twice the pulse's "
                f"largest offset from its carrier ({largest_offset_hz}Hz) at baseband."
            )
        if self.sampling_rate_hz / 2 >= carrier_hz:
            # Bin k of the envelope's spectrum is the physical frequency f_c + f_k, which a
            # propagation model needs to be positive.
            raise ValueError(
                f"Half of sampling_rate_hz ({self.sampling_rate_hz}Hz) must be below the "
                f"carrier ({carrier_hz}Hz) at baseband, so that every frequency bin is a "
                "positive physical frequency."
            )

    @property
    def amplitude(self) -> float:
        """Plateau amplitude in µPa at 1 m, :math:`10^{SL/20}`."""
        return float(10 ** (self.source_level_db / 20.0))

    @property
    @abstractmethod
    def carrier_frequency_hz(self) -> float:
        """Centre frequency, which a baseband pulse is the complex envelope about."""

    @property
    @abstractmethod
    def highest_frequency_hz(self) -> float:
        """Highest frequency in the pulse, which a passband sampling rate must exceed."""

    @property
    @abstractmethod
    def lowest_frequency_hz(self) -> float:
        """Lowest frequency in the pulse."""

    @property
    def demodulation_frequency_hz(self) -> float:
        """Physical frequency of 0 Hz in the generated pulse: the carrier at baseband, else 0."""
        return float(self.carrier_frequency_hz) if self.baseband else 0.0

    @abstractmethod
    def generate(self) -> ComplexArray:
        """Generate the transmit pulse: the analytic signal, or its envelope at baseband.

        Returns
        -------
        ComplexArray
            Complex waveform of shape ``(num_samples,)``.

        """


class CWSignal(ActiveSignal):
    r"""CW (continuous wave) pulse: a single tone for the whole pulse.

    .. math::

        s(t) = A \, e^{j (2 \pi f t + \phi)}, \qquad 0 \le t < T

    At baseband the envelope is the constant :math:`A \, e^{j \phi}`.

    Parameters
    ----------
    frequency_hz : float
        Carrier frequency in Hz.
    source_level_db : float
        Source level in dB re 1 µPa at 1 m.
    phase_rad : float, optional
        Carrier phase offset in radians.
    baseband : bool, optional
        Generate the complex envelope about ``frequency_hz``.
    duration_s : float
        Pulse duration in seconds.
    sampling_rate_hz : int
        Sampling rate in Hz.

    Assumptions
    -----------
    - The pulse switches on and off instantly (rectangular envelope), so its spectrum has
      sinc sidelobes.

    """

    frequency_hz: float = Property(doc="Carrier frequency in Hz")

    @property
    def carrier_frequency_hz(self) -> float:
        """The carrier frequency."""
        return self.frequency_hz

    @property
    def highest_frequency_hz(self) -> float:
        """The carrier frequency."""
        return self.frequency_hz

    @property
    def lowest_frequency_hz(self) -> float:
        """The carrier frequency."""
        return self.frequency_hz

    def generate(self) -> ComplexArray:
        """Generate the CW pulse.

        Returns
        -------
        ComplexArray
            Complex sinusoid of shape ``(num_samples,)``, or a constant at baseband.

        """
        if self.baseband:
            return np.full(
                self.num_samples,
                self.amplitude * np.exp(1j * self.phase_rad),
                dtype=np.complex128,
            )
        t = np.arange(self.num_samples) / self.sampling_rate_hz
        return np.asarray(
            self.amplitude * np.exp(2j * np.pi * self.frequency_hz * t + 1j * self.phase_rad),
            dtype=np.complex128,
        )


class LFMSignal(ActiveSignal):
    r"""LFM (linear frequency modulated) pulse: a chirp with tapered edges.

    The frequency sweeps linearly from :math:`f_0` to :math:`f_1` over the pulse duration
    :math:`T`, so the phase is quadratic, and a quarter-sine taper of length :math:`r` at each
    edge suppresses spectral sidelobes:

    .. math::

        s(t) = A \, w(t) \, e^{j \left(2 \pi \left(f_0 t + \tfrac{1}{2} k t^2\right)
            + \phi\right)}, \qquad k = \frac{f_1 - f_0}{T}

    where :math:`w(t) = \sin(\pi t / 2r)` for :math:`t \le r`,
    :math:`w(t) = \sin(\pi (T - t) / 2r)` for :math:`t > T - r`, and 1 in between.

    At baseband the carrier is the sweep centre :math:`f_c = (f_0 + f_1) / 2`, so the envelope
    is the same chirp with :math:`f_0` replaced by :math:`f_0 - f_c = -(f_1 - f_0) / 2`. The
    taper's spectral skirts reach beyond the sweep, and the part of them beyond half the
    sampling rate aliases. For a 50 ms, 400 Hz sweep with 5 ms tapers sampled at 1 kHz, the
    aliased part is 0.9% of the pulse's RMS (root-mean-square) amplitude; it falls with a
    longer taper, a longer pulse or a higher sampling rate.

    Parameters
    ----------
    freq_min_hz : float
        Start frequency of the sweep in Hz.
    freq_max_hz : float
        End frequency of the sweep in Hz.
    rise_time_s : float
        Duration of the taper at each pulse edge in seconds.
    source_level_db : float
        Source level in dB re 1 µPa at 1 m, at the plateau.
    phase_rad : float, optional
        Carrier phase offset in radians.
    baseband : bool, optional
        Generate the complex envelope about the sweep centre.
    duration_s : float
        Pulse duration in seconds.
    sampling_rate_hz : int
        Sampling rate in Hz.

    Raises
    ------
    ValueError
        If ``freq_max_hz`` does not exceed ``freq_min_hz``, or ``rise_time_s`` is not in
        ``(0, duration_s / 2]``.

    """

    freq_min_hz: float = Property(doc="Start frequency of the LFM sweep in Hz")
    freq_max_hz: float = Property(doc="End frequency of the LFM sweep in Hz")
    rise_time_s: float = Property(doc="Taper duration at each pulse edge in seconds")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the LFM pulse and check its sweep and taper.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        if self.freq_max_hz <= self.freq_min_hz:
            raise ValueError(
                f"freq_max_hz ({self.freq_max_hz}Hz) must exceed freq_min_hz "
                f"({self.freq_min_hz}Hz)."
            )
        if not 0 < self.rise_time_s <= self.duration_s / 2:
            raise ValueError(
                f"rise_time_s ({self.rise_time_s}s) must be positive and at most half of "
                f"duration_s ({self.duration_s}s)."
            )

    @property
    def carrier_frequency_hz(self) -> float:
        """The centre of the sweep."""
        return (self.freq_min_hz + self.freq_max_hz) / 2.0

    @property
    def highest_frequency_hz(self) -> float:
        """The end of the sweep."""
        return self.freq_max_hz

    @property
    def lowest_frequency_hz(self) -> float:
        """The start of the sweep."""
        return self.freq_min_hz

    def _envelope(self, t: FloatArray) -> FloatArray:
        """Compute the quarter-sine rise and fall envelope, in ``[0, 1]``."""
        d = self.duration_s
        r = self.rise_time_s
        envelope = np.ones_like(t)
        envelope = np.where(t <= r, np.sin(np.pi / 2 * t / r), envelope)
        envelope = np.where(t > d - r, np.sin(np.pi / 2 * (t - (d - 2 * r)) / r), envelope)
        envelope = np.where(t > d, 0.0, envelope)
        return np.asarray(envelope, dtype=np.float64)

    def generate(self) -> ComplexArray:
        """Generate the tapered LFM pulse.

        Returns
        -------
        ComplexArray
            Complex LFM waveform of shape ``(num_samples,)``.

        """
        t = np.arange(self.num_samples) / self.sampling_rate_hz
        sweep_rate = (self.freq_max_hz - self.freq_min_hz) / self.duration_s
        start_frequency_hz = self.freq_min_hz - self.demodulation_frequency_hz
        phase = 2 * np.pi * (start_frequency_hz * t + 0.5 * sweep_rate * t**2)
        chirp = np.exp(1j * (phase + self.phase_rad))
        return np.asarray(self.amplitude * self._envelope(t) * chirp, dtype=np.complex128)
