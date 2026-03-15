"""Base signal properties and methods for signal models."""

from abc import ABC
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

if TYPE_CHECKING:
    from stonesoup.types.state import State

ComplexArray: TypeAlias = NDArray[np.complex128]


def _get_source_metadata(source: "State") -> Mapping[str, object]:
    """Validate and return the metadata mapping from a source state.

    Parameters
    ----------
    source : State
        Source state exposing a ``metadata`` attribute.

    Returns
    -------
    Mapping[str, object]
        The validated metadata mapping.

    Raises
    ------
    ValueError
        If ``metadata`` is absent or not mapping-like.

    """
    metadata = getattr(source, "metadata", None)
    if not isinstance(metadata, Mapping):
        msg = "Source state metadata must be mapping-like"
        raise ValueError(msg)
    return metadata


class Signal(Base, ABC):
    """Shared sampling properties for all signal and noise models.

    Provides ``duration_s``, ``sampling_rate_hz``, and the derived
    ``num_samples`` property.  Both :class:`Signal` (per-timestep path) and
    :class:`~bluepebble.signal.anthropogenic.AnthropogenicSignal` (STFT-first
    path) inherit from this class so that they share a common parameter
    contract without one being a subtype of the other.

    Parameters
    ----------
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    duration_s: float = Property(doc="Duration of the signal in seconds")
    sampling_rate_hz: int = Property(doc="Sampling rate in Hertz")

    @property
    def num_samples(self) -> int:
        """Calculate the number of samples based on duration and sampling rate.

        Returns
        -------
        int
            The number of samples in the signal snapshot.

        """
        return int(self.duration_s * self.sampling_rate_hz)

    def get_source_waveform(self, source: "State") -> ComplexArray:
        """Return the full-duration source waveform for this signal model.

        Delegates to :meth:`_generate_base_signal`.

        Parameters
        ----------
        source : State
            Source state passed to the underlying waveform generator.

        Returns
        -------
        ComplexArray
            Full-duration source waveform as ``complex128``.

        """
        return np.asarray(self._generate_base_signal(source), dtype=np.complex128)  # type: ignore[attr-defined]

    def _apply_propagation(
        self,
        base_signal: ComplexArray,
        sensor_delays_s: NDArray[np.float64],
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> ComplexArray:
        """Apply transmission loss and per-sensor phase delays to a base signal.

        Parameters
        ----------
        base_signal : ComplexArray
            1-D complex source waveform with length ``num_samples``.
        sensor_delays_s : NDArray[np.float64]
            1-D array of per-sensor relative delays in seconds.
        tloss_db : ArrayLike | float
            Transmission loss in dB. Either a scalar applied uniformly across all
            frequencies, or a 1-D array of length ``num_samples`` for
            frequency-dependent loss.
        propagation_time_s : float
            Propagation time from the source to the array origin in seconds.

        Returns
        -------
        ComplexArray
            Complex signal matrix with shape ``(num_sensors, num_samples)``.

        """
        num_samples = len(base_signal)
        signal_fft = np.fft.fft(base_signal)

        tloss = np.asarray(tloss_db, dtype=float)
        if tloss.ndim == 0:
            signal_fft = signal_fft * (10.0 ** (-float(tloss) / 20.0))
        elif tloss.ndim == 1:
            if len(tloss) != num_samples:
                msg = (
                    "tloss_db array must have length equal to num_samples when "
                    "frequency-dependent loss is provided"
                )
                raise ValueError(msg)
            signal_fft = signal_fft * (10.0 ** (-tloss / 20.0))
        else:
            msg = "tloss_db must be scalar-like or one-dimensional"
            raise ValueError(msg)

        fft_freqs_hz = np.fft.fftfreq(num_samples, d=1.0 / self.sampling_rate_hz)
        total_delays_s = float(propagation_time_s) + sensor_delays_s
        phase_shifts = np.exp(
            -1j * 2.0 * np.pi * total_delays_s[:, np.newaxis] * fft_freqs_hz[np.newaxis, :]
        )
        signals_fft: ComplexArray = signal_fft[np.newaxis, :] * phase_shifts
        return np.fft.ifft(signals_fft, axis=1).astype(np.complex128)

    def generate(
        self,
        source: "State",
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> ComplexArray:
        """Generate the signal, apply attenuation, and propagate it to a sensor array.

        Parameters
        ----------
        source : State
            The source state.
        sensor_delays_s : ArrayLike
            Relative time delay for each sensor in seconds.
        tloss_db : ArrayLike | float
            Transmission loss in dB to the array origin.
        propagation_time_s : float
            Propagation time from source to origin in seconds.

        Returns
        -------
        ComplexArray
            Complex signal matrix with shape ``(num_sensors, num_samples)``.

        """
        sensor_delays = np.asarray(sensor_delays_s, dtype=float)
        if sensor_delays.ndim != 1:
            msg = "sensor_delays_s must be one-dimensional"
            raise ValueError(msg)

        base_generator = getattr(self, "_generate_base_signal", None)
        if not callable(base_generator):
            msg = (
                f"{type(self).__name__} must implement either generate() or "
                "_generate_base_signal(source)"
            )
            raise NotImplementedError(msg)

        base_signal = np.asarray(base_generator(source), dtype=np.complex128)
        if base_signal.ndim != 1:
            msg = "_generate_base_signal(source) must return a one-dimensional array"
            raise ValueError(msg)

        if len(base_signal) < self.num_samples:
            pad_width = self.num_samples - len(base_signal)
            base_signal = np.concatenate(
                [base_signal, np.zeros(pad_width, dtype=np.complex128)],
            )
        elif len(base_signal) > self.num_samples:
            base_signal = base_signal[: self.num_samples]

        return self._apply_propagation(base_signal, sensor_delays, tloss_db, propagation_time_s)
