"""Base signal properties and methods for signal models."""

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property


class Signal(Base, ABC):
    """Signal base class.

    This class provides a common interface for all signal types. It includes a `generate` method
    that handles signal attenuation and phase-shifting for array propagation, which calls the
    abstract `_generate_base_signal` method that subclasses must implement.

    Parameters
    ----------
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    duration_s = Property(float, doc="Duration of the signal in seconds")
    sampling_rate_hz = Property(int, doc="Sampling rate in Hertz")

    @property
    def num_samples(self) -> int:
        """Calculate the number of samples based on duration and sampling rate.

        Returns
        -------
        int
            The number of samples in the signal snapshot.

        """
        return int(self.duration_s * self.sampling_rate_hz)

    def generate(
        self,
        source: Any,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> np.ndarray:
        """Generate the signal, apply attenuation, and propagate it to a sensor array.

        Parameters
        ----------
        source : State
            The source state.
        sensor_delays_s : numpy.ndarray
            Relative time delay for each sensor in seconds.
        tloss_db : float
            Transmission loss in dB to the array origin.
        propagation_time_s : float
            Propagation time from source to origin in seconds.

        Returns
        -------
        numpy.ndarray
            An array of complex signals with shape (num_sensors, num_samples).

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

        signal_fft = np.fft.fft(base_signal)
        tloss = np.asarray(tloss_db, dtype=float)
        if tloss.ndim == 0:
            signal_fft = signal_fft * (10.0 ** (-float(tloss) / 20.0))
        elif tloss.ndim == 1:
            if len(tloss) != self.num_samples:
                msg = (
                    "tloss_db array must have length equal to num_samples when "
                    "frequency-dependent loss is provided"
                )
                raise ValueError(msg)
            signal_fft = signal_fft * (10.0 ** (-tloss / 20.0))
        else:
            msg = "tloss_db must be scalar-like or one-dimensional"
            raise ValueError(msg)

        fft_freqs_hz = np.fft.fftfreq(self.num_samples, d=1.0 / self.sampling_rate_hz)
        total_delays_s = float(propagation_time_s) + sensor_delays
        phase_shifts = np.exp(
            -1j * 2.0 * np.pi * total_delays_s[:, np.newaxis] * fft_freqs_hz[np.newaxis, :]
        )
        signals_fft: NDArray[np.complex128] = signal_fft[np.newaxis, :] * phase_shifts
        return np.fft.ifft(signals_fft, axis=1).astype(np.complex128)


class DiscreteTimestepSignal(Signal):
    """Discrete timestep signal class."""


class ContinuousTimestepSignal(Signal):
    """Continuous timestep signal class."""
