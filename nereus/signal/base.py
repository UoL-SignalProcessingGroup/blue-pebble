"""Base signal properties and methods for signal models."""

from abc import abstractmethod

import numpy as np
from stonesoup.base import Base, Property


class Signal(Base):
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

    @abstractmethod
    def _generate_base_signal(self, source) -> np.ndarray:
        """Generate the base 1D time-domain signal. Must be implemented by subclasses.

        Parameters
        ----------
        source : State
            The source state, which may be used by some models.

        Returns
        -------
        numpy.ndarray
            A 1D NumPy array representing the base signal.

        """
        pass

    def generate(self, source, sensor_delays_s, tloss_db, propagation_time_s) -> np.ndarray:
        """Generate the signal, apply attenuation, and propagate it to a sensor array.

        This method performs the following steps:
        1. Calls `_generate_base_signal` to get the source waveform.
        2. Attenuates the signal based on transmission loss.
        3. Converts the signal to the frequency domain using an FFT.
        4. Applies phase shifts to simulate propagation to each sensor.
        5. Converts the signals back to the time domain using an IFFT.

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
        # 1. Generate the base signal from the subclass implementation
        base_signal = self._generate_base_signal(source)

        # 2. Attenuate the signal
        amplitude_scaling = 10 ** (-tloss_db / 20.0)
        base_signal *= amplitude_scaling

        # 3. Convert to frequency domain
        base_signal_fft = np.fft.fft(base_signal)
        fft_freqs_hz = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)

        # 4. Calculate and apply phase shifts for propagation
        total_delays_s = propagation_time_s + sensor_delays_s
        phase_shifts = np.exp(
            -1j * 2 * np.pi * total_delays_s[:, np.newaxis] * fft_freqs_hz[np.newaxis, :]
        )
        signals_fft = base_signal_fft[np.newaxis, :] * phase_shifts

        # 5. Convert back to time domain
        signals = np.fft.ifft(signals_fft, axis=1)

        return signals.astype(np.complex128)
