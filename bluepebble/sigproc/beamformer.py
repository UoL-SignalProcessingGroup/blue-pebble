"""Beamforming algorithms for processing signals from an array of sensors."""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange, types
from scipy.linalg import cho_factor, cho_solve
from stonesoup.base import Base, Property
from stonesoup.platform.base import Platform

from ..models.environment import SoundSpeedProfile


class Beamformer(Base, ABC):
    """Abstract base class for beamformers."""

    @abstractmethod
    def beamform(self, sensor_signals: np.ndarray, steering_delays_s: np.ndarray) -> np.ndarray:
        """Process sensor signals to form beams in specified directions.

        Parameters
        ----------
        sensor_signals : np.ndarray
            An array of sensor signals with shape (num_sensors, num_samples).
        steering_delays_s : np.ndarray
            An array of time delays in seconds for each sensor and steering direction,
            with shape (num_directions, num_sensors).

        Returns
        -------
        np.ndarray
            An array of beamformed signals with shape (num_directions, num_samples).

        """
        pass


class DelayAndSumBeamformer(Beamformer):
    """A Delay-and-Sum (DAS) beamformer.

    This class implements the DAS algorithm in either the time or frequency domain. It steers an
    array of sensors by applying time delays to the received signals and summing them to enhance
    the signal from desired directions.
    """

    sampling_rate_hz = Property(float, doc="The sampling frequency of the sensor signals, in Hz")
    shading = Property(
        np.ndarray,
        default=None,
        doc="An array of shading weights applied to each sensor. If None, uniform weights are "
        "used.",
    )
    domain = Property(
        str,
        default="time",
        doc="The domain for beamforming, either 'time' or 'frequency'",
    )
    nfft = Property(int, default=500, doc="STFT window size (samples)")
    overlap = Property(int, default=250, doc="STFT overlap (samples)")
    f0 = Property(float, default=0.0, doc="Carrier frequency for baseband data (Hz)")
    fmin = Property(float, default=None, doc="Minimum frequency to integrate (Hz)")
    fmax = Property(float, default=None, doc="Maximum frequency to integrate (Hz)")

    def __init__(self, *args, **kwargs):
        """Initialise the DelayAndSumBeamformer.

        Parameters
        ----------
        *args
            Positional arguments to pass to the parent class.
        **kwargs
            Keyword arguments to pass to the parent class.
            Specifically expects:
            - sampling_rate_hz : float
                The sampling frequency of the sensor
                    signals, in Hz.
            - shading : np.ndarray | None, optional
                An array of shading weights
                    to apply to each sensor. If None, uniform weights are used.
            - domain : str, optional
                The domain for beamforming, either 'time'
                    or 'frequency'. Defaults to "time".

        Raises
        ------
        ValueError
            If the specified domain is not 'time' or 'frequency'.

        """
        super().__init__(*args, **kwargs)

        if self.shading is not None:
            shading_sum = np.sum(self.shading)
            if not np.isfinite(shading_sum) or np.isclose(shading_sum, 0.0):
                raise ValueError("Shading weights must sum to a finite non-zero value")
            self.shading = self.shading / shading_sum

        # Store number of sensors for consistent shading
        self._num_sensors = None

        if self.domain not in ["time", "frequency", "broadband_power"]:
            raise ValueError("Invalid beamforming domain. Must be 'time' or 'frequency' or 'broadband_power'.")

    def beamform(self, sensor_signals: np.ndarray, steering_delays_s: np.ndarray) -> np.ndarray:
        """Process sensor signals to form beams in specified directions.

        Parameters
        ----------
        sensor_signals : np.ndarray
            An array of sensor signals with shape (num_sensors, num_samples).
        steering_delays_s : np.ndarray
            An array of time delays for each sensor and steering direction, with shape
            (num_directions, num_sensors).

        Returns
        -------
        np.ndarray
            An array of beamformed signals with shape (num_directions, num_samples).

        Raises
        ------
        ValueError
            If the number of sensors in signal_array does not match the number of sensors in
            steering_delays.

        """
        # Numba kernels below are compiled for complex128/float64 C-contiguous arrays.
        # Normalizing inputs here prevents dispatcher type mismatches (e.g. complex64 data).
        sensor_signals = np.ascontiguousarray(sensor_signals, dtype=np.complex128)
        steering_delays_s = np.ascontiguousarray(steering_delays_s, dtype=np.float64)

        num_sensors, _ = sensor_signals.shape

        if num_sensors != steering_delays_s.shape[1]:
            raise ValueError("Number of sensors must match the number of steering delays")

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

        shading_weights = np.ascontiguousarray(shading_weights, dtype=np.float64)

        if self.domain == "time":
            return _time_das(
                sensor_signals,
                steering_delays_s,
                shading_weights,
                self.sampling_rate_hz,
            )
        elif self.domain == "frequency":
            return _frequency_das(
                sensor_signals,
                steering_delays_s,
                shading_weights,
                self.sampling_rate_hz,
            )
        elif self.domain == "broadband_power":
            return self.das_broadband_power(
                sensor_signals,
                self.sampling_rate_hz,
                self.nfft,
                steering_delays_s,
                shading_weights,
                f0=self.f0,
                fmin=self.fmin,
                fmax=self.fmax,
                overlap=self.overlap,
            )

    @staticmethod
    def _stft(x: np.ndarray, nfft: int, overlap: int) -> np.ndarray:
        """Compute the Short-Time Fourier Transform (STFT) of the input signal.

        Parameters
        ----------
        x : np.ndarray
            Input signal array of shape (M, T) where M is the number of sensors and T is the number
            of time samples.
        nfft : int
            The number of FFT points (window size).
        overlap : int
            The number of overlapping samples between windows.

        Returns
        -------
        np.ndarray
            STFT of the input signal with shape (M, n_frames, nfft).

        """
        # x: (M, T) complex
        M, T = x.shape
        if T < nfft:
            raise ValueError(f"Input signal length T={T} is less than window size nfft={nfft}.")
        hop = max(1, nfft - overlap)
        # pad to fit last frame exactly
        n_frames = 1 + (max(0, T - nfft) // hop)
        pad = (n_frames - 1) * hop + nfft - T
        if pad > 0:
            x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

        window = np.hanning(nfft).astype(x.real.dtype)
        # Make a 3D view: (M, n_frames, nfft)
        stride_t = x.strides[1]
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(M, n_frames, nfft),
            strides=(x.strides[0], hop * stride_t, stride_t),
            writeable=False,
        )
        frames = frames * window  # broadcasts over last axis
        return np.fft.fft(frames, axis=2)


class DelayAndSumBeamformerFast(DelayAndSumBeamformer):
    """Vectorized broadband DAS with optional per-frequency output.

    This class preserves the default broadband interface of
    :class:`DelayAndSumBeamformer` for ``domain="broadband_power"``:
    by default it returns shape ``(num_directions, num_frames)``.

    Set ``output_mode="btf"`` to return per-frequency power with shape
    ``(num_directions, num_frames, num_active_freq_bins)``.
    """

    output_mode = Property(
        str,
        default="bt",
        doc=(
            "Broadband output mode: 'bt' -> [direction, frame] (default), "
            "'btf' -> [direction, frame, frequency]."
        ),
    )

    @staticmethod
    def _active_freq_bins(
        *,
        fs: float,
        nfft_actual: int,
        f0: float,
        fmin: float | None,
        fmax: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return active FFT-bin indices and frequencies for broadband integration."""
        k = np.arange(nfft_actual)
        k_centered = np.where(k <= nfft_actual // 2, k, k - nfft_actual)
        f_bins = f0 + (fs / nfft_actual) * k_centered

        fmin_use = f_bins.min() if fmin is None else float(fmin)
        fmax_use = f_bins.max() if fmax is None else float(fmax)
        active = (f_bins >= fmin_use) & (f_bins <= fmax_use)
        active_idx = np.nonzero(active)[0]
        return active_idx, f_bins[active_idx]

    def das_broadband_power_btf(
        self,
        x: np.ndarray,  # (M, T)
        fs: float,
        nfft: int,
        sd: np.ndarray,  # (Ndir, M) steering delays [s]
        shading_weights: np.ndarray,  # (M,)
        f0: float = 0.0,
        fmin: float | None = None,
        fmax: float | None = None,
        overlap: int = 0,
    ) -> np.ndarray:
        """Return DAS power cube with shape ``(dir, frame, freq_active)``."""
        _, _ = x.shape
        x_stft = self._stft(x, nfft, overlap)  # (M, n_frames, nfft)
        m, n_frames, nfft_actual = x_stft.shape

        active_idx, active_freqs = self._active_freq_bins(
            fs=fs,
            nfft_actual=nfft_actual,
            f0=f0,
            fmin=fmin,
            fmax=fmax,
        )
        n_dir = sd.shape[0]
        power_btf = np.zeros((n_dir, n_frames, active_idx.size), dtype=np.float64)
        if active_idx.size == 0:
            return power_btf

        # Process active bins in blocks to reduce Python loop overhead.
        w = np.asarray(shading_weights, dtype=np.float64).reshape(1, m, 1)
        phase_scale = (1j * 2.0 * np.pi) * np.asarray(sd, dtype=np.float64)[:, :, None]
        freq_block = 16

        for block_start in range(0, active_idx.size, freq_block):
            block_idx_local = slice(block_start, block_start + freq_block)
            block_bins = active_idx[block_idx_local]
            f_block = active_freqs[block_idx_local]
            s_block = x_stft[:, :, block_bins]  # (M, n_frames, Fblk)
            a_block = np.exp(phase_scale * f_block[None, None, :])  # (Ndir, M, Fblk)
            y_block = np.einsum(
                "dmf,mtf,dmf->dtf",
                a_block,
                s_block,
                w,
                optimize="greedy",
            )
            power_btf[:, :, block_start : block_start + y_block.shape[2]] = (
                np.abs(y_block) ** 2
            )

        return power_btf

    def das_broadband_power(
        self,
        x: np.ndarray,  # (M, T)
        fs: float,
        nfft: int,
        sd: np.ndarray,  # (Ndir, M) steering delays [s]
        shading_weights: np.ndarray,  # (M,)
        f0: float = 0.0,
        fmin: float | None = None,
        fmax: float | None = None,
        overlap: int = 0,
    ) -> np.ndarray:
        """Perform broadband DAS and return shape ``(direction, frame)``."""
        power_btf = self.das_broadband_power_btf(
            x=x,
            fs=fs,
            nfft=nfft,
            sd=sd,
            shading_weights=shading_weights,
            f0=f0,
            fmin=fmin,
            fmax=fmax,
            overlap=overlap,
        )
        return np.sum(power_btf, axis=2, dtype=np.float64)

    def beamform(
        self, sensor_signals: np.ndarray, steering_delays_s: np.ndarray
    ) -> np.ndarray:
        """Process sensor signals; default broadband output matches base class."""
        if self.domain != "broadband_power":
            return super().beamform(sensor_signals, steering_delays_s)

        num_sensors, _ = sensor_signals.shape
        if num_sensors != steering_delays_s.shape[1]:
            raise ValueError(
                "Number of sensors must match the number of steering delays"
            )

        if self.shading is None:
            shading_weights = np.ones(num_sensors) / num_sensors
        else:
            if len(self.shading) != num_sensors:
                raise ValueError(
                    f"Shading length ({len(self.shading)}) must match "
                    f"number of sensors ({num_sensors})"
                )
            shading_weights = self.shading

        mode = str(self.output_mode).strip().lower()
        if mode in {"bt", "broadband", "power"}:
            return self.das_broadband_power(
                sensor_signals,
                self.sampling_rate_hz,
                self.nfft,
                steering_delays_s,
                shading_weights,
                f0=self.f0,
                fmin=self.fmin,
                fmax=self.fmax,
                overlap=self.overlap,
            )
        if mode in {"btf", "cube", "bands"}:
            return self.das_broadband_power_btf(
                sensor_signals,
                self.sampling_rate_hz,
                self.nfft,
                steering_delays_s,
                shading_weights,
                f0=self.f0,
                fmin=self.fmin,
                fmax=self.fmax,
                overlap=self.overlap,
            )
        raise ValueError(
            f"Unsupported output_mode={self.output_mode!r}. Use 'bt' or 'btf'."
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

    Parameters
    ----------
    sensor_signals : np.ndarray
        An array of sensor signals with shape (num_sensors, num_samples).
    steering_delays_s : np.ndarray
        An array of time delays for each sensor and steering direction, with shape (num_directions,
        num_sensors).
    shading_weights : np.ndarray
        An array of shading weights for each sensor, with shape (num_sensors,).
    sampling_rate_hz : float
        The sampling frequency of the sensor signals, in Hz.

    Returns
    -------
    np.ndarray
        An array of beamformed signals with shape (num_directions, num_samples).

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

    Parameters
    ----------
    sensor_signals  : np.ndarray
        An array of sensor signals with shape (num_sensors, num_samples).
    steering_delays_s  : np.ndarray
        An array of time delays for each sensor and steering direction, with shape (num_directions,
        num_sensors).
    shading_weights  : np.ndarray
        An array of shading weights for each sensor, with shape (num_sensors,).
    sampling_rate_hz  : float
        The sampling frequency of the sensor signals, in Hz.

    Returns
    -------
    np.ndarray
        An array of beamformed signals with shape (num_directions, num_samples).

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
            phase_shift = np.exp(1j * 2 * np.pi * frequency_bins * steering_delays_s[i, j])
            # Apply shading and phase shift, and accumulate
            beamformed_f[i] += signals_f[j] * phase_shift * shading_weights[j]

    # 5. Perform IFFT once on the final result
    beamformed_signals = np.fft.ifft(beamformed_f, axis=1)

    return beamformed_signals


class MinimumVarianceDistortionlessResponseBeamformer(Beamformer):
    """A Minimum Variance Distortionless Response (MVDR) beamformer.

    MVDR is an adaptive beamformer that computes optimal weights from the data covariance matrix.
    Unlike DAS beamformers, shading/tapering is not applied as it would interfere with the adaptive
    optimisation.

    This implementation uses an STFT-based broadband approach: the sensor time-series are
    transformed into short-time frequency bins, a frequency-domain Capon (MVDR) beamformer is
    applied in each bin, and the narrowband outputs are integrated over frequency to produce power
    as a function of steering direction and time-frame.

    """

    sampling_rate_hz = Property(float, doc="The sampling frequency of the sensor signals, in Hz")
    nfft = Property(int, default=500, doc="STFT window size (samples)")
    overlap = Property(int, default=250, doc="STFT overlap (samples)")
    f0 = Property(float, default=0.0, doc="Carrier frequency for baseband data (Hz)")
    fmin = Property(float, default=None, doc="Minimum frequency to integrate (Hz)")
    fmax = Property(float, default=None, doc="Maximum frequency to integrate (Hz)")

    def beamform(self, sensor_signals: np.ndarray, steering_delays_s: np.ndarray) -> np.ndarray:
        """Perform broadband MVDR beamforming and return power time-series.

        Parameters
        ----------
        sensor_signals : np.ndarray
            An array of sensor signals with shape (num_sensors, num_samples).
        steering_delays_s : np.ndarray
            An array of time delays for each sensor and steering direction, with shape
            (num_directions, num_sensors).

        Raises
        ------
        ValueError
            If the number of sensors in signal_array does not match the number of sensors in
            steering_delays.

        Returns
        -------
        np.ndarray
            Array of beamformed power with shape (num_directions, num_time_frames).

        """
        num_sensors, _ = sensor_signals.shape
        if num_sensors != steering_delays_s.shape[1]:
            raise ValueError("Number of sensors must match the number of steering delays")

        return self._mvdr_broadband(
            sensor_signals,
            self.sampling_rate_hz,
            self.nfft,
            steering_delays_s,
            f0=self.f0,
            fmin=self.fmin,
            fmax=self.fmax,
            overlap=self.overlap,
        )

    @staticmethod
    def _stft(x: np.ndarray, nfft: int, overlap: int) -> np.ndarray:
        """Compute the Short-Time Fourier Transform (STFT) of the input signal.

        Parameters
        ----------
        x : np.ndarray
            Input signal array of shape (M, T) where M is the number of sensors and T is the number
            of time samples.
        nfft : int
            The number of FFT points (window size).
        overlap : int
            The number of overlapping samples between windows.

        Returns
        -------
            np.ndarray
                STFT of the input signal with shape (M, n_frames, nfft).

        """
        # x: (M, T) complex
        M, T = x.shape
        if T < nfft:
            raise ValueError(f"Input signal length T={T} is less than window size nfft={nfft}.")
        hop = max(1, nfft - overlap)
        # pad to fit last frame exactly
        n_frames = 1 + (max(0, T - nfft) // hop)
        pad = (n_frames - 1) * hop + nfft - T
        if pad > 0:
            x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

        window = np.hanning(nfft).astype(x.real.dtype)
        # Make a 3D view: (M, n_frames, nfft)
        stride_t = x.strides[1]
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(M, n_frames, nfft),
            strides=(x.strides[0], hop * stride_t, stride_t),
            writeable=False,
        )
        frames = frames * window  # broadcasts over last axis
        return np.fft.fft(frames, axis=2)

    def _mvdr_broadband(
        self,
        x: np.ndarray,
        fs: float,
        nfft: int,
        sd: np.ndarray,
        f0: float = 0,
        fmin: float = None,
        fmax: float = None,
        overlap: int = 0,
    ) -> np.ndarray:
        """Perform broadband MVDR beamforming.

        Parameters
        ----------
        x : np.ndarray
            Input signal array of shape (M, T) where M is the number of sensors and T is the number
            of time samples.
        fs : float
            Sampling frequency in Hz.
        nfft : int
            The number of FFT points (window size).
        sd : np.ndarray
            Steering delays array of shape (Ndir, M) where Ndir is the number of steering
            directions.
        f0 : float, optional
            Carrier frequency for baseband data (Hz). Defaults to 0.
        fmin : float, optional
            Minimum frequency to integrate (Hz). Defaults to None.
        fmax : float, optional
            Maximum frequency to integrate (Hz). Defaults to None.
        overlap : int, optional
            The number of overlapping samples between windows. Defaults to 0.

        Raises
        ------
        ValueError
            If nfft is too small for the array geometry.

        Returns
        -------
        np.ndarray
            Array of beamformed power with shape (Ndir, n_frames).

        """
        # x: (M, T), sd: (Ndir, M)
        M, _ = x.shape
        if nfft / fs < (np.max(sd) - np.min(sd)):
            raise ValueError("nfft too small for this array")

        # STFT (M, n_frames, nfft)
        X = self._stft(x, nfft, overlap)
        M, n_frames, nfft_actual = X.shape

        # frequency bins (full complex spectrum as signal is complex/baseband)
        # bin index -> analog frequency in Hz
        k = np.arange(nfft_actual)
        # map to centered FFT frequency bins: [0 ... nfft/2-1, -nfft/2 ... -1]
        k_centered = np.where(k <= nfft_actual // 2, k, k - nfft_actual)
        f_bins = f0 + (fs / nfft_actual) * k_centered

        # active bins mask
        if fmin is None:
            fmin = f_bins.min()
        if fmax is None:
            fmax = f_bins.max()
        active = (f_bins >= fmin) & (f_bins <= fmax)
        active_idx = np.nonzero(active)[0]

        # Output power accumulator: (Ndir, n_frames)
        P = np.zeros((sd.shape[0], n_frames), dtype=np.float64)

        # Loop only the active bins
        for i in active_idx:
            f = f_bins[i]

            # Snapshots for this bin: (M, n_frames)
            S = X[:, :, i]

            # Covariance: (M, M)
            # Using frames as snapshots, average over time
            R = (S @ S.conj().T) / float(n_frames)

            # Diagonal loading
            # 1e-3 to prevent singular covariance matrices for stability
            dl = 1e-3 * np.trace(R).real / M
            R.flat[:: M + 1] += dl

            # Steering matrix A: (M, Ndir)
            # (we build as (Ndir, M) then transpose for solve)
            A = np.exp(-2j * np.pi * f * sd).T  # (M, Ndir)

            # Solve R X = A  -> X = R^{-1} A using Cholesky once
            # scipy LAPACK is faster than np.linalg.solve or numba
            c, lower = cho_factor(R, overwrite_a=False, check_finite=False)
            RinvA = cho_solve((c, lower), A, overwrite_b=False, check_finite=False)  # (M, Ndir)

            # Denominator: diag(A^H R^{-1} A) -> (Ndir,)
            den = np.sum(A.conj() * RinvA, axis=0)

            # Weights W = R^{-1} a / (a^H R^{-1} a) for all dirs -> (Ndir, M)
            epsilon = np.finfo(np.float64).eps  # prevent division by zero
            W = (RinvA / den[None, :] + epsilon).T  # (Ndir, M)

            # Beamform outputs across frames: (Ndir, M) @ (M, n_frames)
            Y = W.conj() @ S  # (Ndir, n_frames)

            # Accumulate power; not storing per-bin outputs
            P += np.abs(Y) ** 2

        return P


class SteeringCalculator(Base):
    """Computes time delays for beamforming with a horizontal sensor array."""

    ssp = Property(SoundSpeedProfile, doc="Sound speed profile for calculating delays")
    steering_azimuths_rad = Property(np.ndarray, doc="Azimuth angles for steering, in radians")

    def calculate(self, platform: Platform) -> np.ndarray:
        """Calculate steering delays for the current horizontal array geometry.

        This method assumes the platform has an `array` attribute which is an object with
        `state_vector` and `ref_state_vector` attributes, such as the one configured by
        `TowedArrayPlatform`.

        Parameters
        ----------
        platform  : Platform
            The platform containing the sensor array.

        Returns
        -------
        np.ndarray
            An array of steering delays for each sensor relative to the steering direction, with
            shape (num_directions, num_sensors).

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
