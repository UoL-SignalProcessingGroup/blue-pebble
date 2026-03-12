"""Beamforming algorithms for processing signals from an array of sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numba import njit, prange, types
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from stonesoup.base import Base, Property

from ..models.environment import SoundSpeedProfile

if TYPE_CHECKING:
    from stonesoup.platform.base import Platform

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
BeamformerOutput: TypeAlias = ComplexArray | FloatArray
DomainType: TypeAlias = Literal["time", "frequency", "broadband_power"]


class Beamformer(Base, ABC):
    """Abstract interface for beamforming algorithms."""

    @abstractmethod
    def beamform(
        self,
        sensor_signals: ArrayLike,
        steering_delays_s: ArrayLike,
    ) -> BeamformerOutput:
        """Form directional beams from multi-sensor input data.

        Parameters
        ----------
        sensor_signals : ArrayLike
            Sensor signal matrix with shape ``(num_sensors, num_samples)``.
        steering_delays_s : ArrayLike
            Steering-delay matrix (seconds) with shape ``(num_directions, num_sensors)``.

        Returns
        -------
        BeamformerOutput
            Beamformer output matrix. Concrete implementations define whether this
            contains complex beamformed time-series or real-valued beam power.

        """
        pass


class DelayAndSumBeamformer(Beamformer):
    """A Delay-and-Sum (DAS) beamformer.

    Supports three processing modes:

    - ``'time'``: time-domain delay-and-sum.
    - ``'frequency'``: frequency-domain delay-and-sum.
    - ``'broadband_power'``: STFT-based incoherent broadband power integration.

    In all modes, steering is controlled via per-direction per-sensor delay values.
    """

    sampling_rate_hz: float = Property(
        float,
        doc="The sampling frequency of the sensor signals, in Hz",
    )
    shading: FloatArray | None = Property(
        np.ndarray,
        default=None,
        doc="An array of shading weights applied to each sensor. If None, uniform weights are "
        "used.",
    )
    domain: DomainType = Property(
        str,
        default="time",
        doc="Beamforming domain: 'time', 'frequency', or 'broadband_power'.",
    )
    nfft: int = Property(default=500, doc="STFT window size (samples)")
    overlap: int = Property(default=250, doc="STFT overlap (samples)")
    f0: float = Property(default=0.0, doc="Carrier frequency for baseband data (Hz)")
    fmin: float | None = Property(default=None, doc="Minimum frequency to integrate (Hz)")
    fmax: float | None = Property(default=None, doc="Maximum frequency to integrate (Hz)")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the DelayAndSumBeamformer.

        Parameters
        ----------
        *args
            Positional arguments to pass to the parent class.
        **kwargs
            Keyword arguments forwarded to the parent class.
            Common options include ``sampling_rate_hz``, ``shading``, ``domain``,
            ``nfft``, ``overlap``, ``f0``, ``fmin``, and ``fmax``.

        Raises
        ------
        ValueError
            If explicit shading weights do not sum to a finite non-zero value.
        ValueError
            If the specified domain is not ``'time'``, ``'frequency'``, or
            ``'broadband_power'``.

        """
        super().__init__(*args, **kwargs)

        if self.shading is not None:
            shading_sum = np.sum(self.shading)
            if not np.isfinite(shading_sum) or np.isclose(shading_sum, 0.0):
                raise ValueError("Shading weights must sum to a finite non-zero value")
            self.shading = self.shading / shading_sum

        # Store number of sensors for consistent shading
        self._num_sensors: int | None = None

        if self.domain not in ["time", "frequency", "broadband_power"]:
            raise ValueError(
                "Invalid beamforming domain. Must be 'time', 'frequency', or 'broadband_power'"
            )

    def beamform(
        self, sensor_signals: ArrayLike, steering_delays_s: ArrayLike
    ) -> BeamformerOutput:
        """Beamform sensor data using the configured DAS processing domain.

        Parameters
        ----------
        sensor_signals : ArrayLike
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        steering_delays_s : ArrayLike
            Steering-delay matrix with shape ``(num_directions, num_sensors)``.

        Returns
        -------
        BeamformerOutput
            - ``domain='time'`` or ``'frequency'``: complex signals with shape
              ``(num_directions, num_samples_or_cropped_samples)``.
            - ``domain='broadband_power'``: real-valued power map with shape
              ``(num_directions, num_frames)``.

        Raises
        ------
        ValueError
            If the number of sensors in ``sensor_signals`` does not match
            ``steering_delays_s``.
        ValueError
            If explicit shading length does not match the sensor count.

        """
        # Numba kernels below are compiled for complex128/float64 C-contiguous arrays.
        # Normalising inputs here prevents dispatcher type mismatches (e.g. complex64 data).
        sensor_signals_array: ComplexArray = np.ascontiguousarray(
            sensor_signals,
            dtype=np.complex128,
        )
        steering_delays_array: FloatArray = np.ascontiguousarray(
            steering_delays_s,
            dtype=np.float64,
        )

        num_sensors, _ = sensor_signals_array.shape

        if num_sensors != steering_delays_array.shape[1]:
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
                sensor_signals_array,
                steering_delays_array,
                shading_weights,
                self.sampling_rate_hz,
            )
        if self.domain == "frequency":
            return _frequency_das(
                sensor_signals_array,
                steering_delays_array,
                shading_weights,
                self.sampling_rate_hz,
            )
        if self.domain == "broadband_power":
            return self.das_broadband_power(
                sensor_signals_array,
                self.sampling_rate_hz,
                self.nfft,
                steering_delays_array,
                shading_weights,
                f0=self.f0,
                fmin=self.fmin,
                fmax=self.fmax,
                overlap=self.overlap,
            )
        msg = f"Unsupported beamforming domain: {self.domain}"
        raise ValueError(msg)

    @staticmethod
    def _stft(x: ComplexArray, nfft: int, overlap: int) -> ComplexArray:
        """Compute a per-sensor STFT using a Hann window and fixed overlap.

        Parameters
        ----------
        x : ComplexArray
            Complex sensor data with shape ``(num_sensors, num_samples)``.
        nfft : int
            STFT window length (FFT size).
        overlap : int
            Overlap between adjacent windows in samples.

        Returns
        -------
        ComplexArray
            STFT tensor with shape ``(num_sensors, num_frames, nfft)``.

        Raises
        ------
        ValueError
            If ``num_samples < nfft``.

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

    def das_broadband_power(
        self,
        x: ComplexArray,  # (M, T)
        fs: float,
        nfft: int,
        sd: FloatArray,  # (Ndir, M) steering delays [s]
        shading_weights: FloatArray,  # (M,)
        f0: float = 0.0,
        fmin: float | None = None,
        fmax: float | None = None,
        overlap: int = 0,
    ) -> FloatArray:
        """Compute STFT-based broadband DAS power over steering directions.

        The input sensor array data are transformed using STFT, steered at each frequency bin using
        phase shifts derived from ``sd``, and then integrated incoherently (power sum across
        selected frequencies) to produce a direction-time power map.

        Parameters
        ----------
        x : ComplexArray
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        fs : float
            Sampling frequency in Hz.
        nfft : int
            STFT window length (number of FFT points).
        sd : FloatArray
            Steering delays in seconds with shape ``(num_directions, num_sensors)``.
        shading_weights : FloatArray
            Per-sensor beamforming weights with shape ``(num_sensors,)``.
        f0 : float, optional
            Carrier frequency offset in Hz for baseband data. Defaults to ``0.0``.
        fmin : float | None, optional
            Minimum frequency (Hz) included in the power integration. If ``None``, the minimum
            available STFT bin frequency is used.
        fmax : float | None, optional
            Maximum frequency (Hz) included in the power integration. If ``None``, the maximum
            available STFT bin frequency is used.
        overlap : int, optional
            Number of overlapping samples between adjacent STFT frames. Defaults to ``0``.

        Returns
        -------
        FloatArray
            Broadband beam power map with shape ``(num_directions, num_frames)``. Each entry
            contains integrated beam power over the active frequency bins for one steering
            direction and one STFT frame.

        """
        M, _ = x.shape
        X = self._stft(x, nfft, overlap)  # (M, n_frames, nfft)
        M, n_frames, nfft_actual = X.shape

        k = np.arange(nfft_actual)
        k_centered = np.where(k <= nfft_actual // 2, k, k - nfft_actual)
        f_bins = f0 + (fs / nfft_actual) * k_centered

        if fmin is None:
            fmin = f_bins.min()
        if fmax is None:
            fmax = f_bins.max()
        active = (f_bins >= fmin) & (f_bins <= fmax)
        active_idx = np.nonzero(active)[0]

        Ndir = sd.shape[0]
        P = np.zeros((Ndir, n_frames), dtype=np.float64)

        # Normalize weights
        w = shading_weights.reshape(1, M, 1)  # (1, M, 1) for broadcasting

        for i in active_idx:
            f = f_bins[i]

            # Snapshots at this bin: (M, n_frames)
            S = X[:, :, i]

            # Steering phase for all dirs/sensors: (Ndir, M)
            A = np.exp(1j * 2 * np.pi * f * sd)

            # Beamform: Y = sum_m w_m * A(dir,m) * S(m,frame)
            # Result: (Ndir, n_frames)
            Y = np.sum((A[:, :, None] * S[None, :, :]) * w, axis=1)

            # Accumulate power over frequency bins
            P += np.abs(Y) ** 2

        return P


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
    sensor_signals: ComplexArray,
    steering_delays_s: FloatArray,
    shading_weights: FloatArray,
    sampling_rate_hz: float,
) -> ComplexArray:
    """Perform time-domain delay-and-sum beamforming.

    Parameters
    ----------
    sensor_signals : ComplexArray
        Sensor data matrix with shape ``(num_sensors, num_samples)``.
    steering_delays_s : FloatArray
        Steering-delay matrix with shape ``(num_directions, num_sensors)``.
    shading_weights : FloatArray
        Sensor shading/weight vector with shape ``(num_sensors,)``.
    sampling_rate_hz : float
        Sampling frequency in Hz.

    Returns
    -------
    ComplexArray
        Complex beamformed signals with shape
        ``(num_directions, num_samples_or_cropped_samples)``.

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
    sensor_signals: ComplexArray,
    steering_delays_s: FloatArray,
    shading_weights: FloatArray,
    sampling_rate_hz: float,
) -> ComplexArray:
    """Perform frequency-domain delay-and-sum beamforming.

    Parameters
    ----------
    sensor_signals : ComplexArray
        Sensor data matrix with shape ``(num_sensors, num_samples)``.
    steering_delays_s : FloatArray
        Steering-delay matrix with shape ``(num_directions, num_sensors)``.
    shading_weights : FloatArray
        Sensor shading/weight vector with shape ``(num_sensors,)``.
    sampling_rate_hz : float
        Sampling frequency in Hz.

    Returns
    -------
    ComplexArray
        Complex beamformed signals with shape ``(num_directions, num_samples)``.

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
    as a function of steering direction and time frame.

    """

    sampling_rate_hz: float = Property(
        float,
        doc="The sampling frequency of the sensor signals, in Hz",
    )
    nfft: int = Property(default=500, doc="STFT window size (samples)")
    overlap: int = Property(default=250, doc="STFT overlap (samples)")
    f0: float = Property(default=0.0, doc="Carrier frequency for baseband data (Hz)")
    fmin: float | None = Property(default=None, doc="Minimum frequency to integrate (Hz)")
    fmax: float | None = Property(default=None, doc="Maximum frequency to integrate (Hz)")

    def beamform(self, sensor_signals: ArrayLike, steering_delays_s: ArrayLike) -> FloatArray:
        """Perform broadband MVDR beamforming and return power time-series.

        Parameters
        ----------
        sensor_signals : ArrayLike
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        steering_delays_s : ArrayLike
            Steering-delay matrix with shape ``(num_directions, num_sensors)``.

        Raises
        ------
        ValueError
            If the number of sensors in ``sensor_signals`` does not match
            ``steering_delays_s``.

        Returns
        -------
        FloatArray
            Real-valued beamformed power with shape ``(num_directions, num_time_frames)``.

        """
        sensor_signals_array = np.asarray(sensor_signals)
        steering_delays_array = np.asarray(steering_delays_s, dtype=np.float64)

        num_sensors, _ = sensor_signals_array.shape
        if num_sensors != steering_delays_array.shape[1]:
            raise ValueError("Number of sensors must match the number of steering delays")

        return self._mvdr_broadband(
            sensor_signals_array,
            self.sampling_rate_hz,
            self.nfft,
            steering_delays_array,
            f0=self.f0,
            fmin=self.fmin,
            fmax=self.fmax,
            overlap=self.overlap,
        )

    @staticmethod
    def _stft(x: ArrayLike, nfft: int, overlap: int) -> ComplexArray:
        """Compute a per-sensor STFT used by MVDR processing.

        Parameters
        ----------
        x : ArrayLike
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        nfft : int
            STFT window length (FFT size).
        overlap : int
            Overlap between adjacent windows in samples.

        Returns
        -------
        ComplexArray
            STFT tensor with shape ``(num_sensors, num_frames, nfft)``.

        Raises
        ------
        ValueError
            If ``num_samples < nfft``.

        """
        # x: (M, T) complex
        x_array = np.asarray(x)
        M, T = x_array.shape
        if T < nfft:
            raise ValueError(f"Input signal length T={T} is less than window size nfft={nfft}.")
        hop = max(1, nfft - overlap)
        # pad to fit last frame exactly
        n_frames = 1 + (max(0, T - nfft) // hop)
        pad = (n_frames - 1) * hop + nfft - T
        if pad > 0:
            x_array = np.pad(x_array, ((0, 0), (0, pad)), mode="constant")

        window = np.hanning(nfft).astype(np.float64)
        # Make a 3D view: (M, n_frames, nfft)
        stride_t = x_array.strides[1]
        frames = np.lib.stride_tricks.as_strided(
            x_array,
            shape=(M, n_frames, nfft),
            strides=(x_array.strides[0], hop * stride_t, stride_t),
            writeable=False,
        )
        frames = frames * window  # broadcasts over last axis
        return np.fft.fft(frames, axis=2)

    def _mvdr_broadband(
        self,
        x: ArrayLike,
        fs: float,
        nfft: int,
        sd: ArrayLike,
        f0: float = 0.0,
        fmin: float | None = None,
        fmax: float | None = None,
        overlap: int = 0,
    ) -> FloatArray:
        """Compute STFT-based broadband MVDR (Capon) beam power.

        Parameters
        ----------
        x : ArrayLike
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        fs : float
            Sampling frequency in Hz.
        nfft : int
            STFT window length (FFT size).
        sd : ArrayLike
            Steering-delay matrix with shape ``(num_directions, num_sensors)``.
        f0 : float, optional
            Carrier frequency for baseband data in Hz. Defaults to ``0.0``.
        fmin : float | None, optional
            Minimum frequency to integrate in Hz. Defaults to ``None``.
        fmax : float | None, optional
            Maximum frequency to integrate in Hz. Defaults to ``None``.
        overlap : int, optional
            Overlap between adjacent windows in samples. Defaults to ``0``.

        Raises
        ------
        ValueError
            If nfft is too small for the array geometry.

        Returns
        -------
        FloatArray
            Real-valued beamformed power with shape
            ``(num_directions, num_frames)``.

        """
        # x: (M, T), sd: (Ndir, M)
        x_array = np.asarray(x)
        sd_array = np.asarray(sd, dtype=np.float64)

        M, _ = x_array.shape
        if nfft / fs < (np.max(sd_array) - np.min(sd_array)):
            raise ValueError("nfft too small for this array")

        # STFT (M, n_frames, nfft)
        X = self._stft(x_array, nfft, overlap)
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
        P = np.zeros((sd_array.shape[0], n_frames), dtype=np.float64)

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
            A = np.exp(-2j * np.pi * f * sd_array).T  # (M, Ndir)

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
    """Compute steering delays for a horizontal sensor array."""

    ssp: SoundSpeedProfile = Property(
        SoundSpeedProfile,
        doc="Sound speed profile for calculating delays",
    )
    steering_azimuths_rad: FloatArray = Property(
        np.ndarray,
        doc="Azimuth angles for steering, in radians",
    )

    def calculate(self, platform: Platform) -> FloatArray:
        """Calculate per-direction per-sensor steering delays.

        This method assumes the platform has an `array` attribute which is an object with
        `state_vector` and `ref_state_vector` attributes, such as the one configured by
        `TowedArrayPlatform`.

        Parameters
        ----------
        platform : Platform
            The platform containing the sensor array.

        Returns
        -------
        FloatArray
            Steering-delay matrix in seconds with shape
            ``(num_directions, num_sensors)``.

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
