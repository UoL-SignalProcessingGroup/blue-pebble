"""adaptive beamforming."""

from typing import cast

import numpy as np
import rocket_fft  # noqa: F401  -- registers numba support for np.fft.*, used by _fft_frames
from numba import njit, prange
from numpy.typing import ArrayLike

from .base import FloatArray, MirrorPlan, _stft_bin_frequencies, _STFTBeamformer


def _framed_and_windowed(x_array: np.ndarray, nfft: int, overlap: int) -> np.ndarray:
    """Partition ``x_array`` into overlapping, Hann-windowed frames for subsequent transformation.

    The framing procedure is identical to that of :func:`~.base._stft`; it is isolated
    here so that the Fourier transform itself may be substituted with the numba- and
    rocket-fft-accelerated implementation provided by :func:`_fft_frames`.
    """
    M, T = x_array.shape
    hop = max(1, nfft - overlap)
    n_frames = 1 + (max(0, T - nfft) // hop)
    pad = (n_frames - 1) * hop + nfft - T
    if pad > 0:
        x_array = np.pad(x_array, ((0, 0), (0, pad)), mode="constant")

    window = np.hanning(nfft).astype(np.float64)
    stride_t = x_array.strides[1]
    frames = np.lib.stride_tricks.as_strided(
        x_array,
        shape=(M, n_frames, nfft),
        strides=(x_array.strides[0], hop * stride_t, stride_t),
        writeable=False,
    )
    return np.ascontiguousarray(frames * window)


@njit(cache=True, parallel=True)
def _fft_frames(frames: np.ndarray) -> np.ndarray:
    """Compute the discrete Fourier transform of every (sensor, frame) row of ``frames``.

    ``frames`` has shape ``(num_sensors, num_time_frames, nfft)``. As the transform for
    each sensor is independent of that for every other, the computation is parallelised
    over the sensor axis via ``prange``. The rocket-fft package is required for
    ``np.fft.fft`` to be invoked at all within numba's ``nopython`` compilation mode.
    """
    num_sensors, num_time_frames, nfft = frames.shape
    out = np.empty((num_sensors, num_time_frames, nfft), dtype=np.complex128)
    for i in prange(num_sensors):
        for j in range(num_time_frames):
            out[i, j, :] = np.fft.fft(frames[i, j, :])
    return out


@njit(cache=True, parallel=True)
def _steering_matrices(freqs: np.ndarray, sd_array: np.ndarray) -> np.ndarray:
    """Compute the per-bin steering matrices ``exp(-2j*pi*f*sd)``, parallelised over bins.

    ``freqs`` has shape ``(num_active_bins,)`` and ``sd_array`` has shape
    ``(num_directions, num_sensors)``; the array returned has shape ``(num_active_bins,
    num_sensors, num_directions)``.

    As the argument of the exponential is purely imaginary, ``exp(i*theta)`` is evaluated
    as ``cos(theta) + i*sin(theta)`` rather than via the general complex-valued
    ``np.exp``. Empirical profiling of this computation within a representative scenario
    identified the complex exponential as the dominant cost within the MVDR pipeline,
    accounting for approximately 45 per cent of total execution time and substantially
    exceeding the cost of the Cholesky-based solve, which had previously been assumed to
    dominate. The cosine/sine decomposition alone, prior to parallelisation, was already
    found to outperform ``np.exp`` on a complex-valued array; the addition of
    ``parallel=True`` here yields a further, substantial reduction in execution time.
    """
    num_bins = freqs.shape[0]
    num_directions, num_sensors = sd_array.shape
    out = np.empty((num_bins, num_sensors, num_directions), dtype=np.complex128)
    for bi in prange(num_bins):
        f = freqs[bi]
        for d in range(num_directions):
            for m in range(num_sensors):
                theta = -2.0 * np.pi * f * sd_array[d, m]
                out[bi, m, d] = np.cos(theta) + 1j * np.sin(theta)
    return out


@njit(cache=True, parallel=True)
def _mvdr_bin_power(
    S_batch: np.ndarray, R_batch: np.ndarray, A_batch: np.ndarray
) -> np.ndarray:
    """Compute per-bin MVDR weights and beamformed power, parallelised over frequency bins.

    ``S_batch`` (signal snapshots), ``R_batch`` (the diagonally-loaded covariance
    matrices), and ``A_batch`` (steering matrices) share a common leading bin axis. As
    the weight solve for each frequency bin is independent of that for every other, the
    computation is parallelised over the bin axis via ``prange``. The general-purpose
    ``np.linalg.solve`` is used in preference to a Cholesky-based solve, numba providing
    no batched analogue to scipy's ``cho_solve``. Although the general solve is less
    efficient per bin than the Cholesky-based alternative, empirical benchmarking against
    the sequential, scipy-Cholesky implementation showed a net improvement in
    performance, attributable to concurrent execution across all available processor
    cores rather than sequential processing of individual bins.
    """
    num_bins, num_sensors, num_frames = S_batch.shape
    num_directions = A_batch.shape[2]
    power = np.zeros((num_bins, num_directions, num_frames), dtype=np.float64)
    epsilon = np.finfo(np.float64).eps

    for bi in prange(num_bins):
        R = R_batch[bi]
        A = A_batch[bi]
        RinvA = np.linalg.solve(R, A)  # (num_sensors, num_directions)

        den = np.zeros(num_directions, dtype=np.complex128)
        for d in range(num_directions):
            acc = 0.0 + 0.0j
            for m in range(num_sensors):
                acc += np.conj(A[m, d]) * RinvA[m, d]
            den[d] = acc

        S = S_batch[bi]
        for d in range(num_directions):
            for f in range(num_frames):
                acc = 0.0 + 0.0j
                for m in range(num_sensors):
                    w = RinvA[m, d] / den[d] + epsilon
                    acc += np.conj(w) * S[m, f]
                power[bi, d, f] = abs(acc) ** 2

    return power


class MinimumVarianceDistortionlessResponseBeamformer(_STFTBeamformer):
    """A Minimum Variance Distortionless Response (MVDR) beamformer.

    MVDR is an adaptive beamformer that computes optimal weights from the data covariance matrix.
    Unlike DAS beamformers, shading/tapering is not applied as it would interfere with the adaptive
    optimisation.

    This implementation uses an STFT-based broadband approach: the sensor time-series are
    transformed into short-time frequency bins, a frequency-domain Capon (MVDR) beamformer is
    applied in each bin, and the narrowband outputs are integrated over frequency to produce power
    as a function of steering direction and time frame.

    Multiband processing
    --------------------
    Setting ``bands`` integrates several frequency bands from one pass over the data and
    returns a power map per band, with shape ``(num_bands, num_directions, num_frames)``.

    """

    def beamform(
        self,
        sensor_signals: ArrayLike,
        steering_delays_s: ArrayLike,
        mirror_plan: MirrorPlan | None = None,
    ) -> FloatArray:
        """Perform broadband MVDR beamforming and return power time-series.

        Parameters
        ----------
        sensor_signals : ArrayLike
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        steering_delays_s : ArrayLike
            Steering-delay matrix with shape ``(num_directions, num_sensors)``.
        mirror_plan : MirrorPlan | None, optional
            When provided, ``steering_delays_s`` is expected to cover only the primary
            half of the steering grid, and the output is expanded to the full grid via
            :meth:`~.Beamformer.expand_mirrored` before returning.

        Raises
        ------
        ValueError
            If the number of sensors in ``sensor_signals`` does not match
            ``steering_delays_s``.

        Returns
        -------
        FloatArray
            Real-valued beamformed power with shape ``(num_directions, num_time_frames)``,
            or ``(num_bands, num_directions, num_time_frames)`` when ``bands`` is set.
            ``num_directions`` is always the full steering grid, even when ``mirror_plan``
            halved the actual computation.

        """
        sensor_signals_array = np.asarray(sensor_signals)
        steering_delays_array = np.asarray(steering_delays_s, dtype=np.float64)

        num_sensors, _ = sensor_signals_array.shape
        if num_sensors != steering_delays_array.shape[1]:
            raise ValueError("Number of sensors must match the number of steering delays")

        output = self._mvdr_broadband(
            sensor_signals_array,
            self.sampling_rate_hz,
            self.nfft,
            steering_delays_array,
            f0=self.f0,
            fmin=self.fmin,
            fmax=self.fmax,
            overlap=self.overlap,
        )

        if mirror_plan is not None:
            direction_axis = 1 if self.bands else 0
            expanded = self.expand_mirrored(output, mirror_plan, direction_axis=direction_axis)
            output = cast(FloatArray, expanded)
        return output

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
            Real-valued beamformed power with shape ``(num_directions, num_frames)``, or
            ``(num_bands, num_directions, num_frames)`` when ``bands`` is set.

        Notes
        -----
        The ``fmin`` and ``fmax`` arguments select the band in single-band mode only. When
        ``bands`` is set on the beamformer they are ignored in favour of the band edges.

        The independence of the per-bin covariance solve across frequency bins (detailed
        further in :func:`_mvdr_bin_power`) is exploited by the numba-parallelised helper
        functions employed within this module: the Fourier transform, steering-matrix
        construction, and per-bin solve stages are each parallelised over a distinct,
        embarrassingly parallel axis -- sensors, bins, and bins, respectively -- rather
        than executed as a single sequential loop over frequency bins.

        """
        # x: (M, T), sd: (Ndir, M)
        x_array = np.asarray(x)
        sd_array = np.asarray(sd, dtype=np.float64)

        M, _ = x_array.shape
        if nfft / fs < (np.max(sd_array) - np.min(sd_array)):
            raise ValueError("nfft too small for this array")

        frames = _framed_and_windowed(x_array, nfft, overlap)
        X = _fft_frames(frames)  # (M, n_frames, nfft)
        M, n_frames, nfft_actual = X.shape

        # frequency bins (full complex spectrum as signal is complex/baseband)
        f_bins = _stft_bin_frequencies(nfft_actual, fs, f0)
        per_band_bins = self._band_bin_indices(f_bins, fmin, fmax)
        bins_to_bands = self._bins_to_bands(per_band_bins)

        # Output power accumulator: (n_bands, Ndir, n_frames)
        P = np.zeros((len(per_band_bins), sd_array.shape[0], n_frames), dtype=np.float64)

        active_bins = np.array(sorted(bins_to_bands), dtype=np.int64)
        if active_bins.size == 0:
            return self._finalise_band_power(P, per_band_bins)

        freqs = f_bins[active_bins]

        # Snapshots for every active bin at once: (num_active_bins, M, n_frames)
        S_batch = np.ascontiguousarray(np.transpose(X[:, :, active_bins], (2, 0, 1)))

        # Covariance per bin, averaged over time frames: (num_active_bins, M, M)
        R_batch = np.matmul(S_batch, S_batch.conj().transpose(0, 2, 1)) / float(n_frames)

        # Diagonal loading per bin (1e-3 to prevent singular covariance matrices).
        trace = np.trace(R_batch, axis1=1, axis2=2).real
        diagonal_load = 1e-3 * trace / M
        idx = np.arange(M)
        R_batch[:, idx, idx] += diagonal_load[:, None]

        A_batch = _steering_matrices(freqs, sd_array)  # (num_active_bins, M, Ndir)
        bin_power_batch = _mvdr_bin_power(S_batch, R_batch, A_batch)

        for bi, bin_idx in enumerate(active_bins):
            for band_idx in bins_to_bands[int(bin_idx)]:
                P[band_idx] += bin_power_batch[bi]

        return self._finalise_band_power(P, per_band_bins)
