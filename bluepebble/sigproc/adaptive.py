"""adaptive beamforming."""

from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve

from .base import FloatArray, MirrorPlan, _stft, _stft_bin_frequencies, _STFTBeamformer


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

        """
        # x: (M, T), sd: (Ndir, M)
        x_array = np.asarray(x)
        sd_array = np.asarray(sd, dtype=np.float64)

        M, _ = x_array.shape
        if nfft / fs < (np.max(sd_array) - np.min(sd_array)):
            raise ValueError("nfft too small for this array")

        # STFT (M, n_frames, nfft)
        X = _stft(x_array, nfft, overlap)
        M, n_frames, nfft_actual = X.shape

        # frequency bins (full complex spectrum as signal is complex/baseband)
        f_bins = _stft_bin_frequencies(nfft_actual, fs, f0)
        per_band_bins = self._band_bin_indices(f_bins, fmin, fmax)
        bins_to_bands = self._bins_to_bands(per_band_bins)

        # Output power accumulator: (n_bands, Ndir, n_frames)
        P = np.zeros((len(per_band_bins), sd_array.shape[0], n_frames), dtype=np.float64)

        # Loop only the active bins. Everything inside is band-independent — the covariance,
        # the Cholesky solve and the weights depend on the bin alone — so a bin shared by
        # several bands is solved once and its power accumulated into each of them.
        for i in sorted(bins_to_bands):
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
            bin_power = np.abs(Y) ** 2
            for band_idx in bins_to_bands[i]:
                P[band_idx] += bin_power

        return self._finalise_band_power(P, per_band_bins)
