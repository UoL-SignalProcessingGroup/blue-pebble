"""adaptive beamforming."""

from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from stonesoup.base import Property

from .base import (
    BeamformerOutput,
    ComplexArray,
    FloatArray,
    MirrorPlan,
    _active_bin_indices,
    _stft,
    _stft_bin_frequencies,
    _STFTBeamformer,
)

MVDRDomainType: TypeAlias = Literal["frequency", "broadband_power"]


class MinimumVarianceDistortionlessResponseBeamformer(_STFTBeamformer):
    """A Minimum Variance Distortionless Response (MVDR) beamformer.

    MVDR is an adaptive beamformer that computes optimal weights from the data covariance matrix.
    Unlike DAS beamformers, shading/tapering is not applied as it would interfere with the adaptive
    optimisation.

    The sensor time-series are transformed into short-time frequency bins and a Capon (MVDR)
    beamformer is applied in each bin. The STFT frames are the snapshots: each bin's covariance
    is averaged over every frame, so one set of weights per bin is used across the whole input.

    Two output domains are supported:

    - ``'broadband_power'`` (default): the narrowband outputs are integrated incoherently over
      frequency, giving power as a function of steering direction and STFT frame.
    - ``'frequency'``: the per-bin weights are applied to the full-length spectrum of each
      sensor, giving a complex time series with the input's sample count, so sample ``k`` of
      every beam is still time ``k / sampling_rate_hz``. This is the form a matched filter
      needs, for example in :class:`~bluepebble.detector.ActiveSonarDetector`. Bins outside
      ``fmin``/``fmax`` are set to zero, so the output is band-limited to the selected band.

    Snapshot support
    ----------------
    A covariance estimate needs roughly twice as many snapshots as sensors to come within 3 dB
    of the optimal output SNR (Reed, Mallett and Brennan). The number of frames is set by the
    input length, ``nfft`` and ``overlap``; with fewer frames than that, raise
    ``diagonal_loading``. Echoes that are coherent with each other, such as a direct path and
    its surface reflection, are treated as interference and can partly cancel each other.

    Multiband processing
    --------------------
    Setting ``bands`` integrates several frequency bands from one pass over the data and
    returns a power map per band, with shape ``(num_bands, num_directions, num_frames)``.
    This requires ``domain='broadband_power'``.

    """

    domain: MVDRDomainType = Property(
        default="broadband_power",
        doc="Output domain: 'broadband_power' (beam power per STFT frame) or 'frequency' "
        "(complex beam time series with the input's sample count).",
    )
    diagonal_loading: float = Property(
        default=1e-3,
        doc="Diagonal loading added to each bin's covariance, as a fraction of the mean "
        "sensor power in that bin. Larger values make the beamformer more robust to few "
        "snapshots and steering mismatch, and move it towards delay-and-sum.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the MVDR beamformer and validate its configuration.

        Raises
        ------
        ValueError
            If ``domain`` is not ``'frequency'`` or ``'broadband_power'``.
        ValueError
            If ``bands`` is set for a domain other than ``'broadband_power'``.
        ValueError
            If ``diagonal_loading`` is negative or not finite.

        """
        super().__init__(*args, **kwargs)

        if self.domain not in ("frequency", "broadband_power"):
            raise ValueError(
                "Invalid MVDR domain. Must be 'frequency' or 'broadband_power', "
                f"got {self.domain!r}"
            )
        if self.bands is not None and self.domain != "broadband_power":
            raise ValueError(
                f"bands requires domain='broadband_power', got domain={self.domain!r}. "
                "The frequency domain returns one time series per direction, so there is "
                "no per-band power map for bands to select."
            )
        if not np.isfinite(self.diagonal_loading) or self.diagonal_loading < 0:
            raise ValueError(
                f"diagonal_loading must be finite and non-negative, got {self.diagonal_loading}"
            )

    def beamform(
        self,
        sensor_signals: ArrayLike,
        steering_delays_s: ArrayLike,
        mirror_plan: MirrorPlan | None = None,
    ) -> BeamformerOutput:
        """Perform MVDR beamforming in the configured output domain.

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
        BeamformerOutput
            - ``domain='broadband_power'``: real-valued beamformed power with shape
              ``(num_directions, num_time_frames)``, or ``(num_bands, num_directions,
              num_time_frames)`` when ``bands`` is set.
            - ``domain='frequency'``: complex beam time series with shape
              ``(num_directions, num_samples)``.

            ``num_directions`` is always the full steering grid, even when ``mirror_plan``
            halved the actual computation.

        """
        sensor_signals_array = np.asarray(sensor_signals)
        steering_delays_array = np.asarray(steering_delays_s, dtype=np.float64)

        num_sensors, _ = sensor_signals_array.shape
        if num_sensors != steering_delays_array.shape[1]:
            raise ValueError("Number of sensors must match the number of steering delays")

        output: BeamformerOutput
        if self.domain == "frequency":
            output = self._mvdr_time_series(sensor_signals_array, steering_delays_array)
        else:
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
            direction_axis = 1 if (self.bands and self.domain == "broadband_power") else 0
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
            # Snapshots for this bin: (M, n_frames)
            S = X[:, :, i]
            W = self._bin_weights(S, f_bins[i], sd_array)
            if W is None:
                continue

            # Beamform outputs across frames: (Ndir, M) @ (M, n_frames)
            Y = W.conj() @ S  # (Ndir, n_frames)

            # Accumulate power; not storing per-bin outputs
            bin_power = np.abs(Y) ** 2
            for band_idx in bins_to_bands[i]:
                P[band_idx] += bin_power

        return self._finalise_band_power(P, per_band_bins)

    def _bin_weights(
        self,
        snapshots: ComplexArray,
        frequency_hz: float,
        sd: FloatArray,
    ) -> ComplexArray | None:
        """Compute the MVDR weights of one frequency bin for every steering direction.

        Parameters
        ----------
        snapshots : ComplexArray
            The bin's value in every STFT frame, with shape ``(num_sensors, num_frames)``.
        frequency_hz : float
            The bin's frequency in Hz.
        sd : FloatArray
            Steering-delay matrix with shape ``(num_directions, num_sensors)``.

        Returns
        -------
        ComplexArray | None
            Weights ``W = R^-1 a / (a^H R^-1 a)`` with shape ``(num_directions,
            num_sensors)``, applied as ``W.conj() @ snapshots``; ``None`` when the bin holds
            no power, so its output is zero.

        """
        M, n_frames = snapshots.shape

        # Covariance: (M, M), using frames as snapshots, averaged over time
        R = (snapshots @ snapshots.conj().T) / float(n_frames)
        mean_power = np.trace(R).real / M
        if not mean_power > 0.0:
            return None

        # Diagonal loading, as a fraction of the mean sensor power, keeps R well conditioned
        R.flat[:: M + 1] += self.diagonal_loading * mean_power

        # Steering matrix A: (M, Ndir) (built as (Ndir, M) then transposed for the solve)
        A = np.exp(-2j * np.pi * frequency_hz * sd).T

        # Solve R X = A -> X = R^{-1} A using Cholesky once
        # scipy LAPACK is faster than np.linalg.solve or numba
        c, lower = cho_factor(R, overwrite_a=False, check_finite=False)
        RinvA = cho_solve((c, lower), A, overwrite_b=False, check_finite=False)  # (M, Ndir)

        # Denominator: diag(A^H R^{-1} A) -> (Ndir,)
        den = np.sum(A.conj() * RinvA, axis=0)

        # Weights W = R^{-1} a / (a^H R^{-1} a) for all dirs -> (Ndir, M)
        epsilon = np.finfo(np.float64).eps  # prevent division by zero
        return cast(ComplexArray, (RinvA / den[None, :] + epsilon).T)

    def _mvdr_time_series(self, x: ComplexArray, sd: FloatArray) -> ComplexArray:
        """Compute MVDR beam time series by applying per-bin weights to the full spectrum.

        Each bin's covariance, and so its weights, is fixed across the whole input, so the
        beamformer is a linear time-invariant filter on every sensor. The weights are found on
        the STFT bins, where the frames supply the snapshots, and then applied to one
        full-length FFT of each sensor, as ``DelayAndSumBeamformer`` does with
        ``domain='frequency'``. Steering delays are therefore applied exactly, with no frame
        edges.

        Between STFT bins only the weights' departure from delay-and-sum,
        ``G = W * exp(+2j*pi*f*sd)``, is interpolated (linearly, held constant over the outer
        half-bins); the steering phase is evaluated exactly at every full-length bin. Since
        ``sum(G) = 1`` at every STFT bin, the interpolated weights stay distortionless in the
        look direction, and with very heavy loading ``G = 1/M`` the output equals
        delay-and-sum exactly. Bins more than half an STFT bin outside the active band are zero.

        Parameters
        ----------
        x : ComplexArray
            Sensor data matrix with shape ``(num_sensors, num_samples)``.
        sd : FloatArray
            Steering-delay matrix with shape ``(num_directions, num_sensors)``.

        Raises
        ------
        ValueError
            If ``nfft`` is too small for the array geometry.

        Returns
        -------
        ComplexArray
            Complex beam time series with shape ``(num_directions, num_samples)``.

        """
        fs, nfft = self.sampling_rate_hz, self.nfft
        num_sensors, num_samples = x.shape
        num_directions = sd.shape[0]
        if nfft / fs < (np.max(sd) - np.min(sd)):
            raise ValueError("nfft too small for this array")

        X = _stft(x, nfft, self.overlap)
        _, _, nfft_actual = X.shape
        f_bins = _stft_bin_frequencies(nfft_actual, fs, self.f0)
        active = _active_bin_indices(f_bins, self.fmin, self.fmax)
        output = np.zeros((num_directions, num_samples), dtype=np.complex128)
        if len(active) == 0:
            return output
        active = active[np.argsort(f_bins[active])]
        coarse_hz = f_bins[active]

        # Departure from delay-and-sum at each active STFT bin: (n_active, Ndir, M)
        G = np.zeros((len(active), num_directions, num_sensors), dtype=np.complex128)
        for j, i in enumerate(active):
            W = self._bin_weights(X[:, :, i], f_bins[i], sd)
            if W is not None:
                G[j] = W * np.exp(2j * np.pi * f_bins[i] * sd)

        # Full-length spectrum; keep the bins within half an STFT bin of the active band
        spectrum = np.fft.fft(x, axis=1)
        fine_hz = self.f0 + np.fft.fftfreq(num_samples, 1 / fs)
        if len(active) == nfft_actual:
            # Every bin is active: the band wraps round, so keep the whole spectrum
            fine = np.arange(num_samples)
        else:
            half_bin = 0.5 * fs / nfft_actual
            fine = np.nonzero(
                (fine_hz >= coarse_hz[0] - half_bin) & (fine_hz <= coarse_hz[-1] + half_bin)
            )[0]
        f_fine = fine_hz[fine]

        # Linear interpolation of G onto the fine bins, held constant beyond the end bins
        upper = np.clip(np.searchsorted(coarse_hz, f_fine), 1, max(1, len(active) - 1))
        lower = upper - 1
        if len(active) == 1:
            upper = lower = np.zeros_like(upper)
            t = np.zeros(len(f_fine))
        else:
            span = coarse_hz[upper] - coarse_hz[lower]
            t = np.clip((f_fine - coarse_hz[lower]) / span, 0.0, 1.0)

        X_fine = spectrum[:, fine]  # (M, n_fine)
        beams = np.zeros((num_directions, num_samples), dtype=np.complex128)
        for d in range(num_directions):
            G_fine = (1.0 - t)[:, None] * G[lower, d] + t[:, None] * G[upper, d]  # (n_fine, M)
            W_fine = G_fine * np.exp(-2j * np.pi * f_fine[:, None] * sd[d][None, :])
            beams[d, fine] = np.sum(W_fine.conj() * X_fine.T, axis=1)
        return np.fft.ifft(beams, axis=1)
