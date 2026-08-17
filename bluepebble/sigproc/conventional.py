"""conventional beamforming."""

import numpy as np
from numba import njit, prange, types
from numpy.typing import ArrayLike
from stonesoup.base import Property

from .base import (
    BeamformerOutput,
    ComplexArray,
    DomainType,
    FloatArray,
    MirrorPlan,
    _stft,
    _stft_bin_frequencies,
    _STFTBeamformer,
)


class DelayAndSumBeamformer(_STFTBeamformer):
    """A Delay-and-Sum (DAS) beamformer.

    Supports three processing modes:

    - ``'time'``: time-domain delay-and-sum.
    - ``'frequency'``: frequency-domain delay-and-sum.
    - ``'broadband_power'``: STFT-based incoherent broadband power integration.

    In all modes, steering is controlled via per-direction per-sensor delay values.

    Multiband processing
    --------------------
    Setting ``bands`` integrates several frequency bands from one pass over the data and
    returns a power map per band, with shape ``(num_bands, num_directions, num_frames)``.
    This requires ``domain='broadband_power'``: the time and frequency domains have no
    per-bin integration step for bands to select over.
    """

    shading: FloatArray | None = Property(
        default=None,
        doc="An array of shading weights applied to each sensor. If None, uniform weights are "
        "used.",
    )
    domain: DomainType = Property(
        default="time",
        doc="Beamforming domain: 'time', 'frequency', or 'broadband_power'.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the DelayAndSumBeamformer.

        Parameters
        ----------
        *args
            Positional arguments to pass to the parent class.
        **kwargs
            Keyword arguments forwarded to the parent class.
            Common options include ``sampling_rate_hz``, ``shading``, ``domain``,
            ``nfft``, ``overlap``, ``f0``, ``fmin``, ``fmax``, ``bands``, and
            ``normalise_by_bandwidth``.

        Raises
        ------
        ValueError
            If explicit shading weights do not sum to a finite non-zero value.
        ValueError
            If the specified domain is not ``'time'``, ``'frequency'``, or
            ``'broadband_power'``.
        ValueError
            If ``bands`` is set for a domain other than ``'broadband_power'``.

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

        if self.bands is not None and self.domain != "broadband_power":
            raise ValueError(
                f"bands requires domain='broadband_power', got domain={self.domain!r}. "
                "The time and frequency domains integrate no frequency bins, so there is "
                "nothing for bands to select over."
            )

    def beamform(
        self,
        sensor_signals: ArrayLike,
        steering_delays_s: ArrayLike,
        mirror_plan: MirrorPlan | None = None,
    ) -> BeamformerOutput:
        """Beamform sensor data using the configured DAS processing domain.

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

        Returns
        -------
        BeamformerOutput
            - ``domain='time'`` or ``'frequency'``: complex signals with shape
              ``(num_directions, num_samples_or_cropped_samples)``.
            - ``domain='broadband_power'``: real-valued power map with shape
              ``(num_directions, num_frames)``, or ``(num_bands, num_directions,
              num_frames)`` when ``bands`` is set.

            ``num_directions`` is always the full steering grid, even when ``mirror_plan``
            halved the actual computation.

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

        direction_axis = 0
        if self.domain == "time":
            time_das = _time_das_parallel if self.parallelise else _time_das_sequential
            output: BeamformerOutput = time_das(
                sensor_signals_array,
                steering_delays_array,
                shading_weights,
                self.sampling_rate_hz,
            )
        elif self.domain == "frequency":
            frequency_das = (
                _frequency_das_parallel if self.parallelise else _frequency_das_sequential
            )
            output = frequency_das(
                sensor_signals_array,
                steering_delays_array,
                shading_weights,
                self.sampling_rate_hz,
            )
        elif self.domain == "broadband_power":
            output = self.das_broadband_power(
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
            direction_axis = 1 if self.bands else 0
        else:
            msg = f"Unsupported beamforming domain: {self.domain}"
            raise ValueError(msg)

        if mirror_plan is not None:
            output = self.expand_mirrored(output, mirror_plan, direction_axis=direction_axis)
        return output

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
            Broadband beam power map with shape ``(num_directions, num_frames)``, or
            ``(num_bands, num_directions, num_frames)`` when ``bands`` is set on the
            beamformer. Each entry contains integrated beam power over the active frequency
            bins for one steering direction and one STFT frame.

        Notes
        -----
        The ``fmin`` and ``fmax`` arguments select the band in single-band mode only. When
        ``bands`` is set on the beamformer they are ignored in favour of the band edges.

        """
        M, _ = x.shape
        X = _stft(x, nfft, overlap)  # (M, n_frames, nfft)
        M, n_frames, nfft_actual = X.shape

        f_bins = _stft_bin_frequencies(nfft_actual, fs, f0)
        per_band_bins = self._band_bin_indices(f_bins, fmin, fmax)
        bins_to_bands = self._bins_to_bands(per_band_bins)

        Ndir = sd.shape[0]
        P = np.zeros((len(per_band_bins), Ndir, n_frames), dtype=np.float64)

        # Normalize weights
        w = shading_weights.reshape(1, M, 1)  # (1, M, 1) for broadcasting

        # Each bin in the union of all bands is steered once; the resulting power is then
        # accumulated into every band containing it, so overlapping bands cost nothing extra.
        for i in sorted(bins_to_bands):
            f = f_bins[i]

            # Snapshots at this bin: (M, n_frames)
            S = X[:, :, i]

            # Steering phase for all dirs/sensors: (Ndir, M)
            A = np.exp(1j * 2 * np.pi * f * sd)

            # Beamform: Y = sum_m w_m * A(dir,m) * S(m,frame)
            # Result: (Ndir, n_frames)
            Y = np.sum((A[:, :, None] * S[None, :, :]) * w, axis=1)

            # Accumulate power over frequency bins
            bin_power = np.abs(Y) ** 2
            for band_idx in bins_to_bands[i]:
                P[band_idx] += bin_power

        return self._finalise_band_power(P, per_band_bins)


_TIME_DAS_SIGNATURE = (
    types.Array(types.complex128, 2, "C"),
    types.Array(types.float64, 2, "C"),
    types.Array(types.float64, 1, "C"),
    types.float64,
)


def _time_das_impl(
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

    Notes
    -----
    Compiled into both a multi-threaded and a single-threaded variant (see
    ``_time_das_parallel``/``_time_das_sequential`` below); the direction loop below,
    written with ``prange``, degrades to a plain sequential loop under the
    single-threaded compilation.

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


_time_das_parallel = njit(
    _TIME_DAS_SIGNATURE, cache=True, parallel=True, fastmath=True
)(_time_das_impl)
_time_das_sequential = njit(
    _TIME_DAS_SIGNATURE, cache=True, parallel=False, fastmath=True
)(_time_das_impl)


_FREQUENCY_DAS_SIGNATURE = (
    types.Array(types.complex128, 2, "C"),
    types.Array(types.float64, 2, "C"),
    types.Array(types.float64, 1, "C"),
    types.float64,
)


def _frequency_das_impl(
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

    Notes
    -----
    Compiled into both a multi-threaded and a single-threaded variant (see
    ``_frequency_das_parallel``/``_frequency_das_sequential`` below); the direction loop
    below, written with ``prange``, degrades to a plain sequential loop under the
    single-threaded compilation.

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


_frequency_das_parallel = njit(
    _FREQUENCY_DAS_SIGNATURE, cache=True, parallel=True, fastmath=True
)(_frequency_das_impl)
_frequency_das_sequential = njit(
    _FREQUENCY_DAS_SIGNATURE, cache=True, parallel=False, fastmath=True
)(_frequency_das_impl)
