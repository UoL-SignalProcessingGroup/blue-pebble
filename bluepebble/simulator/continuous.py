"""Continuous acoustic sensor simulators module."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy import fft as scipy_fft
from stonesoup.base import Property
from tqdm import tqdm

from ..models.propagation import SpectrumPropagationModel
from ..signal.base import SourceSignal
from ..signal.utils import apply_fade_in, apply_fade_out
from .base import PassiveSonarArraySimulatorBase, SensorBatch

if TYPE_CHECKING:
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]
Complex64Array: TypeAlias = NDArray[np.complex64]
ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]


@dataclass
class _STFTCommonContext:
    """Shared STFT simulation metadata used by all synthesis modes."""

    all_timestamps: list[datetime]
    step_times_s: FloatArray
    n_steps: int
    num_sensors: int
    num_frames: int
    num_freq_bins: int
    frame_len: int
    fs: float
    frequencies: FloatArray
    hop: int
    window: FloatArray


@dataclass
class _STFTTargetHistory:
    """Per-target STFT source and channel histories sampled at simulator knots."""

    source_stft: Complex64Array
    H_hist: Complex64Array
    tau_hist: FloatArray


@dataclass
class _WOLATargetHistory:
    """Per-target STFT source and delay-de-rotated channel histories for WOLA synthesis."""

    source_stft: Complex64Array
    tau_hist: FloatArray
    H_mag_hist: FloatArray
    H_phase_hist: FloatArray
    bulk_hops: IntArray


def _hold_last_finite_delays(delay_history_s: FloatArray) -> FloatArray:
    """Replace non-finite per-sensor delays by holding the nearest finite delay in time.

    A non-finite delay marks a sensor with no arrival (zero channel gain already silences
    it), but left in place it would poison the synthesis phase term
    ``exp(-2j * pi * tau * f)`` with NaN, including into neighbouring interpolated knots.
    Holding the nearest finite delay keeps the trajectory physical across the gap without
    disturbing the zero gain that does the actual silencing.
    """
    delays = np.asarray(delay_history_s, dtype=np.float64).copy()
    finite = np.isfinite(delays)
    if finite.all():
        return delays

    n_steps, n_sensors = delays.shape
    step_idx = np.arange(n_steps)[:, np.newaxis]
    sensor_idx = np.arange(n_sensors)[np.newaxis, :]

    # Index of the nearest finite knot at or before each entry (-1 if there is none)...
    forward = np.maximum.accumulate(np.where(finite, step_idx, -1), axis=0)
    # ...and at or after it (n_steps if there is none).
    backward = np.minimum.accumulate(np.where(finite, step_idx, n_steps)[::-1], axis=0)[::-1]

    source = np.where(forward >= 0, forward, backward)
    has_source = (forward >= 0) | (backward < n_steps)
    held = delays[np.clip(source, 0, n_steps - 1), sensor_idx]

    return np.where(finite, delays, np.where(has_source, held, 0.0))


def _bulk_delay_hops(
    target: _STFTTargetHistory,
    step_idx: IntArray,
    alpha: FloatArray,
    fs: float,
    hop: int,
) -> IntArray:
    """Return one target's bulk delay in whole hops, ``D = rint(tau_ref * fs / hop)``, per frame.

    ``tau_ref`` is the earliest delay at each knot over the sensors the target reaches, or over
    all sensors where it reaches none, so the zero delay held for a never-reached sensor cannot
    cancel the shift. It is interpolated to the frames with ``step_idx`` and ``alpha``;
    ``step_idx`` may be the last knot where ``alpha`` is zero.
    """
    reached = np.any(target.H_hist != 0, axis=-1)
    tau_ref = np.where(reached, target.tau_hist, np.inf).min(axis=-1)
    tau_ref = np.where(np.isfinite(tau_ref), tau_ref, target.tau_hist.min(axis=-1))
    upper = np.minimum(step_idx + 1, len(tau_ref) - 1)
    tau_frames = tau_ref[step_idx] * (1.0 - alpha) + tau_ref[upper] * alpha
    return np.rint(tau_frames * fs / hop).astype(np.int64)


@njit(cache=True, parallel=True)
def _stft_interp_synthesis(
    source_stft,
    H_hist,
    tau_hist,
    bulk_hops,
    step_idx,
    alpha,
    frequencies,
    window,
    hop,
    fs,
    trim,
    out,
):
    """Synthesise every sensor's signal from frame-interpolated channels, in parallel.

    Fuses ``_build_residual_channel_histories``, ``_interpolate_channel_from_residual_histories``
    and ``inverse_stft`` into one loop per sensor, adding the channel and residual delay phases
    before a single exponential. Output frame ``i`` takes source frame ``i - bulk_hops[g, i]``,
    and nothing from target ``g`` if that is out of range. Each thread writes only its own
    sensor's row of ``out``.

    Shapes: ``source_stft`` (targets, frames, bins), ``H_hist`` (targets, steps, sensors, bins),
    ``tau_hist`` (targets, steps, sensors), ``bulk_hops`` (targets, frames), ``out`` (sensors,
    samples). There must be as many bins as the window has samples; the caller checks, since
    nothing here does. ``trim`` samples are dropped from each end of the overlap-added signal, as
    ``inverse_stft`` does.
    """
    n_targets, n_steps, n_sensors, n_bins = H_hist.shape
    n_frames = source_stft.shape[1]
    frame_len = window.shape[0]
    two_pi = 2 * np.pi

    for s in prange(n_sensors):
        total = np.zeros((n_frames, n_bins), np.complex64)
        mag = np.empty((n_steps, n_bins))
        phase = np.empty((n_steps, n_bins))
        for g in range(n_targets):
            for k in range(n_steps):
                for j in range(n_bins):
                    residual = H_hist[g, k, s, j] * np.exp(
                        2j * np.pi * tau_hist[g, k, s] * frequencies[j]
                    )
                    mag[k, j] = np.abs(residual)
                    phase[k, j] = np.arctan2(residual.imag, residual.real)
            # numpy.unwrap along the step axis, following its algorithm exactly so the
            # unwrapped phases are identical to the vectorised implementation's.
            for j in range(n_bins):
                correction = 0.0
                previous = phase[0, j]
                for k in range(1, n_steps):
                    current = phase[k, j]
                    dd = current - previous
                    ddmod = np.mod(dd + np.pi, two_pi) - np.pi
                    if ddmod == -np.pi and dd > 0:
                        ddmod = np.pi
                    if abs(dd) >= np.pi:
                        correction += ddmod - dd
                    previous = current
                    phase[k, j] = current + correction
            for i in range(n_frames):
                src = i - bulk_hops[g, i]
                if src < 0 or src >= n_frames:
                    continue
                k = step_idx[i]
                a = alpha[i]
                w0 = 1.0 - a
                tau = tau_hist[g, k, s] * w0 + tau_hist[g, k + 1, s] * a
                tau -= bulk_hops[g, i] * hop / fs
                for j in range(n_bins):
                    m = mag[k, j] * w0 + mag[k + 1, j] * a
                    p = phase[k, j] * w0 + phase[k + 1, j] * a - two_pi * tau * frequencies[j]
                    total[i, j] += source_stft[g, src, j] * complex(m * np.cos(p), m * np.sin(p))

        signal = np.zeros((n_frames - 1) * hop + frame_len, np.complex64)
        window_sum = np.zeros(signal.shape[0], np.float32)
        for i in range(n_frames):
            start = i * hop
            frame = np.fft.ifft(total[i])
            for n in range(frame_len):
                signal[start + n] += frame[n] * window[n]
                window_sum[start + n] += window[n] ** 2
        for n in range(signal.shape[0]):
            if window_sum[n] > 1e-10:
                signal[n] /= window_sum[n]
        out[s, :] = signal[trim : signal.shape[0] - trim]


def _stft_interp_receiver(
    ctx: _STFTCommonContext,
    targets_data: list[_STFTTargetHistory],
    step_idx: IntArray,
    alpha: FloatArray,
) -> Complex64Array:
    """Return every sensor's reconstructed signal for ``stft_interp`` synthesis, before fades.

    Parameters
    ----------
    ctx : _STFTCommonContext
        Shared STFT context.
    targets_data : list of _STFTTargetHistory
        Per-target source and propagation histories.
    step_idx : IntArray
        Lower interpolation knot for each frame.
    alpha : FloatArray
        Interpolation fraction for each frame.

    Returns
    -------
    Complex64Array
        Receiver signals with shape ``(num_sensors, num_samples)``; zero when there are no
        targets.

    Raises
    ------
    ValueError
        If ``ctx.num_freq_bins`` is not ``ctx.frame_len``.

    Notes
    -----
    Speed is bought with memory. The targets' channel histories are copied into one array for
    the kernel (8 bytes per step, sensor and bin, per target: about 46 MB each at 180 steps,
    64 sensors and 500 bins), and every Numba thread holds its own sensor's spectrum and
    signal buffers. Both scale with the run, so long runs with many targets need the headroom;
    ``NUMBA_NUM_THREADS`` caps the per-thread share.

    """
    # Overlap-adding ends a hop short of full coverage at each end, so those samples are
    # trimmed, unless the signal is too short to spare them.
    full_len = (ctx.num_frames - 1) * ctx.hop + ctx.frame_len
    trim = ctx.hop if full_len > 2 * ctx.hop else 0
    receiver = np.zeros((ctx.num_sensors, full_len - 2 * trim), dtype=np.complex64)
    if not targets_data:
        return receiver

    # The kernel reads frame_len samples from each inverse transform without a bounds check.
    if ctx.num_freq_bins != ctx.frame_len:
        raise ValueError(
            f"Invalid STFT shape ({ctx.num_frames}, {ctx.num_freq_bins}). Expected "
            f"num_freq_bins to be frame_len ({ctx.frame_len})."
        )

    _stft_interp_synthesis(
        np.ascontiguousarray([t.source_stft for t in targets_data], dtype=np.complex64),
        np.ascontiguousarray([t.H_hist for t in targets_data], dtype=np.complex64),
        np.ascontiguousarray([t.tau_hist for t in targets_data], dtype=np.float64),
        np.array([_bulk_delay_hops(t, step_idx, alpha, ctx.fs, ctx.hop) for t in targets_data]),
        np.ascontiguousarray(step_idx),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(ctx.frequencies, dtype=np.float64),
        np.ascontiguousarray(ctx.window),
        ctx.hop,
        float(ctx.fs),
        trim,
        receiver,
    )
    return receiver


class ContinuousSTFTPassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Continuous broadband passive-sonar simulator with selectable STFT synthesis mode.

    Use ``mode`` for channel update method. All three modes use the same
    propagation history, then
    differ in reconstruction method:

    - ``stft_interp`` and ``wola_interp`` common interpolation steps:
        1. De-rotate channel phase using per-sensor delay history (remove delay phase).
        2. Interpolate residual channel magnitude and unwrapped phase at frame-center times.
        3. Shift source frames by the bulk delay, re-apply the rest of the interpolated delay
           phase and form per-target frame spectra (see Propagation delay).

    - ``stft_interp``
        4. Sum target spectra in the frequency domain.
        5. Reconstruct sensor time series by inverse FFT and overlap-add, in one compiled pass
           per sensor, with sensors processed in parallel.

    - ``wola_interp``
        4. IFFT each frame and accumulate with weighted overlap-add (WOLA).
        5. Normalize overlap energy to produce output sensor signals.

    - ``cola``
        1. Select the nearest previous knot transfer function for each frame.
        2. Apply knot transfer function, less the bulk delay's phase, to each target's
           bulk-shifted source spectrum.
        3. IFFT and overlap-add frame signals.
        4. Apply COLA-style overlap normalization.

    All modes share fades, noise addition, beamforming, and timestamp chunk output.
    In practice, each method produces largely similar results. The default ``stft_interp``
    and ``wola_interp`` methods are recommended. The ``cola`` method can produce
    interference artefacts, but is good as a fast baseline.

    Propagation delay
    -----------------
    A frame spectrum's phase can only shift its content circularly within the frame, so each
    target's delay is split into a bulk part, applied as a whole-hop shift of source frames,
    and a residual, applied as phase. Output frame ``i`` of sensor ``s`` is

    ``Y_s[i, f] = X[i - D_i, f] * R_s(t_i, f) * exp(-2j * pi * f * (tau_s(t_i) - D_i * hop / fs))``

    where ``X`` is the source STFT, ``R_s`` the delay-de-rotated channel, ``tau_s`` the
    sensor's delay, ``D_i = rint(tau_ref(t_i) * fs / hop)`` and ``tau_ref`` the earliest delay
    over the sensors the target reaches. ``cola`` takes the channel and ``D_i`` from the
    nearest previous knot instead of interpolating.

    The residual, at most half a hop plus the array's delay spread, is still a circular shift
    within each frame, so those samples wrap round the frame's edges, where the window is
    small. Measured against the exactly delayed signal with a Hann window and
    ``hop = frame_len / 10``, a constant half-hop residual costs 0.15 dB of power with an
    error 36 dB below the signal. ``D_i`` steps by one every ``hop * c / fs`` of range
    change; frames either side of a step agree, since the source-frame shift cancels the
    residual's jump, but the error rises to about 27 dB below the signal for the hops around
    it (20 to 120 Hz noise, 3 m/s range rate). Other windows and hop factors were not
    measured.

    Assumptions
    -----------
    - The source emits nothing before the first timestep: a target contributes nothing to
      frames with ``i - D_i < 0``, so it is first heard about ``tau_ref`` seconds in.
    - One ``D_i`` serves every sensor, which keeps them mutually coherent; the delay spread
      across the array is assumed small next to a frame, since it adds to the residual.

    Doppler
    -------
    Doppler emerges from the time-varying delay ``tau(t) = range(t) / c``, carried here as a
    per-frame phase term. It's only correct below the frame-rate Nyquist limit
    ``max_freq * max|v_radial| / c < fs / (2 * hop)``; above that it aliases silently. Raise
    ``hop_factor`` to lift the limit, or use
    :class:`ContinuousFractionalDelayPassiveSonarArraySimulator` (no frame-rate limit). See the
    ``doppler_cpa`` gallery example.
    """

    signal_models: list[SourceSignal] = Property(
        doc="Source signal models (one per target, or a single-element list for all); any "
        ":class:`~bluepebble.signal.SourceSignal`, anthropogenic or biological",
    )
    mode: str = Property(
        default="stft_interp",
        doc="Synthesis mode: stft_interp, wola_interp, or cola",
    )
    fade_in_ms: float = Property(default=100.0, doc="Fade-in duration at arrival (ms)")
    fade_out_ms: float = Property(default=100.0, doc="Fade-out duration at end (ms)")
    norm_floor_ratio: float = Property(
        default=1e-3,
        doc=(
            "Relative floor (fraction of max overlap weight) below which COLA/WOLA "
            "normalization is not applied to avoid edge gain blow-up"
        ),
    )

    @staticmethod
    def _valid_modes() -> set[str]:
        """Return supported synthesis mode names."""
        return {"stft_interp", "wola_interp", "cola"}

    def _validate_mode(self) -> str:
        """Validate and return the configured synthesis mode.

        Returns
        -------
        str
            Validated synthesis mode string.

        Raises
        ------
        ValueError
            If ``self.mode`` is not one of the supported modes.

        """
        selected_mode = str(self.mode)
        if selected_mode not in self._valid_modes():
            msg = (
                f"Unsupported mode: {selected_mode!r}. "
                f"Expected one of {sorted(self._valid_modes())}."
            )
            raise ValueError(msg)
        return selected_mode

    @staticmethod
    def _interp_index_alpha(time_s: float, knot_times_s: FloatArray) -> tuple[int, float]:
        """Return interpolation index and mixing weight for knot times.

        Parameters
        ----------
        time_s : float
            Query time in seconds.
        knot_times_s : numpy.ndarray
            Monotonic knot times in seconds.

        Returns
        -------
        tuple of (int, float)
            Lower knot index and interpolation fraction in ``[0, 1]``.

        """
        if len(knot_times_s) < 2:
            return 0, 0.0

        idx = int(np.searchsorted(knot_times_s, time_s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(knot_times_s) - 2))

        t0 = float(knot_times_s[idx])
        t1 = float(knot_times_s[idx + 1])
        dt = max(t1 - t0, 1e-12)
        alpha = float(np.clip((time_s - t0) / dt, 0.0, 1.0))
        return idx, alpha

    @staticmethod
    def _ifft_frame(frame_spec: ComplexArray, frame_len: int, num_freq_bins: int) -> ComplexArray:
        """Inverse-transform one STFT frame.

        Parameters
        ----------
        frame_spec : numpy.ndarray
            Complex frame spectrum for one sensor and one frame.
        frame_len : int
            Time-domain frame length in samples.
        num_freq_bins : int
            Number of frequency bins represented in ``frame_spec``.

        Returns
        -------
        numpy.ndarray
            Reconstructed time-domain frame.

        Raises
        ------
        ValueError
            If ``num_freq_bins`` is not ``frame_len``.

        """
        if num_freq_bins != frame_len:
            raise ValueError(f"Invalid STFT bin count: {num_freq_bins}. Expected {frame_len}.")
        return np.fft.ifft(frame_spec, n=frame_len)

    def _build_common_context(
        self,
        all_timestamps: list[datetime],
        signal_models_list: "list[SourceSignal]",
    ) -> _STFTCommonContext:
        """Build shared STFT metadata for synthesis.

        Parameters
        ----------
        all_timestamps : list of datetime
            Simulation timestamps used as propagation knots.
        signal_models_list : list
            Resolved list of signal models; only ``signal_models_list[0]`` is
            used to derive geometry.

        Returns
        -------
        _STFTCommonContext
            Shared context containing timing, STFT geometry, and array metadata.

        """
        ref_model = signal_models_list[0]
        num_freq_bins, frequencies, hop, window, num_frames = ref_model.stft_geometry()
        frame_len = int(ref_model.frame_len)
        fs = float(ref_model.sampling_rate_hz)
        num_sensors = int(self._array().num_sensors)

        t0 = all_timestamps[0]
        step_times_s = np.array(
            [(ts - t0).total_seconds() for ts in all_timestamps],
            dtype=np.float64,
        )
        return _STFTCommonContext(
            all_timestamps=all_timestamps,
            step_times_s=step_times_s,
            n_steps=len(step_times_s),
            num_sensors=num_sensors,
            num_frames=num_frames,
            num_freq_bins=num_freq_bins,
            frame_len=frame_len,
            fs=fs,
            frequencies=np.asarray(frequencies, dtype=np.float64),
            hop=int(hop),
            window=np.asarray(window),
        )

    @staticmethod
    def _source_stfts(
        ctx: _STFTCommonContext,
        ground_truth_paths: "Sequence[Iterable[State]]",
        signal_models_list: "list[SourceSignal]",
    ) -> list[Complex64Array]:
        """Compute every target's source STFT.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.
        ground_truth_paths : Sequence of Iterable of State
            Target trajectories.
        signal_models_list : list
            Resolved signal-model list aligned with targets.

        Returns
        -------
        list of numpy.ndarray
            One source STFT per target, each of shape ``(num_frames, num_freq_bins)``.

        Raises
        ------
        ValueError
            If a target path has no states.
        RuntimeError
            If a target's STFT does not have the shape ``ctx`` took from the first model's
            :meth:`~bluepebble.signal.SourceSignal.stft_geometry`.

        """
        source_stfts: list[Complex64Array] = []
        for target_idx, target_path in enumerate(ground_truth_paths):
            try:
                target_first_state = next(iter(target_path))
            except StopIteration:
                msg = f"ground_truth_paths[{target_idx}] has no states"
                raise ValueError(msg) from None
            source_stft, _, _, _ = signal_models_list[target_idx].compute_stft(target_first_state)
            if source_stft.shape != (ctx.num_frames, ctx.num_freq_bins):
                msg = (
                    "All target source STFTs must share the same shape. "
                    f"Expected {(ctx.num_frames, ctx.num_freq_bins)}, "
                    f"got {source_stft.shape}."
                )
                raise RuntimeError(msg)
            source_stfts.append(source_stft)

        return source_stfts

    def _build_target_histories(
        self,
        ctx: _STFTCommonContext,
        ground_truth_paths: "Sequence[Iterable[State]]",
        signal_models_list: "list[SourceSignal]",
        progress_bar: bool = False,
    ) -> list[_STFTTargetHistory]:
        """Build per-target source and channel histories at knot times.

        This is the dominant cost of :meth:`sensor_data_gen` for scenarios with targets: every
        knot of every target requires its own ``propagate_spectrum`` call, all of it done here
        before the first sensor snapshot is yielded.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.
        ground_truth_paths : Sequence of Iterable of State
            Target trajectories.
        signal_models_list : list
            Resolved signal-model list aligned with targets.
        progress_bar : bool, optional
            If True, show a progress bar over every (target, knot) propagation call, by
            default False.

        Returns
        -------
        list of _STFTTargetHistory
            Source STFT and propagation history for each target.

        Raises
        ------
        TypeError
            If the propagation model does not implement ``propagate_spectrum``.
        RuntimeError
            If target STFT shapes are inconsistent across targets.

        """
        targets_data: list[_STFTTargetHistory] = []
        if not ground_truth_paths:
            return targets_data

        if not isinstance(self.propagation_model, SpectrumPropagationModel):
            msg = (
                f"{type(self.propagation_model).__name__} does not implement "
                "'propagate_spectrum', which is required for STFT-based simulation"
            )
            raise TypeError(msg)
        spectrum_propagation_model = cast(SpectrumPropagationModel, self.propagation_model)

        source_stfts = self._source_stfts(ctx, ground_truth_paths, signal_models_list)

        with tqdm(
            total=len(ground_truth_paths) * len(ctx.all_timestamps),
            disable=not progress_bar,
            desc="Propagating signals",
        ) as bar:
            for target_path, target_source_stft in zip(
                ground_truth_paths, source_stfts, strict=True
            ):
                H_hist = np.zeros(
                    (ctx.n_steps, ctx.num_sensors, ctx.num_freq_bins), dtype=np.complex64
                )
                tau_hist = np.zeros((ctx.n_steps, ctx.num_sensors), dtype=np.float64)

                states_by_time = self._states_by_timestamp(target_path)
                for step_idx, timestamp in enumerate(ctx.all_timestamps):
                    platform_state = self._platform_state_at(timestamp)
                    target_state = states_by_time.get(timestamp)
                    if target_state is None:
                        bar.update(1)
                        continue

                    H_sensors, sensor_delays_s = spectrum_propagation_model.propagate_spectrum(
                        platform_state,
                        target_state,
                        ctx.frequencies,
                    )
                    H_hist[step_idx, :, :] = np.asarray(H_sensors, dtype=np.complex64)
                    tau_hist[step_idx, :] = np.asarray(sensor_delays_s, dtype=np.float64)
                    bar.update(1)

                targets_data.append(
                    _STFTTargetHistory(
                        source_stft=target_source_stft,
                        H_hist=H_hist,
                        tau_hist=_hold_last_finite_delays(tau_hist),
                    )
                )

        return targets_data

    def _frame_interp_indices(self, ctx: _STFTCommonContext) -> tuple[IntArray, FloatArray]:
        """Map frame centres to neighbouring knot indices and weights.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Lower knot indices per frame and interpolation fractions per frame.

        """
        frame_times_s = (
            np.arange(ctx.num_frames, dtype=np.float64) * ctx.hop + 0.5 * ctx.frame_len
        ) / ctx.fs
        step_idx = np.searchsorted(ctx.step_times_s, frame_times_s, side="right") - 1
        step_idx = np.clip(step_idx, 0, ctx.n_steps - 2).astype(np.int32)

        t0 = ctx.step_times_s[step_idx]
        t1 = ctx.step_times_s[step_idx + 1]
        dt = np.maximum(t1 - t0, 1e-12)
        alpha = np.clip((frame_times_s - t0) / dt, 0.0, 1.0)
        return step_idx, alpha

    def _build_residual_channel_histories(
        self,
        H_hist: Complex64Array,
        tau_hist: FloatArray,
        frequencies_hz: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Build delay-de-rotated channel histories.

        Parameters
        ----------
        H_hist : numpy.ndarray
            Complex channel transfer history with shape
            ``(n_steps, n_sensors, n_freq_bins)``.
        tau_hist : numpy.ndarray
            Absolute delay history in seconds with shape
            ``(n_steps, n_sensors)``.
        frequencies_hz : numpy.ndarray
            Frequency bins in hertz.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Residual magnitude history and unwrapped residual phase history.

        """
        phase_derotate = np.exp(
            2j * np.pi * tau_hist[:, :, np.newaxis] * frequencies_hz[np.newaxis, np.newaxis, :]
        )
        H_residual = H_hist * phase_derotate
        H_mag_hist = np.asarray(np.abs(H_residual), dtype=np.float64)
        H_phase_hist = np.asarray(np.unwrap(np.angle(H_residual), axis=0), dtype=np.float64)
        return H_mag_hist, H_phase_hist

    def _interpolate_channel_from_residual_histories(
        self,
        H_mag_hist: FloatArray,
        H_phase_hist: FloatArray,
        tau_hist: FloatArray,
        step_idx: int | IntArray,
        alpha: float | ArrayLike,
        frequencies_hz: FloatArray,
        bulk_delay_s: float | ArrayLike = 0.0,
    ) -> ComplexArray:
        """Interpolate residual channel history and re-apply the delay phase, less a bulk delay.

        Parameters
        ----------
        H_mag_hist : numpy.ndarray
            Residual magnitude history.
        H_phase_hist : numpy.ndarray
            Residual unwrapped phase history.
        tau_hist : numpy.ndarray
            Delay history in seconds.
        step_idx : int or numpy.ndarray
            Lower interpolation index or indices.
        alpha : float or array-like
            Interpolation fraction(s) in ``[0, 1]``.
        frequencies_hz : numpy.ndarray
            Frequency bins in hertz.
        bulk_delay_s : float or array-like, optional
            Delay (per frame, for array ``alpha``) the caller applies by shifting source
            frames, so it is left out of the re-applied phase.

        Returns
        -------
        numpy.ndarray
            Interpolated complex transfer function(s).

        """
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        w0 = 1.0 - alpha_arr

        if np.ndim(alpha_arr) == 0:
            H_mag = H_mag_hist[step_idx, :] * w0 + H_mag_hist[step_idx + 1, :] * alpha_arr
            H_phase = H_phase_hist[step_idx, :] * w0 + H_phase_hist[step_idx + 1, :] * alpha_arr
            tau = tau_hist[step_idx] * w0 + tau_hist[step_idx + 1] * alpha_arr - bulk_delay_s
            phase_rerotate = np.exp(-2j * np.pi * frequencies_hz * tau)
            return H_mag * np.exp(1j * H_phase) * phase_rerotate

        H_mag = (
            H_mag_hist[step_idx, :] * w0[:, np.newaxis]
            + H_mag_hist[step_idx + 1, :] * alpha_arr[:, np.newaxis]
        )
        H_phase = (
            H_phase_hist[step_idx, :] * w0[:, np.newaxis]
            + H_phase_hist[step_idx + 1, :] * alpha_arr[:, np.newaxis]
        )
        tau = tau_hist[step_idx] * w0 + tau_hist[step_idx + 1] * alpha_arr - bulk_delay_s
        phase_rerotate = np.exp(-2j * np.pi * tau[:, np.newaxis] * frequencies_hz[np.newaxis, :])
        return H_mag * np.exp(1j * H_phase) * phase_rerotate

    def _slice_uniform_step_samples(
        self,
        receiver_signals: Complex64Array,
        n_steps: int,
    ) -> IntArray:
        """Create equal-length sample boundaries for each timestep.

        Parameters
        ----------
        receiver_signals : numpy.ndarray
            Rendered receiver signals with shape ``(n_sensors, n_samples)``.
        n_steps : int
            Number of simulation steps.

        Returns
        -------
        numpy.ndarray
            Monotonic sample index boundaries of length ``n_steps + 1``.

        """
        actual_signal_len = receiver_signals.shape[1]
        samples_per_step = actual_signal_len // n_steps
        sample_idx = np.arange(n_steps + 1, dtype=np.int64) * samples_per_step
        sample_idx[-1] = actual_signal_len
        return sample_idx

    def _slice_knot_step_samples(self, ctx: _STFTCommonContext, out_len: int) -> IntArray:
        """Create sample boundaries from knot times.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context with knot times and sample rate.
        out_len : int
            Output signal length in samples.

        Returns
        -------
        numpy.ndarray
            Monotonic sample boundaries aligned to knot times.

        """
        step_sample_idx = np.rint(ctx.step_times_s * ctx.fs).astype(int)
        step_sample_idx = np.clip(step_sample_idx, 0, out_len)
        step_sample_idx = np.maximum.accumulate(step_sample_idx)
        return step_sample_idx

    def _apply_fades(
        self,
        receiver_signals: Complex64Array,
        fs: float,
        *,
        do_fade_out: bool,
    ) -> Complex64Array:
        """Apply configured fade-in and optional fade-out to each sensor.

        Parameters
        ----------
        receiver_signals : numpy.ndarray
            Sensor signals to fade in place.
        fs : float
            Sampling rate in hertz.
        do_fade_out : bool
            Whether to apply fade-out using ``fade_out_ms``.

        Returns
        -------
        numpy.ndarray
            Faded sensor signals.

        """
        if self.fade_in_ms > 0:
            fade_samples = int(self.fade_in_ms * fs / 1000.0)
            for sensor_idx in range(receiver_signals.shape[0]):
                receiver_signals[sensor_idx, :] = apply_fade_in(
                    receiver_signals[sensor_idx, :],
                    fade_samples,
                )

        if do_fade_out and self.fade_out_ms > 0:
            fade_samples = int(self.fade_out_ms * fs / 1000.0)
            for sensor_idx in range(receiver_signals.shape[0]):
                receiver_signals[sensor_idx, :] = apply_fade_out(
                    receiver_signals[sensor_idx, :],
                    fade_samples,
                )
        return receiver_signals

    def _synthesise_stft_interp(
        self,
        ctx: _STFTCommonContext,
        targets_data: list[_STFTTargetHistory],
    ) -> tuple[Complex64Array, IntArray]:
        """Synthesize receiver signals using frame-wise channel interpolation.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.
        targets_data : list of _STFTTargetHistory
            Per-target source and propagation histories.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Receiver signals and timestep sample boundaries.

        """
        step_idx, alpha = self._frame_interp_indices(ctx)
        receiver = _stft_interp_receiver(ctx, targets_data, step_idx, alpha)
        receiver = self._apply_fades(receiver, ctx.fs, do_fade_out=False)
        return receiver, self._slice_uniform_step_samples(receiver, ctx.n_steps)

    def _synthesise_wola_interp(
        self,
        ctx: _STFTCommonContext,
        targets_data: list[_STFTTargetHistory],
    ) -> tuple[Complex64Array, IntArray]:
        """Synthesize receiver signals using WOLA with channel interpolation.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.
        targets_data : list of _STFTTargetHistory
            Per-target source and propagation histories.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Receiver signals and knot-aligned sample boundaries.

        """
        out_len = (ctx.num_frames - 1) * ctx.hop + ctx.frame_len
        receiver_accum = np.zeros((ctx.num_sensors, out_len), dtype=np.complex64)
        norm_accum = np.zeros(out_len, dtype=np.float64)
        window_arr = np.asarray(ctx.window, dtype=np.float32)
        window_sq = np.asarray(ctx.window, dtype=np.float64) ** 2

        # Precompute residual channel magnitude/phase histories once per target.
        # Doing this inside the frame loop is the dominant runtime cost.
        frame_step_idx, frame_alpha = self._frame_interp_indices(ctx)
        wola_histories: list[_WOLATargetHistory] = []
        for target_data in targets_data:
            H_mag_hist, H_phase_hist = self._build_residual_channel_histories(
                H_hist=target_data.H_hist,
                tau_hist=target_data.tau_hist,
                frequencies_hz=ctx.frequencies,
            )
            wola_histories.append(
                _WOLATargetHistory(
                    source_stft=target_data.source_stft,
                    tau_hist=target_data.tau_hist,
                    H_mag_hist=H_mag_hist,
                    H_phase_hist=H_phase_hist,
                    bulk_hops=_bulk_delay_hops(
                        target_data, frame_step_idx, frame_alpha, ctx.fs, ctx.hop
                    ),
                )
            )

        for frame_idx in range(ctx.num_frames):
            start = frame_idx * ctx.hop
            end = start + ctx.frame_len
            frame_center_s = (start + 0.5 * ctx.frame_len) / ctx.fs
            step_idx, alpha = self._interp_index_alpha(frame_center_s, ctx.step_times_s)

            norm_accum[start:end] += window_sq

            for sensor_idx in range(ctx.num_sensors):
                frame_sensor_sum = np.zeros(ctx.frame_len, dtype=np.complex64)

                for wola_data in wola_histories:
                    bulk_hops = int(wola_data.bulk_hops[frame_idx])
                    source_idx = frame_idx - bulk_hops
                    if not 0 <= source_idx < ctx.num_frames:
                        continue
                    source_spec = wola_data.source_stft[source_idx, :]
                    H_mag_hist = wola_data.H_mag_hist[:, sensor_idx, :]
                    H_phase_hist = wola_data.H_phase_hist[:, sensor_idx, :]
                    tau_hist = wola_data.tau_hist[:, sensor_idx]

                    H_interp = self._interpolate_channel_from_residual_histories(
                        H_mag_hist=H_mag_hist,
                        H_phase_hist=H_phase_hist,
                        tau_hist=tau_hist,
                        step_idx=step_idx,
                        alpha=alpha,
                        frequencies_hz=ctx.frequencies,
                        bulk_delay_s=bulk_hops * ctx.hop / ctx.fs,
                    )

                    frame_spec = source_spec * H_interp
                    frame_td = self._ifft_frame(frame_spec, ctx.frame_len, ctx.num_freq_bins)
                    frame_sensor_sum += np.asarray(frame_td, dtype=np.complex64) * window_arr

                receiver_accum[sensor_idx, start:end] += frame_sensor_sum

        norm_floor = max(1e-12, float(self.norm_floor_ratio) * float(np.max(norm_accum)))
        valid = norm_accum > norm_floor
        receiver_accum[:, valid] /= norm_accum[valid][np.newaxis, :]
        receiver_accum[:, ~valid] = 0.0

        receiver_accum = self._apply_fades(receiver_accum, ctx.fs, do_fade_out=True)
        return receiver_accum, self._slice_knot_step_samples(ctx, out_len)

    def _synthesise_cola(
        self,
        ctx: _STFTCommonContext,
        targets_data: list[_STFTTargetHistory],
    ) -> tuple[Complex64Array, IntArray]:
        """Synthesize receiver signals via nearest-knot COLA rendering.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.
        targets_data : list of _STFTTargetHistory
            Per-target source and propagation histories.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Receiver signals and knot-aligned sample boundaries.

        """
        out_len = (ctx.num_frames - 1) * ctx.hop + ctx.frame_len
        receiver_accum = np.zeros((ctx.num_sensors, out_len), dtype=np.complex64)
        norm_accum = np.zeros(out_len, dtype=np.float64)
        window_arr = np.asarray(ctx.window, dtype=np.float32)
        window_sq = np.asarray(ctx.window, dtype=np.float64) ** 2

        frame_centres_s = (
            np.arange(ctx.num_frames, dtype=np.float64) * ctx.hop + 0.5 * ctx.frame_len
        ) / ctx.fs
        knot_indices = np.searchsorted(ctx.step_times_s, frame_centres_s, side="right") - 1
        knot_indices = np.clip(knot_indices, 0, ctx.n_steps - 1)
        no_interp = np.zeros(ctx.num_frames)
        bulk_hops = [
            _bulk_delay_hops(target_data, knot_indices, no_interp, ctx.fs, ctx.hop)
            for target_data in targets_data
        ]

        for frame_idx in range(ctx.num_frames):
            start = frame_idx * ctx.hop
            end = start + ctx.frame_len
            knot_idx = int(knot_indices[frame_idx])
            # The knot channels carry the whole delay's phase, so the part the frame shift
            # applies is taken out of the shifted source spectrum (None before it arrives).
            shifted_sources: list[ComplexArray | None] = []
            for target_data, hops in zip(targets_data, bulk_hops, strict=True):
                source_idx = frame_idx - int(hops[frame_idx])
                if not 0 <= source_idx < ctx.num_frames:
                    shifted_sources.append(None)
                    continue
                bulk_delay_s = hops[frame_idx] * ctx.hop / ctx.fs
                derotate = np.exp(2j * np.pi * ctx.frequencies * bulk_delay_s).astype(np.complex64)
                shifted_sources.append(target_data.source_stft[source_idx, :] * derotate)

            norm_accum[start:end] += window_sq

            for sensor_idx in range(ctx.num_sensors):
                frame_sensor_sum = np.zeros(ctx.frame_len, dtype=np.complex64)
                for target_data, source_spec in zip(targets_data, shifted_sources, strict=True):
                    if source_spec is None:
                        continue
                    H_knot = target_data.H_hist[knot_idx, sensor_idx, :]
                    frame_spec = source_spec * H_knot
                    frame_td = self._ifft_frame(frame_spec, ctx.frame_len, ctx.num_freq_bins)
                    frame_sensor_sum += np.asarray(frame_td, dtype=np.complex64) * window_arr

                receiver_accum[sensor_idx, start:end] += frame_sensor_sum

        norm_floor = max(1e-12, float(self.norm_floor_ratio) * float(np.max(norm_accum)))
        valid = norm_accum > norm_floor
        receiver_accum[:, valid] /= norm_accum[valid][np.newaxis, :]
        receiver_accum[:, ~valid] = 0.0

        receiver_accum = self._apply_fades(receiver_accum, ctx.fs, do_fade_out=True)
        return receiver_accum, self._slice_knot_step_samples(ctx, out_len)

    def sensor_data_gen(self, progress_bar: bool = False) -> Iterator[SensorBatch]:
        """Yield continuous broadband sensor snapshots using selected STFT mode.

        Every target's propagation history is built in full before the first snapshot is
        yielded (see :meth:`_build_target_histories`), so with targets present there is a
        delay before this starts yielding that ``progress_bar`` reports on but a caller
        iterating the result cannot otherwise observe.

        Parameters
        ----------
        progress_bar : bool, optional
            If True, show a progress bar over target propagation while it builds, by
            default False. Covers only that up-front phase; wrap the returned iterator
            separately (e.g. via :meth:`~.PassiveSonarDetector.detections_gen`'s own
            ``progress_bar``) to track per-timestep yields too.

        Yields
        ------
        tuple of (datetime, set of PassiveSonarSensorData)
            Timestamp and simulated sensor-data set for that timestep.

        Raises
        ------
        ValueError
            If fewer than two timesteps are available or the synthesis mode is
            unsupported.

        """
        selected_mode = self._validate_mode()
        all_timestamps = self._sorted_timestamps()
        if len(all_timestamps) < 2:
            msg = "Need at least 2 timesteps for broadband processing"
            raise ValueError(msg)

        ground_truth_paths = self.ground_truth_paths or []
        ref_signal_model = self.signal_models[0]
        signal_models_list = self._resolve_models(
            self.signal_models,
            len(ground_truth_paths),
            "signal models",
        )
        ctx = self._build_common_context(all_timestamps, [ref_signal_model])
        targets_data = self._build_target_histories(
            ctx, ground_truth_paths, signal_models_list, progress_bar=progress_bar
        )

        if selected_mode == "stft_interp":
            receiver_signals, step_sample_idx = self._synthesise_stft_interp(ctx, targets_data)
        elif selected_mode == "wola_interp":
            receiver_signals, step_sample_idx = self._synthesise_wola_interp(ctx, targets_data)
        else:
            receiver_signals, step_sample_idx = self._synthesise_cola(ctx, targets_data)

        for step_idx, timestamp in enumerate(ctx.all_timestamps):
            start = int(step_sample_idx[step_idx])
            end = (
                int(step_sample_idx[step_idx + 1])
                if step_idx < ctx.n_steps - 1
                else receiver_signals.shape[1]
            )
            sensor_signals = receiver_signals[:, start:end].copy()

            noise = self._generate_noise(
                num_sensors=ctx.num_sensors,
                num_samples=sensor_signals.shape[1],
            )
            if noise is not None:
                sensor_signals += noise

            beamformed_data, steering_bearings_rad = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
                steering_bearings_rad=steering_bearings_rad,
            )
            yield timestamp, {sensor_data}


class ContinuousFractionalDelayPassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Broadband simulator with sample-domain fractional-delay rendering.

    This method avoids frame-wise channel interpolation and renders each sensor using
    time-varying fractional delay plus gain envelope:

    ``y_s[n] = a_s[n] * x(n - tau_s[n] * fs)``

    Here ``tau_s[n]`` is interpolated absolute delay and ``a_s[n]`` is interpolated broadband
    gain. Broadband gain is estimated from ``propagate_spectrum`` using RMS magnitude across
    frequency bins.

    Algorithm overview
    ------------------
    1. Build per-target source signals with matched lengths.
    2. Compute per-knot transfer functions and sensor delay history.
    3. Derive broadband gain history from ``|H_s(f)|``.
    4. Interpolate gain and delay to per-sample trajectories.
    5. Fractionally resample and accumulate target contributions.
    6. Apply fades, noise, beamforming, and timestamp slicing.

    (A knot is a tie point between simulation steps)

    Each source's waveform comes from ``get_source_waveform``, which caches it, so a second
    run repeats the first run's waveforms unless each model's ``reset()`` is called first.

    Tradeoffs
    ---------
    - Strong arrival-time fidelity under fast geometry changes.
    - **Doppler-faithful**: time-scaling via ``x(n - tau[n]*fs)`` reproduces the frequency shift
      with no frame-rate limit, unlike the STFT modes. Recommended for Doppler-critical work; see
      the ``doppler_cpa`` gallery example.
    - Lower spectral-detail fidelity than full complex frame-wise synthesis: the per-sensor channel
      is collapsed to a single broadband RMS gain (frequency-flat), so frequency-dependent
      transmission loss and multipath colour in ``H(f)`` are not reproduced.
    """

    signal_models: list[SourceSignal] = Property(
        doc="Source signal models (one per target, or a single-element list for all); any "
        ":class:`~bluepebble.signal.SourceSignal`, anthropogenic or biological",
    )
    fade_in_ms: float = Property(default=100.0, doc="Fade-in duration at arrival (ms)")
    fade_out_ms: float = Property(default=100.0, doc="Fade-out duration at end (ms)")

    @staticmethod
    def _fractional_delay_resample(
        source_signal: ArrayLike,
        delay_s: FloatArray,
        fs: float,
    ) -> Complex64Array:
        """Apply time-varying fractional delay to one source waveform.

        Parameters
        ----------
        source_signal : array-like
            Source waveform samples.
        delay_s : numpy.ndarray
            Per-sample delay trajectory in seconds.
        fs : float
            Sampling rate in hertz.

        Returns
        -------
        numpy.ndarray
            Fractionally delayed waveform as ``complex64``.

        """
        source = np.asarray(source_signal)
        out_len = len(delay_s)
        src_idx = np.arange(out_len, dtype=np.float64) - (
            np.asarray(delay_s, dtype=np.float64) * fs
        )
        src_n = np.arange(len(source), dtype=np.float64)

        delayed = np.interp(src_idx, src_n, source.astype(np.complex128), left=0.0, right=0.0)
        return delayed.astype(np.complex64)

    def sensor_data_gen(self, progress_bar: bool = False) -> Iterator[SensorBatch]:
        """Yield broadband sensor snapshots via fractional-delay rendering.

        Every target's propagation history is built in full before the first snapshot is
        yielded, so with targets present there is a delay before this starts yielding that
        ``progress_bar`` reports on but a caller iterating the result cannot otherwise observe.

        Parameters
        ----------
        progress_bar : bool, optional
            If True, show a progress bar over target propagation while it builds, by
            default False. Covers only that up-front phase; wrap the returned iterator
            separately (e.g. via :meth:`~.PassiveSonarDetector.detections_gen`'s own
            ``progress_bar``) to track per-timestep yields too.

        Yields
        ------
        tuple of (datetime, set of PassiveSonarSensorData)
            Timestamp and simulated sensor-data set for that timestep.

        Raises
        ------
        ValueError
            If fewer than two timesteps are available.
        TypeError
            If targets are present and the propagation model does not implement
            ``propagate_spectrum``.
        RuntimeError
            If target source-signal lengths are inconsistent.

        """
        all_timestamps = self._sorted_timestamps()
        if len(all_timestamps) < 2:
            msg = "Need at least 2 timesteps for broadband processing"
            raise ValueError(msg)

        ground_truth_paths = self.ground_truth_paths or []
        ref_signal_model = self.signal_models[0]
        signal_models_list = self._resolve_models(
            self.signal_models,
            len(ground_truth_paths),
            "signal models",
        )
        fs = float(ref_signal_model.sampling_rate_hz)
        num_sensors = int(self._array().num_sensors)

        frame_len = int(ref_signal_model.frame_len)
        frequencies_hz = np.fft.rfftfreq(frame_len, d=1.0 / fs)

        t0 = all_timestamps[0]
        step_times_s = np.array(
            [(ts - t0).total_seconds() for ts in all_timestamps], dtype=np.float64
        )
        n_steps = len(step_times_s)

        out_len = ref_signal_model.num_samples
        sample_times_s = np.arange(out_len, dtype=np.float64) / fs
        receiver_accum = np.zeros((num_sensors, out_len), dtype=np.complex64)

        if ground_truth_paths and not isinstance(self.propagation_model, SpectrumPropagationModel):
            msg = (
                f"{type(self.propagation_model).__name__} does not implement "
                "'propagate_spectrum', which is required for STFT-based simulation"
            )
            raise TypeError(msg)
        spectrum_propagation_model = cast(SpectrumPropagationModel, self.propagation_model)

        with tqdm(
            total=len(ground_truth_paths) * n_steps,
            disable=not progress_bar,
            desc="Propagating targets",
        ) as bar:
            for target_idx, target_path in enumerate(ground_truth_paths):
                target_signal_model = signal_models_list[target_idx]
                try:
                    target_first_state = next(iter(target_path))
                except StopIteration:
                    msg = f"ground_truth_paths[{target_idx}] has no states"
                    raise ValueError(msg) from None

                source_signal = target_signal_model.get_source_waveform(target_first_state)

                if len(source_signal) != out_len:
                    msg = (
                        "All target source signals must have the same sample length. "
                        f"Expected {out_len}, got {len(source_signal)}."
                    )
                    raise RuntimeError(msg)

                broadband_rms = np.zeros((n_steps, num_sensors), dtype=np.float64)
                sensor_delay_history_s = np.zeros((n_steps, num_sensors), dtype=np.float64)

                states_by_time = self._states_by_timestamp(target_path)
                for step_idx, timestamp in enumerate(all_timestamps):
                    platform_state = self._platform_state_at(timestamp)
                    target_state = states_by_time.get(timestamp)
                    if target_state is None:
                        bar.update(1)
                        continue

                    H_sensors, sensor_delays_s = spectrum_propagation_model.propagate_spectrum(
                        platform_state,
                        target_state,
                        frequencies_hz,
                    )

                    sensor_delay_history_s[step_idx, :] = np.asarray(
                        sensor_delays_s,
                        dtype=np.float64,
                    )

                    H_abs = np.abs(np.asarray(H_sensors, dtype=np.complex64))
                    broadband_rms[step_idx, :] = np.sqrt(np.mean(H_abs**2, axis=1)).astype(
                        np.float64
                    )
                    bar.update(1)

                sensor_delay_history_s = _hold_last_finite_delays(sensor_delay_history_s)

                for sensor_idx in range(num_sensors):
                    amp_s = np.interp(
                        sample_times_s,
                        step_times_s,
                        broadband_rms[:, sensor_idx],
                        left=0.0,
                        right=0.0,
                    )
                    tau_s = np.interp(
                        sample_times_s,
                        step_times_s,
                        sensor_delay_history_s[:, sensor_idx],
                        left=0.0,
                        right=0.0,
                    )

                    delayed = self._fractional_delay_resample(source_signal, tau_s, fs)
                    receiver_accum[sensor_idx, :] += delayed * amp_s.astype(np.float32)

        if self.fade_in_ms > 0:
            fade_samples = int(self.fade_in_ms * fs / 1000.0)
            for sensor_idx in range(num_sensors):
                receiver_accum[sensor_idx, :] = apply_fade_in(
                    receiver_accum[sensor_idx, :], fade_samples
                )

        if self.fade_out_ms > 0:
            fade_samples = int(self.fade_out_ms * fs / 1000.0)
            for sensor_idx in range(num_sensors):
                receiver_accum[sensor_idx, :] = apply_fade_out(
                    receiver_accum[sensor_idx, :],
                    fade_samples,
                )

        step_sample_idx = np.rint(step_times_s * fs).astype(int)
        step_sample_idx = np.clip(step_sample_idx, 0, out_len)
        step_sample_idx = np.maximum.accumulate(step_sample_idx)

        for step_idx, timestamp in enumerate(all_timestamps):
            start = int(step_sample_idx[step_idx])
            end = int(step_sample_idx[step_idx + 1]) if step_idx < n_steps - 1 else out_len

            sensor_signals = receiver_accum[:, start:end].copy()

            noise = self._generate_noise(
                num_sensors=num_sensors,
                num_samples=sensor_signals.shape[1],
            )
            if noise is not None:
                sensor_signals += noise

            beamformed_data, steering_bearings_rad = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
                steering_bearings_rad=steering_bearings_rad,
            )
            yield timestamp, {sensor_data}


def _lagrange_cubic(samples: ComplexArray, positions: FloatArray) -> ComplexArray:
    """Read ``samples`` at fractional ``positions`` by four-point Lagrange interpolation.

    Positions outside the samples read as zero, as ``np.interp(..., left=0, right=0)`` does.
    """
    padded = np.concatenate(([0.0], samples, [0.0, 0.0])).astype(np.complex128)
    inside = (positions >= 0.0) & (positions <= samples.size - 1)
    clipped = np.clip(positions, 0.0, samples.size - 1)
    index = np.floor(clipped).astype(np.int64)
    u = clipped - index
    # Points index-1 .. index+2 sit at padded index .. index+3.
    um1, up1, um2 = u - 1.0, u + 1.0, u - 2.0
    out = padded[index] * (-u * um1 * um2 / 6.0)
    out += padded[index + 1] * (up1 * um1 * um2 / 2.0)
    out += padded[index + 2] * (-up1 * u * um2 / 2.0)
    out += padded[index + 3] * (up1 * u * um1 / 6.0)
    out[~inside] = 0.0
    return out


@dataclass
class ChannelHistory:
    """One source's propagation channel, sampled at its own knot times.

    Attributes
    ----------
    knot_times_s : numpy.ndarray
        Knot times, shape ``(num_knots,)``, in seconds from the simulation start.
    transfer : numpy.ndarray
        Complex transfer functions, shape ``(num_knots, num_sensors, num_frequencies)``.
    delays_s : numpy.ndarray
        Per-sensor arrival delays, shape ``(num_knots, num_sensors)``, in seconds.

    """

    knot_times_s: FloatArray
    transfer: Complex64Array
    delays_s: FloatArray


class ContinuousChannelFilterPassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Broadband simulator combining a true bulk delay with per-sensor channel filters.

    Each source is rendered in two stages::

        x_ref(t) = x(t - tau_ref(t))
        y_s(t)   = crossfade over knots of (h_s,k * x_ref)(t)

    ``tau_ref`` is one delay per source, the median of its sensors' arrivals at each
    knot, applied in the time domain from an oversampled waveform. ``h_s,k`` is a short
    filter holding what remains of the channel once that delay is removed: the small
    differences between sensors and the multipath structure of ``H(f)``.

    Compared with the other continuous simulators
    ---------------------------------------------
    :class:`ContinuousSTFTPassiveSonarArraySimulator` keeps the full ``H(f)`` but applies
    the delay per frame, so its Doppler is right only below ``fs / (2 * hop)``.
    :class:`ContinuousFractionalDelayPassiveSonarArraySimulator` delays in the time domain
    but reduces ``H(f)`` to one broadband gain, which removes multipath colour. Separating
    the bulk delay from the channel filter keeps both: Doppler with no frame-rate limit and
    a frequency-dependent channel.

    Knots
    -----
    Each source's knots are the states of its ground-truth path, so a slow source needs
    few and a fast one many. The source is silent outside its path, fading over one knot
    interval at each end. Space knots closely enough that the channel changes little
    between them. It is the source's movement that matters: a shallow-water channel can
    decorrelate within about 10 m.

    Timing
    ------
    Output sample 0 is at ``start_time``. Signal models must cover
    ``duration_s + source_preroll_s``, because sound arriving at the start of the run
    left the source earlier, so ``source_preroll_s`` must exceed the longest delay.

    Output is one-sided (analytic), like the source waveforms and the other simulators'
    output: the real pressure is ``sqrt(2) * Re{y}``.

    Assumptions
    -----------
    - The channel is evaluated only inside ``frequency_band_hz``, with a raised-cosine taper
      at its edges; there is no output outside the band.
    - Every arrival at a knot lies within half a filter length of that knot's bulk delay.
    """

    signal_models: list[SourceSignal] = Property(
        doc="One source signal model per ground-truth path, or one shared by all paths",
    )
    start_time: datetime = Property(doc="Absolute time of the first output sample")
    duration_s: float = Property(doc="Length of the rendered output, in seconds")
    source_preroll_s: float = Property(
        default=10.0,
        doc="Length of waveform before start_time; must exceed the longest propagation delay",
    )
    snapshot_duration_s: float = Property(
        default=2.0,
        doc="Length of each snapshot yielded by sensor_data_gen, in seconds",
    )
    frequency_band_hz: tuple[float, float] = Property(
        default=(800.0, 1700.0),
        doc="Band the channel is computed over, in Hz; there is no output outside it",
    )
    band_taper_hz: float = Property(
        default=50.0,
        doc="Width of the raised-cosine taper inside each band edge, in Hz",
    )
    filter_length: int = Property(
        default=2048,
        doc="Channel filter length in samples; also sets the channel frequency spacing "
        "to sampling rate / filter_length",
    )
    oversampling: int = Property(
        default=4,
        doc="Oversampling of the source waveform for the bulk delay's interpolation",
    )
    delay_interpolation: str = Property(
        default="linear",
        doc="Interpolation reading the oversampled waveform at the bulk delay: 'linear', "
        "or 'cubic' (four-point Lagrange). Linear loses up to 20 log10 cos(pi f / (fs * "
        "oversampling)) dB, varying with the fractional delay; cubic is flat to about "
        "0.01 dB below a quarter of the oversampled Nyquist",
    )
    sensor_gains: FloatArray | None = Property(
        default=None,
        doc="Per-sensor linear gains applied to source signals, not to noise",
    )
    channel_histories: list[ChannelHistory] | None = Property(
        default=None,
        doc="Precomputed channel per path, from compute_channel_history; computed if None",
    )

    @property
    def _sample_rate_hz(self) -> float:
        """Sampling rate shared by the signal models, in Hz."""
        return float(self.signal_models[0].sampling_rate_hz)

    @property
    def channel_frequencies_hz(self) -> FloatArray:
        """In-band frequencies the channel is evaluated at, in Hz."""
        frequencies = np.fft.rfftfreq(int(self.filter_length), 1.0 / self._sample_rate_hz)
        low, high = self.frequency_band_hz
        return frequencies[(frequencies >= low) & (frequencies <= high)]

    def _elapsed_s(self, timestamp: datetime) -> float:
        """Convert an absolute time to seconds from ``start_time``."""
        return (timestamp - self.start_time).total_seconds()

    def compute_channel_history(self, path_index: int) -> ChannelHistory:
        """Compute one source's channel at each state of its ground-truth path.

        Parameters
        ----------
        path_index : int
            Index into ``ground_truth_paths``.

        Returns
        -------
        ChannelHistory
            The channel at each of the path's knots. It can be cached and passed back
            through ``channel_histories``, since ray tracing dominates the cost of a run.

        Raises
        ------
        TypeError
            If the propagation model cannot produce a transfer function.
        ValueError
            If the path has no states.

        """
        if not isinstance(self.propagation_model, SpectrumPropagationModel):
            raise TypeError(
                f"{type(self.propagation_model).__name__} does not implement propagate_spectrum"
            )
        states = list(self.ground_truth_paths[path_index])
        if not states:
            raise ValueError(f"ground_truth_paths[{path_index}] has no states")

        frequencies = self.channel_frequencies_hz
        num_sensors = int(self._array().num_sensors)
        transfer = np.zeros((len(states), num_sensors, frequencies.size), dtype=np.complex64)
        delays = np.full((len(states), num_sensors), np.nan, dtype=np.float64)
        for knot, state in enumerate(states):
            platform_state = self._platform_state_at(state.timestamp)
            transfer[knot], delays[knot] = self.propagation_model.propagate_spectrum(
                platform_state, state, frequencies
            )
        return ChannelHistory(
            knot_times_s=np.array([self._elapsed_s(state.timestamp) for state in states]),
            transfer=transfer,
            delays_s=delays,
        )

    def _band_taper(self, frequencies_hz: FloatArray) -> FloatArray:
        """Return the raised-cosine taper that keeps the channel filters short."""
        low, high = self.frequency_band_hz
        width = max(float(self.band_taper_hz), 1e-9)
        ramp = np.clip(np.minimum(frequencies_hz - low, high - frequencies_hz) / width, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * ramp)

    def _filter_spectra(
        self,
        transfer: ComplexArray,
        reference_delay_s: FloatArray,
    ) -> Complex64Array:
        """Turn in-band transfer functions ``(K, S, F)`` into filter spectra, delay removed.

        The spectra have all ``filter_length`` bins in ``numpy.fft.fftfreq`` order, zero at
        negative frequencies, so the filters pass a one-sided waveform unchanged in form. They
        are centred in their length so that they can be causal, and the renderer takes that
        latency back off the bulk delay.
        """
        transfer = np.asarray(transfer)
        reference_delay_s = np.asarray(reference_delay_s, dtype=np.float64).reshape(-1)
        length = int(self.filter_length)
        positive_frequencies = np.fft.rfftfreq(length, 1.0 / self._sample_rate_hz)
        frequencies = self.channel_frequencies_hz
        low_bin = int(np.searchsorted(positive_frequencies, frequencies[0]))
        taper = self._band_taper(frequencies)
        latency_s = 0.5 * length / self._sample_rate_hz

        phase = np.exp(
            2j * np.pi * frequencies[None, :] * (reference_delay_s[:, None] - latency_s)
        ).astype(np.complex64)
        spectra = np.zeros((transfer.shape[0], transfer.shape[1], length), dtype=np.complex64)
        spectra[:, :, low_bin : low_bin + frequencies.size] = transfer * phase[:, None, :] * taper
        return spectra

    def _knot_filters(self, spectra: Complex64Array) -> ComplexArray:
        """Return the impulse responses of filter spectra, along their last axis."""
        return np.fft.ifft(spectra, n=int(self.filter_length), axis=-1)

    def _render_source(
        self, model: SourceSignal, path_index: int, history: ChannelHistory
    ) -> ComplexArray:
        """Render one source at every sensor, in the units of its waveform."""
        reference_delay_s = self._reference_delays(history)
        filters = self._knot_filters(self._filter_spectra(history.transfer, reference_delay_s))
        return self._render_knots(
            model, path_index, history.knot_times_s, reference_delay_s, filters.__getitem__
        )

    @staticmethod
    def _reference_delays(history: ChannelHistory) -> FloatArray:
        """One bulk delay per knot: the median of its sensors' finite arrivals."""
        # A knot with no arrivals at all (a shadowed source) holds the nearest delay in time;
        # its zero transfer function is what silences it.
        arrivals = np.asarray(history.delays_s, dtype=np.float64)
        counted = np.where(np.isfinite(arrivals), arrivals, np.nan)
        any_arrival = np.any(np.isfinite(counted), axis=1)
        reference_delay_s = np.full(counted.shape[0], np.nan)
        reference_delay_s[any_arrival] = np.nanmedian(counted[any_arrival], axis=1)
        return _hold_last_finite_delays(reference_delay_s[:, np.newaxis])[:, 0]

    def _source_waveform(
        self, model: SourceSignal, path_index: int
    ) -> tuple[ComplexArray, int, int]:
        """Return a source's waveform, the preroll and the output length, in samples."""
        fs = self._sample_rate_hz
        first_state = next(iter(self.ground_truth_paths[path_index]))
        preroll = int(round(self.source_preroll_s * fs))
        num_output = int(round(self.duration_s * fs))
        waveform = np.asarray(model.get_source_waveform(first_state), dtype=np.complex128)
        if waveform.size < num_output + preroll:
            raise ValueError(
                f"Signal model {path_index} provides {waveform.size} samples but the run needs "
                f"{num_output + preroll}: duration_s plus source_preroll_s"
            )
        return waveform, preroll, num_output

    @staticmethod
    def _oversample(waveform: ComplexArray, factor: int) -> ComplexArray:
        """Band-limited interpolation of ``waveform`` to ``factor`` times its rate."""
        num_samples = waveform.size
        spectrum = np.fft.fft(waveform)
        half = (num_samples + 1) // 2
        padded = np.zeros(num_samples * factor, dtype=np.complex128)
        padded[:half] = spectrum[:half]
        padded[num_samples * factor - (num_samples - half) :] = spectrum[half:]
        return np.fft.ifft(padded) * factor

    def _render_knots(
        self,
        model: SourceSignal,
        path_index: int,
        knots_s: FloatArray,
        reference_delay_s: FloatArray,
        filter_at: Callable[[int], ComplexArray],
    ) -> ComplexArray:
        """Apply the bulk delay, then crossfade the knots' filters over the output.

        Parameters
        ----------
        model : SourceSignal
            The source's signal model.
        path_index : int
            Source whose waveform is rendered.
        knots_s : numpy.ndarray
            Knot times from ``start_time``, increasing.
        reference_delay_s : numpy.ndarray
            Bulk delay at each knot.
        filter_at : callable
            Returns knot ``k``'s filters, shape ``(num_sensors, filter_length)``; called once
            per knot, in order, so filters can be computed as they are needed.

        Returns
        -------
        numpy.ndarray
            The source at every sensor.

        """
        fs = self._sample_rate_hz
        waveform, preroll, num_output = self._source_waveform(model, path_index)
        knots_s = np.asarray(knots_s, dtype=np.float64)
        reference_delay_s = np.asarray(reference_delay_s, dtype=np.float64)

        length = int(self.filter_length)
        latency_samples = 0.5 * length
        if (float(np.max(reference_delay_s)) * fs - latency_samples) > preroll:
            raise ValueError(
                f"source_preroll_s {self.source_preroll_s:g} is shorter than the longest delay "
                f"{float(np.max(reference_delay_s)):.3f} s of source {path_index}"
            )

        # Bulk delay: read the oversampled waveform at a time-varying position. Oversampling
        # is what lets linear interpolation be used here: read directly at 10 kHz it would
        # lose over 1 dB at 1650 Hz between samples, varying with the fractional delay.
        factor = int(self.oversampling)
        oversampled = self._oversample(waveform, factor)

        # Read the waveform half a filter ahead, to take back the filters' centring latency.
        # Output at t leaves the filters from input at t - latency, so that input must carry
        # the bulk delay of time t, not of t - latency: evaluating it there would shift a
        # moving source by its delay rate times the latency.
        times_s = np.arange(num_output, dtype=np.float64) / fs
        delay_s = np.interp(times_s + latency_samples / fs, knots_s, reference_delay_s)
        positions = (np.arange(num_output) + preroll - delay_s * fs + latency_samples) * factor
        if self.delay_interpolation == "cubic":
            delayed = _lagrange_cubic(oversampled, positions)
        elif self.delay_interpolation == "linear":
            delayed = np.interp(
                positions, np.arange(oversampled.size), oversampled, left=0.0, right=0.0
            )
        else:
            raise ValueError(
                "delay_interpolation must be 'linear' or 'cubic', got "
                f"{self.delay_interpolation!r}"
            )
        del oversampled

        # Crossfade: each knot's filter applies over a triangle reaching its neighbours,
        # with the outer two extended by one knot interval so the source fades in and out.
        num_knots = knots_s.size
        interval_s = float(np.median(np.diff(knots_s))) if num_knots > 1 else 1.0
        edges_s = np.concatenate(([knots_s[0] - interval_s], knots_s, [knots_s[-1] + interval_s]))
        output = np.zeros((int(self._array().num_sensors), num_output), dtype=np.complex128)
        for knot in range(num_knots):
            first = max(int(np.ceil(edges_s[knot] * fs)), 0)
            last = min(int(np.floor(edges_s[knot + 2] * fs)) + 1, num_output)
            if last <= first:
                continue
            knot_filters = filter_at(knot)
            # The filter reaches back length samples, so read that much more input. Before
            # the start of the output the waveform is simply taken as zero; that region is
            # discarded along with the rest of the filter's start-up.
            lead = min(first, length)
            block = delayed[first - lead : last]
            size = scipy_fft.next_fast_len(block.size + length)
            convolved = scipy_fft.ifft(
                scipy_fft.fft(block, size) * scipy_fft.fft(knot_filters, size, axis=1),
                size,
                axis=1,
            )[:, lead : lead + last - first]
            segment_s = np.arange(first, last) / fs
            weight = np.interp(
                segment_s,
                edges_s[knot : knot + 3],
                [0.0, 1.0, 0.0],
            )
            output[:, first:last] += convolved * weight
        return output

    def _render_path(self, model: SourceSignal, path_index: int) -> ComplexArray:
        """Render one source from its precomputed or freshly computed channel history."""
        histories = self.channel_histories
        history = (
            histories[path_index]
            if histories is not None
            else self.compute_channel_history(path_index)
        )
        return self._render_source(model, path_index, history)

    def render(self, include_noise: bool = True) -> ComplexArray:
        """Render the whole run.

        Parameters
        ----------
        include_noise : bool, optional
            Whether to add the noise model's output.

        Returns
        -------
        numpy.ndarray
            One-sided sensor signals, shape ``(num_sensors, num_samples)``.

        Raises
        ------
        ValueError
            If the signal models or channel histories do not match the paths.

        """
        paths = self.ground_truth_paths or []
        models = self._resolve_models(self.signal_models, len(paths), "signal_models")
        histories = self.channel_histories
        if histories is not None and len(histories) != len(paths):
            raise ValueError(f"{len(histories)} channel histories for {len(paths)} paths")

        fs = self._output_rate_hz()
        num_output = int(round(self.duration_s * fs))
        num_sensors = int(self._array().num_sensors)
        output = np.zeros((num_sensors, num_output), dtype=np.complex128)
        for index in range(len(paths)):
            output += self._render_path(models[index], index)
        if self.sensor_gains is not None:
            output = output * np.asarray(self.sensor_gains, dtype=np.float64)[:, np.newaxis]

        if include_noise:
            noise = self._generate_noise(num_sensors, num_output)
            if noise is not None:
                output = output + noise
        return output

    def _output_rate_hz(self) -> float:
        """Return the output sampling rate: the sources', or the noise model's without any."""
        if self.ground_truth_paths or self.noise_model is None:
            return self._sample_rate_hz
        return float(self.noise_model.sampling_rate_hz)

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield contiguous snapshots of the rendered run, stamped at their centres.

        Yields
        ------
        tuple of (datetime, set of PassiveSonarSensorData)
            Timestamp and simulated sensor-data set for that snapshot.

        """
        signals = self.render()
        size = int(round(self.snapshot_duration_s * self._output_rate_hz()))
        for snapshot in range(signals.shape[1] // size):
            offset_s = (snapshot + 0.5) * self.snapshot_duration_s
            timestamp = self.start_time + timedelta(seconds=offset_s)
            sensor_signals = signals[:, snapshot * size : (snapshot + 1) * size]
            beamformed, bearings = self._beamform_if_configured(timestamp, sensor_signals)
            yield (
                timestamp,
                {self._make_sensor_data(timestamp, sensor_signals, beamformed, bearings)},
            )
