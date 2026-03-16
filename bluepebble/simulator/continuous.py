"""Continuous acoustic sensor simulators module."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property

from ..models.propagation import SpectrumPropagationModel
from ..signal.anthropogenic import Anthropogenic
from ..signal.utils import apply_fade_in, apply_fade_out, inverse_stft
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


class ContinuousSTFTPassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Continuous broadband passive-sonar simulator with selectable STFT synthesis mode.

    Use ``mode`` for channel update method. All three modes use the same
    propagation history, then
    differ in reconstruction method:

    - ``stft_interp`` and ``wola_interp`` common interpolation steps:
        1. De-rotate channel phase using per-sensor delay history (remove delay phase).
        2. Interpolate residual channel magnitude and unwrapped phase at frame-center times.
        3. Re-apply interpolated delay phase and form per-target frame spectra.

    - ``stft_interp``
        4. Sum target spectra in the frequency domain.
        5. Reconstruct sensor time series with ``inverse_stft``.

    - ``wola_interp``
        4. IFFT each frame and accumulate with weighted overlap-add (WOLA).
        5. Normalize overlap energy to produce output sensor signals.

    - ``cola``
        1. Select the nearest previous knot transfer function for each frame.
        2. Apply knot transfer function to each target source spectrum.
        3. IFFT and overlap-add frame signals.
        4. Apply COLA-style overlap normalization.

    All modes share fades, noise addition, beamforming, and timestamp chunk output.
    In practice, each method produces largely similar results. The default ``stft_interp``
    and ``wola_interp`` methods are recommended. The ``cola`` method can produce
    interference artefacts, but is good as a fast baseline.
    """

    signal_models: list[Anthropogenic] = Property(
        doc="List of broadband signal models (one per target, or single-element list for all)",
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
    def _ifft_frame(
        frame_spec: ComplexArray,
        frame_len: int,
        num_freq_bins: int,
    ) -> ComplexArray | FloatArray:
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
            If ``num_freq_bins`` does not match valid one-sided or two-sided
            STFT conventions for ``frame_len``.

        """
        if num_freq_bins == frame_len:
            return np.fft.ifft(frame_spec, n=frame_len)

        if num_freq_bins == frame_len // 2 + 1:
            return np.fft.irfft(frame_spec, n=frame_len)

        msg = (
            f"Invalid STFT bin count: {num_freq_bins}. Expected {frame_len} "
            f"(two-sided) or {frame_len // 2 + 1} (one-sided)."
        )
        raise ValueError(msg)

    def _build_common_context(
        self,
        all_timestamps: list[datetime],
        signal_models_list: "list[Anthropogenic]",
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
        num_sensors = int(self.platform.num_sensors)

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

    def _build_target_histories(
        self,
        ctx: _STFTCommonContext,
        ground_truth_paths: "list[Iterable[State]]",
        signal_models_list: "list[Anthropogenic]",
    ) -> list[_STFTTargetHistory]:
        """Build per-target source and channel histories at knot times.

        Parameters
        ----------
        ctx : _STFTCommonContext
            Shared STFT context.
        ground_truth_paths : list of Iterable of State
            Target trajectories.
        signal_models_list : list
            Resolved signal-model list aligned with targets.

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

        for target_idx, target_path in enumerate(ground_truth_paths):
            try:
                target_first_state = next(iter(target_path))
            except StopIteration:
                msg = f"ground_truth_paths[{target_idx}] has no states"
                raise ValueError(msg) from None
            target_signal_model = signal_models_list[target_idx]
            target_source_stft, target_freqs_hz, _, _ = target_signal_model.compute_stft(
                target_first_state
            )

            if target_idx == 0:
                # Reconcile ctx geometry with the actual STFT shape and frequencies.
                # stft_geometry() cannot know whether the source is real or complex
                # (which determines rfft vs. fft bin count), so we correct here.
                ctx.num_frames, ctx.num_freq_bins = target_source_stft.shape
                ctx.frequencies = np.asarray(target_freqs_hz, dtype=np.float64)
            elif target_source_stft.shape != (ctx.num_frames, ctx.num_freq_bins):
                msg = (
                    "All target source STFTs must share the same shape. "
                    f"Expected {(ctx.num_frames, ctx.num_freq_bins)}, "
                    f"got {target_source_stft.shape}."
                )
                raise RuntimeError(msg)

            H_hist = np.zeros(
                (ctx.n_steps, ctx.num_sensors, ctx.num_freq_bins), dtype=np.complex64
            )
            tau_hist = np.zeros((ctx.n_steps, ctx.num_sensors), dtype=np.float64)

            for step_idx, timestamp in enumerate(ctx.all_timestamps):
                platform_state = self.platform.get_platform_state_at(timestamp)
                target_state = self._target_state_at(target_path, timestamp)
                if target_state is None:
                    continue

                H_sensors, prop_time_s = spectrum_propagation_model.propagate_spectrum(
                    platform_state,
                    target_state,
                    ctx.frequencies,
                )
                H_hist[step_idx, :, :] = np.asarray(H_sensors, dtype=np.complex64)

                sensor_delays_s = self.propagation_model.compute_sensor_delays(
                    platform_state,
                    target_state,
                )
                tau_hist[step_idx, :] = np.asarray(prop_time_s + sensor_delays_s, dtype=np.float64)

            targets_data.append(
                _STFTTargetHistory(
                    source_stft=target_source_stft,
                    H_hist=H_hist,
                    tau_hist=tau_hist,
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
    ) -> ComplexArray:
        """Interpolate residual channel history and re-apply delay phase.

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
            tau = tau_hist[step_idx] * w0 + tau_hist[step_idx + 1] * alpha_arr
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
        tau = tau_hist[step_idx] * w0 + tau_hist[step_idx + 1] * alpha_arr
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
        receiver_signals = []

        for sensor_idx in range(ctx.num_sensors):
            stft_total = np.zeros((ctx.num_frames, ctx.num_freq_bins), dtype=np.complex64)

            for target_data in targets_data:
                H_mag_hist, H_phase_hist = self._build_residual_channel_histories(
                    H_hist=np.asarray(target_data.H_hist[:, sensor_idx : sensor_idx + 1, :]),
                    tau_hist=np.asarray(target_data.tau_hist[:, sensor_idx : sensor_idx + 1]),
                    frequencies_hz=ctx.frequencies,
                )
                H_interp = self._interpolate_channel_from_residual_histories(
                    H_mag_hist=H_mag_hist[:, 0, :],
                    H_phase_hist=H_phase_hist[:, 0, :],
                    tau_hist=np.asarray(target_data.tau_hist[:, sensor_idx], dtype=np.float64),
                    step_idx=step_idx,
                    alpha=alpha,
                    frequencies_hz=ctx.frequencies,
                )
                stft_total += target_data.source_stft * H_interp

            signal_reconstructed = inverse_stft(stft_total, ctx.frame_len, ctx.hop, ctx.window)
            receiver_signals.append(np.asarray(signal_reconstructed, dtype=np.complex64))

        max_len = max(len(sig) for sig in receiver_signals)
        padded_signals = []
        for signal in receiver_signals:
            if len(signal) < max_len:
                pad_len = max_len - len(signal)
                signal = np.concatenate([signal, np.zeros(pad_len, dtype=np.complex64)])
            padded_signals.append(signal)

        receiver = np.asarray(padded_signals, dtype=np.complex64)
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
        wola_histories = []
        for target_data in targets_data:
            H_mag_hist, H_phase_hist = self._build_residual_channel_histories(
                H_hist=target_data.H_hist,
                tau_hist=target_data.tau_hist,
                frequencies_hz=ctx.frequencies,
            )
            wola_histories.append(
                {
                    "source_stft": target_data.source_stft,
                    "tau_hist": target_data.tau_hist,
                    "H_mag_hist": H_mag_hist,
                    "H_phase_hist": H_phase_hist,
                }
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
                    source_spec = wola_data["source_stft"][frame_idx, :]
                    H_mag_hist = wola_data["H_mag_hist"][:, sensor_idx, :]
                    H_phase_hist = wola_data["H_phase_hist"][:, sensor_idx, :]
                    tau_hist = wola_data["tau_hist"][:, sensor_idx]

                    H_interp = self._interpolate_channel_from_residual_histories(
                        H_mag_hist=H_mag_hist,
                        H_phase_hist=H_phase_hist,
                        tau_hist=tau_hist,
                        step_idx=step_idx,
                        alpha=alpha,
                        frequencies_hz=ctx.frequencies,
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

        for frame_idx in range(ctx.num_frames):
            start = frame_idx * ctx.hop
            end = start + ctx.frame_len
            frame_center_s = (start + 0.5 * ctx.frame_len) / ctx.fs

            knot_idx = int(np.searchsorted(ctx.step_times_s, frame_center_s, side="right") - 1)
            knot_idx = int(np.clip(knot_idx, 0, ctx.n_steps - 1))

            norm_accum[start:end] += window_sq

            for sensor_idx in range(ctx.num_sensors):
                frame_sensor_sum = np.zeros(ctx.frame_len, dtype=np.complex64)
                for target_data in targets_data:
                    source_spec = target_data.source_stft[frame_idx, :]
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

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield continuous broadband sensor snapshots using selected STFT mode.

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
        targets_data = self._build_target_histories(ctx, ground_truth_paths, signal_models_list)

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

            beamformed_data = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
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

    Tradeoffs
    ---------
    - Strong arrival-time fidelity under fast geometry changes.
    - Lower spectral-detail fidelity than full complex frame-wise synthesis.
    """

    signal_models: list[Anthropogenic] = Property(
        doc="List of broadband signal models (one per target, or single-element list for all)",
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

        if np.iscomplexobj(source):
            y_real = np.interp(src_idx, src_n, np.real(source), left=0.0, right=0.0)
            y_imag = np.interp(src_idx, src_n, np.imag(source), left=0.0, right=0.0)
            return (y_real + 1j * y_imag).astype(np.complex64)

        y = np.interp(src_idx, src_n, source.astype(np.float64), left=0.0, right=0.0)
        return y.astype(np.complex64)

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield broadband sensor snapshots via fractional-delay rendering.

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
        num_sensors = int(self.platform.num_sensors)

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

        if ground_truth_paths:
            if not isinstance(self.propagation_model, SpectrumPropagationModel):
                msg = (
                    f"{type(self.propagation_model).__name__} does not implement "
                    "'propagate_spectrum', which is required for STFT-based simulation"
                )
                raise TypeError(msg)
            spectrum_propagation_model = cast(SpectrumPropagationModel, self.propagation_model)

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

            for step_idx, timestamp in enumerate(all_timestamps):
                platform_state = self.platform.get_platform_state_at(timestamp)
                target_state = self._target_state_at(target_path, timestamp)
                if target_state is None:
                    continue

                H_sensors, prop_time_s = spectrum_propagation_model.propagate_spectrum(
                    platform_state,
                    target_state,
                    frequencies_hz,
                )
                sensor_delays_s = self.propagation_model.compute_sensor_delays(
                    platform_state,
                    target_state,
                )

                sensor_delay_history_s[step_idx, :] = np.asarray(
                    prop_time_s + sensor_delays_s,
                    dtype=np.float64,
                )

                H_abs = np.abs(np.asarray(H_sensors, dtype=np.complex64))
                broadband_rms[step_idx, :] = np.sqrt(np.mean(H_abs**2, axis=1)).astype(np.float64)

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

            beamformed_data = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
            )
            yield timestamp, {sensor_data}
