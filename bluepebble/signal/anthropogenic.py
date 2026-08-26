"""Anthropogenic signal models for sensor arrays."""

import warnings
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal
from scipy.io import wavfile
from stonesoup.base import Property

from .._seed import _spawn_rng
from .base import ComplexArray, Signal, _get_source_metadata
from .utils import compute_stft

if TYPE_CHECKING:
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]
CachedStftResult: TypeAlias = tuple[NDArray[np.complex64], FloatArray, int, FloatArray]


def _extract_tonal_metadata(
    source: "State",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract and validate tonal metadata from a source state.

    Parameters
    ----------
    source : State
        Source state exposing metadata containing tonal parameters.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        Tuple of ``(amplitudes_upa, frequencies_hz, phases_rad)``.

    Raises
    ------
    ValueError
        If metadata is missing, malformed, or arrays are shape-incompatible.

    """
    metadata = _get_source_metadata(source)

    required_keys = ("amplitudes_upa", "frequencies_hz", "phases_rad")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = f"Source metadata missing required keys: {missing}"
        raise ValueError(msg)

    amplitudes_upa = np.asarray(metadata["amplitudes_upa"], dtype=float)
    frequencies_hz = np.asarray(metadata["frequencies_hz"], dtype=float)
    phases_rad = np.asarray(metadata["phases_rad"], dtype=float)

    if amplitudes_upa.ndim != 1 or frequencies_hz.ndim != 1 or phases_rad.ndim != 1:
        msg = "Tonal metadata arrays must be one-dimensional"
        raise ValueError(msg)

    num_tonals = len(amplitudes_upa)
    if len(frequencies_hz) != num_tonals or len(phases_rad) != num_tonals:
        msg = (
            "Source tonal metadata arrays must have matching lengths: "
            f"len(amplitudes_upa)={len(amplitudes_upa)}, "
            f"len(frequencies_hz)={len(frequencies_hz)}, "
            f"len(phases_rad)={len(phases_rad)}"
        )
        raise ValueError(msg)

    return (
        cast(FloatArray, amplitudes_upa),
        cast(FloatArray, frequencies_hz),
        cast(FloatArray, phases_rad),
    )


class AnthropogenicSignal(Signal, ABC):
    """Base class for STFT-first anthropogenic signal models.

    Lifecycle
    ---------
    These models generate their full-duration source waveform once and cache it
    as an STFT matrix for frequency-domain propagation by
    :class:`~bluepebble.simulator.continuous.ContinuousPassiveSonarArraySimulator`.

    The expected call sequence for a single simulation run is:

    1. Call :meth:`compute_stft` once with the source state to build and cache
       the STFT.
    2. Pass the model to the simulator, which reads the cached STFT via
       :meth:`get_stft` for each frame.

    To reuse the same model instance across multiple simulation runs (e.g. with
    a different source state or after changing signal parameters), call
    :meth:`reset` before the next :meth:`compute_stft` call.  Calling
    :meth:`compute_stft` a second time without resetting raises
    :exc:`RuntimeError`.
    """

    frame_len: int = Property(default=1024, doc="STFT frame length in samples")
    hop_factor: int = Property(
        default=4,
        doc="Hop factor (hop = frame_len // hop_factor)",
    )
    window_type: str = Property(default="hann", doc="Window type for STFT")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise shared STFT caches."""
        super().__init__(*args, **kwargs)
        self._stft_cache: NDArray[np.complex64] | None = None
        self._frequencies: FloatArray | None = None
        self._hop: int | None = None
        self._window: FloatArray | None = None
        self._source_signal: ComplexArray | None = None
        self._cached_source: "State | None" = None  # noqa: UP037

    @abstractmethod
    def _generate_base_signal(self, source: "State") -> ComplexArray:
        """Generate full-duration source waveform for STFT processing."""

    def compute_stft(self, source: "State") -> CachedStftResult:
        """Compute and cache STFT outputs for the source signal.

        This method may only be called once per simulation run.  If the cache
        is already populated, :exc:`RuntimeError` is raised — call
        :meth:`reset` first to clear it before recomputing.

        Parameters
        ----------
        source : State
            Source state used by concrete implementations to build the waveform.

        Returns
        -------
        CachedStftResult
            Cached STFT tuple ``(stft, frequencies_hz, hop_samples, window)``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_stft` has already been called on this instance.
            Call :meth:`reset` to clear the cache before recomputing.

        """
        if self._stft_cache is not None:
            msg = (
                f"{type(self).__name__}.compute_stft() has already been called. "
                "Call reset() first to clear the cache before recomputing."
            )
            raise RuntimeError(msg)

        self._source_signal = self._generate_base_signal(source)

        stft, freq_normalized, hop, window = compute_stft(
            self._source_signal, self.frame_len, self.hop_factor, self.window_type
        )
        stft = np.asarray(stft, dtype=np.complex64)
        frequencies = cast(FloatArray, freq_normalized * self.sampling_rate_hz)
        window_float = np.asarray(window, dtype=np.float64)

        self._stft_cache = stft
        self._frequencies = frequencies
        self._hop = hop
        self._window = window_float
        self._cached_source = source

        return stft, frequencies, hop, window_float

    def get_stft(self) -> CachedStftResult:
        """Return cached STFT data.

        Returns
        -------
        CachedStftResult
            Cached STFT tuple ``(stft, frequencies_hz, hop_samples, window)``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_stft` has not been called yet.

        """
        if self._stft_cache is None:
            msg = "STFT not computed yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return (
            self._stft_cache,
            cast(FloatArray, self._frequencies),
            cast(int, self._hop),
            cast(FloatArray, self._window),
        )

    def get_source_signal(self) -> ComplexArray:
        """Return the cached full-duration source signal.

        Returns
        -------
        ComplexArray
            Cached source waveform.

        Raises
        ------
        RuntimeError
            If :meth:`compute_stft` has not been called yet.

        """
        if self._source_signal is None:
            msg = "Source signal not generated yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._source_signal

    def get_source_waveform(self, source: "State") -> ComplexArray:
        """Return the cached source waveform, computing it on first call.

        If the cache is already populated but ``source`` is not the same object
        that was passed to :meth:`compute_stft`, a :exc:`UserWarning` is raised.
        This guards against stale-cache bugs in multi-source simulations where
        the same model instance is accidentally reused across different targets.
        Call :meth:`reset` then :meth:`compute_stft` to update the cache.

        Parameters
        ----------
        source : State
            Source state used to generate the waveform if not yet cached.

        Returns
        -------
        ComplexArray
            Full-duration source waveform.

        """
        if self._source_signal is None:
            self.compute_stft(source)
        elif source is not self._cached_source:
            warnings.warn(
                f"{type(self).__name__}.get_source_waveform() was called with a different "
                "source state than the one used to build the cached STFT. "
                "The cached result will be returned unchanged. "
                "Call reset() then compute_stft() to recompute for the new source.",
                UserWarning,
                stacklevel=2,
            )
        return cast(ComplexArray, self._source_signal)

    def stft_geometry(self) -> tuple[int, FloatArray, int, FloatArray, int]:
        """Return STFT geometry derived purely from signal model properties.

        Returns
        -------
        tuple of (int, FloatArray, int, FloatArray, int)
            ``(num_freq_bins, frequencies_hz, hop, window, num_frames)``.
            No source State is required.

        """
        stft, freq_normalized, hop, window = compute_stft(
            np.zeros(self.num_samples, dtype=np.complex64),
            self.frame_len,
            self.hop_factor,
            self.window_type,
        )
        num_frames, num_freq_bins = stft.shape
        freqs = np.asarray(freq_normalized * self.sampling_rate_hz, dtype=np.float64)
        return num_freq_bins, freqs, int(hop), np.asarray(window, dtype=np.float64), num_frames

    def reset(self) -> None:
        """Clear the cached STFT and source-signal state.

        Call this before reusing a signal model instance across multiple
        simulation runs, or after changing signal parameters, so that the next
        call to :meth:`compute_stft` generates a fresh waveform and STFT.

        This method is safe to call even if :meth:`compute_stft` has never been
        called — it is a no-op in that case.

        Subclasses that maintain additional caches (e.g. noise realisations in
        :class:`SyntheticSignal`) override this method and call ``super().reset()``
        to ensure all state is cleared.
        """
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None
        self._cached_source = None


class SyntheticAnthropogenicSignal(AnthropogenicSignal):
    """Generates ship signals with broadband tonals and coloured noise.

    This signal model combines:
    1. Broadband tonals with finite bandwidth
    2. Wideband coloured noise
    3. STFT-based frequency-domain processing for efficient propagation

    Parameters
    ----------
    frame_len : int, optional
        STFT frame length in samples (power of 2 recommended). Default is 1024.
    hop_factor : int, optional
        Hop factor, where hop size = frame_len // hop_factor. Default is 4.
    window_type : str, optional
        Window type for STFT (e.g., 'hann'). Default is 'hann'.
    tonal_bandwidth_hz : float, optional
        Bandwidth of each tonal component in Hz. Creates realistic spectral spreading around
        nominal frequencies. Default is 2.0.
    noise_amplitude_upa : float, optional
        RMS amplitude of background noise in µPa. Set to 0.0 to disable noise. Default is 0.0.
    noise_spectral_exponent : float, optional
        Spectral shape exponent for coloured noise. -1.0 is pink noise (1/f), -2.0 is
        red/brownian noise (1/f^2), and 0.0 is white. Default is -1.0.
    noise_freq_range_hz : tuple, optional
        Tuple of (min_freq, max_freq) for noise generation. Default is (20.0, 200.0), covering
        typical machinery noise ranges.
    noise_variance : float, optional
        Variance multiplier applied to all generated white noise before any bandlimiting or
        normalisation (default 1.0). This controls the base random field variance.
    tonal_noise_is_constant : bool, optional
        If True, reuse the same band-limited tonal noise across calls; phase and amplitude are
        still applied per call. Default is False.
    use_powerlaw_noise : bool, optional
        If True, build broadband noise deterministically from the power-law spectrum (no random
        white-noise seed). Default is False.
    noise_is_constant : bool, optional
        If True, use same noise realization for all signal generations (constant scalar over time).
        If False, generate new random noise each time. Default is True.
    seed : int or None, optional
        Seed for the random number generator. When ``None`` (default), defers to the global seed
        set by ``bluepebble.set_seed()`` if called, otherwise non-deterministic. Provide an
        integer for a reproducible independent stream.

    """

    tonal_bandwidth_hz: float = Property(default=2.0, doc="Bandwidth of each tonal component (Hz)")
    noise_amplitude_upa: float = Property(
        default=0.0, doc="RMS amplitude of background noise (µPa)"
    )
    noise_spectral_exponent: float = Property(
        default=-1.0,
        doc="Spectral shape exponent (-1=pink, -2=red/brownian, 0=white)",
    )
    noise_freq_range_hz: tuple[float, float] = Property(
        default=(20.0, 200.0), doc="Frequency range for noise (Hz)"
    )
    noise_variance: float = Property(
        default=1.0,
        doc=(
            "Variance multiplier for generated white noise before shaping; std = sqrt(variance)."
        ),
    )
    tonal_noise_is_constant: bool = Property(
        default=False,
        doc=(
            "If True, reuse the same band-limited tonal noise across calls; "
            "phase and amplitude are still applied per call."
        ),
    )
    use_powerlaw_noise: bool = Property(
        default=False,
        doc=(
            "If True, build broadband noise deterministically from the power-law spectrum (no "
            "random white-noise seed)."
        ),
    )
    noise_is_constant: bool = Property(
        default=True,
        doc="If True, use same noise realization across calls; "
        "if False, generate new noise each time",
    )
    seed: int | None = Property(
        default=None,
        doc="Seed for the random number generator. ``None`` defers to the global seed set by "
        "``bluepebble.set_seed()`` if called, otherwise gives non-deterministic output; "
        "an explicit integer always produces a reproducible independent stream.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise realistic ship signal generator."""
        super().__init__(*args, **kwargs)
        self._validate_tonal_bandwidth(self.tonal_bandwidth_hz)
        self._validate_noise_variance(self.noise_variance)
        self._rng = _spawn_rng(self.seed)
        self._noise_realization: ComplexArray | None = None
        self._tonal_realizations: list[ComplexArray] | None = None

    @staticmethod
    def _validate_tonal_bandwidth(bandwidth_hz: float) -> float:
        """Validate and return tonal bandwidth in Hz."""
        bandwidth = float(bandwidth_hz)
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            msg = "tonal_bandwidth_hz must be finite and > 0"
            raise ValueError(msg)
        return bandwidth

    @staticmethod
    def _validate_noise_variance(noise_variance: float) -> float:
        """Validate and return white-noise variance multiplier."""
        variance = float(noise_variance)
        if not np.isfinite(variance) or variance < 0.0:
            msg = "noise_variance must be finite and >= 0"
            raise ValueError(msg)
        return variance

    def _generate_band_limited_tonal(self, freq_hz: float, bandwidth_hz: float) -> ComplexArray:
        """Generate a unit-RMS band-limited noise component centred on ``freq_hz``.

        Generates complex white noise, applies a symmetric Gaussian bandpass filter
        in the frequency domain, and normalises the result to unit RMS.

        Parameters
        ----------
        freq_hz : float
            Centre frequency of the tonal in Hz.
        bandwidth_hz : float
            -3 dB bandwidth of the Gaussian bandpass filter in Hz.

        Returns
        -------
        ComplexArray
            Unit-RMS band-limited complex noise of length ``num_samples``.

        """
        noise = self._rng.standard_normal(self.num_samples) + 1j * self._rng.standard_normal(
            self.num_samples
        )
        freq_bins = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)
        bandwidth = self._validate_tonal_bandwidth(bandwidth_hz)

        # Symmetric Gaussian bandpass (positive and negative frequencies).
        # sigma = bandwidth / (2 * sqrt(2 * ln(2))) ≈ bandwidth / 2.355
        sigma_hz = bandwidth / 2.355
        bandpass_filter = np.exp(-((freq_bins - freq_hz) ** 2) / (2 * sigma_hz**2))
        bandpass_filter += np.exp(-((freq_bins + freq_hz) ** 2) / (2 * sigma_hz**2))

        filtered = np.fft.ifft(np.fft.fft(noise) * bandpass_filter)
        rms = np.sqrt(np.mean(np.abs(filtered) ** 2))
        return cast(ComplexArray, filtered if rms == 0 else filtered / rms)

    def _generate_tonal_signal(
        self,
        frequencies_hz: "FloatArray",
        amplitudes_upa: "FloatArray",
        phases_rad: "FloatArray",
    ) -> ComplexArray:
        """Build the combined tonal signal, using the tonal noise cache when available.

        Parameters
        ----------
        frequencies_hz : FloatArray
            Centre frequencies of the tonals in Hz.
        amplitudes_upa : FloatArray
            Amplitude of each tonal in µPa.
        phases_rad : FloatArray
            Phase offset of each tonal in radians.

        Returns
        -------
        ComplexArray
            Combined tonal signal of length ``num_samples``.

        """
        cache_valid = (
            self.tonal_noise_is_constant
            and self._tonal_realizations is not None
            and len(self._tonal_realizations) == len(frequencies_hz)
        )
        cached = self._tonal_realizations if cache_valid else None
        new_cache: list[ComplexArray] = []

        signal = np.zeros(self.num_samples, dtype=np.complex128)
        for idx, (freq, amp, phase) in enumerate(
            zip(frequencies_hz, amplitudes_upa, phases_rad, strict=False)
        ):
            base_noise = (
                cached[idx]
                if cached is not None
                else self._generate_band_limited_tonal(freq, float(self.tonal_bandwidth_hz))
            )
            if self.tonal_noise_is_constant and cached is None:
                new_cache.append(base_noise)
            signal += amp * base_noise * np.exp(1j * phase)

        if self.tonal_noise_is_constant and cached is None:
            self._tonal_realizations = new_cache

        return cast(ComplexArray, signal)

    def _generate_coloured_noise(self, amplitude_upa: float) -> ComplexArray:
        """Generate wideband coloured noise scaled to ``amplitude_upa``.

        Builds a power-law spectral envelope with a bandpass mask, then either
        uses it deterministically (``use_powerlaw_noise=True``) or shapes
        stochastic white noise with it.  The result is RMS-normalised and
        scaled.  The realisation is cached when ``noise_is_constant=True``.

        Parameters
        ----------
        amplitude_upa : float
            Target RMS amplitude of the output noise in µPa.

        Returns
        -------
        ComplexArray
            Coloured noise of length ``num_samples``.

        """
        if self.noise_is_constant and self._noise_realization is not None:
            return self._noise_realization

        freq_bins = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)
        freq_abs = np.abs(freq_bins)
        freq_abs[freq_abs < 1.0] = 1.0  # avoid division by zero at DC

        spectral_shape = freq_abs ** (float(self.noise_spectral_exponent) / 2.0)
        freq_min, freq_max = self.noise_freq_range_hz
        bandpass = np.where((freq_abs >= freq_min) & (freq_abs <= freq_max), 1.0, 0.0)
        noise_filter = spectral_shape * bandpass

        if self.use_powerlaw_noise:
            colored_noise_fft = noise_filter
        else:
            noise_std = np.sqrt(self._validate_noise_variance(self.noise_variance))
            white_noise = noise_std * (
                self._rng.standard_normal(self.num_samples)
                + 1j * self._rng.standard_normal(self.num_samples)
            )
            colored_noise_fft = np.fft.fft(white_noise) * noise_filter

        colored_noise = np.fft.ifft(colored_noise_fft)
        rms = np.sqrt(np.mean(np.abs(colored_noise) ** 2))
        if rms > 0:
            colored_noise = colored_noise * (amplitude_upa / rms)

        result = cast(ComplexArray, colored_noise)
        if self.noise_is_constant:
            self._noise_realization = result.copy()
        return result

    def _generate_base_signal(self, source: "State") -> ComplexArray:
        """Generate the complete source signal with broadband tonals and coloured noise.

        Parameters
        ----------
        source : State
            Source state with tonal parameters in metadata.

        Returns
        -------
        ComplexArray
            Complex source signal with shape ``(num_samples,)``.

        """
        amplitudes_upa, frequencies_hz, phases_rad = _extract_tonal_metadata(source)
        signal = self._generate_tonal_signal(frequencies_hz, amplitudes_upa, phases_rad)

        noise_amplitude_upa = float(self.noise_amplitude_upa)
        if noise_amplitude_upa > 0:
            signal = cast(
                ComplexArray, signal + self._generate_coloured_noise(noise_amplitude_upa)
            )

        return signal

    def reset(self) -> None:
        """Clear cached STFT and source signal data.

        Call this when starting a new simulation with different source parameters.
        """
        super().reset()
        self._noise_realization = None
        self._tonal_realizations = None


class RecordedAnthropogenicSignal(AnthropogenicSignal):
    """Generates broadband source signals from measured WAV recordings.

    This signal model loads a measured waveform from disk, resamples it to the
    simulator sampling rate, matches the requested simulation duration, scales to
    a target RMS level in dB re 1 µPa, and then computes/caches an STFT for
    frequency-domain propagation.

    Parameters
    ----------
    wav_path : str
        Path to the measured WAV file.
    frame_len : int, optional
        STFT frame length in samples. Default is 1024.
    hop_factor : int, optional
        Hop factor, where hop size = frame_len // hop_factor. Default is 4.
    window_type : str, optional
        STFT window type. Default is "hann".
    segment_start_s : float, optional
        Start time (seconds) within the WAV to extract. Default is 0.0.
    segment_duration_s : float, optional
        Duration (seconds) to extract before duration matching. If <= 0, uses to
        the end of file.
    duration_match_mode : str, optional
        Method to match requested duration when audio is shorter than required.
        Supported values: "tile", "zero_pad". Default is "tile".
    level_db_re_1upa : float, optional
        Target RMS level of the source signal in dB re 1 µPa. Default is 85.0.

    """

    wav_path: str = Property(doc="Path to measured WAV recording")
    segment_start_s: float = Property(
        default=0.0,
        doc="Segment start time in WAV (seconds)",
    )
    segment_duration_s: float = Property(
        default=0.0,
        doc="Segment duration in WAV (seconds); <=0 uses to end of recording",
    )
    duration_match_mode: str = Property(
        default="tile",
        doc='Duration matching mode when audio is short: "tile" or "zero_pad"',
    )
    level_db_re_1upa: float = Property(
        default=85.0,
        doc="Target RMS source level in dB re 1 µPa",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise measured broadband signal generator."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def _to_float_mono(audio: ArrayLike) -> FloatArray:
        """Convert waveform to mono ``float64`` in approximately ``[-1, 1]``.

        Parameters
        ----------
        audio : ArrayLike
            Input waveform as mono or multi-channel samples.

        Returns
        -------
        FloatArray
            Mono waveform as one-dimensional ``float64`` samples.

        """
        audio_array = np.asarray(audio)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        audio_dtype = audio_array.dtype
        if np.issubdtype(audio_dtype, np.floating):
            return np.asarray(audio_array, dtype=np.float64)

        if np.issubdtype(audio_dtype, np.signedinteger):
            info = np.iinfo(audio_dtype)
            denom = max(abs(info.min), info.max)
            return np.asarray(audio_array, dtype=np.float64) / float(denom)

        if np.issubdtype(audio_dtype, np.unsignedinteger):
            info = np.iinfo(audio_dtype)
            midpoint = info.max / 2.0
            return (np.asarray(audio_array, dtype=np.float64) - midpoint) / midpoint

        return np.asarray(audio_array, dtype=np.float64)

    def _resample_to_sim_rate(
        self,
        signal: FloatArray,
        source_fs_hz: float,
    ) -> FloatArray:
        """Resample waveform to the simulator sampling rate.

        Parameters
        ----------
        signal : FloatArray
            Input mono waveform.
        source_fs_hz : float
            Source sample rate in Hz.

        Returns
        -------
        FloatArray
            Resampled mono waveform.

        """
        target_fs_hz = float(self.sampling_rate_hz)
        if np.isclose(source_fs_hz, target_fs_hz):
            return signal

        ratio = Fraction(target_fs_hz / source_fs_hz).limit_denominator(1000)
        return scipy_signal.resample_poly(signal, ratio.numerator, ratio.denominator)

    def _match_duration(self, signal: FloatArray) -> FloatArray:
        """Match waveform length to required simulation sample count.

        Parameters
        ----------
        signal : FloatArray
            Input mono waveform.

        Returns
        -------
        FloatArray
            Waveform trimmed, tiled, or padded to ``self.num_samples``.

        Raises
        ------
        ValueError
            If ``duration_match_mode`` is unsupported.

        """
        target_samples = self.num_samples

        if len(signal) >= target_samples:
            return signal[:target_samples]

        if self.duration_match_mode == "zero_pad":
            return np.pad(signal, (0, target_samples - len(signal)))

        if self.duration_match_mode == "tile":
            reps = int(np.ceil(target_samples / max(len(signal), 1)))
            return np.tile(signal, reps)[:target_samples]

        msg = (
            f"Unknown duration_match_mode: {self.duration_match_mode}. "
            "Expected 'tile' or 'zero_pad'."
        )
        raise ValueError(msg)

    def _apply_level(self, signal: FloatArray) -> FloatArray:
        """Scale waveform to target RMS level in dB re 1 µPa.

        Parameters
        ----------
        signal : FloatArray
            Input mono waveform.

        Returns
        -------
        FloatArray
            Level-adjusted mono waveform.

        """
        target_rms_upa = 10 ** (self.level_db_re_1upa / 20.0)
        current_rms = np.sqrt(np.mean(signal**2))
        if current_rms <= 0:
            return signal
        return signal * (target_rms_upa / current_rms)

    def _generate_base_signal(self, source: "State") -> ComplexArray:
        """Generate full-duration source signal from measured WAV data.

        Parameters
        ----------
        source : State
            Source state (unused placeholder for interface compatibility).

        Returns
        -------
        ComplexArray
            Complex source signal with shape ``(num_samples,)``.

        Raises
        ------
        FileNotFoundError
            If the configured WAV file does not exist.
        ValueError
            If the selected WAV segment is empty.

        """
        _ = source
        wav_file = Path(self.wav_path)
        if not wav_file.exists():
            msg = f"Measured WAV file not found: {wav_file}"
            raise FileNotFoundError(msg)

        fs_hz, audio = wavfile.read(str(wav_file))
        waveform = self._to_float_mono(audio)

        start_sample = int(max(self.segment_start_s, 0.0) * fs_hz)
        if self.segment_duration_s > 0:
            end_sample = start_sample + int(self.segment_duration_s * fs_hz)
            waveform = waveform[start_sample:end_sample]
        else:
            waveform = waveform[start_sample:]

        if len(waveform) == 0:
            msg = "Selected WAV segment is empty. Check segment_start_s and segment_duration_s."
            raise ValueError(msg)

        waveform = self._resample_to_sim_rate(waveform, fs_hz)
        waveform = self._match_duration(waveform)
        waveform = self._apply_level(waveform)

        return np.asarray(waveform, dtype=np.complex128)
