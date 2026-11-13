"""Anthropogenic signal models for sensor arrays."""

from abc import ABC
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as scipy_signal
from scipy.io import wavfile
from stonesoup.base import Property

from .._seed import _spawn_rng
from .base import ComplexArray, FloatArray, SourceSignal, _get_source_metadata

if TYPE_CHECKING:
    from stonesoup.types.state import State


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


# Source characteristics that were SyntheticAnthropogenicSignal arguments before v0.8.0 and are
# now read only from each source's metadata.
_SOURCE_METADATA_KEYS = ("tonal_bandwidth_hz", "noise_amplitude_upa", "noise_spectral_exponent")


class AnthropogenicSignal(SourceSignal, ABC):
    """Base class for anthropogenic (man-made) source models, such as ships.

    The waveform and STFT lifecycle is :class:`~bluepebble.signal.base.SourceSignal`'s.
    """


class SyntheticAnthropogenicSignal(AnthropogenicSignal):
    """Generates ship signals with broadband tonals and coloured noise.

    This signal model combines:
    1. Broadband tonals with finite bandwidth
    2. Wideband coloured noise
    3. STFT-based frequency-domain processing for efficient propagation

    What the source emits comes from the source state's metadata: the tonals
    (``amplitudes_upa``, ``frequencies_hz``, ``phases_rad``), their ``tonal_bandwidth_hz``, and
    the broadband noise, ``noise_amplitude_upa`` (0 for none) and ``noise_spectral_exponent``
    (needed only when ``noise_amplitude_upa`` is positive). ``tonal_bandwidth_hz`` is each
    tonal's bandwidth in Hz, ``noise_amplitude_upa`` the broadband noise's RMS amplitude in µPa,
    and ``noise_spectral_exponent`` its spectral slope (-1 pink, -2 red/brownian, 0 white). A
    missing key raises ``ValueError``. This model holds only how the waveform is synthesised.
    The propagation model also reads ``position_mapping`` from the metadata, to locate the
    source.

    Parameters
    ----------
    frame_len : int, optional
        STFT frame length in samples (power of 2 recommended). Default is 1024.
    hop_factor : int, optional
        Hop factor, where hop size = frame_len // hop_factor. Default is 4.
    window_type : {'hann', 'hamming', 'blackman'}, optional
        Window type for STFT. Default is 'hann'.
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

    Examples
    --------
    Merchant vessel with propeller tonals and machinery noise. The source is described in its
    state's metadata:

    >>> metadata = {
    ...     "amplitudes_upa": [10**(110/20), 10**(105/20)],
    ...     "frequencies_hz": [50.0, 100.0],
    ...     "phases_rad": [0.0, 0.0],
    ...     "tonal_bandwidth_hz": 3.0,  # Broader tonals
    ...     "noise_amplitude_upa": 10**(50/20),  # 50 dB re 1 µPa broadband noise
    ...     "noise_spectral_exponent": -1.0,  # Pink noise
    ...     "position_mapping": [0, 2, 4],
    ... }

    and the model says how to synthesise it:

    >>> signal_model = SyntheticAnthropogenicSignal(
    ...     duration_s=60.0,
    ...     sampling_rate_hz=500.0,
    ...     frame_len=500,
    ...     hop_factor=4,
    ...     noise_freq_range_hz=(30.0, 150.0)
    ... )

    """

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
        """Initialise realistic ship signal generator.

        Raises
        ------
        TypeError
            If a removed source characteristic (``tonal_bandwidth_hz``, ``noise_amplitude_upa``
            or ``noise_spectral_exponent``) is passed.

        """
        # Caught before Stone Soup's Base, whose "unexpected keyword argument" names the mistake
        # but not its replacement.
        removed = [key for key in _SOURCE_METADATA_KEYS if key in kwargs]
        if removed:
            raise TypeError(
                f"SyntheticAnthropogenicSignal no longer takes {', '.join(removed)} (removed in "
                "v0.8.0). Set them in each source's metadata instead, alongside the tonals."
            )
        super().__init__(*args, **kwargs)
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

    def _source_characteristics(self, source: "State") -> tuple[float, float, float | None]:
        """Return ``(tonal_bandwidth_hz, noise_amplitude_upa, noise_spectral_exponent)``.

        ``noise_spectral_exponent`` is ``None`` when there is no broadband noise to shape.
        """
        metadata = _get_source_metadata(source)

        def read(key: str) -> float | None:
            # Metadata is user-supplied, so float() is what validates it.
            value: Any = metadata.get(key)
            return None if value is None else float(value)

        bandwidth_hz = read("tonal_bandwidth_hz")
        amplitude_upa = read("noise_amplitude_upa")
        exponent = read("noise_spectral_exponent")
        required = {"tonal_bandwidth_hz": bandwidth_hz, "noise_amplitude_upa": amplitude_upa}
        missing = [key for key, value in required.items() if value is None]
        if amplitude_upa is not None and amplitude_upa > 0 and exponent is None:
            missing.append("noise_spectral_exponent")
        if missing:
            raise ValueError(f"Source metadata missing required keys: {', '.join(missing)}")
        amplitude_upa = cast(float, amplitude_upa)
        if not np.isfinite(amplitude_upa) or amplitude_upa < 0:
            raise ValueError("noise_amplitude_upa must be finite and >= 0")
        bandwidth_hz = self._validate_tonal_bandwidth(cast(float, bandwidth_hz))
        return bandwidth_hz, amplitude_upa, exponent

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
        bandwidth_hz: float,
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
        bandwidth_hz : float
            Bandwidth of each tonal in Hz.

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
                else self._generate_band_limited_tonal(freq, bandwidth_hz)
            )
            if self.tonal_noise_is_constant and cached is None:
                new_cache.append(base_noise)
            signal += amp * base_noise * np.exp(1j * phase)

        if self.tonal_noise_is_constant and cached is None:
            self._tonal_realizations = new_cache

        return cast(ComplexArray, signal)

    def _generate_coloured_noise(
        self, amplitude_upa: float, spectral_exponent: float
    ) -> ComplexArray:
        """Generate wideband coloured noise scaled to ``amplitude_upa``.

        Builds a power-law spectral envelope with a bandpass mask, then either
        uses it deterministically (``use_powerlaw_noise=True``) or shapes
        stochastic white noise with it.  The result is RMS-normalised and
        scaled.  The realisation is cached when ``noise_is_constant=True``.

        Parameters
        ----------
        amplitude_upa : float
            Target RMS amplitude of the output noise in µPa.
        spectral_exponent : float
            Power-law slope of the noise spectrum (-1 pink, -2 red/brownian, 0 white).

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

        spectral_shape = freq_abs ** (spectral_exponent / 2.0)
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
        bandwidth_hz, noise_amplitude_upa, noise_exponent = self._source_characteristics(source)
        signal = self._generate_tonal_signal(
            frequencies_hz, amplitudes_upa, phases_rad, bandwidth_hz
        )

        if noise_amplitude_upa > 0:
            # _source_characteristics requires an exponent whenever there is noise to shape.
            noise = self._generate_coloured_noise(noise_amplitude_upa, cast(float, noise_exponent))
            signal = signal + noise

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
        # Read before averaging channels, since the mean of integer samples is float64 and would
        # otherwise skip the integer normalisation below.
        audio_dtype = audio_array.dtype
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

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
