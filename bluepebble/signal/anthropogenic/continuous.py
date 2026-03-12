"""Anthropogenic signal models for sensor arrays."""

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal
from scipy.io import wavfile
from stonesoup.base import Property

from .base import BroadbandStftSignalBase

if TYPE_CHECKING:
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]
Complex128Array: TypeAlias = NDArray[np.complex128]


def _extract_tonal_metadata(source: State) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract and validate tonal metadata from a source state."""
    metadata = getattr(source, "metadata", None)
    if metadata is None:
        msg = "Source state must define metadata for tonal synthesis"
        raise ValueError(msg)
    if not isinstance(metadata, Mapping):
        msg = "Source metadata must be mapping-like"
        raise ValueError(msg)

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


class BroadbandSyntheticSignal(BroadbandStftSignalBase):
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
        Spectral shape exponent for coloured noise. -2.0 is pink noise (1/f), -1.0 is flicker, 0.0
        is white. Default is -2.0.
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

    Examples
    --------
    Merchant vessel with propeller tonals and machinery noise:

    >>> signal_model = BroadbandSyntheticSignal(
    ...     duration_s=60.0,
    ...     sampling_rate_hz=500.0,
    ...     frame_len=500,
    ...     hop_factor=4,
    ...     tonal_bandwidth_hz=3.0,  # Broader tonals
    ...     noise_amplitude_upa=10**(50/20),  # 50 dB re 1 µPa background
    ...     noise_spectral_exponent=-2.0,  # Pink noise
    ...     noise_freq_range_hz=(30.0, 150.0)
    ... )

    """

    tonal_bandwidth_hz = Property(float, default=2.0, doc="Bandwidth of each tonal component (Hz)")
    noise_amplitude_upa = Property(
        float, default=0.0, doc="RMS amplitude of background noise (µPa)"
    )
    noise_spectral_exponent = Property(
        float, default=-2.0, doc="Spectral shape exponent (-2=pink, 0=white)"
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

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise realistic ship signal generator."""
        super().__init__(*args, **kwargs)
        self._noise_realization: Complex128Array | None = None
        self._tonal_realizations: list[Complex128Array] | None = None

    def _generate_source_signal(self, source: State) -> Complex128Array:
        """Generate the complete source signal with broadband tonals and noise.

        This method creates:
        1. Broadband tonals using band-limited white noise modulated by tonal frequencies
        2. Wideband colored noise for background machinery/cavitation sounds

        Parameters
        ----------
        source : State
            Source state with tonal parameters in metadata.

        Returns
        -------
        Complex128Array
            Complex source signal with shape ``(num_samples,)``.

        """
        amplitudes_upa, frequencies_hz, phases_rad = _extract_tonal_metadata(source)
        tonal_bandwidth_hz = float(self.tonal_bandwidth_hz)
        noise_amplitude_upa = float(self.noise_amplitude_upa)
        noise_spectral_exponent = float(self.noise_spectral_exponent)
        noise_variance = float(self.noise_variance)

        # Initialise output signal
        signal = np.zeros(self.num_samples, dtype=np.complex128)

        tonal_cache_available = (
            self.tonal_noise_is_constant
            and self._tonal_realizations is not None
            and len(self._tonal_realizations) == len(frequencies_hz)
        )
        tonal_realizations = self._tonal_realizations if tonal_cache_available else None
        tonal_cache: list[Complex128Array] = []

        # Generate broadband tonals (each tonal has finite bandwidth)
        for idx, (freq, amp, phase) in enumerate(
            zip(frequencies_hz, amplitudes_upa, phases_rad, strict=False)
        ):
            # Create narrow-band noise centered at tonal frequency
            # Bandwidth determined by tonal_bandwidth_hz
            if tonal_realizations is not None:
                base_noise = tonal_realizations[idx]
            else:
                noise_real = np.random.randn(self.num_samples)
                noise_imag = np.random.randn(self.num_samples)
                noise = noise_real + 1j * noise_imag

                # Bandpass filter: Create filter in frequency domain
                freq_bins = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)

                # Gaussian bandpass centered at tonal frequency
                # Bandwidth controls the spectral width
                # (sigma = bandwidth / 2sqrt2ln2 ~= bandwidth / 2.355)
                sigma_hz = tonal_bandwidth_hz / 2.355
                bandpass_filter = np.exp(-((freq_bins - freq) ** 2) / (2 * sigma_hz**2))
                bandpass_filter += np.exp(
                    -((freq_bins + freq) ** 2) / (2 * sigma_hz**2)
                )  # Negative freq

                # Apply filter in frequency domain
                noise_fft = np.fft.fft(noise)
                filtered_noise_fft = noise_fft * bandpass_filter
                filtered_noise = np.fft.ifft(filtered_noise_fft)

                # Normalise to unit RMS for later amplitude scaling
                rms = np.sqrt(np.mean(np.abs(filtered_noise) ** 2))
                base_noise = filtered_noise if rms == 0 else filtered_noise / rms

                if self.tonal_noise_is_constant:
                    tonal_cache.append(base_noise)

            phase_shift = np.exp(1j * phase)
            tonal_component = amp * base_noise * phase_shift

            signal += tonal_component

        if self.tonal_noise_is_constant and not tonal_cache_available:
            self._tonal_realizations = tonal_cache

        # Add wideband colored noise if amplitude > 0
        if noise_amplitude_upa > 0:
            # Check cache for constant mode
            if self.noise_is_constant and self._noise_realization is not None:
                colored_noise = self._noise_realization
            else:
                freq_bins = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)
                freq_abs = np.abs(freq_bins)
                freq_abs[freq_abs < 1.0] = 1.0  # Avoid division by zero at DC

                # Spectral envelope (power-law) and bandpass mask
                spectral_shape = freq_abs ** (noise_spectral_exponent / 2.0)
                freq_min, freq_max = self.noise_freq_range_hz
                bandpass = np.where(
                    (freq_abs >= freq_min) & (freq_abs <= freq_max),
                    1.0,
                    0.0,
                )
                noise_filter = spectral_shape * bandpass

                if self.use_powerlaw_noise:
                    # Deterministic: use the magnitude spectrum directly
                    colored_noise_fft = noise_filter
                else:
                    # Stochastic: start from white noise then shape
                    noise_std = np.sqrt(noise_variance)
                    noise_real = noise_std * np.random.randn(self.num_samples)
                    noise_imag = noise_std * np.random.randn(self.num_samples)
                    white_noise = noise_real + 1j * noise_imag
                    white_noise_fft = np.fft.fft(white_noise)
                    colored_noise_fft = white_noise_fft * noise_filter

                colored_noise = np.fft.ifft(colored_noise_fft)

                # Normalise to desired RMS amplitude
                rms = np.sqrt(np.mean(np.abs(colored_noise) ** 2))
                if rms > 0:
                    colored_noise = colored_noise * (noise_amplitude_upa / rms)

                # Cache for constant mode
                if self.noise_is_constant:
                    self._noise_realization = colored_noise.copy()

            signal += colored_noise

        return signal

    def reset(self) -> None:
        """Clear cached STFT and source signal data.

        Call this when starting a new simulation with different source parameters.
        """
        super().reset()
        self._noise_realization = None
        self._tonal_realizations = None


class BroadbandRecordedSignal(BroadbandStftSignalBase):
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

    def _generate_source_signal(self, source: State) -> Complex128Array:
        """Generate full-duration source signal from measured WAV data.

        Parameters
        ----------
        source : State
            Source state (unused placeholder for interface compatibility).

        Returns
        -------
        Complex128Array
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
